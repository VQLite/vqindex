#!/usr/bin/env python3
"""Run the ScaNN docs/example.ipynb flow against vqindex_api.

This uses the same GloVe-100 angular ANN benchmark dataset and Tree-AH shape as
the upstream ScaNN notebook, then verifies vqindex dump/load uses the packed
LUT16 artifact format.
"""

from __future__ import annotations

import argparse
import ctypes
import platform
import shutil
import time
from pathlib import Path

import h5py
import numpy as np
import requests


REPO_ROOT = Path(__file__).resolve().parents[1]
LIB_NAME = "libvqindex_api.dylib" if platform.system() == "Darwin" else "libvqindex_api.so"
LIB_PATH = REPO_ROOT / "libs" / LIB_NAME

TRAIN_TYPE_DEFAULT = 0
STORAGE_FILE = 0
INDEX_TYPE_SCANN = 0
RET_CODE_OK = 0


class IndexConfig(ctypes.Structure):
    _fields_ = [
        ("index_type_", ctypes.c_int),
        ("dim_", ctypes.c_uint32),
        ("brute_threshold_", ctypes.c_uint64),
        ("partitioning_train_sample_rate_", ctypes.c_float),
        ("hash_train_sample_rate_", ctypes.c_float),
        ("storage_type_", ctypes.c_int),
    ]


class ParamsSearch(ctypes.Structure):
    _fields_ = [
        ("topk_", ctypes.c_uint32),
        ("reorder_topk_", ctypes.c_uint32),
        ("nprobe_", ctypes.c_uint32),
    ]


class ResultSearch(ctypes.Structure):
    _fields_ = [
        ("idx_", ctypes.c_uint64),
        ("vid_", ctypes.c_int64),
        ("score_", ctypes.c_float),
    ]


class IndexStats(ctypes.Structure):
    _fields_ = [
        ("datasets_size_", ctypes.c_int64),
        ("vid_size_", ctypes.c_int64),
        ("index_size_", ctypes.c_int64),
        ("brute_threshold_", ctypes.c_int64),
        ("index_nlist_", ctypes.c_int32),
        ("dim_", ctypes.c_int32),
        ("is_brute_", ctypes.c_int8),
        ("current_status_", ctypes.c_int),
    ]


def log(message: str) -> None:
    print(message, flush=True)


def assert_ok(code: int, action: str) -> None:
    if code != RET_CODE_OK:
        raise RuntimeError(f"{action} failed with ret_code={code}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--url",
        default="http://ann-benchmarks.com/glove-100-angular.hdf5",
        help="HDF5 dataset URL from the upstream ScaNN example notebook.",
    )
    parser.add_argument(
        "--data-path",
        default="/tmp/glove-100-angular.hdf5",
        help="Where to cache the downloaded HDF5 dataset.",
    )
    parser.add_argument(
        "--index-dir",
        default="/tmp/vqindex-scann-example-index",
        help="Temporary vqindex directory to rebuild for this run.",
    )
    parser.add_argument("--num-leaves", type=int, default=2000)
    parser.add_argument("--nprobe", type=int, default=100)
    parser.add_argument("--reorder-topk", type=int, default=100)
    parser.add_argument("--alt-nprobe", type=int, default=150)
    parser.add_argument("--alt-reorder-topk", type=int, default=250)
    parser.add_argument("--topk", type=int, default=10)
    parser.add_argument("--train-threads", type=int, default=0)
    parser.add_argument("--add-batch-rows", type=int, default=50_000)
    parser.add_argument("--search-batch-rows", type=int, default=500)
    parser.add_argument(
        "--reload-queries",
        type=int,
        default=1000,
        help="Number of queries to run after reload.",
    )
    parser.add_argument(
        "--min-recall",
        type=float,
        default=0.86,
        help="Fail if the notebook-style search recall@topk is below this.",
    )
    return parser.parse_args()


def load_library() -> ctypes.CDLL:
    if not LIB_PATH.exists():
        raise FileNotFoundError(f"{LIB_PATH} is missing; run ./build.sh vqindex_api first")

    lib = ctypes.CDLL(str(LIB_PATH))
    lib.vqindex_init.argtypes = [ctypes.c_char_p, IndexConfig]
    lib.vqindex_init.restype = ctypes.c_void_p
    lib.vqindex_release.argtypes = [ctypes.c_void_p]
    lib.vqindex_release.restype = None
    lib.vqindex_add.argtypes = [
        ctypes.c_void_p,
        ctypes.POINTER(ctypes.c_float),
        ctypes.c_uint64,
        ctypes.POINTER(ctypes.c_int64),
    ]
    lib.vqindex_add.restype = ctypes.c_int
    lib.vqindex_train.argtypes = [
        ctypes.c_void_p,
        ctypes.c_int,
        ctypes.c_uint32,
        ctypes.c_int32,
    ]
    lib.vqindex_train.restype = ctypes.c_int
    lib.vqindex_search.argtypes = [
        ctypes.c_void_p,
        ctypes.POINTER(ctypes.c_float),
        ctypes.c_int,
        ctypes.POINTER(ResultSearch),
        ParamsSearch,
    ]
    lib.vqindex_search.restype = ctypes.c_int
    lib.vqindex_dump.argtypes = [ctypes.c_void_p]
    lib.vqindex_dump.restype = ctypes.c_int
    lib.vqindex_stats.argtypes = [ctypes.c_void_p]
    lib.vqindex_stats.restype = IndexStats
    return lib


def config(dim: int) -> IndexConfig:
    return IndexConfig(
        index_type_=INDEX_TYPE_SCANN,
        dim_=dim,
        brute_threshold_=0,
        partitioning_train_sample_rate_=0.2,
        hash_train_sample_rate_=0.1,
        storage_type_=STORAGE_FILE,
    )


def download_data(url: str, data_path: Path) -> None:
    if data_path.exists() and data_path.stat().st_size > 400_000_000:
        log(f"Using cached dataset: {data_path} ({data_path.stat().st_size} bytes)")
        return

    log(f"Downloading {url} -> {data_path}")
    data_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = data_path.with_suffix(data_path.suffix + ".part")
    with requests.get(url, stream=True, timeout=60) as response:
        response.raise_for_status()
        with tmp_path.open("wb") as out:
            total = 0
            for chunk in response.iter_content(chunk_size=8 * 1024 * 1024):
                if not chunk:
                    continue
                out.write(chunk)
                prev_bucket = total // (128 * 1024 * 1024)
                total += len(chunk)
                if total // (128 * 1024 * 1024) != prev_bucket:
                    log(f"  downloaded {total / (1024**2):.0f} MiB")
    tmp_path.replace(data_path)
    log(f"Download complete: {data_path.stat().st_size} bytes")


def normalize_rows(rows: np.ndarray) -> np.ndarray:
    rows = np.asarray(rows, dtype=np.float32, order="C")
    norms = np.linalg.norm(rows, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return np.ascontiguousarray(rows / norms, dtype=np.float32)


def add_dataset(
    lib: ctypes.CDLL,
    handler: ctypes.c_void_p,
    train: h5py.Dataset,
    batch_rows: int,
) -> None:
    n_points = train.shape[0]
    for start in range(0, n_points, batch_rows):
        end = min(start + batch_rows, n_points)
        chunk = normalize_rows(train[start:end])
        vids = np.arange(start, end, dtype=np.int64)
        assert_ok(
            lib.vqindex_add(
                handler,
                chunk.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                chunk.size,
                vids.ctypes.data_as(ctypes.POINTER(ctypes.c_int64)),
            ),
            f"add {start}:{end}",
        )
        if start == 0 or end == n_points or end % 200_000 == 0:
            log(f"  added {end}/{n_points}")


def compute_recall(found: np.ndarray, truth: np.ndarray) -> float:
    total = 0
    for row, gt in zip(found, truth):
        total += np.intersect1d(row, gt).shape[0]
    return total / truth.size


def search_batched(
    lib: ctypes.CDLL,
    handler: ctypes.c_void_p,
    queries: h5py.Dataset,
    *,
    topk: int,
    reorder_topk: int,
    nprobe: int,
    batch_rows: int,
    limit: int | None = None,
) -> np.ndarray:
    n_queries = min(queries.shape[0], limit) if limit is not None else queries.shape[0]
    params = ParamsSearch(topk_=topk, reorder_topk_=reorder_topk, nprobe_=nprobe)
    all_neighbors = []
    for start in range(0, n_queries, batch_rows):
        end = min(start + batch_rows, n_queries)
        q = np.ascontiguousarray(queries[start:end], dtype=np.float32)
        result = (ResultSearch * (q.shape[0] * topk))()
        assert_ok(
            lib.vqindex_search(
                handler,
                q.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                q.size,
                result,
                params,
            ),
            f"search {start}:{end}",
        )
        neighbors = np.empty((q.shape[0], topk), dtype=np.int64)
        for i in range(q.shape[0]):
            for j in range(topk):
                neighbors[i, j] = result[i * topk + j].vid_
        all_neighbors.append(neighbors)
    return np.vstack(all_neighbors)


def verify_packed_artifacts(index_dir: Path) -> None:
    index_path = index_dir / "index"
    expected = [
        "leaf_lut16_packed_dataset.npy",
        "leaf_lut16_packed_meta.npy",
        "scann_assets.pbtxt",
        "datapoint_to_token.npy",
        "int8_dataset.npy",
    ]
    for name in expected:
        path = index_path / name
        if not path.exists():
            raise AssertionError(f"missing artifact: {path}")
        log(f"artifact {name}: {path.stat().st_size} bytes")

    old_hash = index_path / "hashed_dataset.npy"
    if old_hash.exists():
        raise AssertionError(f"old expanded AH artifact should not exist: {old_hash}")
    log("artifact hashed_dataset.npy: missing as expected")


def main() -> None:
    args = parse_args()
    data_path = Path(args.data_path)
    index_dir = Path(args.index_dir)

    download_data(args.url, data_path)
    shutil.rmtree(index_dir, ignore_errors=True)

    with h5py.File(data_path, "r") as glove_h5py:
        train = glove_h5py["train"]
        queries = glove_h5py["test"]
        truth = np.asarray(glove_h5py["neighbors"][:, : args.topk], dtype=np.int64)
        log(f"dataset shape: {train.shape}")
        log(f"queries shape: {queries.shape}")

        lib = load_library()
        handler = lib.vqindex_init(str(index_dir).encode("utf-8"), config(train.shape[1]))
        if not handler:
            raise RuntimeError("vqindex_init returned null")
        try:
            t0 = time.time()
            add_dataset(lib, handler, train, args.add_batch_rows)
            log(f"add time: {time.time() - t0:.3f}s")

            t0 = time.time()
            assert_ok(
                lib.vqindex_train(
                    handler,
                    TRAIN_TYPE_DEFAULT,
                    args.num_leaves,
                    args.train_threads,
                ),
                "train",
            )
            train_time = time.time() - t0
            stats = lib.vqindex_stats(handler)
            log(
                "train time: "
                f"{train_time:.3f}s; stats index_size={stats.index_size_} "
                f"nlist={stats.index_nlist_} brute={stats.is_brute_}"
            )

            t0 = time.time()
            neighbors = search_batched(
                lib,
                handler,
                queries,
                topk=args.topk,
                reorder_topk=args.reorder_topk,
                nprobe=args.nprobe,
                batch_rows=args.search_batch_rows,
            )
            search_time = time.time() - t0
            recall = compute_recall(neighbors, truth)
            log(
                f"search recall@{args.topk} nprobe={args.nprobe} "
                f"reorder={args.reorder_topk}: {recall:.6f}; time={search_time:.3f}s"
            )
            if recall < args.min_recall:
                raise AssertionError(f"recall {recall:.6f} < minimum {args.min_recall:.6f}")

            t0 = time.time()
            neighbors = search_batched(
                lib,
                handler,
                queries,
                topk=args.topk,
                reorder_topk=args.alt_reorder_topk,
                nprobe=args.alt_nprobe,
                batch_rows=args.search_batch_rows,
            )
            log(
                f"search recall@{args.topk} nprobe={args.alt_nprobe} "
                f"reorder={args.alt_reorder_topk}: {compute_recall(neighbors, truth):.6f}; "
                f"time={time.time() - t0:.3f}s"
            )

            t0 = time.time()
            assert_ok(lib.vqindex_dump(handler), "dump")
            log(f"dump time: {time.time() - t0:.3f}s")
        finally:
            lib.vqindex_release(handler)

        verify_packed_artifacts(index_dir)

        t0 = time.time()
        reloaded = lib.vqindex_init(str(index_dir).encode("utf-8"), config(train.shape[1]))
        load_time = time.time() - t0
        if not reloaded:
            raise RuntimeError("reload vqindex_init returned null")
        try:
            stats = lib.vqindex_stats(reloaded)
            log(
                f"load time: {load_time:.3f}s; stats index_size={stats.index_size_} "
                f"nlist={stats.index_nlist_} brute={stats.is_brute_}"
            )
            reload_limit = min(args.reload_queries, queries.shape[0])
            t0 = time.time()
            neighbors = search_batched(
                lib,
                reloaded,
                queries,
                topk=args.topk,
                reorder_topk=args.reorder_topk,
                nprobe=args.nprobe,
                batch_rows=args.search_batch_rows,
                limit=reload_limit,
            )
            reload_truth = truth[:reload_limit]
            log(
                f"reload search recall@{args.topk} first{reload_limit}: "
                f"{compute_recall(neighbors, reload_truth):.6f}; time={time.time() - t0:.3f}s"
            )
        finally:
            lib.vqindex_release(reloaded)


if __name__ == "__main__":
    main()
