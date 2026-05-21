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
import re
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

VARIANT_BASELINE = "baseline"
VARIANT_SOAR = "soar"
VARIANT_AUTOPILOT = "autopilot"
VARIANT_AUTOPILOT_SOAR = "autopilot_soar"
AVAILABLE_VARIANTS = (
    VARIANT_BASELINE,
    VARIANT_SOAR,
    VARIANT_AUTOPILOT,
    VARIANT_AUTOPILOT_SOAR,
)


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
        ("pending_size_", ctypes.c_int64),
        ("last_load_ms_", ctypes.c_int64),
        ("last_dump_ms_", ctypes.c_int64),
        ("last_train_ms_", ctypes.c_int64),
        ("last_rebalance_ms_", ctypes.c_int64),
        ("artifact_format_", ctypes.c_int),
        ("use_autopilot_", ctypes.c_int8),
        ("enable_soar_", ctypes.c_int8),
    ]


class IndexTuningConfig(ctypes.Structure):
    _fields_ = [
        ("use_autopilot_", ctypes.c_uint32),
        ("enable_soar_", ctypes.c_uint32),
        ("topk_", ctypes.c_uint32),
        ("reorder_topk_", ctypes.c_uint32),
        ("nprobe_", ctypes.c_uint32),
        ("autopilot_reordering_dtype_", ctypes.c_uint32),
        ("soar_lambda_", ctypes.c_float),
        ("soar_overretrieve_factor_", ctypes.c_float),
        ("autopilot_l1_size_", ctypes.c_uint64),
        ("autopilot_l3_size_", ctypes.c_uint64),
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
    parser.add_argument("--extra-nprobe", type=int, default=200)
    parser.add_argument("--extra-reorder-topk", type=int, default=400)
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
    parser.add_argument(
        "--variants",
        default=",".join(AVAILABLE_VARIANTS),
        help=(
            "Comma-separated variants to test. Available: "
            + ",".join(AVAILABLE_VARIANTS)
        ),
    )
    return parser.parse_args()


def parse_variants(raw_variants: str) -> list[str]:
    variants = [item.strip() for item in raw_variants.split(",") if item.strip()]
    invalid = [item for item in variants if item not in AVAILABLE_VARIANTS]
    if invalid:
        raise ValueError(
            f"unknown variants {invalid}; available variants are {list(AVAILABLE_VARIANTS)}"
        )
    if not variants:
        raise ValueError("at least one variant is required")
    return variants


def search_settings(args: argparse.Namespace) -> list[tuple[int, int]]:
    settings = [
        (args.nprobe, args.reorder_topk),
        (args.alt_nprobe, args.alt_reorder_topk),
        (args.extra_nprobe, args.extra_reorder_topk),
    ]
    deduped = []
    for nprobe, reorder_topk in settings:
        if nprobe <= 0 or reorder_topk <= 0:
            continue
        item = (nprobe, reorder_topk)
        if item not in deduped:
            deduped.append(item)
    return deduped


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
        ctypes.c_uint64,
        ctypes.POINTER(ResultSearch),
        ctypes.c_uint64,
        ctypes.POINTER(ctypes.c_uint64),
        ParamsSearch,
    ]
    lib.vqindex_search.restype = ctypes.c_int
    lib.vqindex_dump.argtypes = [ctypes.c_void_p]
    lib.vqindex_dump.restype = ctypes.c_int
    lib.vqindex_stats.argtypes = [ctypes.c_void_p]
    lib.vqindex_stats.restype = IndexStats
    lib.vqindex_set_tuning.argtypes = [ctypes.c_void_p, IndexTuningConfig]
    lib.vqindex_set_tuning.restype = ctypes.c_int
    lib.vqindex_suggest_config.argtypes = [
        ctypes.c_void_p,
        ctypes.c_uint64,
        ctypes.c_uint32,
        ctypes.c_char_p,
        ctypes.c_uint64,
    ]
    lib.vqindex_suggest_config.restype = ctypes.c_int
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


def tuning_for_variant(variant: str) -> IndexTuningConfig:
    return IndexTuningConfig(
        use_autopilot_=1 if variant in (VARIANT_AUTOPILOT, VARIANT_AUTOPILOT_SOAR) else 0,
        enable_soar_=1 if variant in (VARIANT_SOAR, VARIANT_AUTOPILOT_SOAR) else 0,
        topk_=0,
        reorder_topk_=0,
        nprobe_=0,
        autopilot_reordering_dtype_=0,
        soar_lambda_=1.5,
        soar_overretrieve_factor_=2.0,
        autopilot_l1_size_=0,
        autopilot_l3_size_=0,
    )


def suggest_config(
    lib: ctypes.CDLL,
    handler: ctypes.c_void_p,
    *,
    dataset_size: int,
    num_leaves: int,
) -> str:
    buf = ctypes.create_string_buffer(1 << 20)
    assert_ok(
        lib.vqindex_suggest_config(handler, dataset_size, num_leaves, buf, len(buf)),
        "suggest config",
    )
    return buf.value.decode("utf-8")


def config_summary(config_pbtxt: str) -> str:
    def field(pattern: str, default: str = "n/a", text: str = config_pbtxt) -> str:
        match = re.search(pattern, text, re.MULTILINE | re.DOTALL)
        return match.group(1) if match else default

    query_spilling_block = field(r"query_spilling\s*{\s*(.*?)\n\s*}", "")
    database_spilling_block = field(r"database_spilling\s*{\s*(.*?)\n\s*}", "")
    expected_samples = re.findall(r"expected_sample_size:\s*(\d+)", config_pbtxt)

    num_children = field(r"num_children:\s*(\d+)")
    query_leaves = field(r"max_spill_centers:\s*(\d+)", text=query_spilling_block)
    reorder = field(r"approx_num_neighbors:\s*(\d+)")
    part_sample = expected_samples[0] if expected_samples else "n/a"
    hash_sample = expected_samples[-1] if len(expected_samples) > 1 else "n/a"
    projection_blocks = field(r"num_blocks:\s*(\d+)")
    database_spilling = "none"
    if database_spilling_block:
        database_spilling = field(
            r"spilling_type:\s*(\w+)", default="present", text=database_spilling_block
        )
    if "brute_force" in config_pbtxt:
        return "brute_force=true"
    return (
        f"num_children={num_children} query_leaves={query_leaves} "
        f"reorder={reorder} partition_sample={part_sample} "
        f"hash_sample={hash_sample} projection_blocks={projection_blocks} "
        f"database_spilling={database_spilling}"
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
        result_count = ctypes.c_uint64(0)
        assert_ok(
            lib.vqindex_search(
                handler,
                q.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                q.size,
                result,
                q.shape[0] * topk,
                ctypes.byref(result_count),
                params,
            ),
            f"search {start}:{end}",
        )
        assert result_count.value <= q.shape[0] * topk, result_count.value
        neighbors = np.empty((q.shape[0], topk), dtype=np.int64)
        for i in range(q.shape[0]):
            for j in range(topk):
                neighbors[i, j] = result[i * topk + j].vid_
        all_neighbors.append(neighbors)
    return np.vstack(all_neighbors)


def verify_packed_artifacts(index_dir: Path, variant: str) -> dict[str, int]:
    index_path = index_dir / "index"
    expected = [
        "leaf_lut16_packed_dataset.npy",
        "leaf_lut16_packed_meta.npy",
        "scann_assets.pbtxt",
        "datapoint_to_token.npy",
        "int8_dataset.npy",
    ]
    sizes = {}
    for name in expected:
        path = index_path / name
        if not path.exists():
            raise AssertionError(f"missing artifact: {path}")
        sizes[name] = path.stat().st_size
        log(f"[{variant}] artifact {name}: {sizes[name]} bytes")

    old_hash = index_path / "hashed_dataset.npy"
    if old_hash.exists():
        raise AssertionError(f"old expanded AH artifact should not exist: {old_hash}")
    log(f"[{variant}] artifact hashed_dataset.npy: missing as expected")
    return sizes


def run_variant(
    *,
    args: argparse.Namespace,
    lib: ctypes.CDLL,
    variant: str,
    index_dir: Path,
    train: h5py.Dataset,
    queries: h5py.Dataset,
    truth: np.ndarray,
    settings: list[tuple[int, int]],
) -> dict:
    shutil.rmtree(index_dir, ignore_errors=True)
    log(f"=== variant: {variant} ===")

    handler = lib.vqindex_init(str(index_dir).encode("utf-8"), config(train.shape[1]))
    if not handler:
        raise RuntimeError(f"{variant}: vqindex_init returned null")
    try:
        tuning = tuning_for_variant(variant)
        assert_ok(lib.vqindex_set_tuning(handler, tuning), f"{variant}: set tuning")
        config_pbtxt = suggest_config(
            lib,
            handler,
            dataset_size=train.shape[0],
            num_leaves=args.num_leaves,
        )
        log(f"[{variant}] config: {config_summary(config_pbtxt)}")

        t0 = time.time()
        add_dataset(lib, handler, train, args.add_batch_rows)
        add_time = time.time() - t0
        log(f"[{variant}] add time: {add_time:.3f}s")

        t0 = time.time()
        assert_ok(
            lib.vqindex_train(
                handler,
                TRAIN_TYPE_DEFAULT,
                args.num_leaves,
                args.train_threads,
            ),
            f"{variant}: train",
        )
        train_time = time.time() - t0
        stats = lib.vqindex_stats(handler)
        log(
            f"[{variant}] train time: {train_time:.3f}s; "
            f"stats index_size={stats.index_size_} nlist={stats.index_nlist_} "
            f"brute={stats.is_brute_}"
        )

        searches = []
        for idx, (nprobe, reorder_topk) in enumerate(settings):
            t0 = time.time()
            neighbors = search_batched(
                lib,
                handler,
                queries,
                topk=args.topk,
                reorder_topk=reorder_topk,
                nprobe=nprobe,
                batch_rows=args.search_batch_rows,
            )
            search_time = time.time() - t0
            recall = compute_recall(neighbors, truth)
            searches.append(
                {
                    "nprobe": nprobe,
                    "reorder_topk": reorder_topk,
                    "recall": recall,
                    "time": search_time,
                }
            )
            log(
                f"[{variant}] search recall@{args.topk} nprobe={nprobe} "
                f"reorder={reorder_topk}: {recall:.6f}; time={search_time:.3f}s"
            )
            if idx == 0 and recall < args.min_recall:
                raise AssertionError(
                    f"{variant}: recall {recall:.6f} < minimum {args.min_recall:.6f}"
                )

        t0 = time.time()
        assert_ok(lib.vqindex_dump(handler), f"{variant}: dump")
        dump_time = time.time() - t0
        log(f"[{variant}] dump time: {dump_time:.3f}s")
    finally:
        lib.vqindex_release(handler)

    artifact_sizes = verify_packed_artifacts(index_dir, variant)

    t0 = time.time()
    reloaded = lib.vqindex_init(str(index_dir).encode("utf-8"), config(train.shape[1]))
    load_time = time.time() - t0
    if not reloaded:
        raise RuntimeError(f"{variant}: reload vqindex_init returned null")
    try:
        stats = lib.vqindex_stats(reloaded)
        log(
            f"[{variant}] load time: {load_time:.3f}s; "
            f"stats index_size={stats.index_size_} nlist={stats.index_nlist_} "
            f"brute={stats.is_brute_}"
        )
        reload_limit = min(args.reload_queries, queries.shape[0])
        t0 = time.time()
        neighbors = search_batched(
            lib,
            reloaded,
            queries,
            topk=args.topk,
            reorder_topk=settings[0][1],
            nprobe=settings[0][0],
            batch_rows=args.search_batch_rows,
            limit=reload_limit,
        )
        reload_time = time.time() - t0
        reload_truth = truth[:reload_limit]
        reload_recall = compute_recall(neighbors, reload_truth)
        log(
            f"[{variant}] reload search recall@{args.topk} first{reload_limit}: "
            f"{reload_recall:.6f}; time={reload_time:.3f}s"
        )
    finally:
        lib.vqindex_release(reloaded)

    return {
        "variant": variant,
        "add_time": add_time,
        "train_time": train_time,
        "dump_time": dump_time,
        "load_time": load_time,
        "artifact_sizes": artifact_sizes,
        "searches": searches,
        "reload_recall": reload_recall,
        "reload_time": reload_time,
    }


def log_summary(results: list[dict], settings: list[tuple[int, int]]) -> None:
    log("=== summary ===")
    for result in results:
        variant = result["variant"]
        artifact_sizes = result["artifact_sizes"]
        log(
            f"[{variant}] train={result['train_time']:.3f}s "
            f"dump={result['dump_time']:.3f}s load={result['load_time']:.3f}s "
            f"packed={artifact_sizes['leaf_lut16_packed_dataset.npy']} bytes"
        )
        for search in result["searches"]:
            log(
                f"[{variant}] nprobe={search['nprobe']} "
                f"reorder={search['reorder_topk']} recall={search['recall']:.6f} "
                f"time={search['time']:.3f}s"
            )

    baseline = next((result for result in results if result["variant"] == VARIANT_BASELINE), None)
    if baseline is None:
        return
    baseline_by_setting = {
        (item["nprobe"], item["reorder_topk"]): item for item in baseline["searches"]
    }
    for result in results:
        if result["variant"] == VARIANT_BASELINE:
            continue
        for search in result["searches"]:
            key = (search["nprobe"], search["reorder_topk"])
            base = baseline_by_setting.get(key)
            if base is None:
                continue
            log(
                f"[compare:{result['variant']}] nprobe={key[0]} reorder={key[1]} "
                f"recall_delta={search['recall'] - base['recall']:+.6f} "
                f"time_ratio={search['time'] / base['time']:.3f}x"
            )


def main() -> None:
    args = parse_args()
    variants = parse_variants(args.variants)
    settings = search_settings(args)
    if not settings:
        raise ValueError("at least one positive search setting is required")
    data_path = Path(args.data_path)
    index_root = Path(args.index_dir)

    download_data(args.url, data_path)
    shutil.rmtree(index_root, ignore_errors=True)

    with h5py.File(data_path, "r") as glove_h5py:
        train = glove_h5py["train"]
        queries = glove_h5py["test"]
        truth = np.asarray(glove_h5py["neighbors"][:, : args.topk], dtype=np.int64)
        log(f"dataset shape: {train.shape}")
        log(f"queries shape: {queries.shape}")
        log(f"variants: {','.join(variants)}")
        log(
            "search settings: "
            + ", ".join(f"nprobe={nprobe}/reorder={reorder}" for nprobe, reorder in settings)
        )

        lib = load_library()
        results = []
        multi_variant = len(variants) > 1
        for variant in variants:
            variant_index_dir = index_root / variant if multi_variant else index_root
            results.append(
                run_variant(
                    args=args,
                    lib=lib,
                    variant=variant,
                    index_dir=variant_index_dir,
                    train=train,
                    queries=queries,
                    truth=truth,
                    settings=settings,
                )
            )
        log_summary(results, settings)


if __name__ == "__main__":
    main()
