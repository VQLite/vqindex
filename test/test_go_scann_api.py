#!/usr/bin/env python3
"""Smoke-test the VQLite Go wrapper contract against libvqindex_api.

This mirrors /Users/owlwang/OSProjects/VQLite-dev/engine/go-scann/scann_index.go:
STORAGE_FILE, default brute threshold, default training, dump/reload, and
Go-style search result handling.
"""

from __future__ import annotations

import ctypes
import ast
import math
import os
import platform
import random
import shutil
import struct
import sys
import tempfile
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
LIB_NAME = "libvqindex_api.dylib" if platform.system() == "Darwin" else "libvqindex_api.so"
LIB_PATH = REPO_ROOT / "libs" / LIB_NAME

TRAIN_TYPE_DEFAULT = 0
STORAGE_FILE = 0
INDEX_TYPE_SCANN = 0

INDEX_STATE_NOINDEX = 2
INDEX_STATE_READY = 3

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


class IndexHealthStats(ctypes.Structure):
    _fields_ = [
        ("partition_weighted_avg_relative_imbalance_", ctypes.c_double),
        ("partition_avg_relative_positive_imbalance_", ctypes.c_double),
        ("avg_quantization_error_", ctypes.c_double),
        ("sum_partition_sizes_", ctypes.c_uint64),
    ]


def load_library() -> ctypes.CDLL:
    if not LIB_PATH.exists():
        raise AssertionError(f"{LIB_PATH} is missing; run ./build.sh vqindex_api first")

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
    lib.vqindex_rebalance.argtypes = [ctypes.c_void_p, ctypes.c_char_p, ctypes.c_int32]
    lib.vqindex_rebalance.restype = ctypes.c_int
    lib.vqindex_initialize_health_stats.argtypes = [ctypes.c_void_p]
    lib.vqindex_initialize_health_stats.restype = ctypes.c_int
    lib.vqindex_health_stats.argtypes = [ctypes.c_void_p, ctypes.POINTER(IndexHealthStats)]
    lib.vqindex_health_stats.restype = ctypes.c_int
    return lib


def assert_ok(code: int, action: str) -> None:
    assert code == RET_CODE_OK, f"{action} failed with ret_code={code}"


def make_config(dim: int) -> IndexConfig:
    return IndexConfig(
        index_type_=INDEX_TYPE_SCANN,
        dim_=dim,
        brute_threshold_=0,
        partitioning_train_sample_rate_=0.2,
        hash_train_sample_rate_=0.1,
        storage_type_=STORAGE_FILE,
    )


def unit_vector(rng: random.Random, dim: int) -> list[float]:
    vec = [rng.random() - 0.5 for _ in range(dim)]
    norm = math.sqrt(sum(x * x for x in vec))
    return [x / norm for x in vec]


def flatten(vectors: list[list[float]]) -> list[float]:
    return [value for vector in vectors for value in vector]


def add_vectors(
    lib: ctypes.CDLL, handler: ctypes.c_void_p, vectors: list[list[float]], vids: list[int]
) -> None:
    assert len(vectors) == len(vids)
    flat = flatten(vectors)
    data = (ctypes.c_float * len(flat))(*flat)
    vid_data = (ctypes.c_int64 * len(vids))(*vids)
    assert_ok(lib.vqindex_add(handler, data, len(flat), vid_data), "add")


def search_one(
    lib: ctypes.CDLL,
    handler: ctypes.c_void_p,
    vector: list[float],
    topk: int = 10,
    reorder_topk: int = 32,
    nprobe: int = 32,
) -> list[ResultSearch]:
    query = (ctypes.c_float * len(vector))(*vector)
    result = (ResultSearch * topk)()
    params = ParamsSearch(topk_=topk, reorder_topk_=reorder_topk, nprobe_=nprobe)
    assert_ok(lib.vqindex_search(handler, query, len(vector), result, params), "search")
    return [result[i] for i in range(topk) if result[i].score_ >= 0]


def init_index(lib: ctypes.CDLL, index_dir: Path, dim: int) -> ctypes.c_void_p:
    handler = lib.vqindex_init(str(index_dir).encode("utf-8"), make_config(dim))
    assert handler, "vqindex_init returned NULL"
    return handler


def assert_stats(stats: IndexStats, *, dataset_size: int, index_size: int, dim: int) -> None:
    assert stats.datasets_size_ == dataset_size, stats.datasets_size_
    assert stats.vid_size_ == dataset_size, stats.vid_size_
    assert stats.index_size_ == index_size, stats.index_size_
    assert stats.dim_ == dim, stats.dim_
    assert stats.brute_threshold_ == 4096, stats.brute_threshold_


def suggest_config(
    lib: ctypes.CDLL, handler: ctypes.c_void_p, dataset_size: int, nlist: int = 0
) -> str:
    buf = ctypes.create_string_buffer(1 << 20)
    assert_ok(
        lib.vqindex_suggest_config(handler, dataset_size, nlist, buf, len(buf)),
        "suggest config",
    )
    return buf.value.decode("utf-8")


def read_npy(path: Path) -> tuple[tuple[int, ...], str, bytes]:
    data = path.read_bytes()
    if data[:6] != b"\x93NUMPY":
        raise AssertionError(f"{path} is not an NPY file")
    major = data[6]
    if major == 1:
        header_len = struct.unpack_from("<H", data, 8)[0]
        offset = 10
    elif major == 2:
        header_len = struct.unpack_from("<I", data, 8)[0]
        offset = 12
    else:
        raise AssertionError(f"unsupported NPY version {major} in {path}")
    header = ast.literal_eval(data[offset : offset + header_len].decode("latin1"))
    if header.get("fortran_order"):
        raise AssertionError(f"fortran-order NPY is not supported: {path}")
    return tuple(header["shape"]), header["descr"], data[offset + header_len :]


def write_npy_u1(path: Path, shape: tuple[int, ...], payload: bytes | bytearray) -> None:
    header = f"{{'descr': '|u1', 'fortran_order': False, 'shape': {shape}, }}"
    header_len = len(header) + 1
    padding = (16 - ((10 + header_len) % 16)) % 16
    header_bytes = (header + (" " * padding) + "\n").encode("latin1")
    path.write_bytes(b"\x93NUMPY\x01\x00" + struct.pack("<H", len(header_bytes)) + header_bytes + payload)


def unpack_lut16_leaf(packed: bytes, num_datapoints: int, num_blocks: int) -> bytearray:
    unpacked = bytearray(num_datapoints * num_blocks)
    idx = 0
    for dp_block in range(num_datapoints // 32):
        out_idx = 32 * dp_block
        for dim in range(num_blocks):
            for offset in range(16):
                value = packed[idx]
                idx += 1
                unpacked[((out_idx | offset) * num_blocks) + dim] = value & 0x0F
                unpacked[((out_idx | 16 | offset) * num_blocks) + dim] = value >> 4
    if num_datapoints % 32:
        out_idx = num_datapoints - (num_datapoints % 32)
        for dim in range(num_blocks):
            for offset in range(16):
                value = packed[idx]
                idx += 1
                idx1 = out_idx | offset
                idx2 = out_idx | 16 | offset
                if idx1 < num_datapoints:
                    unpacked[(idx1 * num_blocks) + dim] = value & 0x0F
                if idx2 < num_datapoints:
                    unpacked[(idx2 * num_blocks) + dim] = value >> 4
    return unpacked


def convert_packed_artifacts_to_legacy_hashed(index_dir: Path, *, has_soar: bool) -> None:
    index_path = index_dir / "index"
    token_shape, token_descr, token_payload = read_npy(index_path / "datapoint_to_token.npy")
    if token_descr not in ("<i4", "|i4"):
        raise AssertionError(f"unexpected token dtype: {token_descr}")
    tokens = list(struct.unpack("<" + "i" * token_shape[0], token_payload))

    data_shape, data_descr, packed_payload = read_npy(
        index_path / "leaf_lut16_packed_dataset.npy"
    )
    if data_descr not in ("|u1", "<u1"):
        raise AssertionError(f"unexpected packed dtype: {data_descr}")
    meta_shape, meta_descr, meta_payload = read_npy(index_path / "leaf_lut16_packed_meta.npy")
    if meta_descr not in ("<u8", "|u8"):
        raise AssertionError(f"unexpected packed metadata dtype: {meta_descr}")
    meta = list(struct.unpack("<" + "Q" * (meta_shape[0] * meta_shape[1]), meta_payload))

    n_leaves = meta_shape[0]
    num_blocks = next(meta[4 * leaf + 3] for leaf in range(n_leaves) if meta[4 * leaf + 2] > 0)
    total_datapoints = len(tokens) // 2 if has_soar else len(tokens)
    hashed = bytearray(total_datapoints * num_blocks)
    soar_hashed = bytearray(total_datapoints * num_blocks) if has_soar else None
    datapoints_by_token: list[list[int]] = [[] for _ in range(n_leaves)]
    for idx, token in enumerate(tokens):
        if token >= 0:
            datapoints_by_token[token].append(idx)

    for leaf in range(n_leaves):
        offset, byte_size, leaf_num_datapoints, leaf_blocks = meta[4 * leaf : 4 * leaf + 4]
        indices = datapoints_by_token[leaf]
        assert len(indices) == leaf_num_datapoints, (leaf, len(indices), leaf_num_datapoints)
        if leaf_num_datapoints == 0:
            continue
        assert leaf_blocks == num_blocks, (leaf, leaf_blocks, num_blocks)
        leaf_payload = packed_payload[offset : offset + byte_size]
        unpacked = unpack_lut16_leaf(leaf_payload, leaf_num_datapoints, num_blocks)
        for local_idx, token_idx in enumerate(indices):
            global_idx = token_idx // 2 if has_soar else token_idx
            dst = global_idx * num_blocks
            src = local_idx * num_blocks
            if has_soar and token_idx % 2:
                assert soar_hashed is not None
                soar_hashed[dst : dst + num_blocks] = unpacked[src : src + num_blocks]
            else:
                hashed[dst : dst + num_blocks] = unpacked[src : src + num_blocks]

    write_npy_u1(index_path / "hashed_dataset.npy", (total_datapoints, num_blocks), hashed)
    if soar_hashed is not None:
        write_npy_u1(index_path / "hashed_dataset_soar.npy", (total_datapoints, num_blocks), soar_hashed)
    (index_path / "leaf_lut16_packed_dataset.npy").unlink()
    (index_path / "leaf_lut16_packed_meta.npy").unlink()
    assert data_shape[0] == len(packed_payload), data_shape


def main() -> None:
    dim = 32
    base_count = 9000
    target_vid = 424242
    added_vid = 424243

    rng = random.Random(20260515)
    base_vectors = [unit_vector(rng, dim) for _ in range(base_count)]
    base_vids = [100000 + i for i in range(base_count)]
    target_vector = unit_vector(rng, dim)
    added_vector = unit_vector(rng, dim)

    lib = load_library()
    index_dir = Path(tempfile.mkdtemp(prefix="vqindex-go-scann-api-"))
    keep_dir = os.environ.get("KEEP_VQINDEX_TEST_DIR")
    handler = None

    try:
        handler = init_index(lib, index_dir, dim)
        stats = lib.vqindex_stats(handler)
        assert_stats(stats, dataset_size=0, index_size=0, dim=dim)
        assert stats.current_status_ == INDEX_STATE_NOINDEX, stats.current_status_

        add_vectors(lib, handler, base_vectors, base_vids)
        add_vectors(lib, handler, [target_vector], [target_vid])
        stats = lib.vqindex_stats(handler)
        assert_stats(stats, dataset_size=base_count + 1, index_size=0, dim=dim)

        assert_ok(lib.vqindex_train(handler, TRAIN_TYPE_DEFAULT, 0, 8), "initial train")
        stats = lib.vqindex_stats(handler)
        assert_stats(stats, dataset_size=base_count + 1, index_size=base_count + 1, dim=dim)
        assert stats.current_status_ == INDEX_STATE_READY, stats.current_status_
        assert stats.index_nlist_ > 0, stats.index_nlist_
        assert stats.is_brute_ == 0, stats.is_brute_

        results = search_one(lib, handler, target_vector)
        assert results, "expected at least one search result"
        assert results[0].vid_ == target_vid, results[0].vid_
        assert results[0].score_ > 0.99, results[0].score_

        assert_ok(lib.vqindex_initialize_health_stats(handler), "initialize health stats")
        health = IndexHealthStats()
        assert_ok(lib.vqindex_health_stats(handler, ctypes.byref(health)), "health stats")
        assert health.sum_partition_sizes_ == base_count + 1, health.sum_partition_sizes_

        tuning = IndexTuningConfig(
            use_autopilot_=1,
            enable_soar_=1,
            topk_=10,
            reorder_topk_=64,
            nprobe_=32,
            autopilot_reordering_dtype_=3,
            soar_lambda_=1.5,
            soar_overretrieve_factor_=2.0,
            autopilot_l1_size_=256,
            autopilot_l3_size_=1 << 20,
        )
        assert_ok(lib.vqindex_set_tuning(handler, tuning), "set tuning")
        tuned_config = suggest_config(lib, handler, base_count + 1, stats.index_nlist_)
        assert "autopilot" not in tuned_config, tuned_config
        assert "database_spilling" in tuned_config, tuned_config
        assert "overretrieve_factor" in tuned_config, tuned_config
        assert_ok(
            lib.vqindex_rebalance(handler, tuned_config.encode("utf-8"), 8),
            "rebalance",
        )
        stats = lib.vqindex_stats(handler)
        assert_stats(stats, dataset_size=base_count + 1, index_size=base_count + 1, dim=dim)
        results = search_one(lib, handler, target_vector, topk=10, reorder_topk=64, nprobe=32)
        assert results[0].vid_ == target_vid, results[0].vid_

        assert_ok(lib.vqindex_dump(handler), "dump")
        assert (index_dir / "datasets.vql").exists()
        assert (index_dir / "vids.vql").exists()
        assert (index_dir / "index" / "scann_config.pb").exists()
        assert (index_dir / "index" / "scann_assets.pbtxt").exists()
        assert (index_dir / "index" / "leaf_lut16_packed_dataset.npy").exists()
        assert (index_dir / "index" / "leaf_lut16_packed_meta.npy").exists()
        assert not (index_dir / "index" / "hashed_dataset.npy").exists()
        convert_packed_artifacts_to_legacy_hashed(index_dir, has_soar=True)
        assert (index_dir / "index" / "hashed_dataset.npy").exists()
        assert (index_dir / "index" / "hashed_dataset_soar.npy").exists()
        assert not (index_dir / "index" / "leaf_lut16_packed_dataset.npy").exists()
        assert not (index_dir / "index" / "leaf_lut16_packed_meta.npy").exists()

        lib.vqindex_release(handler)
        handler = None
        handler = init_index(lib, index_dir, dim)
        stats = lib.vqindex_stats(handler)
        assert_stats(stats, dataset_size=base_count + 1, index_size=base_count + 1, dim=dim)
        assert stats.current_status_ == INDEX_STATE_READY, stats.current_status_
        results = search_one(lib, handler, target_vector)
        assert results[0].vid_ == target_vid, results[0].vid_
        assert_ok(lib.vqindex_dump(handler), "dump after legacy load")
        assert (index_dir / "index" / "leaf_lut16_packed_dataset.npy").exists()
        assert (index_dir / "index" / "leaf_lut16_packed_meta.npy").exists()
        assert not (index_dir / "index" / "hashed_dataset.npy").exists()
        assert not (index_dir / "index" / "hashed_dataset_soar.npy").exists()

        add_vectors(lib, handler, [added_vector], [added_vid])
        stats = lib.vqindex_stats(handler)
        assert_stats(stats, dataset_size=base_count + 2, index_size=base_count + 1, dim=dim)
        assert_ok(lib.vqindex_train(handler, TRAIN_TYPE_DEFAULT, 0, 8), "incremental train")
        stats = lib.vqindex_stats(handler)
        assert_stats(stats, dataset_size=base_count + 2, index_size=base_count + 2, dim=dim)

        results = search_one(lib, handler, added_vector)
        assert results, "expected incremental search result"
        assert results[0].vid_ == added_vid, results[0].vid_
        assert results[0].score_ > 0.99, results[0].score_

        assert_ok(lib.vqindex_dump(handler), "dump after incremental train")
        print(f"ok: Go ScaNN C API contract passed in {index_dir}")
    finally:
        if handler:
            lib.vqindex_release(handler)
        if not keep_dir:
            shutil.rmtree(index_dir, ignore_errors=True)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"FAIL: {exc}", file=sys.stderr)
        raise
