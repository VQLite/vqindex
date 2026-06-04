# VQLite ScaNN Integration Notes

This document describes the VQIndex fork-specific ScaNN wrapper used by
VQLite. It focuses on the C API contract, build modes, persisted artifacts,
tuning hooks, and the regression tests that should be run before changing the
wrapper.

## Scope

VQIndex embeds the current Google ScaNN C++ core and exposes a small C ABI in
`scann/scann_ops/cc/vqindex_api.h`. The ABI is intended for the VQLite Go
wrapper and supports file-backed add, train, search, dump, load, stats, tuning,
rebalance, and health stats.

The fork no longer depends on TensorFlow for the C++ core build. TensorFlow ops
from upstream ScaNN are not part of the VQLite runtime path.

## Build

Build the shared C API library:

```bash
./build.sh vqindex_api
```

The output is written to:

```text
libs/libvqindex_api.so     # Linux
libs/libvqindex_api.dylib  # macOS
libs/vqindex_api.h
```

`build.sh` requires Clang and Bazel 7.x. It searches in this order:

1. `BAZEL_BIN` if provided.
2. `.tools/bazel`, `bazel`, or `bazelisk` if they are Bazel 7 or newer.
3. Downloads Bazel into `.tools/bazel`.

Useful build environment variables:

```bash
BAZEL_BIN=/path/to/bazel
BAZEL_DOWNLOAD_VERSION=7.6.1
VQINDEX_CPU_OPT=native        # default
VQINDEX_CPU_OPT=portable
VQINDEX_HWY_TARGETS=all       # default
VQINDEX_HWY_TARGETS=static
VQINDEX_EXTRA_COPTS="-DFOO=1"
```

`VQINDEX_CPU_OPT=native` optimizes for the build machine. On x86 it uses
`-march=native` / `-mtune=native` and appends supported feature flags such as
SSE4, AVX, AVX2, FMA, F16C, BMI, AES, PCLMUL, SHA, and supported AVX512
features. On ARM/aarch64 it tries `-mcpu=native` and falls back to an ARMv8
SIMD baseline when native flags are unavailable.

`VQINDEX_HWY_TARGETS=all` defines `HWY_COMPILE_ALL_ATTAINABLE` so Highway can
compile runtime-dispatched implementations for the attainable targets. This
increases cold build time and binary size, but is the recommended performance
mode for deployment on a known machine family.

Use `portable/static` only when you need a more conservative binary. It can
reduce final search performance on machines where native vector extensions are
available.

## C API Contract

The public header is `scann/scann_ops/cc/vqindex_api.h`.

Basic lifecycle:

```c
void *handler = vqindex_init(index_dir, config);
vqindex_add(handler, vectors, vector_float_count, vids);
vqindex_train(handler, TRAIN_TYPE_DEFAULT, 0, nthreads);
vqindex_search(handler, queries, query_float_count, results,
               result_capacity, &result_count, params);
vqindex_dump(handler);
vqindex_release(handler);
```

Important API notes:

- `vqindex_add` appends vectors to the file-backed dataset. It does not update
  the trained ScaNN searcher until `vqindex_train` is called.
- `vqindex_insert` is only an alias for `vqindex_add`; it has no special
  upsert semantics.
- Live `delete` and `upsert` APIs are intentionally not exposed. VQLite should
  continue using its upper-layer tombstone logic.
- `vqindex_flush` is an alias for flushing/dumping current index artifacts.
- `vqindex_search` requires `res_capacity` and returns `res_count`. Callers
  must allocate at least `num_queries * topk` result slots.
- `vqindex_stats` returns an expanded `index_stats_t`; callers must recompile
  against the current header.
- `vqindex_last_error` exposes the last detailed C++ error for the handler.

### Breaking Changes From The Old ABI

Compared with the pre-upgrade `4d2f75e...` version:

- `vqindex_search` changed from:

```c
ret_code_t vqindex_search(void *handler, const float *queries, int len,
                          result_search_t *res, params_search_t params);
```

to:

```c
ret_code_t vqindex_search(void *handler, const float *queries, uint64_t len,
                          result_search_t *res, uint64_t res_capacity,
                          uint64_t *res_count, params_search_t params);
```

- `index_stats_t` has additional fields for pending data, timings, artifact
  format, autopilot, and SOAR.
- TensorFlow runtime libraries are no longer required beside
  `libvqindex_api.so`.

## Search Parameters

`params_search_t` controls per-call search behavior:

```c
struct params_search_s {
  uint32_t topk_;         // final number of neighbors
  uint32_t reorder_topk_; // exact reordering candidate count
  uint32_t nprobe_;       // number of leaves to search
};
```

Higher `nprobe_` generally improves recall and increases latency. Higher
`reorder_topk_` can improve top-k quality after AH scoring, but also increases
the exact reordering work. In service, check the Go-side defaults and request
parameters before attributing search latency changes to the ScaNN core.

Current wrapper defaults when a field is zero:

```text
topk          30
reorder_topk 256
nprobe       128
```

The VQLite Go layer may override these. At the time this was written,
`core.CheckSearchOpt` defaults to:

```text
topk          30
reorder_topk 128
nprobe       128
```

## Artifact Format

Index files are stored under:

```text
<index_dir>/index/
```

Brute-force indexes store `dataset.npy`.

Tree-AH indexes now prefer packed LUT16 leaf artifacts:

```text
leaf_lut16_packed_dataset.npy
leaf_lut16_packed_meta.npy
```

`leaf_lut16_packed_dataset.npy` contains the concatenated packed leaf code
bytes. `leaf_lut16_packed_meta.npy` is a `uint64` metadata table with one row
per leaf:

```text
offset, size, num_datapoints, num_blocks
```

The loader validates that metadata offsets and sizes are inside the packed data
file before constructing leaf searchers.

Legacy indexes with:

```text
hashed_dataset.npy
hashed_dataset_soar.npy
```

are still supported for loading. Load priority is:

1. If packed LUT16 files are present, load packed.
2. Otherwise load legacy hashed artifacts.

Flush/dump always writes the new packed format when the trained searcher can
export packed leaf data. After dumping a legacy loaded Tree-AH index, the old
hashed artifacts are removed in favor of the packed artifacts.

## Dump And Recovery

`vqindex_dump` and `vqindex_flush` use a staged write path:

1. ScaNN artifacts are serialized into a sibling temporary directory named
   `index.tmp.<pid>.<timestamp>`.
2. `datasets.vql` is atomically replaced when the in-memory dataset cache is
   dirty or memory-backed storage is used.
3. `vids.vql` is atomically replaced after the dataset file.
4. The temporary index directory is validated, fsynced, and then promoted to
   `<index_dir>/index/` through a backup directory named
   `index.backup.<pid>.<timestamp>`.

`vids.vql` remains required because ScaNN returns internal row ids and VQIndex
maps them back to caller-visible vids through this file. `datasets.vql` may be
deleted for immutable fully-trained segments when there are no pending vectors,
but full retrain/rebalance from raw vectors will no longer be possible until new
vectors are added and the file is recreated.

If a process is interrupted during dump, the next `vqindex_init` handles the
common recovery cases:

- If a valid `index/` still exists, stale `index.tmp.*` or `index.backup.*`
  directories are ignored and cleaned after a successful load.
- If `index/` is missing and a complete `index.tmp.*` exists, it is promoted to
  `index/`.
- If `index/` is missing and only a complete `index.backup.*` exists, the
  backup is restored.

The dump path is crash-tolerant for the wrapper's normal local filesystem use,
but it is not a multi-version transaction manager. Do not run concurrent dumps
or mutate the same index directory from multiple processes.

## Tuning API

Use `vqindex_set_tuning` before training to control wrapper-generated configs:

```c
struct index_tuning_config_s {
  uint32_t use_autopilot_;
  uint32_t enable_soar_;
  uint32_t topk_;
  uint32_t reorder_topk_;
  uint32_t nprobe_;
  uint32_t autopilot_reordering_dtype_;
  float soar_lambda_;
  float soar_overretrieve_factor_;
  uint64_t autopilot_l1_size_;
  uint64_t autopilot_l3_size_;
};
```

`use_autopilot_` asks ScaNN autopilot to generate a Tree-AH configuration from
dataset size and dimensionality. It is heuristic config generation; it does not
run a parameter sweep over recall and latency.

`autopilot_reordering_dtype_` values:

```text
0 keep default
1 FLOAT32
2 BFLOAT16
3 INT8
```

`enable_soar_` enables SOAR database spilling in the generated Tree-AH config.
SOAR can improve recall, but typically increases search work because a datapoint
may be assigned redundantly. Benchmark recall and latency together before
turning it on in production.

`vqindex_suggest_config` returns the config text that would be generated for a
given dataset size and optional `nlist`.

`vqindex_current_config` returns the currently trained or loaded ScaNN config.

## Rebalance And Health Stats

`vqindex_rebalance` retrains/rebuilds the searcher from the wrapper's persisted
dataset, optionally using caller-supplied config pbtxt. It is an offline
maintenance operation from the C API perspective; callers should serialize it
with train/dump/search at a higher layer if needed.

Health stats:

```c
vqindex_initialize_health_stats(handler);
vqindex_health_stats(handler, &stats);
```

The current stats include partition imbalance and average quantization error.
They are useful for deciding whether retraining or changing `nlist` is worth
testing.

## Testing

Run the C API contract test:

```bash
./build.sh vqindex_api
python3 test/test_go_scann_api.py
```

This covers:

- init/add/train/search/dump/load
- `vqindex_insert` alias
- search result capacity and count
- last error handling
- version/capabilities
- tuning, SOAR, config suggestion
- health stats
- packed LUT16 dump/load
- legacy hashed load fallback
- atomic dump temp/backup recovery
- corrupted packed metadata rejection

Run the upstream ScaNN notebook-style benchmark:

```bash
test/run_scann_example_benchmark.sh
```

The script builds `vqindex_api`, downloads or reuses the GloVe-100 angular ANN
benchmark dataset, trains Tree-AH, checks recall, verifies packed artifacts,
reloads the index, and searches again. If the active Python environment lacks
dependencies, it installs `uv` when needed and creates a virtual environment
under `.tools/`.

Useful options:

```bash
VQINDEX_SKIP_BUILD=1 test/run_scann_example_benchmark.sh
test/run_scann_example_benchmark.sh --variants baseline
test/run_scann_example_benchmark.sh --variants baseline,soar,autopilot,autopilot_soar
```

Before pushing ScaNN wrapper changes, run both tests. For performance-sensitive
changes, also run a fixed old-vs-new benchmark on a Linux machine using the
same `topk`, `nprobe`, `reorder_topk`, batch size, and build flags expected in
production.

## Performance Checklist

If search appears slower after an upgrade:

1. Confirm the deployed shared library was built with `VQINDEX_CPU_OPT=native`
   and `VQINDEX_HWY_TARGETS=all` when targeting a known CPU family.
2. Confirm `nprobe_` and `reorder_topk_` did not increase in the caller.
3. Confirm SOAR and autopilot were not enabled unexpectedly.
4. Measure C API search directly before blaming Go-layer queuing.
5. Check VQLite `cgoSearchPoolSize` and `GOMAXPROCS`; queueing can look like
   slower ScaNN search.
6. Compare single-query and batch-query latency separately.
7. Compare freshly trained search and dump/load search separately.

The TensorFlow removal is not expected to slow the search hot path. The C++ core
no longer links TensorFlow, and measured direct C API search on Linux showed the
current ScaNN core faster than the old TensorFlow-dependent baseline under the
same search parameters.
