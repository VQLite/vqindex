# VQIndex

A simple vector search library based on Google ScaNN (see the upstream README
below), with a small C API (`libvqindex_api.so`) and a Python extension
(`vqindex_py`).

This is VQIndex 2: the same API on top of ScaNN 1.4.2 instead of ScaNN 1.2.9.

* No TensorFlow. The library no longer links `libtensorflow_framework.so.2`
  and nothing from TensorFlow is needed to build or run it.
* Drop-in for programs built against VQIndex 1: every function and struct of
  the old `vqindex_api.h` is unchanged (one function, `vqindex_dump_format`,
  was added), so an existing binary runs against the new `.so` without being
  rebuilt.
* Indexes dumped by VQIndex 1 load as they are and can be added to, trained
  and dumped again. Search results are the same as with VQIndex 1.
* A new on-disk form of the AH codes (`hashed_dataset_packed.bin`) that loads
  and dumps without a temporary copy of the codes. On a 92M point, 128-dim
  index: load 25-28 s instead of 56-62 s, peak memory while loading 16.8 GB
  instead of 20.5 GB, same memory afterwards.
* Fixes to locking between add / train / dump / search / stats, to vids of
  results with a negative inner product, and to child-process training.

Built and tested on Linux x86-64 (Ubuntu 24.04). The macOS build of VQIndex 1
has not been re-verified for this version.

Chinese documentation with the full details (toolchain, build options, tests,
index file formats, upgrading from VQIndex 1, troubleshooting):
[docs/vqindex.md](docs/vqindex.md).

## Build

Requires Bazel 7.x, clang (upstream targets clang 19; clang 18 also works) and
python3 (with development headers for the Python module).

```
tools/install_bazel.sh    # once; fetches Bazel 7.6.1 into tools/bazel/
./build.sh vqindex_py     # -> test/vqindex_py.so
./build.sh vqindex_api    # -> libs/libvqindex_api.so, libs/vqindex_api.h
test/run_tests.sh         # needs numpy
```

The baseline instruction set defaults to `-mavx` (upstream ScaNN uses
`-mavx -mfma`), so the library runs on any CPU with AVX; the AVX2/AVX-512
kernels are selected at runtime. Override with `VQINDEX_COPTS=...`. The script
also pads conditional jumps away from 32-byte boundaries (JCC erratum on
Skylake-family CPUs), which is worth several percent of search throughput.

See [docs/vqindex.md](docs/vqindex.md) for the toolchain, build options, tests
and troubleshooting.

## Usage

```python
# index_dir, dim, storage_type[1], brute_threshold
handler = vqindex_py.init(index_dir, 128, 1, 4096)

# handler, vector list, vid list. Only stores the vectors; they become
# searchable after the next train().
ret = vqindex_py.add(handler, normalized_datasets.tolist(), [100]*npoint)

# handler, train_type, nlist[0 default, sqrt(dataset.size)], nthreads
#   train_type 0: default, 1: new (retrain everything),
#              2: add (index the vectors added since the last train, keeping
#                 the existing partitions and codebook)
ret = vqindex_py.train(handler, 0, 0, 8)

# handler, queries, topk, reorder_topk, nprobe
ret = vqindex_py.search(handler, normalized_queries.tolist(), 16, 32, 16)

ret = vqindex_py.dump(handler)
stats = vqindex_py.stats(handler)
vqindex_py.release(handler)
```

`dump(handler, format)` selects the on-disk form of the AH codes (0 packed,
1 legacy, 2 both; see below). `test/test.py` is a minimal example and
`test/test_add.py` exercises incremental adds.

Notes:

* Vectors are expected to be L2-normalized; the score is the inner product.
  Reordering uses int8-quantized vectors, so scores are accurate to about 1e-3.
* Below `brute_threshold` points the index is searched by brute force and
  `train` does not build partitions.
* `add` only stores vectors. They are searchable after the next `train`;
  `train_type` 2 indexes them into the existing partitions and codebook.

### C API

`./build.sh vqindex_api` writes `libs/libvqindex_api.so` and
`libs/vqindex_api.h`; the header documents every call.

```c
#include "vqindex_api.h"

index_config_t config = {0};
config.index_type_ = INDEX_TYPE_SCANN;
config.dim_ = 128;
config.brute_threshold_ = 4096;
config.partitioning_train_sample_rate_ = 0.2f;
config.hash_train_sample_rate_ = 0.1f;
config.storage_type_ = STORAGE_MEMORY;
void *h = vqindex_init("/path/to/index_dir", config);

vqindex_add(h, vectors, npoint * 128, vids);       /* len counts floats */
vqindex_train(h, TRAIN_TYPE_DEFAULT, 0, 8);        /* nlist 0: default */

params_search_t params = {.topk_ = 16, .reorder_topk_ = 32, .nprobe_ = 16};
result_search_t res[16];                           /* topk_ per query */
vqindex_search(h, query, 128, res, params);

vqindex_dump(h);
vqindex_release(h);
```

Link with `-lvqindex_api`; the library depends only on libstdc++, libm and
libc. It requires the glibc and libstdc++ of the machine it was built on or
newer, so build on the oldest system it has to run on.

## Index files and upgrading from VQIndex 1

An index directory holds the AH codes in one of two equivalent files:

| file | layout | size | read by |
|---|---|---|---|
| `hashed_dataset.npy` (legacy) | one byte per code, in insertion order | 64 B/point at 128 dims | every version |
| `hashed_dataset_packed.bin` (packed) | 4 bits per code, grouped by partition, as laid out in memory | 32 B/point | VQIndex 2 |

Loading uses the packed file when the directory has one and the legacy file
otherwise. A packed file that no longer matches the rest of the directory
(for example after VQIndex 1 dumped into it) is detected and not used.

| call | writes |
|---|---|
| `vqindex_dump(h)` | the default form: packed, or the value of the environment variable `VQINDEX_DUMP_FORMAT` (0, 1 or 2) when the index was opened |
| `vqindex_dump_format(h, DUMP_FORMAT_PACKED / _LEGACY / _BOTH)` | the given form |

A dump removes the file of the form it does not write. **The default dump
therefore leaves a directory VQIndex 1 cannot load**; to keep the way back
open, dump with format 1 or 2 (or set `VQINDEX_DUMP_FORMAT`), or convert
afterwards.

`tools/vqindex_convert.py` converts a dumped index between the two forms. It
needs only Python 3 and numpy, not this library:

```
tools/vqindex_convert.py info      <index_dir>/index
tools/vqindex_convert.py to-packed <index_dir>/index [--out-dir DIR | --remove-source] [--memory-gb N]
tools/vqindex_convert.py to-legacy <index_dir>/index [--out-dir DIR | --remove-source] [--memory-gb N]
```

The source file is kept unless `--remove-source` is given. `--out-dir` writes
the converted file, and nothing else, to another directory and does not touch
the index directory (for indexes on slow or read-only network storage; copy
the file next to the other index files afterwards). The source is read
strictly sequentially, in several passes when the codes do not fit in the
memory limit `--memory-gb` (default: half of the available memory).

Upgrading a deployment:

1. Replace `libvqindex_api.so` with the new build and restart. Programs built
   against the old header do not need to be rebuilt, and
   `libtensorflow_framework.so.2` is no longer needed.
2. Existing indexes load unchanged. To get the faster load, convert them with
   `to-packed` (offline, the service does not have to be stopped for the
   conversion itself, but the index must not be dumped or trained meanwhile)
   or let the next dump write the packed form.
3. To go back to VQIndex 1, make sure `hashed_dataset.npy` exists
   (`to-legacy`, or dump format 1 / 2) before switching the library back.

One visible difference: `index_nlist_` in the stats is the configured number
of partitions; VQIndex 1 reported the number of leaves actually built.

## Changes relative to upstream ScaNN

* `scann/scann_ops/cc/vqindex_api.{h,cc}`, `vqindex_py.cc`: the VQIndex API.
  Incremental add uses ScaNN's own mutation API.
  The index is configured with `use_global_topn: False` (also forced when
  loading older indexes): with it this ScaNN scans leaves through the float
  LUT16 kernel, which on AVX2 machines with large leaves is much slower than the
  int16 kernel that ScaNN 1.2.9 used for batched search.
* `scann/utils/packed_leaves_file.{h,cc}` and its users
  (`asymmetric_hashing2/searcher`, `tree_ah_hybrid_residual`,
  `tree_x_hybrid_factory`, `scann_ops/cc/scann.cc`, `scann_assets.proto`): an
  index can keep its AH codes in `hashed_dataset_packed.bin`, already in the
  LUT16 layout and grouped by partition, instead of `hashed_dataset.npy`.
  Loading and dumping then need no unpacked copy of the codes (and dumping no
  copy of the int8 dataset). `vqindex_dump` writes this form by default;
  `vqindex_dump_format` / `VQINDEX_DUMP_FORMAT` select the legacy form or both,
  and `tools/vqindex_convert.py` converts a dumped index either way. See
  docs/vqindex.md, section 4.
* `scann/tree_x_hybrid/tree_ah_hybrid_residual.{h,cc}`,
  `scann/base/internal/tree_x_hybrid_factory.cc`: when
  `hash.asymmetric_hash.expected_sample_size` is set, the AH codebook is trained
  on the residuals of a sample of that size instead of materializing residuals
  for the whole dataset.
* `scann/oss_wrappers/scann_cpu_info.cc`, `scann/oss_wrappers/BUILD.bazel`:
  `PLATFORM_IS_X86` is defined on x86-64 builds (and the CPUID code fixed to
  compile), so the AVX2/AVX-512 kernels are picked at runtime when the CPU
  supports them. Upstream leaves this off, which pins those kernels to SSE4.

---

# ScaNN

ScaNN (Scalable Nearest Neighbors) is a method for efficient vector similarity
search at scale. This code implements [1, 2], which includes search space
pruning and quantization for Maximum Inner Product Search and also supports
other distance functions such as Euclidean distance. The implementation is
optimized for x86 processors with AVX support. ScaNN achieves state-of-the-art
performance on [ann-benchmarks.com](http://ann-benchmarks.com) as shown on the
glove-100-angular dataset below:

![glove-100-angular](https://github.com/google-research/google-research/raw/master/scann/docs/glove_bench.png)

ScaNN can be configured to fit datasets with different sizes and distributions.
It has both TensorFlow and Python APIs. The library shows strong performance
with large datasets [1, 2]. The code is released for research purposes. For more
details on the academic description of algorithms, please see below.

References:

1. ```
   @inproceedings{avq_2020,
     title={Accelerating Large-Scale Inference with Anisotropic Vector Quantization},
     author={Guo, Ruiqi and Sun, Philip and Lindgren, Erik and Geng, Quan and Simcha, David and Chern, Felix and Kumar, Sanjiv},
     booktitle={International Conference on Machine Learning},
     year={2020},
     URL={https://arxiv.org/abs/1908.10396}
   }
   ```

1. ```
   @inproceedings{soar_2023,
     title={SOAR: Improved Indexing for Approximate Nearest Neighbor Search},
     author={Sun, Philip and Simcha, David and Dopson, Dave and Guo, Ruiqi and Kumar, Sanjiv},
     booktitle={Neural Information Processing Systems},
     year={2023},
     URL={https://arxiv.org/abs/2404.00774}
   }
   ```

## Installation

`manylinux_2_27`-compatible wheels are available on
[PyPI](https://pypi.org/project/scann/):

```
pip install scann
```

ScaNN supports Linux environments running Python versions 3.9-3.13. See
[docs/releases.md](docs/releases.md) for release notes; the page also contains
download links for ScaNN wheels prior to version 1.1.0, which were not released
on PyPI. **The x86 wheels require AVX and FMA instruction set support, while
the ARM wheels require NEON**.

In accordance with the
[`manylinux_2_27` specification](https://peps.python.org/pep-0600/), ScaNN
requires `libstdc++` version 3.4.23 or above from the operating system. See
[here](https://stackoverflow.com/questions/10354636) for an example of how
to find your system's `libstdc++` version; it can generally be upgraded by
installing a newer version of `g++`.

### Using ScaNN with TensorFlow

ScaNN has optional TensorFlow op bindings that allow ScaNN's nearest neighbor
search functionality to be embedded into a TensorFlow SavedModel. **As of
ScaNN 1.4.0, this functionality is no longer enabled by default, and `pip
install scann[tf]` is now needed to enable the TensorFlow integration.**

If `scann[tf]` is installed, the TensorFlow ops can be accessed via
`scann.scann_ops`, and the API is almost identical to `scann.scann_ops_pybind`.
For users not already in the TensorFlow ecosystem, the native Python bindings
in `scann.scann_ops_pybind` are a better fit, which is why the TensorFlow ops
are an optional extra.

#### Integration with TensorFlow Serving

We provide custom Docker images of
[TF Serving](https://github.com/tensorflow/serving) that are linked to the ScaNN
TF ops. See the [`tf_serving` directory](tf_serving/README.md) for further
information.

## Usage

See the example in [docs/example.ipynb](docs/example.ipynb). For a more in-depth
explanation of ScaNN techniques, see [docs/algorithms.md](docs/algorithms.md).

## Building from source

To build ScaNN from source, first install the build tool
[bazel](https://bazel.build) (use version 7.x), Clang 19, and libstdc++ headers
for C++17 (which are provided with GCC 9). Additionally, ScaNN requires a modern
version of Python (3.9.x or later) and Tensorflow 2.20 installed on that version
of Python. Once these prerequisites are satisfied, run the following command in
the root directory of the repository:

```
python configure.py
CC=clang-19 bazel build -c opt --features=thin_lto --copt=-mavx --copt=-mfma --cxxopt="-std=c++17" --copt=-fsized-deallocation --copt=-w :build_pip_pkg
./bazel-bin/build_pip_pkg
```

To build an ARM binary from an ARM machine, the prerequisites are the same, but
the compile flags are slightly modified:

```
python configure.py
CC=clang-19 bazel build -c opt --features=thin_lto --copt=-march=armv8-a+simd --cxxopt="-std=c++17" --copt=-fsized-deallocation --copt=-w :build_pip_pkg
./bazel-bin/build_pip_pkg
```

A .whl file should appear in the root of the repository upon successful
completion of these commands. This .whl can be installed via pip.
