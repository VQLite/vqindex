# VQIndex

A simple vector search library based on the latest Google ScaNN, it only support Linux and MacOS.

This fork keeps the existing `vqindex_api.h` C ABI used by VQLite and wraps
ScaNN with add/train/search/dump/stats APIs.

## Build requirements

VQIndex builds only the ScaNN C++ core and does not require TensorFlow. Latest
ScaNN requires Bazel 7.x and Clang. `build.sh` will use `BAZEL_BIN` when set,
then a local/system Bazel 7.x, and otherwise download Bazel into `.tools/bazel`.
Override the downloaded version with `BAZEL_DOWNLOAD_VERSION` when needed.
`vqindex_py` also needs Python headers from the Python selected by
`PYTHON_BIN_PATH` or `python3`.

## Build vqindex py
centos need "-lstdc++fs"
```
./build.sh vqindex_py
```
You can use "test/test.py" to test.
```python
# index_dir, dim, storage_type[1], brute_threshold
handler = vqindex_py.init(index_dir, 128, 1, 4096)

npoint = 8000
datasets = np.random.rand(npoint, 128)
normalized_datasets = datasets / np.linalg.norm(datasets, axis=1)[:, np.newaxis]
# handler, vector list, vid list
ret = vqindex_py.add(handler, normalized_datasets.tolist(), [100]*npoint)

#handler, train_type[0], nlist[0 default, sqrt(dataset.size)], nthreads
ret = vqindex_py.train(handler, 0, 0, 8)

queries = np.random.rand(1, 128)
normalized_queries = queries / np.linalg.norm(queries, axis=1)[:, np.newaxis]
#handler, queres, topk, reorder_topk, nprobe
ret = vqindex_py.search(handler, normalized_queries.tolist(), 16, 32, 16)

#handler
#ret = vqindex_py.dump(handler)

#handler
stats = vqindex_py.stats(handler)

#handler
vqindex_py.release(handler)
```

## Build vqindex api
```
./build.sh vqindex_api
```
It will build "libs/libvqindex_api.so", you can use the "vqindex_api.h" for development.

## Test the VQLite C API contract
```
./build.sh vqindex_api
python3 test/test_go_scann_api.py
```
This test mirrors the Go wrapper in VQLite: file-backed storage, default
training, search, dump/reload, and incremental add/train/search.
