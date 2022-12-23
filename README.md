# VQIndex

A simple vector search library based on Google scaNN, it only support Linux and MacOS.

## Build vqindex py
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
