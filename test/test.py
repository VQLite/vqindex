import os, sys, time
import numpy as np
import vqindex_py

index_dir = sys.argv[1]

# index_dir, dim, storage_type[1], brute_threshold
handler = vqindex_py.init(index_dir, 128, 1, 4096)

npoint = 8000
datasets = np.random.rand(npoint, 128)
normalized_datasets = datasets / np.linalg.norm(datasets, axis=1)[:, np.newaxis]
# handler, vector list, vid list
ret = vqindex_py.add(handler, normalized_datasets.tolist(), [100]*npoint)
print("add 1:", ret)
queries = np.random.rand(1, 128)
normalized_queries = queries / np.linalg.norm(queries, axis=1)[:, np.newaxis]
ret = vqindex_py.add(handler, normalized_queries.tolist(), [101])
print("add 2:", ret)

#handler, train_type[0], nlist[0 default], nthreads
ret = vqindex_py.train(handler, 0, 0, 8)
print("train:", ret)

#handler, queres, topk, reorder_topk, nprobe
ret = vqindex_py.search(handler, normalized_queries.tolist(), 20, 40, 20)
print("search:", ret)

#handler
#ret = vqindex_py.dump(handler)
#print("dump:", ret)

#handler
stats = vqindex_py.stats(handler)
print("stats:", stats)

#handler
vqindex_py.release(handler)
