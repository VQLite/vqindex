"""Incremental-add checks: odd batch sizes, dump/reload consistency, FILE storage, fork training.

Usage: python test_add.py <dir containing vqindex_py.so> <scratch dir>
"""
import os, shutil, sys, time
import numpy as np
sys.path.insert(0, sys.argv[1]); import vqindex_py as vq
BASE = sys.argv[2]; DIM = 128; rng = np.random.default_rng(3); FAILS = []
def check(name, ok, detail=""):
    print(f"[{'PASS' if ok else 'FAIL'}] {name} {detail}", flush=True)
    if not ok: FAILS.append(name)
norm = lambda x: (x / np.linalg.norm(x, axis=1, keepdims=True)).astype(np.float32)
cent = rng.standard_normal((300, DIM))
def clustered(n): return norm(cent[rng.integers(0, 300, n)] + 0.35 * rng.standard_normal((n, DIM)))
def fresh(name):
    d = os.path.join(BASE, name); shutil.rmtree(d, ignore_errors=True); os.makedirs(d); return d
def search(h, q, k, reorder, nprobe):
    r = vq.search(h, q.tolist(), k, reorder, nprobe); return None if r is None else r[0]
def selfhit(h, data, ids, nprobe=64):
    r = search(h, data[ids], 1, 100, nprobe)
    return float(np.mean([row[0]['idx'] == i and row[0]['vid'] == i + 7 for row, i in zip(r, ids)]))
def recall(h, data, q, nprobe=64, reorder=200, k=10):
    gt = np.argsort(-(q @ data.T), axis=1)[:, :k]
    r = search(h, q, k, reorder, nprobe)
    return sum(len(set(x['idx'] for x in row) & set(g.tolist())) for row, g in zip(r, gt)) / gt.size
def ids_of(res): return [[(x['idx'], x['vid'], round(x['score'], 5)) for x in row] for row in res]

N = 200_000
PIECES = [1, 31, 32, 33, 1, 1000, 4095, 7, 64, 20_000, 3]   # exercises LUT16 32-point block boundaries
M = sum(PIECES)
data = clustered(N + M); vids = (np.arange(N + M) + 7).tolist()

for storage, sname in ((1, "MEMORY"), (0, "FILE")):
    print(f"--- storage={sname}")
    d = fresh("add_" + sname)
    h = vq.init(d, DIM, storage, 4096)
    for s in range(0, N, 50_000): assert vq.add(h, data[s:s+50_000].tolist(), vids[s:s+50_000])
    t = time.time(); assert vq.train(h, 1, 0, 16); t_train = time.time() - t
    old_ids = rng.integers(0, N, 500); q_old = norm(data[old_ids] + 0.05 * rng.standard_normal((500, DIM)))
    base_hit = selfhit(h, data, old_ids); base_rec = recall(h, data[:N], q_old)
    print(f"   train {N}: {t_train:.1f}s  old self-hit={base_hit:.3f} recall@10={base_rec:.3f}")
    pos = N; t_add = 0.0
    for p in PIECES:
        assert vq.add(h, data[pos:pos+p].tolist(), vids[pos:pos+p]); pos += p
        t = time.time(); ok = vq.train(h, 2, 0, 16); t_add += time.time() - t
        st = vq.stats(h)[0]
        if not (ok and st['index_size'] == pos and st['vid_size'] == pos):
            check(f"{sname} piece {p}", False, str(st)); break
    else:
        check(f"{sname} all {len(PIECES)} incremental adds accepted, index_size tracks", True, f"index_size={pos} add-train total {t_add:.2f}s ({M/t_add:.0f} pts/s)")
    new_ids = rng.integers(N, N + M, 500); q_new = norm(data[new_ids] + 0.05 * rng.standard_normal((500, DIM)))
    edge = np.array([N, N+1, N+31, N+32, N+33, N+63, N+64, N+65, N+97, N+M-3, N+M-2, N+M-1])
    hit_old, hit_new, hit_edge = selfhit(h, data, old_ids), selfhit(h, data, new_ids), selfhit(h, data, edge, 512)
    rec_old, rec_new = recall(h, data, q_old), recall(h, data, q_new)
    check(f"{sname} old points unaffected by add", hit_old >= base_hit - 0.01, f"self-hit {base_hit:.3f} -> {hit_old:.3f}")
    check(f"{sname} new points searchable like old ones", hit_new >= hit_old - 0.03, f"self-hit new={hit_new:.3f} old={hit_old:.3f}")
    check(f"{sname} block-boundary points all found", hit_edge == 1.0, f"{hit_edge:.3f}")
    check(f"{sname} recall on queries near new points ~ old", rec_new >= rec_old - 0.05, f"recall@10 new={rec_new:.3f} old={rec_old:.3f} (pre-add {base_rec:.3f})")
    r = search(h, data[new_ids[:100]], 5, 100, 64)
    err = max(abs(x['score'] - float(data[i] @ data[x['idx']])) for row, i in zip(r, new_ids[:100]) for x in row)
    check(f"{sname} scores of results == dot product (int8 reorder tolerance)", err < 0.02, f"max|err|={err:.4f}")
    before = ids_of(search(h, np.concatenate([q_old[:100], q_new[:100]]), 10, 200, 64))
    check(f"{sname} dump after adds", vq.dump(h) is True)
    vq.release(h)
    h = vq.init(d, DIM, storage, 4096); st = vq.stats(h)[0]
    check(f"{sname} reload stats", st['index_size'] == N + M and st['vid_size'] == N + M and st['datasets_size'] == N + M, str(st))
    after = ids_of(search(h, np.concatenate([q_old[:100], q_new[:100]]), 10, 200, 64))
    # Ties at the pre-reordering cutoff may be broken differently once the leaves
    # are rebuilt from disk, so allow a couple of queries to differ.
    same = sum(a == b for a, b in zip(before, after))
    check(f"{sname} search consistent after dump+reload", same >= 196, f"{same}/200 queries identical")
    # add again on a reloaded index
    more = clustered(777); mv = (np.arange(777) + 7 + N + M).tolist()
    assert vq.add(h, more.tolist(), mv) and vq.train(h, 2, 0, 16)
    r = search(h, more[:50], 1, 100, 64)
    f = sum(row[0]['idx'] == N + M + i and row[0]['vid'] == N + M + i + 7 for i, row in enumerate(r))
    check(f"{sname} add on reloaded index", vq.stats(h)[0]['index_size'] == N + M + 777 and f >= 45, f"{f}/50")
    # fresh full retrain over everything for comparison
    t = time.time(); assert vq.train(h, 1, 0, 16)
    alld = np.concatenate([data, more])
    print(f"   full retrain {N+M+777}: {time.time()-t:.1f}s  recall@10 old={recall(h, alld, q_old):.3f} new={recall(h, alld, q_new):.3f}")
    vq.release(h)

print("--- train in child process (fork) + dump, then reload")
d = fresh("fork"); h = vq.init(d, DIM, 0, 4096)
assert vq.add(h, data[:50_000].tolist(), vids[:50_000])
ok = vq.train(h, 0, 0, 8, 1)
check("fork train returns True", ok is True)
vq.release(h); h = vq.init(d, DIM, 0, 4096); st = vq.stats(h)[0]
check("fork-trained index loads", st['index_size'] == 50_000, str(st))
ids = rng.integers(0, 50_000, 200)
check("fork-trained index searchable", selfhit(h, data, ids) > 0.9, f"{selfhit(h, data, ids):.3f}")
vq.release(h)
print("\nFAILED:", FAILS if FAILS else "none"); sys.exit(1 if FAILS else 0)
