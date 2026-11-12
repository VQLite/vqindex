"""Locking checks: concurrent adds, search/stats during train, negative scores, incremental dump.

Usage: python test_concurrency.py <dir containing vqindex_py.so> <scratch dir>
"""
import os, shutil, sys, threading, time
import numpy as np
sys.path.insert(0, sys.argv[1]); import vqindex_py as vq
BASE = sys.argv[2]; DIM = 128; OFF = 1000; rng = np.random.default_rng(11); FAILS = []
def check(name, ok, detail=""):
    print(f"[{'PASS' if ok else 'FAIL'}] {name} {detail}", flush=True)
    if not ok: FAILS.append(name)
norm = lambda x: (x / np.linalg.norm(x, axis=1, keepdims=True)).astype(np.float32)
cent = rng.standard_normal((200, DIM))
def clustered(n): return norm(cent[rng.integers(0, 200, n)] + 0.35 * rng.standard_normal((n, DIM)))
def fresh(name):
    d = os.path.join(BASE, name); shutil.rmtree(d, ignore_errors=True); os.makedirs(d); return d
def search(h, q, k, reorder, nprobe):
    r = vq.search(h, q.tolist(), k, reorder, nprobe); return None if r is None else r[0]

# ---------- A. concurrent adds keep vectors and vids aligned ----------
for storage, sname in ((1, "MEMORY"), (0, "FILE")):
    d = fresh("cadd_" + sname); h = vq.init(d, DIM, storage, 4096)
    T, ROUNDS, B = 8, 40, 250
    data = clustered(T * ROUNDS * B)
    accepted = [[] for _ in range(T)]
    def adder(t):
        for r in range(ROUNDS):
            s = (t * ROUNDS + r) * B
            while not vq.add(h, data[s:s + B].tolist(), list(range(s + OFF, s + B + OFF))):
                time.sleep(0.0005)          # busy -> RET_CODE_NOREADY -> retry
            accepted[t].append(s)
    ths = [threading.Thread(target=adder, args=(t,)) for t in range(T)]
    [t.start() for t in ths]; [t.join() for t in ths]
    st = vq.stats(h)[0]; total = T * ROUNDS * B
    check(f"A {sname} sizes after concurrent add", st["vid_size"] == total and st["datasets_size"] == total, str(st))
    check(f"A {sname} train + dump", vq.train(h, 1, 0, 16) is True and vq.dump(h) is True)
    # Batches land in arrival order; row i of the stored dataset must be the
    # vector that was added together with vid i.
    fv = np.fromfile(os.path.join(d, "vids.vql"), dtype=np.int64)
    fd = np.fromfile(os.path.join(d, "datasets.vql"), dtype=np.float32).reshape(-1, DIM)
    check(f"A {sname} every vid stored once", np.array_equal(np.sort(fv), np.arange(OFF, total + OFF)))
    check(f"A {sname} vid stays attached to its vector", fd.shape[0] == total and np.array_equal(fd, data[fv - OFF]))
    ids = rng.integers(0, total, 500)
    r = search(h, fd[ids], 1, 100, 128)
    ok = sum(row[0]["vid"] == fv[row[0]["idx"]] for row in r)
    check(f"A {sname} search returns the vid of the row it found", ok == 500, f"{ok}/500")
    vq.release(h)

# ---------- B. results with negative inner product keep their vid ----------
for n, mode in ((2000, "brute"), (30000, "ann")):
    d = fresh("neg_" + mode); h = vq.init(d, DIM, 1, 4096)
    data = norm(np.abs(clustered(n)))      # all coordinates >= 0, so <-x, y> < 0 for every pair
    assert vq.add(h, data.tolist(), list(range(OFF, n + OFF))) and vq.train(h, 1, 0, 8)
    nprobe = vq.stats(h)[0]["index_nlist"] or 1
    r = search(h, -data[:50], 5, 200, nprobe)
    rows = [x for row in r for x in row]
    neg = [x for x in rows if x["score"] < 0]
    check(f"B {mode} all results have a negative score", len(neg) == len(rows) == 250, f"{len(neg)}/{len(rows)}")
    bad = [x for x in rows if x["vid"] != x["idx"] + OFF]
    check(f"B {mode} every result has its vid", not bad, f"bad={len(bad)} e.g. {bad[:2]}")
    vq.release(h)

# ---------- C. search + stats while training / adding ----------
d = fresh("mixed"); h = vq.init(d, DIM, 1, 4096)
N = 60000; data = clustered(N + 20000)
assert vq.add(h, data[:N].tolist(), list(range(OFF, N + OFF))) and vq.train(h, 1, 0, 8)
stop = threading.Event(); res_ok = [0]; res_none = [0]; res_bad = []; stat_bad = []
def searcher(seed):
    r_ = np.random.default_rng(seed)
    while not stop.is_set():
        q = data[r_.integers(0, N, 8)]
        r = search(h, q, 10, 100, 32)
        if r is None: res_none[0] += 1; time.sleep(0.001); continue
        res_ok[0] += 1
        for row in r:
            for x in row:
                if x["vid"] != x["idx"] + OFF and not (x["score"] == -1 and x["vid"] == 0): res_bad.append(x)
def poller():
    while not stop.is_set():
        s = vq.stats(h)[0]
        if not (0 <= s["index_size"] <= s["vid_size"] <= N + 20000 and s["dim"] == DIM): stat_bad.append(s)
ths = [threading.Thread(target=searcher, args=(i,)) for i in range(6)] + [threading.Thread(target=poller)]
[t.start() for t in ths]
ops = []; cur = N
for rnd in range(4):
    ops.append(vq.train(h, 1, 0, 8)); time.sleep(0.3)
    while not vq.add(h, data[cur:cur + 5000].tolist(), list(range(cur + OFF, cur + 5000 + OFF))): time.sleep(0.001)
    cur += 5000
    ops.append(vq.train(h, 2, 0, 8)); time.sleep(0.3)
stop.set(); [t.join() for t in ths]
check("C all train calls succeeded", all(ops), str(ops))
check("C searches ran both between and during trains", res_ok[0] > 0 and res_none[0] > 0, f"ok={res_ok[0]} rejected={res_none[0]}")
check("C no result with a wrong vid", not res_bad, f"bad={len(res_bad)} e.g. {res_bad[:2]}")
check("C stats stayed consistent", not stat_bad, str(stat_bad[:1]))
st = vq.stats(h)[0]; check("C final index size", st["index_size"] == cur and st["vid_size"] == cur, str(st))

# ---------- D. incremental dump (MEMORY) ----------
dpath, vpath = os.path.join(d, "datasets.vql"), os.path.join(d, "vids.vql")
def files_match(n):
    return (np.array_equal(np.fromfile(dpath, dtype=np.float32), data[:n].ravel())
            and np.array_equal(np.fromfile(vpath, dtype=np.int64), np.arange(OFF, n + OFF)))
check("D first dump", vq.dump(h) is True and files_match(cur))
t = time.time(); ok = vq.dump(h); t_noop = time.time() - t
check("D dump with nothing new keeps files intact", ok is True and files_match(cur))
more = clustered(3000); data = np.concatenate([data[:cur], more])
assert vq.add(h, more.tolist(), list(range(cur + OFF, cur + 3000 + OFF))) and vq.train(h, 2, 0, 8); cur += 3000
check("D dump after add appends the tail", vq.dump(h) is True and files_match(cur))
with open(dpath, "r+b") as f: f.truncate(1000 * DIM * 4)       # file no longer matches what was persisted
check("D dump rewrites a file that changed underneath", vq.dump(h) is True and files_match(cur))
vq.release(h)
h = vq.init(d, DIM, 1, 4096); st = vq.stats(h)[0]
check("D reload after incremental dumps", st["index_size"] == cur and st["datasets_size"] == cur and st["vid_size"] == cur, str(st))
r = search(h, data[[5, cur - 1]], 1, 100, 64)
check("D reloaded index maps vids", [row[0]["vid"] for row in r] == [5 + OFF, cur - 1 + OFF])
assert vq.add(h, more[:10].tolist(), list(range(cur + OFF, cur + 10 + OFF))) and vq.dump(h)
check("D dump after reload + add", np.fromfile(vpath, dtype=np.int64).size == cur + 10 and os.path.getsize(dpath) == (cur + 10) * DIM * 4)
vq.release(h)

# ---------- E. child-process training reports failures ----------
d = fresh("proc"); h = vq.init(d, DIM, 1, 4096)
x = clustered(20000); assert vq.add(h, x.tolist(), list(range(20000)))
# train(ADD) needs an existing index -> RET_CODE_NOREADY in the child
check("E failure in the child is reported", vq.train(h, 2, 0, 4, 1) is False)
check("E train in child succeeds", vq.train(h, 1, 0, 4, 1) is True)
check("E handler still usable after child training", vq.add(h, x[:5].tolist(), [1] * 5) is True and vq.stats(h)[0]["vid_size"] == 20005)
vq.release(h)
h = vq.init(d, DIM, 1, 4096); check("E child-trained index loads", vq.stats(h)[0]["index_size"] == 20000); vq.release(h)

print("\nFAILED:", FAILS if FAILS else "none")
sys.exit(1 if FAILS else 0)
