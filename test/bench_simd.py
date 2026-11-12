"""Checks which SIMD level ScaNN picked at runtime and compares speed/results across levels.

The levels are switched inside one process by overwriting ScaNN's detection
flags, so this only measures the hand-written kernels that honour them.

Usage: python bench_simd.py <dir containing vqindex_py.so> <scratch dir>
"""
import ctypes, os, shutil, subprocess, sys, time
import numpy as np
so_dir = sys.argv[1]; BASE = sys.argv[2]
sys.path.insert(0, so_dir); import vqindex_py as vq
so = os.path.join(so_dir, "vqindex_py.so"); lib = ctypes.CDLL(so)
syms = subprocess.run(["nm", "-D", "--defined-only", so], capture_output=True, text=True).stdout.split()
flag = lambda n: ctypes.c_bool.in_dll(lib, [s for s in syms if s.endswith(n)][0])
f2, f512, fvnni = flag("15should_use_avx2E"), flag("17should_use_avx512E"), flag("22should_use_avx512_vnniE")
detected = (f2.value, f512.value, fvnni.value)
print("cpu flags:", " ".join(sorted(set(open("/proc/cpuinfo").read().split()) & {"avx", "avx2", "fma", "avx512f", "avx512bw", "avx512dq", "avx512_vnni"})))
print("runtime detection: should_use_avx2=%s should_use_avx512=%s should_use_avx512_vnni=%s" % detected, flush=True)

DIM = 128; rng = np.random.default_rng(0)
norm = lambda x: (x / np.linalg.norm(x, axis=1, keepdims=True)).astype(np.float32)
cent = rng.standard_normal((500, DIM))
N = 500_000
data = norm(cent[rng.integers(0, 500, N)] + 0.35 * rng.standard_normal((N, DIM)))
q = norm(data[rng.integers(0, N, 2000)] + 0.05 * rng.standard_normal((2000, DIM)))
gt = np.argsort(-(q[:300] @ data.T), axis=1)[:, :10]
ql = q.tolist()

def set_level(a2, a512, vnni): f2.value, f512.value, fvnni.value = a2, a512, vnni
levels = [("SSE4 (forced)", (False, False, False))]
if detected[0]: levels.append(("AVX2 (forced)" if detected[1] else "AVX2 (auto)", (True, False, False)))
if detected[1]: levels.append(("AVX-512 (auto)", detected))

def build():
    d = os.path.join(BASE, "simd"); shutil.rmtree(d, ignore_errors=True); os.makedirs(d)
    h = vq.init(d, DIM, 1, 4096)
    for s in range(0, N, 100_000): assert vq.add(h, data[s:s+100_000].tolist(), list(range(s, s+100_000)))
    t = time.time(); assert vq.train(h, 1, 0, 16); return h, time.time() - t

def bench(h, cfg, reps=3):
    best = 1e9
    for _ in range(reps):
        t = time.time(); r = vq.search(h, ql, *cfg)[0]; best = min(best, time.time() - t)
    return best / len(ql) * 1e3, r

set_level(*detected)
h, t_train = build()
print("train %d pts with detected level: %.1fs, nlist=%d" % (N, t_train, vq.stats(h)[0]["index_nlist"]), flush=True)
cfgs = [(10, 100, 16), (10, 200, 64), (10, 1000, 256)]
ref = {}
print("%-16s %-18s %10s %10s  %s" % ("level", "topk,reorder,nprobe", "ms/query", "recall@10", "same ids as SSE4"))
for name, lv in levels:
    set_level(*lv)
    for cfg in cfgs:
        ms, r = bench(h, cfg)
        ids = [[x["idx"] for x in row] for row in r]
        rec = sum(len(set(a) & set(g.tolist())) for a, g in zip(ids[:300], gt)) / gt.size
        if name.startswith("SSE4"): ref[cfg] = ids; same = "-"
        else: same = "%d/%d" % (sum(a == b for a, b in zip(ids, ref[cfg])), len(ids))
        print("%-16s %-18s %10.4f %10.4f  %s" % (name, cfg, ms, rec, same), flush=True)
# training time per level
for name, lv in levels:
    set_level(*lv); vq.release(h); h, t = build(); print("train time @ %-16s %.1fs" % (name, t), flush=True)
vq.release(h); shutil.rmtree(os.path.join(BASE, "simd"), ignore_errors=True)
