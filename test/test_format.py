"""Dump formats: packed / legacy / both give the same index, stale files are not used.

Usage: python test_format.py <dir containing vqindex_py.so> <scratch dir> [path to tools/vqindex_convert.py]
"""
import filecmp, os, re, shutil, subprocess, sys
import numpy as np
sys.path.insert(0, sys.argv[1]); import vqindex_py as vq
BASE = sys.argv[2]; CONVERT = sys.argv[3] if len(sys.argv) > 3 else None
DIM = 128; OFF = 1000; rng = np.random.default_rng(23); FAILS = []
PACKED, LEGACY = "hashed_dataset_packed.bin", "hashed_dataset.npy"
def check(name, ok, detail=""):
    print(f"[{'PASS' if ok else 'FAIL'}] {name} {detail}", flush=True)
    if not ok: FAILS.append(name)
norm = lambda x: (x / np.linalg.norm(x, axis=1, keepdims=True)).astype(np.float32)
cent = rng.standard_normal((300, DIM))
def clustered(n): return norm(cent[rng.integers(0, 300, n)] + 0.35 * rng.standard_normal((n, DIM)))
def fresh(name):
    d = os.path.join(BASE, name); shutil.rmtree(d, ignore_errors=True); os.makedirs(d); return d
def seg(d):
    subs = [os.path.join(d, x) for x in os.listdir(d) if os.path.isfile(os.path.join(d, x, "scann_config.pb"))]
    assert len(subs) == 1, subs
    return subs[0]
def forms(d): return (os.path.exists(os.path.join(seg(d), PACKED)), os.path.exists(os.path.join(seg(d), LEGACY)))
def results(h, q):
    r = vq.search(h, q.tolist(), 10, 200, 64)[0]
    return [[(x["idx"], x["vid"], x["score"]) for x in row] for row in r]
def reopen(d):
    h = vq.init(d, DIM, 1, 4096)
    return h, results(h, Q)

# An index whose leaves were partly built by training and partly grown by adds.
d = fresh("fmt"); h = vq.init(d, DIM, 1, 4096)
N1, N2, N3 = 60000, 15000, 5000
data = clustered(N1 + N2 + N3); Q = clustered(300)
assert vq.add(h, data[:N1].tolist(), list(range(OFF, OFF + N1))) and vq.train(h, 1, 0, 16)
assert vq.add(h, data[N1:N1 + N2].tolist(), list(range(OFF + N1, OFF + N1 + N2))) and vq.train(h, 2, 0, 16)
st = vq.stats(h)[0]
check("index holds trained + added points", st["index_size"] == N1 + N2 and not st["is_brute"], str(st))

# ---------- A. every format loads to the same index ----------
# The reference is the index as loaded from the legacy file.  (Results taken
# before the first dump are not comparable: a freshly trained index may order
# ties differently from a loaded one.)
check("A dump(legacy)", vq.dump(h, 1) is True)
check("A legacy dump writes only the npy file", forms(d) == (False, True), str(forms(d)))
vq.release(h)
h, R0 = reopen(d)
check("A legacy: results are complete", all(len(row) == 10 for row in R0))

check("A default dump", vq.dump(h) is True)
check("A default dump writes only the packed file", forms(d) == (True, False), str(forms(d)))
vq.release(h)
h, r = reopen(d); check("A packed: reloaded results identical", r == R0)
h2, r = reopen(d); check("A packed: second load identical", r == R0); vq.release(h2)

check("A dump(both)", vq.dump(h, 2) is True)
check("A both files written", forms(d) == (True, True), str(forms(d)))
vq.release(h)
h, r = reopen(d); check("A both: reloaded results identical", r == R0)
check("A unknown format is rejected", vq.dump(h, 7) is False and forms(d) == (True, True))

# ---------- B. converter agrees with the library byte for byte ----------
# h was loaded from disk, so every leaf has the canonical tail padding.
vq.dump(h, 2)
lib = fresh("fmt_lib"); shutil.copytree(seg(d), os.path.join(lib, "seg"))
libseg = os.path.join(lib, "seg")
if CONVERT:
    def convert(cmd, src_seg, *flags):
        w = fresh("fmt_conv"); t = os.path.join(w, "seg"); shutil.copytree(src_seg, t)
        return t, subprocess.run([sys.executable, CONVERT, cmd, t, *flags], capture_output=True, text=True)
    t, p = convert("to-legacy", libseg, "--remove-source")
    check("B to-legacy runs", p.returncode == 0, p.stderr.strip())
    check("B to-legacy output equals the library's npy", filecmp.cmp(os.path.join(t, LEGACY), os.path.join(libseg, LEGACY), False))
    check("B --remove-source removes the packed file", not os.path.exists(os.path.join(t, PACKED)))
    t2 = t; p = subprocess.run([sys.executable, CONVERT, "to-packed", t2], capture_output=True, text=True)
    check("B to-packed runs", p.returncode == 0, p.stderr.strip())
    check("B to-packed output equals the library's packed file", filecmp.cmp(os.path.join(t2, PACKED), os.path.join(libseg, PACKED), False))
    check("B the source is kept by default", os.path.exists(os.path.join(t2, LEGACY)))
    p = subprocess.run([sys.executable, CONVERT, "info", t2], capture_output=True, text=True)
    check("B info reports a current packed file", p.returncode == 0 and "matches datapoint_to_token.npy: yes" in p.stdout, p.stdout.strip())
    # --out-dir: only the converted file appears there, the index directory is untouched.
    for cmd, name, src_only in (("to-packed", PACKED, LEGACY), ("to-legacy", LEGACY, PACKED)):
        w = fresh("fmt_src"); src = os.path.join(w, "seg"); shutil.copytree(libseg, src); os.remove(os.path.join(src, name))
        before = {x: os.stat(os.path.join(src, x)).st_mtime_ns for x in os.listdir(src)}
        out = os.path.join(fresh("fmt_out"), "sub")
        p = subprocess.run([sys.executable, CONVERT, cmd, src, "--out-dir", out], capture_output=True, text=True)
        check(f"B {cmd} --out-dir runs", p.returncode == 0, p.stderr.strip())
        check(f"B {cmd} --out-dir creates only the converted file", os.listdir(out) == [name], str(os.listdir(out)))
        check(f"B {cmd} --out-dir output equals the library's file", filecmp.cmp(os.path.join(out, name), os.path.join(libseg, name), False))
        check(f"B {cmd} --out-dir leaves the index directory untouched", before == {x: os.stat(os.path.join(src, x)).st_mtime_ns for x in os.listdir(src)})
    # A buffer far smaller than the codes: many passes, same bytes.
    for cmd, name in (("to-packed", PACKED), ("to-legacy", LEGACY)):
        out = fresh("fmt_out")
        p = subprocess.run([sys.executable, CONVERT, cmd, libseg, "--out-dir", out], capture_output=True, text=True, env=dict(os.environ, VQINDEX_CONVERT_BUFFER_ROWS="9000"))
        npass = int((re.search(r"(\d+) pass", p.stdout) or [0, 0])[1])
        check(f"B {cmd} in several passes equals the library's file", p.returncode == 0 and npass > 3 and filecmp.cmp(os.path.join(out, name), os.path.join(libseg, name), False), f"{npass} passes {p.stderr.strip()}")
    p = subprocess.run([sys.executable, CONVERT, "to-packed", libseg, "--out-dir", out, "--memory-gb", "0.1"], capture_output=True, text=True)
    check("B a memory limit that is too small is refused", p.returncode == 1 and "not enough memory" in p.stderr, p.stderr.strip())
    p = subprocess.run([sys.executable, CONVERT, "to-packed", libseg, "--out-dir", out, "--remove-source"], capture_output=True, text=True)
    check("B --out-dir with --remove-source is refused", p.returncode == 2 and os.path.exists(os.path.join(libseg, LEGACY)))
    # A converted directory loads like a dumped one.
    for name, drop in (("packed only", LEGACY), ("legacy only", PACKED)):
        w = fresh("fmt_load"); shutil.copytree(d, w, dirs_exist_ok=True)
        s = seg(w); shutil.rmtree(s); shutil.copytree(t2, s); os.remove(os.path.join(s, drop))
        hh, r = reopen(w); check(f"B converted index ({name}) gives identical results", r == R0); vq.release(hh)
else:
    print("[SKIP] B no converter path given")

# ---------- C. mutating an index that was loaded from the packed file ----------
vq.dump(h, 0); vq.release(h)
h, r = reopen(d)
lo = N1 + N2
assert vq.add(h, data[lo:lo + N3].tolist(), list(range(OFF + lo, OFF + lo + N3))) and vq.train(h, 2, 0, 16)
check("C added to a packed-loaded index", vq.stats(h)[0]["index_size"] == N1 + N2 + N3)
old_packed = os.path.join(BASE, "old_packed.bin"); shutil.copy(os.path.join(seg(d), PACKED), old_packed)
check("C dump(legacy) after add", vq.dump(h, 1) is True); vq.release(h)
h, R1 = reopen(d)
check("C new points are searchable", R1 != R0)
check("C dump(packed)", vq.dump(h, 0) is True and forms(d) == (True, False)); vq.release(h)
h, r = reopen(d); check("C packed: reloaded results identical", r == R1)
check("C dump(both)", vq.dump(h, 2) is True and forms(d) == (True, True)); vq.release(h)
h, r = reopen(d); check("C both: reloaded results identical", r == R1); vq.release(h)

# ---------- D. a packed file from an older state is not used ----------
# What a library without packed support leaves behind when it dumps into a
# directory that already has a packed file.
shutil.copy(old_packed, os.path.join(seg(d), PACKED))
h, r = reopen(d); check("D stale packed file: legacy file is loaded instead", r == R1)
vq.release(h)
os.remove(os.path.join(seg(d), LEGACY))
h = vq.init(d, DIM, 1, 4096)
check("D stale packed file alone is refused", h == 0, str(h))
if h: vq.release(h)

# ---------- E. VQINDEX_DUMP_FORMAT sets the default ----------
d2 = fresh("fmt_env"); os.environ["VQINDEX_DUMP_FORMAT"] = "1"
h = vq.init(d2, DIM, 1, 4096)
assert vq.add(h, data[:30000].tolist(), list(range(OFF, OFF + 30000))) and vq.train(h, 1, 0, 16)
check("E env default = legacy", vq.dump(h) is True and forms(d2) == (False, True), str(forms(d2)))
check("E explicit format still wins", vq.dump(h, 0) is True and forms(d2) == (True, False), str(forms(d2)))
vq.release(h); del os.environ["VQINDEX_DUMP_FORMAT"]

print("FAILED: " + ", ".join(FAILS) if FAILS else "format tests passed")
sys.exit(1 if FAILS else 0)
