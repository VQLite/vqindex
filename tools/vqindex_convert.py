#!/usr/bin/env python3
"""Convert the AH codes of a dumped VQIndex between the two on-disk forms.

    hashed_dataset.npy          one byte per code, datapoint order (legacy,
                                readable by every library version)
    hashed_dataset_packed.bin   LUT16-packed, one block per partition (loads
                                and dumps without an extra unpacked copy)

usage: vqindex_convert.py <command> <index_dir> [--out-dir DIR | --remove-source]
                                                 [--memory-gb N]

    index_dir   directory holding scann_config.pb, e.g. <root>/index

    to-packed   hashed_dataset.npy        -> hashed_dataset_packed.bin
    to-legacy   hashed_dataset_packed.bin -> hashed_dataset.npy
    info        show which forms are present and whether they are current

    --remove-source   remove the source file once the other form is written
                      (by default it is kept, leaving both forms in the
                      directory; the library then loads the packed one)
    --out-dir DIR     write the converted file into DIR instead of index_dir
                      (e.g. a local disk when index_dir is a slow network
                      mount).  Only that one file is created in DIR; index_dir
                      is only read.  Copy the file next to the other index
                      files afterwards.
    --memory-gb N     keep the memory use of the conversion below N GB
                      (default: half of the available memory)
    --keep            accepted for compatibility; keeping is the default

Needs only Python 3 and numpy; it does not load the index or use the VQIndex
library.  It reads datapoint_to_token.npy and the file being converted.

Both files are read front to back and the output is written front to back, so
the conversion does not depend on the page cache and works on network mounts.
The codes are regrouped in a memory buffer.  When the memory limit lets the
buffer hold them all, the source is read once; otherwise it is read once per
buffer-full.  Besides the buffer the conversion needs 4 bytes per datapoint
(8 for to-legacy) and about 0.25 GB.  The index must not be dumped or trained
while it runs.

The packed layout is documented in scann/utils/packed_leaves_file.h.
"""
import os
import re
import struct
import sys

import numpy as np

LEGACY = "hashed_dataset.npy"
PACKED = "hashed_dataset_packed.bin"
TOKENS = "datapoint_to_token.npy"
MAGIC = b"VQAHPK16"
VERSION = 1
HEADER = struct.Struct("<8sIIIIQQQ16x")  # 64 bytes
M64 = (1 << 64) - 1


class ConvertError(Exception):
    pass


# ---- tokenization fingerprint (TokenizationFingerprint in packed_leaves_file.cc)

def _mix_array(x):
    x = x + np.uint64(0x9E3779B97F4A7C15)
    x = (x ^ (x >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9)
    x = (x ^ (x >> np.uint64(27))) * np.uint64(0x94D049BB133111EB)
    return x ^ (x >> np.uint64(31))


def _mix(x):
    x = (x + 0x9E3779B97F4A7C15) & M64
    x = ((x ^ (x >> 30)) * 0xBF58476D1CE4E5B9) & M64
    x = ((x ^ (x >> 27)) * 0x94D049BB133111EB) & M64
    return x ^ (x >> 31)


def fingerprint(tokens, num_leaves):
    total = 0
    step = 1 << 22
    with np.errstate(over="ignore"):
        for start in range(0, len(tokens), step):
            t = tokens[start:start + step].astype(np.uint32).astype(np.uint64)
            dp = np.arange(start, start + len(t), dtype=np.uint64)
            total += int(np.add.reduce(_mix_array((t << np.uint64(32)) ^ dp)))
    total &= M64
    return _mix(total ^ _mix(len(tokens)) ^ _mix((num_leaves << 40) & M64))


# ---- file helpers

def packed_leaf_bytes(n, num_blocks):
    return num_blocks * ((n + 31) & ~31) // 2


def data_offset(num_leaves):
    return (HEADER.size + 4 * num_leaves + 63) & ~63


def num_leaves_from_partitioner(index_dir):
    """SerializedPartitioner.n_tokens: field 1, a varint, written first."""
    with open(os.path.join(index_dir, "serialized_partitioner.pb"), "rb") as f:
        head = f.read(16)
    if not head or head[0] != 0x08:
        raise ConvertError("serialized_partitioner.pb does not start with the partition count")
    value, shift = 0, 0
    for b in head[1:]:
        value |= (b & 0x7F) << shift
        shift += 7
        if not b & 0x80:
            return value
    raise ConvertError("serialized_partitioner.pb: cannot read the partition count")


def load_tokens(index_dir, num_leaves):
    tokens = np.load(os.path.join(index_dir, TOKENS))
    if tokens.ndim != 1 or tokens.dtype != np.int32:
        raise ConvertError(f"{TOKENS} is not a one-dimensional int32 array "
                           "(an index with SOAR spilling cannot be converted)")
    if len(tokens) and (tokens.min() < 0 or tokens.max() >= num_leaves):
        raise ConvertError(f"{TOKENS} refers to partitions outside 0..{num_leaves - 1}")
    return tokens


CHUNK_ROWS = 1 << 20


def leaf_slots(leaf_ids, cursor):
    """Where consecutive datapoints go when grouped by partition.

    leaf_ids: partition of each datapoint, relative to cursor.  cursor[p] is
    the next free slot of partition p and is advanced.  Fed with the
    datapoints in index order, every partition comes out in ascending order.
    """
    by_leaf = np.argsort(leaf_ids, kind="stable")
    sorted_ids = leaf_ids[by_leaf]
    pos = np.arange(len(sorted_ids), dtype=np.int64)
    run_start = np.zeros(len(sorted_ids), dtype=np.int64)
    if len(sorted_ids) > 1:
        changes = np.flatnonzero(sorted_ids[1:] != sorted_ids[:-1]) + 1
        run_start[changes] = changes
        np.maximum.accumulate(run_start, out=run_start)
    slots = np.empty(len(leaf_ids), dtype=np.int64)
    slots[by_leaf] = cursor[sorted_ids] + (pos - run_start)
    cursor += np.bincount(leaf_ids, minlength=len(cursor))
    return slots


def leaf_begins(tokens, num_leaves):
    counts = np.bincount(tokens, minlength=num_leaves).astype(np.int64)
    return counts, np.concatenate(([0], np.cumsum(counts)))


FIXED_BYTES = 1 << 28   # one chunk of codes, its index arrays, the interpreter


def buffer_rows(memory_gb, n, num_blocks, bytes_per_datapoint):
    """Rows of codes the regrouping buffer may hold within the memory limit."""
    if os.environ.get("VQINDEX_CONVERT_BUFFER_ROWS"):   # for tests
        return max(1, min(n, int(os.environ["VQINDEX_CONVERT_BUFFER_ROWS"])))
    if memory_gb is None:
        budget = 8 << 30
        try:
            with open("/proc/meminfo") as f:
                for line in f:
                    if line.startswith("MemAvailable:"):
                        budget = int(line.split()[1]) * 1024 // 2
        except OSError:
            pass
    else:
        budget = int(memory_gb * (1 << 30))
    budget -= FIXED_BYTES + bytes_per_datapoint * n
    if budget < min(n * num_blocks, 64 << 20):
        need = (FIXED_BYTES + bytes_per_datapoint * n + (64 << 20)) / 2**30
        raise ConvertError(f"not enough memory: this index needs at least {need:.1f} GB "
                           "(see --memory-gb)")
    return max(1, min(n, budget // num_blocks))


def open_legacy(path):
    """-> (file positioned at the data, n, num_blocks)"""
    f = open(path, "rb")
    try:
        version = np.lib.format.read_magic(f)
        read_header = (np.lib.format.read_array_header_1_0 if version == (1, 0)
                       else np.lib.format.read_array_header_2_0)
        shape, fortran, dtype = read_header(f)
    except (ValueError, OSError) as e:
        f.close()
        raise ConvertError(f"{path} is not a readable .npy file: {e}")
    if len(shape) != 2 or dtype != np.uint8 or fortran:
        f.close()
        raise ConvertError(f"{LEGACY} is not a two-dimensional uint8 array")
    if os.path.getsize(path) != f.tell() + shape[0] * shape[1]:
        f.close()
        raise ConvertError(f"{path} is truncated")
    return f, shape[0], shape[1]


def read_exact(f, view, what):
    if f.readinto(view) != view.nbytes:
        raise ConvertError(f"{what} is truncated")


def read_packed_header(path):
    size = os.path.getsize(path)
    with open(path, "rb") as f:
        raw = f.read(HEADER.size)
        if len(raw) < HEADER.size:
            raise ConvertError(f"{path} is truncated")
        magic, version, num_blocks, num_leaves, _, n, fp, offset = HEADER.unpack(raw)
        if magic != MAGIC:
            raise ConvertError(f"{path} is not a packed AH leaves file")
        if version != VERSION:
            raise ConvertError(f"{path} has unsupported version {version}")
        if offset != data_offset(num_leaves) or offset > size:
            raise ConvertError(f"{path} has an inconsistent header")
        leaf_sizes = np.frombuffer(f.read(4 * num_leaves), dtype="<u4").astype(np.int64)
    if len(leaf_sizes) != num_leaves or int(leaf_sizes.sum()) != n:
        raise ConvertError(f"{path}: partition sizes do not add up to {n} datapoints")
    expected = offset + sum(packed_leaf_bytes(int(s), num_blocks) for s in leaf_sizes)
    if expected != size:
        raise ConvertError(f"{path} is {size} bytes, expected {expected}")
    return num_blocks, num_leaves, n, fp, offset, leaf_sizes


def npy_header(n, num_blocks):
    """The header ScaNN writes, so the output is byte-identical to a dump."""
    d = "{'descr':'<u1', 'fortran_order':False, 'shape':(%d,%d,)}" % (n, num_blocks)
    while (10 + len(d)) % 64 != 63:
        d += " "
    d += "\n"
    return b"\x93NUMPY\x01\x00" + struct.pack("<H", len(d)) + d.encode("ascii")


def update_assets(index_dir, have_packed, have_legacy):
    """Keeps scann_assets.pbtxt in step.  VQIndex does not read the asset
    list, so this is best effort."""
    path = os.path.join(index_dir, "scann_assets.pbtxt")
    try:
        with open(path) as f:
            text = f.read()
        block = re.compile(r"assets\s*\{[^{}]*\}\s*")
        path_of = lambda b: (re.search(r'asset_path:\s*"([^"]*)"', b) or [None, ""])[1]
        prefix = None      # how the list spells the directory of its files
        kept = []
        for m in block.finditer(text):
            b = m.group(0)
            if re.search(r"asset_type:\s*(AH_DATASET_NPY|AH_DATASET_PACKED_LEAVES)\b", b):
                prefix = os.path.dirname(path_of(b))
            else:
                kept.append(b.rstrip() + "\n")
                if prefix is None and path_of(b):
                    prefix = os.path.dirname(path_of(b))
        rest = block.sub("", text).strip()

        def entry(name, kind):
            return 'assets {\n  asset_type: %s\n  asset_path: "%s"\n}\n' % (
                kind, os.path.join(prefix or "", name))
        if have_packed:
            kept.append(entry(PACKED, "AH_DATASET_PACKED_LEAVES"))
        if have_legacy:
            kept.append(entry(LEGACY, "AH_DATASET_NPY"))
        with open(path + ".tmp", "w") as f:
            f.write("".join(kept) + (rest + "\n" if rest else ""))
        os.replace(path + ".tmp", path)
    except OSError as e:
        if os.path.exists(path):
            print(f"warning: cannot rewrite {path}: {e}", file=sys.stderr)


# ---- packing: PackedDataset::bit_packed_data for one partition

def pack_leaf(codes):
    """codes: (n, num_blocks) uint8, values 0..15 -> packed bytes.

    Datapoints go in groups of 32; for each group and each block there are 16
    bytes, byte m holding datapoint m in the low nibble and datapoint m + 16
    in the high nibble.  A short last group is filled with the last datapoint.
    """
    n, num_blocks = codes.shape
    pad = -n % 32
    if pad:
        codes = np.concatenate([codes, np.repeat(codes[-1:], pad, axis=0)])
    groups = codes.reshape(-1, 32, num_blocks)
    packed = groups[:, :16, :] | (groups[:, 16:, :] << 4)   # (groups, 16, blocks)
    return np.ascontiguousarray(packed.transpose(0, 2, 1)).tobytes()


def unpack_leaf(raw, n, num_blocks):
    packed = np.frombuffer(raw, dtype=np.uint8).reshape(-1, num_blocks, 16).transpose(0, 2, 1)
    groups = np.concatenate([packed & 15, packed >> 4], axis=1)   # (groups, 32, blocks)
    return groups.reshape(-1, num_blocks)[:n]


# ---- commands

def to_packed(index_dir, keep, out_dir=None, memory_gb=None):
    legacy_path = os.path.join(index_dir, LEGACY)
    packed_path = os.path.join(out_dir or index_dir, PACKED)
    num_leaves = num_leaves_from_partitioner(index_dir)
    tokens = load_tokens(index_dir, num_leaves)
    src, n, num_blocks = open_legacy(legacy_path)
    with src:
        if n != len(tokens):
            raise ConvertError(f"{LEGACY} does not hold one row per datapoint of {TOKENS}")
        data_start = src.tell()
        counts, begins = leaf_begins(tokens, num_leaves)

        # Consecutive partitions that fit the buffer together are regrouped in
        # one pass over the source.
        rows = max(buffer_rows(memory_gb, n, num_blocks, 4), int(counts.max()) if n else 1)
        passes, a = [], 0
        while a < num_leaves:
            b = int(np.searchsorted(begins, begins[a] + rows, side="right")) - 1
            b = min(max(b, a + 1), num_leaves)
            passes.append((a, b))
            a = b
        print(f"{n} datapoints, {num_leaves} partitions, {num_blocks} codes per datapoint; "
              f"{len(passes)} pass(es) over {LEGACY} with a "
              f"{min(rows, n) * num_blocks / 2**30:.2f} GB buffer", flush=True)

        tmp = packed_path + ".tmp"
        offset = data_offset(num_leaves)
        chunk = np.empty((min(CHUNK_ROWS, max(n, 1)), num_blocks), dtype=np.uint8)
        with open(tmp, "wb") as f:
            f.write(HEADER.pack(MAGIC, VERSION, num_blocks, num_leaves, 0, n,
                                fingerprint(tokens, num_leaves), offset))
            f.write(counts.astype("<u4").tobytes())
            f.write(b"\0" * (offset - f.tell()))
            for a, b in passes:
                base = int(begins[a])
                buf = np.empty((int(begins[b]) - base, num_blocks), dtype=np.uint8)
                cursor = begins[a:b] - base
                everything = a == 0 and b == num_leaves
                src.seek(data_start)
                for start in range(0, n, CHUNK_ROWS):
                    t = tokens[start:start + CHUNK_ROWS]
                    if not everything:
                        wanted = (t >= a) & (t < b)
                        if not wanted.any():
                            src.seek(len(t) * num_blocks, os.SEEK_CUR)
                            continue
                    part = chunk[:len(t)]
                    read_exact(src, part, legacy_path)
                    if part.max() > 15:
                        raise ConvertError(f"{LEGACY} holds codes above 15; only 16-center "
                                           "(LUT16) indexes have a packed form")
                    if everything:
                        buf[leaf_slots(t.astype(np.int64), cursor)] = part
                    else:
                        buf[leaf_slots(t[wanted].astype(np.int64) - a, cursor)] = part[wanted]
                for leaf in range(a, b):
                    if counts[leaf]:
                        f.write(pack_leaf(buf[begins[leaf] - base:begins[leaf + 1] - base]))
                del buf
    os.replace(tmp, packed_path)
    if out_dir is None:
        if not keep:
            os.remove(legacy_path)
        update_assets(index_dir, True, keep)
    print(f"wrote {packed_path}")


def to_legacy(index_dir, keep, out_dir=None, memory_gb=None):
    legacy_path = os.path.join(out_dir or index_dir, LEGACY)
    packed_path = os.path.join(index_dir, PACKED)
    num_blocks, num_leaves, n, fp, offset, leaf_sizes = read_packed_header(packed_path)
    tokens = load_tokens(index_dir, num_leaves)
    if len(tokens) != n or fingerprint(tokens, num_leaves) != fp:
        raise ConvertError(f"{PACKED} was not written for the tokenization in {TOKENS} "
                           "(it is stale)")
    counts, begins = leaf_begins(tokens, num_leaves)
    if not np.array_equal(counts, leaf_sizes):
        raise ConvertError(f"{PACKED} and {TOKENS} disagree on partition sizes")

    # order[begins[p]:begins[p + 1]]: the datapoints of partition p, ascending.
    order = np.empty(n, dtype=np.uint32)
    cursor = begins[:-1].copy()
    for start in range(0, n, CHUNK_ROWS):
        t = tokens[start:start + CHUNK_ROWS].astype(np.int64)
        order[leaf_slots(t, cursor)] = np.arange(start, start + len(t), dtype=np.uint32)
    del tokens, cursor

    # Each pass reads the packed file once and produces a range of datapoints.
    rows = buffer_rows(memory_gb, n, num_blocks, 8)
    passes = [(lo, min(lo + rows, n)) for lo in range(0, n, rows)]
    print(f"{n} datapoints, {num_leaves} partitions, {num_blocks} codes per datapoint; "
          f"{len(passes)} pass(es) over {PACKED} with a "
          f"{min(rows, n) * num_blocks / 2**30:.2f} GB buffer", flush=True)

    tmp = legacy_path + ".tmp"
    with open(tmp, "wb") as f, open(packed_path, "rb") as src:
        f.write(npy_header(n, num_blocks))
        for lo, hi in passes:
            buf = np.empty((hi - lo, num_blocks), dtype=np.uint8)
            everything = lo == 0 and hi == n
            src.seek(offset)
            for leaf in range(num_leaves):
                size = int(counts[leaf])
                if not size:
                    continue
                nbytes = packed_leaf_bytes(size, num_blocks)
                dps = order[begins[leaf]:begins[leaf + 1]]
                i0, i1 = (0, size) if everything else np.searchsorted(dps, [lo, hi])
                if i0 == i1:
                    src.seek(nbytes, os.SEEK_CUR)
                    continue
                raw = src.read(nbytes)
                if len(raw) != nbytes:
                    raise ConvertError(f"{packed_path} is truncated")
                buf[dps[i0:i1].astype(np.int64) - lo] = unpack_leaf(raw, size, num_blocks)[i0:i1]
            f.write(memoryview(buf).cast("B"))
            del buf
    os.replace(tmp, legacy_path)
    if out_dir is None:
        if not keep:
            os.remove(packed_path)
        update_assets(index_dir, keep, True)
    print(f"wrote {legacy_path}")


def info(index_dir):
    legacy_path = os.path.join(index_dir, LEGACY)
    packed_path = os.path.join(index_dir, PACKED)
    print(f"{LEGACY}: {'present' if os.path.exists(legacy_path) else 'absent'}")
    if not os.path.exists(packed_path):
        print(f"{PACKED}: absent")
        return
    num_blocks, num_leaves, n, fp, _, _ = read_packed_header(packed_path)
    print(f"{PACKED}: {n} datapoints, {num_leaves} partitions, {num_blocks} codes per datapoint")
    tokens = load_tokens(index_dir, num_leaves)
    fresh = len(tokens) == n and fingerprint(tokens, num_leaves) == fp
    print(f"  matches {TOKENS}: " + ("yes" if fresh else
          "NO (stale; the library falls back to the legacy file if there is one)"))


def main(argv):
    argv = list(argv)
    values = {"--out-dir": None, "--memory-gb": None}
    i = 0
    while i < len(argv):
        name, eq, value = argv[i].partition("=")
        if name not in values:
            i += 1
            continue
        if not eq:
            if i + 1 >= len(argv):
                print(f"{name} needs a value", file=sys.stderr)
                return 2
            value = argv[i + 1]
            del argv[i + 1]
        del argv[i]
        values[name] = value
    out_dir, memory_gb = values["--out-dir"], values["--memory-gb"]
    if memory_gb is not None:
        try:
            memory_gb = float(memory_gb)
            if not memory_gb > 0:
                raise ValueError
        except ValueError:
            print("--memory-gb needs a positive number", file=sys.stderr)
            return 2
    args = [a for a in argv if a not in ("--keep", "--remove-source")]
    keep = "--remove-source" not in argv
    if out_dir is not None and not keep:
        print("--out-dir leaves index_dir untouched; it cannot be combined with "
              "--remove-source", file=sys.stderr)
        return 2
    if "--keep" in argv and not keep:
        print("--keep and --remove-source cannot be combined", file=sys.stderr)
        return 2
    commands = {"to-packed": to_packed, "to-legacy": to_legacy, "info": info}
    if len(args) != 2 or args[0] not in commands:
        print(__doc__, file=sys.stderr)
        return 2
    command, index_dir = args
    if out_dir is not None:
        if command == "info":
            print("--out-dir does not apply to info", file=sys.stderr)
            return 2
        os.makedirs(out_dir, exist_ok=True)
        if os.path.samefile(out_dir, index_dir):
            out_dir = None
    if not os.path.isfile(os.path.join(index_dir, "scann_config.pb")):
        print(f"{index_dir} does not hold a dumped index (no scann_config.pb)", file=sys.stderr)
        return 1
    try:
        if command == "info":
            info(index_dir)
        else:
            commands[command](index_dir, keep, out_dir, memory_gb)
    except (ConvertError, OSError) as e:
        print(f"error: {e}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
