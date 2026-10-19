#!/usr/bin/env python3
"""The serving engine's log-probs over the quality panel, for p6_kld.py --mode serving (standard library only).

The capture server runs with STEP5_CAPTURE_DIR (step5.py: every compute_logits call saves the fp16 log-probs of its
rows as cap_NNNNN.npy), STEP5_CAPTURE_MIN_ROWS=1 and STEP5_CAPTURE_STRIDE=1, without speculative decoding or prefix
caching. `run` sends every --stride-th panel sequence (every domain, in panel order) one at a time as a prompt-logprob
request (max_tokens 1, top-5 prompt log-probs) and appends to <capdir>/manifest.jsonl, per sequence, the capture files
its request wrote and the server's own prompt log-probs. kld_eval.py pack places every captured row at its position
from those: chunked prefill also captures a one-row sampling call per chunk.

    python3 kld_capture.py stats PANEL INDEX [--stride K] [--max-seqs N]
    python3 kld_capture.py run PANEL CAPDIR [--stride K] [--max-seqs N] [--base http://127.0.0.1:8888]
    python3 kld_capture.py check PANEL CAPDIR    places every file as pack will, reading two values a row
"""
from __future__ import annotations

import argparse
import array
import ast
import json
import os
import struct
import sys
import time
import urllib.request

DTYPES = {"I64": "q", "I32": "i", "I16": "h", "U8": "B", "BOOL": "B", "F32": "f"}
TOP = 5


def read_safetensors(path: str) -> dict[str, list]:
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        header = json.loads(f.read(n))
        base = 8 + n
        out = {}
        for name, t in header.items():
            if name == "__metadata__":
                continue
            a, b = t["data_offsets"]
            f.seek(base + a)
            arr = array.array(DTYPES[t["dtype"]])
            arr.frombytes(f.read(b - a))
            if sys.byteorder != "little":
                arr.byteswap()
            out[name] = arr.tolist()
    return out


def panel(path: str, stride: int, max_seqs: int) -> list[tuple[int, list[int]]]:
    """(panel index, tokens) of every stride-th sequence, at most max_seqs of them (0: all)."""
    d = read_safetensors(path)
    toks, offs = d["tokens"], d["offsets"]
    picked = list(range(0, len(offs) - 1, stride))[:max_seqs or None]
    return [(i, toks[offs[i]:offs[i + 1]]) for i in picked]


def npy_header(path: str) -> tuple[int, dict]:
    """(data offset, header dict) of a .npy file."""
    with open(path, "rb") as f:
        if f.read(6) != b"\x93NUMPY":
            raise SystemExit(f"{path}: not a .npy file")
        major = f.read(2)[0]
        size = struct.unpack("<H" if major == 1 else "<I", f.read(2 if major == 1 else 4))[0]
        header = ast.literal_eval(f.read(size).decode("latin1"))
        return f.tell(), header


def npy_rows(path: str) -> int:
    return int(npy_header(path)[1]["shape"][0])


def captures(capdir: str) -> set[str]:
    return {f for f in os.listdir(capdir) if f.startswith("cap_") and f.endswith(".npy") and ".tmp" not in f}


def cmd_stats(a) -> int:
    seqs = panel(a.panel, a.stride, a.max_seqs)
    index = json.load(open(a.index))
    by: dict[str, list[int]] = {}
    for i, s in seqs:
        d = by.setdefault(str(index[i].get("domain", "all")), [0, 0])
        d[0] += 1
        d[1] += len(s)
    total = sum(len(s) for _, s in seqs)
    print(f"{len(seqs)} sequences, {total:,} tokens, longest {max(len(s) for _, s in seqs):,}")
    for k, (n, t) in sorted(by.items(), key=lambda kv: -kv[1][1]):
        print(f"  {k:14s} {n:4d} sequences {t:9,d} tokens")
    return 0


def post(base: str, body: dict, timeout: float) -> dict:
    req = urllib.request.Request(base + "/v1/completions", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


def cmd_run(a) -> int:
    seqs = panel(a.panel, a.stride, a.max_seqs)
    if not os.path.isdir(a.capdir) or not os.access(a.capdir, os.W_OK):
        raise SystemExit(f"{a.capdir}: need an existing writable capture directory")
    path = os.path.join(a.capdir, "manifest.jsonl")
    if os.path.exists(path):
        raise SystemExit(f"{path} exists: one run a capture directory")
    with open(path, "w") as m:
        m.write(json.dumps({"ignored": sorted(captures(a.capdir)), "sequences": [i for i, _ in seqs], "top": TOP})
                + "\n")
        t0, done = time.time(), 0
        for k, (i, s) in enumerate(seqs):
            before = captures(a.capdir)
            r = post(a.base, {"model": a.model, "prompt": s, "max_tokens": 1, "temperature": 0.0,
                              "prompt_logprobs": TOP}, a.timeout)
            new = sorted(captures(a.capdir) - before)
            plp = r["choices"][0].get("prompt_logprobs")
            if not plp or len(plp) != len(s):
                raise SystemExit(f"sequence {i}: {len(plp or [])} prompt log-prob entries for {len(s)} tokens")
            # position j (1..n-1): {token id: log-prob} over the prompt token and the top-5
            lp = [None] + [{t: v["logprob"] for t, v in e.items()} for e in plp[1:]]
            m.write(json.dumps({"seq": i, "tokens": len(s), "files": [[f, npy_rows(os.path.join(a.capdir, f))]
                                                                          for f in new], "prompt_logprobs": lp}) + "\n")
            m.flush()
            done += len(s)
            print(f"{time.strftime('%H:%M:%S')} sequence {k + 1}/{len(seqs)} (panel {i}): {len(s)} tokens, "
                  f"{len(new)} files, {done / (time.time() - t0):.0f} tokens/s", flush=True)
    return 0


ABS_TOL, REL_TOL = 0.02, 0.004      # fp16 log-probs against the server's fp32 ones (kld_eval.py uses the same)


def file_matches(path: str, start: int, toks: list[int], plp: list) -> bool:
    """Whether the file's rows are positions start.. : each row's log-probs of the true next token and of the server's
    top-1 token agree with the server's (two values read a row)."""
    off, h = npy_header(path)
    if h["descr"] != "<f2" or h["fortran_order"] or len(h["shape"]) != 2:
        raise SystemExit(f"{path}: want a C-order fp16 [rows, vocab] array, got {h}")
    rows, cols = h["shape"]
    with open(path, "rb") as f:
        def value(r: int, c: int) -> float:
            f.seek(off + 2 * (r * cols + c))
            return struct.unpack("<e", f.read(2))[0]

        for j in range(rows):
            p = start + j
            if p + 1 >= len(toks):
                return False
            ref = plp[p + 1]
            top = max(ref.items(), key=lambda kv: kv[1])
            for tok, want in ((str(toks[p + 1]), ref[str(toks[p + 1])]), top):
                if abs(value(j, int(tok)) - want) > ABS_TOL + REL_TOL * abs(want):
                    return False
    return True


def same_rows(path_a: str, row_a: int, path_b: str, row_b: int, cols: list[int]) -> bool:
    """Whether two capture rows agree in the given columns (within the tolerance: a one-row and a many-row lm_head
    call round differently)."""
    out = []
    for path, row in ((path_a, row_a), (path_b, row_b)):
        off, h = npy_header(path)
        with open(path, "rb") as f:
            vals = []
            for c in cols:
                f.seek(off + 2 * (row * h["shape"][1] + c))
                vals.append(struct.unpack("<e", f.read(2))[0])
            out.append(vals)
    return all(abs(x - y) <= ABS_TOL + REL_TOL * abs(x) for x, y in zip(*out))


def placement(capdir: str, rec: dict, toks: list[int]) -> tuple[list[tuple[str, int]], str | None, str]:
    """A request's files as (file, first position) prompt-row blocks and the final sampling file (position n - 1).

    Each engine step of a one-request prompt-logprob pass writes a one-row sampling call (the step's last position)
    and then the step's prompt rows; the final step's prompt rows stop at n - 2 and are absent when the step holds one
    token. Every block must match the server's log-probs at its positions and every sampling row but the last must
    equal the last row of its step's block. Returns the error text, or "" when the placement holds.
    """
    n, plp, files = rec["tokens"], rec["prompt_logprobs"], rec["files"]
    blocks, p = [], 0
    for k in range(0, len(files), 2):
        s_name, s_rows = files[k]
        if s_rows != 1:
            return blocks, None, f"call {k}: {s_name} has {s_rows} rows where a one-row sampling call belongs"
        if k + 1 == len(files):
            break
        b_name, b_rows = files[k + 1]
        if p + b_rows > n - 1 or not file_matches(os.path.join(capdir, b_name), p, toks, plp):
            return blocks, None, f"call {k + 1}: {b_name} ({b_rows} rows) does not match positions {p}.."
        blocks.append((b_name, p))
        p += b_rows
        if p < n - 1:      # not the final step: its sampling row is the block's last position
            cols = [toks[p], int(max(plp[p].items(), key=lambda kv: kv[1])[0])]
            if not same_rows(os.path.join(capdir, s_name), 0, os.path.join(capdir, b_name), b_rows - 1, cols):
                return blocks, None, f"call {k}: sampling row {s_name} differs from the last row of {b_name}"
    if p != n - 1:
        return blocks, None, f"positions 0..{p - 1} placed, want 0..{n - 2}"
    return blocks, files[-1][0] if len(files) % 2 else files[-2][0], ""


def cmd_check(a) -> int:
    """Place every capture file of the manifest as kld_eval.py pack will, without numpy (safe beside a server)."""
    lines = [json.loads(x) for x in open(os.path.join(a.capdir, "manifest.jsonl"))]
    seqs = dict(panel(a.panel, 1, 0))
    bad = 0
    for rec in lines[1:]:
        blocks, final, err = placement(a.capdir, rec, seqs[rec["seq"]])
        bad += bool(err)
        print(f"panel {rec['seq']}: {rec['tokens']} tokens, {len(rec['files'])} files, {len(blocks)} prompt blocks, "
              f"final sampling row {final}: {err or 'OK'}", flush=True)
    print("PASS" if not bad else f"FAIL: {bad} sequences", flush=True)
    return 1 if bad else 0


def main() -> int:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("check")
    c.add_argument("panel")
    c.add_argument("capdir")
    s = sub.add_parser("stats")
    s.add_argument("panel")
    s.add_argument("index")
    s.add_argument("--stride", type=int, default=1)
    s.add_argument("--max-seqs", type=int, default=0)
    r = sub.add_parser("run")
    r.add_argument("panel")
    r.add_argument("capdir")
    r.add_argument("--stride", type=int, default=1)
    r.add_argument("--max-seqs", type=int, default=0)
    r.add_argument("--base", default="http://127.0.0.1:8888")
    r.add_argument("--model", default="step5")
    r.add_argument("--timeout", type=float, default=1800)
    a = ap.parse_args()
    return {"stats": cmd_stats, "run": cmd_run, "check": cmd_check}[a.cmd](a)


if __name__ == "__main__":
    raise SystemExit(main())
