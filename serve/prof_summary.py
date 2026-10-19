#!/usr/bin/env python3
"""Summarise a torch-profiler (Kineto) trace of decode: GPU busy time by kernel, idle gaps, per-step timing.

    python3 serve/prof_summary.py TRACE.json[.gz] [--top 40] [--step-gap-us 1000]

Steps are split at GPU idle gaps longer than --step-gap-us on the busiest stream. Kernel names are grouped into
coarse classes by substring (see CLASSES); unmatched names fall into "other".
"""
from __future__ import annotations

import argparse
import collections
import gzip
import json
import re
import statistics

CLASSES = [
    ("exl3_moe", r"exl3_moe|moe_kernel|exl3.*moe"),
    ("exl3_gemv", r"exl3|gemv|trellis|had_|hadamard|reconstruct"),
    ("allreduce", r"allreduce|all_reduce|rocenante|b12x|nccl|oneshot|one_shot|allgather|all_gather"),
    ("attention", r"flash|fmha|attn|attention|paged|reshape_and_cache|kv_cache"),
    ("gemm", r"gemm|cutlass|cublas|sm\d+_xmma|ampere|hopper|matmul|splitk|nvjet"),
    ("norm", r"rms|norm"),
    ("rope", r"rotary|rope"),
    ("topk_route", r"topk|sort|argsort|scatter|gather|index|bincount|cumsum|radix"),
    ("sampling", r"argmax|softmax|sample|rejection|logit|eagle|spec"),
    ("elementwise", r"elementwise|vectorized|unrolled|copy|cast|fill|add|mul|silu|sigmoid|reduce"),
]


def classify(name: str) -> str:
    low = name.lower()
    for cls, pat in CLASSES:
        if re.search(pat, low):
            return cls
    return "other"


def load(path: str) -> dict:
    op = gzip.open if path.endswith(".gz") else open
    with op(path, "rt") as f:
        return json.load(f)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("trace")
    ap.add_argument("--top", type=int, default=40)
    ap.add_argument("--step-gap-us", type=float, default=1000.0)
    a = ap.parse_args()
    tr = load(a.trace)
    ev = tr["traceEvents"] if isinstance(tr, dict) else tr
    kern = [e for e in ev if e.get("ph") == "X" and e.get("cat") in ("kernel", "gpu_memcpy", "gpu_memset")]
    if not kern:
        print("no GPU kernel events in", a.trace)
        cats = collections.Counter(e.get("cat") for e in ev)
        print("categories:", cats.most_common(20))
        return 1
    by_stream = collections.Counter()
    for e in kern:
        by_stream[(e.get("pid"), e.get("tid"))] += e["dur"]
    main_stream = by_stream.most_common(1)[0][0]
    ks = sorted((e for e in kern if (e.get("pid"), e.get("tid")) == main_stream), key=lambda e: e["ts"])
    t0, t1 = ks[0]["ts"], ks[-1]["ts"] + ks[-1]["dur"]
    busy = sum(e["dur"] for e in ks)
    gaps = [ks[i + 1]["ts"] - (ks[i]["ts"] + ks[i]["dur"]) for i in range(len(ks) - 1)]
    print(f"streams with kernels: {len(by_stream)}; main stream {main_stream}: {len(ks)} kernels, "
          f"window {(t1 - t0) / 1e3:.1f} ms, busy {busy / 1e3:.1f} ms ({100 * busy / (t1 - t0):.1f}%)")
    for (pid, tid), d in by_stream.most_common(6):
        print(f"   stream pid={pid} tid={tid}: busy {d / 1e3:.1f} ms")
    # steps: split at long idle gaps
    steps, cur = [], [ks[0]]
    for i, g in enumerate(gaps):
        if g > a.step_gap_us:
            steps.append(cur); cur = []
        cur.append(ks[i + 1])
    steps.append(cur)
    spans = [(s[-1]["ts"] + s[-1]["dur"] - s[0]["ts"]) / 1e3 for s in steps]
    sbusy = [sum(e["dur"] for e in s) / 1e3 for s in steps]
    nks = [len(s) for s in steps]
    long_gaps = [g / 1e3 for g in gaps if g > a.step_gap_us]
    print(f"segments split at gaps > {a.step_gap_us:.0f} us: {len(steps)}; span median {statistics.median(spans):.2f} ms, "
          f"busy median {statistics.median(sbusy):.2f} ms, kernels median {statistics.median(nks):.0f}; "
          f"gap median {statistics.median(long_gaps) if long_gaps else 0:.2f} ms")
    small = [g for g in gaps if 0 < g <= a.step_gap_us]
    print(f"in-segment idle: {sum(small) / 1e3:.1f} ms over {len(small)} gaps (median {statistics.median(small) if small else 0:.1f} us)")
    # per class and per kernel
    per_cls, per_k, cnt_k = collections.Counter(), collections.Counter(), collections.Counter()
    for e in ks:
        per_cls[classify(e["name"])] += e["dur"]
        per_k[e["name"]] += e["dur"]
        cnt_k[e["name"]] += 1
    print("\nbusy time by class (main stream):")
    for c, d in per_cls.most_common():
        print(f"  {c:12s} {d / 1e3:9.1f} ms  {100 * d / busy:5.1f}%")
    print(f"\ntop {a.top} kernels by total time:")
    for n, d in per_k.most_common(a.top):
        print(f"  {d / 1e3:8.1f} ms {100 * d / busy:5.1f}%  x{cnt_k[n]:<6d} avg {d / cnt_k[n]:7.1f} us  {n[:110]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
