#!/usr/bin/env python3
"""Compare experiment runs under ~/step5/runs/exp: decode (sets b/a), concurrency and prefill, one line per run.

    python3 serve/cmp_runs.py base cfgA [NAME ...]
"""
from __future__ import annotations

import json
import os
import sys

EXP = os.path.expanduser("~/step5/runs/exp")


def load(name: str, kind: str):
    path = os.path.join(EXP, f"{name}-{kind}.json")
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)


def main() -> int:
    for name in sys.argv[1:]:
        parts = [f"{name:10s}"]
        for s in ("b", "a"):
            d = load(name, f"decode-{s}")
            if d:
                acc = (d.get("spec") or {}).get("mean_acceptance_length")
                parts.append(f"set {s} {d['decode_tps_median_of_medians']:.2f} tok/s (acc {acc:.3f})" if acc else
                             f"set {s} {d['decode_tps_median_of_medians']:.2f} tok/s")
        c = load(name, "concurrent")
        if c:
            agg: dict[int, list[float]] = {}
            for r in c["rows"]:
                agg.setdefault(r["streams"], []).append(r["aggregate_tps"])
            parts.append("conc " + " ".join(f"{k}x {sum(v) / len(v):.1f}" for k, v in sorted(agg.items())))
        p = load(name, "prefill")
        if p:
            parts.append("prefill " + " ".join(f"{r['length'] // 1024}K {r['prefill_tps']:.0f}" for r in p["result"]))
        print(" | ".join(parts))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
