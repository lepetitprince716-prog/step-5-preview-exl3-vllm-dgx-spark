#!/usr/bin/env python3
"""Equal-bytes uniform control plan: copy a p2_allocate.py plan and replace every routed expert's K with a per-layer
constant (K3 for the first N MoE layers and the floor layers, K2 elsewhere) so that the average K matches the plan.

    p2_uniform.py plans/plan.json plans/plan-uniform.json [--k3-layers 3-34,88-90]

The default K3 set (32 + 3 layers of 88) gives (35*3 + 53*2) / 88 = 2.398 bpw against the 2.4 bpw target.
"""
from __future__ import annotations

import argparse
import json
import sys


def parse_ranges(spec: str) -> set[int]:
    out: set[int] = set()
    for part in spec.split(","):
        lo, _, hi = part.partition("-")
        out.update(range(int(lo), int(hi or lo) + 1))
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("plan")
    ap.add_argument("out")
    ap.add_argument("--k3-layers", default="3-34,88-90")
    a = ap.parse_args()
    k3 = parse_ranges(a.k3_layers)
    plan = json.load(open(a.plan))
    total = 0
    count = 0
    for layer, entry in plan["layers"].items():
        k = 3 if int(layer) in k3 else 2
        entry["k"] = [k] * len(entry["k"])
        total += k * len(entry["k"])
        count += len(entry["k"])
    plan["summary"] = {"control": "uniform", "k3_layers": sorted(k3), "bpw": total / count, "from": a.plan}
    json.dump(plan, open(a.out, "w"), indent=1)
    print(f"{a.out}: {len(k3)} K3 layers, average K {total / count:.4f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
