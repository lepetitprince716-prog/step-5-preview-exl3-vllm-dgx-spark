# SPDX-License-Identifier: Apache-2.0
"""Domain-weighted pass-0 statistics (ported from sudoingx dsv41_recipe scripts/quantize/p4_merge_stats.py,
Apache-2.0). The allocator's importance alpha_l * tshare_e / tokens_l is a per-token average over the calibration tokens,
so the calibration token mix decides it (sudoingx FINDINGS 16: a code-heavy set lost IFEval). This writes stats whose
per-token averages are a chosen mix of domains instead:

  x_merged[e] / tokens_merged = sum_d w_d * x_d[e] / tokens_d          (x = count, top1, sum_g, sum_g2, out_energy, tshare)

with each domain's sums pooled across the input runs first (p0_stats.py --index output: "domains" per layer). Domains
without a weight (or with no tokens) are dropped; weights are renormalised over the domains present. out_max and the
resid maxima are maxima over the runs; the per-layer energies are token-weighted. The output dir is a drop-in for
p2_allocate.py's stats argument. Pure Python.

Default weights (sudoingx's start point): the eight panel domains code, long, agentic, chat, writing, knowledge, math,
multilingual at 1, instruct 1.5, std (exllamav3 standard text) 0.5 (his random-token rows no longer exist).

  python3 p0b_merge_stats.py /out/stats_w /out/stats [/out/stats_more ...] [--weights code=1,chat=1,...]
"""
from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from p_util import DEFAULT_DOMAIN_WEIGHTS, atomic_json  # noqa: E402

FIELDS = ("count", "top1", "sum_g", "sum_g2", "out_energy", "tshare")
ENERGIES = ("routed_out_energy", "shared_out_energy", "resid_in_energy", "resid_out_energy")
MAXIMA = ("resid_in_max", "resid_out_max")


def parse_weights(spec: str) -> dict[str, float]:
    if not spec:
        return dict(DEFAULT_DOMAIN_WEIGHTS)
    return {k: float(v) for k, v in (x.split("=") for x in spec.split(","))}


def merge_layer(recs: list[dict], w: dict[str, float]) -> dict:
    pooled: dict[str, dict] = {}                     # domain -> {"tokens", fields...} summed over runs
    for rec in recs:
        if "domains" not in rec:
            raise SystemExit(f"layer {rec.get('layer')}: no per-domain stats (run p0_stats.py with --index)")
        for d, s in rec["domains"].items():
            n_e = len(s["count"])
            p = pooled.setdefault(d, {"tokens": 0.0, **{k: [0.0] * n_e for k in FIELDS}})
            p["tokens"] += s["tokens"]
            for k in FIELDS:
                p[k] = [x + y for x, y in zip(p[k], s.get(k, [0.0] * n_e))]
    use = {d: w[d] for d in pooled if w.get(d, 0) > 0 and pooled[d]["tokens"] > 0}
    if not use:
        raise SystemExit(f"layer {recs[0].get('layer')}: no weighted domain present (have {sorted(pooled)})")
    tot_w = sum(use.values())
    T = sum(pooled[d]["tokens"] for d in use)
    n_e = len(next(iter(pooled.values()))["count"])
    out = {"tokens": T, "layer": recs[0]["layer"], "n_experts": n_e}
    for k in FIELDS:
        out[k] = [T * sum(use[d] / tot_w * pooled[d][k][e] / pooled[d]["tokens"] for d in use) for e in range(n_e)]
    tok_all = sum(r["tokens"] for r in recs)
    for k in ENERGIES:
        if all(k in r for r in recs):
            out[k] = sum(r[k] for r in recs) * T / tok_all
    for k in MAXIMA:
        if all(k in r for r in recs):
            out[k] = max(r[k] for r in recs)
    if all("out_max" in r for r in recs):
        out["out_max"] = [max(v) for v in zip(*(r["out_max"] for r in recs))]
    out["merge"] = {"weights": {d: round(use[d] / tot_w, 4) for d in sorted(use)},
                    "tokens": {d: pooled[d]["tokens"] for d in sorted(pooled)},
                    "dropped": sorted(set(pooled) - set(use))}
    return out


def main(argv: list[str] | None = None) -> dict:
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("runs", nargs="+", help="p0_stats.py output dirs (layer_LL.json with domains)")
    ap.add_argument("--weights", default="", help="domain=weight,... (renormalised; default sudoingx's start point)")
    a = ap.parse_args(argv)
    w = parse_weights(a.weights)
    os.makedirs(a.out, exist_ok=True)
    layers = sorted({f for r in a.runs for f in os.listdir(r) if f.startswith("layer_") and f.endswith(".json")})
    report = {}
    for f in layers:
        recs = [json.load(open(os.path.join(r, f))) for r in a.runs if os.path.exists(os.path.join(r, f))]
        if len(recs) != len(a.runs):
            raise SystemExit(f"{f}: missing in some runs")
        if "count" not in recs[0]:
            continue                                 # dense layers carry no expert statistics
        out = merge_layer(recs, w)
        out["merge"]["runs"] = [os.path.basename(r.rstrip("/")) for r in a.runs]
        atomic_json(os.path.join(a.out, f), out)
        report = out["merge"]
    print(f"{len(layers)} layer files merged into {a.out}")
    print("weights:", report.get("weights"))
    print("tokens per domain:", {d: int(t) for d, t in report.get("tokens", {}).items()},
          "dropped:", report.get("dropped"))
    return report


if __name__ == "__main__":
    main()
