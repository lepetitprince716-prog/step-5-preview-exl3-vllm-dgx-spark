# SPDX-License-Identifier: Apache-2.0
"""Bit allocation (ported from sudoingx dsv41_recipe scripts/quantize/p4_allocate.py): one EXL3 K per routed expert
(gate / up / down share it) from pass-0 statistics and the measured per-layer sensitivity. Pure Python.

Model: a K-bit expert adds output noise of relative energy eps(K) = rel_err(K)^2, rel_err measured by sudoingx on
DeepSeek-V4.1 layer 0 (FINDINGS 15: K2 16.2%, K3 6.5%, K4 2.4%, K5 1.0%; --rel-err overrides, e.g. with Step 5's own
pass-1 numbers). p1_sensitivity.py gives alpha_l = KL per unit of per-token relative noise energy in layer l's routed
output, so expert e costs alpha_l * eps(K_e) * tshare_e / tokens_l (importance = sensitivity x token share; FINDINGS 13:
energy share misprices super experts).

Constraints:
  K in {2, 3, 4, 5} (integer K: the serving image's exllamav3 1.5.3 decodes integer K only), one K per expert,
  at most --max-kinds (3) distinct K per layer (each kind is one more kernel launch per MoE layer; when the greedy
  produces more, the smallest class is merged into its neighbouring class, downward when its floors allow),
  memory budget: average K over all routed experts <= --bpw,
  optional decode cap: usage-weighted average K <= --decode-bpw (what decode reads per token; hot experts pay),
  super experts at K >= --super-k (5): >= --super-share of their layer's routed-output energy, or with --super-max
  Su et al.'s criterion (arXiv 2507.23279 Eq. 6: p0's out_max > pooled P99.5 and > a_max / 10, first 75% of the MoE
  layers, i.e. 3-68), either one triggers,
  --layer-guard 5,4: each layer's two most important experts at K >= 5 and >= 4 (GEMQ Table 17's insurance),
  --floor-layers (default 88-90:3: the layers Step 5 ships in BF16 are not starved).
--rel-err takes the table inline or p1b_relerr.py's JSON (pooled "rel_err" plus optional per-layer "band" tables).
The predicted cost is printed before and after the kinds limit (a gap under ~1% means no ILP is needed).
Minimises the total cost by greedy marginal upgrades (cost convex in K -> Lagrangian optimum on the grid), the decode
cap by bisection on a usage penalty (sudoingx's method).

--prune-cold N records the N least-used experts of every layer under plan["prune"] (experimental; nothing applies it).

  python3 p2_allocate.py /out/stats_w --alpha /out/alpha.json --bpw 2.4 --decode-bpw 2.6 --rel-err /out/rel_err.json
      --super-share 0.20 --super-k 5 --super-max --floor-layers 88-90:3 --max-kinds 3
      --attn-k 4 --shared-k 4 --mlp-k 4 -o /plan/plan.json
  python3 p2_allocate.py /out/stats --alpha /out/alpha.json --score /plan/plan.json
"""
from __future__ import annotations

import argparse
import heapq
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from p_util import atomic_json, parse_layers, percentile  # noqa: E402

from step5q.plan import bytes_per_node, expert_bytes, validate  # noqa: E402  (lane A; pure Python)

GRID = (2, 3, 4, 5)
REL_ERR = {2: 0.162, 3: 0.065, 4: 0.024, 5: 0.010}


def load_stats(stats_dir: str, layers: list[int] | None = None) -> dict[int, dict]:
    out = {}
    for f in sorted(os.listdir(stats_dir)):
        if f.startswith("layer_") and f.endswith(".json"):
            l = int(f[6:-5])
            if layers is None or l in layers:
                rec = json.load(open(os.path.join(stats_dir, f)))
                if "count" in rec:
                    out[l] = rec
    if not out:
        raise SystemExit(f"no layer_LL.json with expert statistics in {stats_dir}")
    return out


def load_rel_err(spec: str) -> tuple[dict[int, float], list[tuple[int, int, dict[int, float]]]]:
    """--rel-err: "2:0.162,3:0.065,..." or a JSON file (p1b_relerr.py output): {"rel_err": {"2": ...}, "band":
    {"3-16": {"2": ...}, ...}} (a flat {"2": ...} also works). -> (pooled table, [(first, last, table)])."""
    def table(d: dict) -> dict[int, float]:
        t = {int(k): float(v) for k, v in d.items()}
        if set(t) != set(GRID):
            raise SystemExit(f"--rel-err: need K {GRID}, got {sorted(t)}")
        return t
    if not spec:
        return dict(REL_ERR), []
    if spec.endswith(".json") or os.path.isfile(spec):
        d = json.load(open(spec))
        bands = [(int(r.split("-")[0]), int(r.split("-")[-1]), table(t)) for r, t in d.get("band", {}).items()]
        return table(d.get("rel_err", {k: v for k, v in d.items() if k.isdigit()})), bands
    return table(dict(x.split(":") for x in spec.split(","))), []


def eps_of(layer: int, pooled: dict[int, float], bands: list) -> dict[int, float]:
    """Relative noise energy per K for one layer: its band's table if any, else the pooled one; eps = err^2."""
    t = next((tb for a, b, tb in bands if a <= layer <= b), pooled)
    return {k: v * v for k, v in t.items()}


def super_max_flags(st: dict[int, dict], frac: float = 0.75) -> dict[int, set[int]]:
    """Su et al. (arXiv 2507.23279, Eq. 6): an expert whose max |down_proj output| exceeds the P99.5 of the pooled
    per-expert maxima AND a_max / 10, in the first `frac` of the MoE layers. -> {layer: {expert}}."""
    if any("out_max" not in r for r in st.values()):
        raise SystemExit("--super-max needs out_max in the stats (p0_stats.py from this version)")
    pool = [v for r in st.values() for v in r["out_max"]]
    thr = max(percentile(pool, 99.5), max(pool) / 10)
    cand = sorted(st)[: int(frac * len(st))]
    return {l: {e for e, v in enumerate(st[l]["out_max"]) if v > thr} for l in cand}


def parse_floor_layers(spec: str) -> dict[int, int]:
    """'88-90:3' or '88,89:3;3-10:4' -> {layer: K floor}."""
    out: dict[int, int] = {}
    for part in filter(None, spec.split(";")):
        ls, k = part.split(":")
        for l in parse_layers(ls, []):
            out[l] = int(k)
    return out


def build_experts(st: dict[int, dict], alpha: dict[int, float], super_share: float, super_k: int,
                  floors: dict[int, int], rel_err: tuple = (REL_ERR, []), super_max: bool = False,
                  guard: tuple[int, ...] = ()) -> list[dict]:
    """One record per expert: layer, e, imp (alpha * tshare / tokens), use (slots per token), floor K, super flag
    ("share" / "max" trigger), eps per K. guard: per layer the most important experts get these K floors (5,4)."""
    by_max = super_max_flags(st) if super_max else {}
    ex = []
    for l in sorted(st):
        r = st[l]
        tok = max(float(r["tokens"]), 1.0)
        etot = max(sum(r["out_energy"]), 1e-30)
        if l not in alpha:
            raise SystemExit(f"no sensitivity for layer {l} in --alpha")
        eps = eps_of(l, *rel_err)
        row = []
        for e, (c, ts, en) in enumerate(zip(r["count"], r["tshare"], r["out_energy"])):
            why = [t for t, hit in (("share", en / etot >= super_share), ("max", e in by_max.get(l, ()))) if hit]
            fl = max(GRID[0], floors.get(l, GRID[0]), super_k if why else GRID[0])
            row.append({"layer": l, "e": e, "imp": alpha[l] * ts / tok, "use": c / tok, "floor": min(fl, GRID[-1]),
                        "super": bool(why), "super_by": "+".join(why), "eps": eps})
        for x, k in zip(sorted(row, key=lambda x: -x["imp"]), guard):
            x["floor"] = min(max(x["floor"], k), GRID[-1])
        ex += row
    return ex


def cost(ex: list[dict], K: list[int]) -> float:
    return sum(x["imp"] * x["eps"][k] for x, k in zip(ex, K))


def greedy(ex: list[dict], budget: float, lam: float) -> list[int]:
    K = [x["floor"] for x in ex]
    tot = sum(K)
    heap = []

    def push(i: int) -> None:
        k = K[i]
        if k < GRID[-1]:
            g = ex[i]["imp"] * (ex[i]["eps"][k] - ex[i]["eps"][k + 1]) - lam * ex[i]["use"]
            if g > 0:
                heapq.heappush(heap, (-g, i))
    for i in range(len(ex)):
        push(i)
    while heap:
        _, i = heapq.heappop(heap)
        if tot + 1 > budget + 1e-9:
            break
        K[i] += 1
        tot += 1
        push(i)
    return K


def enforce_kinds(ex: list[dict], K: list[int], max_kinds: int) -> int:
    """Merge the smallest K class of every layer with more than max_kinds into its neighbouring class (down when no
    expert of it would fall below its floor, else up). Returns the number of experts moved."""
    by_layer: dict[int, list[int]] = {}
    for i, x in enumerate(ex):
        by_layer.setdefault(x["layer"], []).append(i)
    moved = 0
    for idx in by_layer.values():
        while True:
            kinds = sorted({K[i] for i in idx})
            if len(kinds) <= max_kinds:
                break
            classes = sorted(kinds, key=lambda k: (sum(1 for i in idx if K[i] == k),
                                                    sum(ex[i]["imp"] for i in idx if K[i] == k)))
            for c in classes:
                j = kinds.index(c)
                members = [i for i in idx if K[i] == c]
                if j > 0 and all(ex[i]["floor"] <= kinds[j - 1] for i in members):
                    target = kinds[j - 1]
                elif j + 1 < len(kinds):
                    target = kinds[j + 1]
                else:
                    continue
                for i in members:
                    K[i] = target
                moved += len(members)
                break
            else:
                raise SystemExit("cannot reduce K kinds (floors)")
    return moved


def repair_budget(ex: list[dict], K: list[int], budget: float, max_kinds: int) -> None:
    """Downgrade the cheapest experts (cost increase per bit) until sum(K) <= budget, never below a floor and never
    into a K that would add a kind beyond max_kinds."""
    by_layer: dict[int, list[int]] = {}
    for i, x in enumerate(ex):
        by_layer.setdefault(x["layer"], []).append(i)
    while sum(K) > budget + 1e-9:
        best, bi = None, -1
        for idx in by_layer.values():
            cnt: dict[int, int] = {}
            for i in idx:
                cnt[K[i]] = cnt.get(K[i], 0) + 1
            for i in idx:
                k = K[i] - 1               # a new kind is fine only when expert i empties its own class
                if k < ex[i]["floor"] or k < GRID[0] or (k not in cnt and len(cnt) >= max_kinds and cnt[K[i]] > 1):
                    continue
                d = ex[i]["imp"] * (ex[i]["eps"][k] - ex[i]["eps"][K[i]])
                if best is None or d < best:
                    best, bi = d, i
        if bi < 0:
            raise SystemExit(f"budget {budget / len(K):.3f} bpw unreachable under the floors")
        K[bi] -= 1


def run(ex: list[dict], budget: float, lam: float, max_kinds: int) -> tuple[list[int], list[int]]:
    """(final K, the greedy K before the kinds limit and the budget repair)."""
    Kg = greedy(ex, budget, lam)
    K = list(Kg)
    enforce_kinds(ex, K, max_kinds)
    repair_budget(ex, K, budget, max_kinds)
    return K, Kg


def decode_bpw(ex: list[dict], K: list[int]) -> float:
    u = sum(x["use"] for x in ex)
    return sum(x["use"] * k for x, k in zip(ex, K)) / max(u, 1e-30)


def allocate(ex: list[dict], bpw: float, dec_cap: float = 0.0,
             max_kinds: int = 3) -> tuple[list[int], float, list[int]]:
    """(K, decode-cap lambda, greedy K before enforce_kinds at that lambda)."""
    budget = bpw * len(ex)
    if sum(x["floor"] for x in ex) > budget + 1e-9:
        raise SystemExit(f"floors alone need {sum(x['floor'] for x in ex) / len(ex):.3f} bpw > --bpw {bpw}")
    (K, Kg), lam = run(ex, budget, 0.0, max_kinds), 0.0
    if dec_cap and decode_bpw(ex, K) > dec_cap:
        lo, hi = 0.0, max(x["imp"] * x["eps"][GRID[0]] / x["use"] for x in ex if x["use"] > 0)
        best = None
        for _ in range(60):
            mid = (lo + hi) / 2
            Km = run(ex, budget, mid, max_kinds)
            if decode_bpw(ex, Km[0]) > dec_cap:
                lo = mid
            else:
                hi, best = mid, Km
        if best is None or decode_bpw(ex, best[0]) > dec_cap + 1e-6:
            raise SystemExit(f"decode cap {dec_cap} unreachable")
        (K, Kg), lam = best, hi
    return K, lam, Kg


def bytes_report(ex: list[dict], K: list[int], tp: int = 2) -> dict:
    """Routed-expert bytes on one node (STEP5_EXPERT_TP=1 slicing, step5q.plan.expert_bytes) and the expert bytes one
    decoded token reads per node (each expert weighted by its routed slots per token)."""
    exp_b = sum(expert_bytes(k, tp) for k in K)
    dec_b = sum(x["use"] * expert_bytes(k, tp) for x, k in zip(ex, K))
    return {"tp": tp, "expert_tp": True, "experts": int(exp_b), "experts_gib": round(exp_b / 2 ** 30, 2),
            "decode_expert_bytes_per_token": int(dec_b)}


def make_plan(ex: list[dict], K: list[int], args, lam: float, Kg: list[int]) -> dict:
    body = {"attn": args.attn_k, "shared": args.shared_k, "mlp": args.mlp_k, "head": "bf16"}
    layers: dict[str, dict] = {}
    for x, k in zip(ex, K):
        layers.setdefault(str(x["layer"]), {"k": []})["k"].append(int(k))
    plan = {"format": "step5_exl3_plan", "version": 1, "codebook": "mul1", "bpw_target": args.bpw,
            "n_experts": len(layers[next(iter(layers))]["k"]), "layers": layers, "body": body,
            "bytes_per_node": bytes_report(ex, K)}
    if args.prune_cold:
        plan["prune"] = prune_cold(ex, args.prune_cold)
    hist = {str(k): K.count(k) for k in GRID}
    plan["summary"] = {"bpw": round(sum(K) / len(K), 4), "decode_bpw": round(decode_bpw(ex, K), 4), "hist": hist,
                       "lambda_decode": lam, "predicted_cost": cost(ex, K), "cost_before_kinds": cost(ex, Kg),
                       "super_experts": sum(1 for x in ex if x["super"]),
                       "super_by": {t: sum(1 for x in ex if x["super_by"] == t) for t in ("share", "max", "share+max")},
                       "rel_err": {str(k): v for k, v in sorted(args.rel_err_map[0].items())},
                       "rel_err_bands": {f"{a}-{b}": {str(k): v for k, v in t.items()} for a, b, t in args.rel_err_map[1]},
                       "floor_layers": args.floor_layers, "max_kinds": args.max_kinds, "super_max": args.super_max,
                       "layer_guard": args.layer_guard}
    try:                                     # a full plan (layers 3-90): lane A's checks and per-component bytes
        validate(plan)
    except ValueError as e:
        print(f"note: not a complete serving plan ({e}); per-component bytes omitted")
    else:
        plan["bytes_per_node"]["components"] = bytes_per_node(plan)
    return plan


def prune_cold(ex: list[dict], n: int) -> dict[str, list[int]]:
    """The n least-used experts per layer (by routed slot count, ties by index)."""
    out: dict[str, list[int]] = {}
    for l in sorted({x["layer"] for x in ex}):
        xs = sorted((x for x in ex if x["layer"] == l), key=lambda x: (x["use"], x["e"]))
        out[str(l)] = sorted(x["e"] for x in xs[:n])
    return out


def parse_args(argv: list[str] | None = None):
    ap = argparse.ArgumentParser()
    ap.add_argument("stats")
    ap.add_argument("--alpha", required=True, help="p1_sensitivity.py output {alpha: {layer: KL per unit noise}}")
    ap.add_argument("--bpw", type=float, default=2.4, help="average K over all routed experts (memory budget)")
    ap.add_argument("--decode-bpw", type=float, default=0.0, help="cap on the usage-weighted average K (0 = none)")
    ap.add_argument("--rel-err", default="", help="relative output error per K (eps = err^2): '2:0.162,3:...' or "
                    "p1b_relerr.py's JSON (optional per-layer bands); default sudoingx's DSV4.1 layer-0 table")
    ap.add_argument("--super-share", type=float, default=0.20)
    ap.add_argument("--super-k", type=int, default=5)
    ap.add_argument("--super-max", action="store_true", help="also Su et al.'s out_max criterion (first 75%% of the "
                    "MoE layers, out_max > pooled P99.5 and > a_max/10) for the super-expert floor")
    ap.add_argument("--layer-guard", default="", help="K floors for each layer's most important experts, e.g. 5,4")
    ap.add_argument("--floor-layers", default="88-90:3")
    ap.add_argument("--max-kinds", type=int, default=3)
    ap.add_argument("--layers", default="", help="restrict to these layers' stats (default all files)")
    ap.add_argument("--attn-k", type=int, default=4)
    ap.add_argument("--shared-k", type=int, default=4)
    ap.add_argument("--mlp-k", type=int, default=4)
    ap.add_argument("--prune-cold", type=int, default=0)
    ap.add_argument("--score", default="", help="score this plan instead of allocating")
    ap.add_argument("-o", "--out", default="")
    a = ap.parse_args(argv)
    a.rel_err_map = load_rel_err(a.rel_err)
    a.guard = tuple(int(k) for k in a.layer_guard.split(",")) if a.layer_guard else ()
    return a


def main(argv: list[str] | None = None) -> dict | None:
    a = parse_args(argv)
    st = load_stats(a.stats, parse_layers(a.layers, []) or None)
    alpha = {int(k): float(v) for k, v in json.load(open(a.alpha))["alpha"].items()}
    ex = build_experts(st, alpha, a.super_share, a.super_k, parse_floor_layers(a.floor_layers), a.rel_err_map,
                       a.super_max, a.guard)
    if a.score:
        pl = json.load(open(a.score))
        K = [int(pl["layers"][str(x["layer"])]["k"][x["e"]]) for x in ex]
        print(json.dumps({"plan": a.score, "bpw": sum(K) / len(K), "decode_bpw": decode_bpw(ex, K),
                          "predicted_cost": cost(ex, K)}))
        return None
    if not a.out:
        raise SystemExit("-o plan.json is required")
    K, lam, Kg = allocate(ex, a.bpw, a.decode_bpw, a.max_kinds)
    cb, ca = cost(ex, Kg), cost(ex, K)
    print(f"predicted_cost before enforce_kinds {cb:.6g}, after {ca:.6g} ({100 * (ca - cb) / max(cb, 1e-30):+.2f}%)")
    plan = make_plan(ex, K, a, lam, Kg)
    atomic_json(a.out, plan, indent=1)
    print(json.dumps({**plan["summary"], "bytes_per_node": plan["bytes_per_node"]}, indent=1))
    for l in sorted(st):
        row = plan["layers"][str(l)]["k"]
        print(f"layer {l:2d}: avg K {sum(row) / len(row):.3f}  " + " ".join(f"K{k}:{row.count(k)}" for k in GRID
                                                                           if row.count(k)))
    return plan


if __name__ == "__main__":
    main()
