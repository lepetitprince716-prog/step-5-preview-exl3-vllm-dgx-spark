"""plan.json: per-expert EXL3 K for layers 3-90 plus the body format, and the bytes it costs per Spark.

{"bpw_target": 2.4, "layers": {"3": {"k": [352 ints]}, ..., "90": {...}},
 "body": {"attn": 4, "shared": 4, "mlp": 4, "head": "bf16"}}

bytes_per_node follows the budget in research/2026-10-15-step5-2spark-plan.md (TP2, STEP5_EXPERT_TP=1): per EXL3
linear trellis in*out*K/8 bytes + suh fp16 (in) + svh fp16 (out) + a 4-byte codebook marker, sliced the way 0xSero's
loader slices it (exl3_quant.py: experts gate/up by output, down by input at 128-multiples; body q/k/v, shared and
dense gate/up by output, o_proj and down by input); BF16 embed and lm_head vocab-split; FP32 router gate on every
node; MTP layers 92-94 BF16 (attention, dense MLP, eh_proj) TP-split, their three shared heads vocab-split. Omitted:
items under 50 MB per node (g_proj, norms, router_bias, MTP enorm/hnorm), the vision tower (text-only serving).
"""
from __future__ import annotations

import json
import os

HIDDEN, EXPERT_INTER, DENSE_INTER, VOCAB = 4096, 1536, 13824, 128896
N_EXPERTS = 352
HEADS, KV_HEADS, HEAD_DIM = 64, 4, 192
MAIN_LAYERS = range(92)
MOE_LAYERS = range(3, 91)
DENSE_LAYERS = (0, 1, 2, 91)
MTP_LAYERS = 3
BF16, FP32, MARKER = 2, 4, 4
EXPERT_K = (2, 3, 4, 5)
MAX_K_PER_LAYER = 3
BODY_KEYS = ("attn", "shared", "mlp", "head")
GIB = 2 ** 30


def validate(plan: dict) -> None:
    """Raise ValueError unless the plan is complete and within the serving constraints."""
    float(plan["bpw_target"])
    layers = plan["layers"]
    if sorted(layers, key=int) != [str(l) for l in MOE_LAYERS]:
        raise ValueError(f"plan layers must be exactly {MOE_LAYERS.start}..{MOE_LAYERS.stop - 1}")
    for l, ent in layers.items():
        ks = ent["k"]
        if len(ks) != N_EXPERTS or any(type(k) is not int or k not in EXPERT_K for k in ks):
            raise ValueError(f"layer {l}: k must be {N_EXPERTS} ints in {EXPERT_K}")
        if len(set(ks)) > MAX_K_PER_LAYER:
            raise ValueError(f"layer {l}: {len(set(ks))} distinct K, at most {MAX_K_PER_LAYER}")
    body = plan["body"]
    if set(body) != set(BODY_KEYS):
        raise ValueError(f"body keys must be {BODY_KEYS}")
    for key in ("attn", "shared", "mlp"):
        if type(body[key]) is not int or not 2 <= body[key] <= 8:
            raise ValueError(f"body.{key}: integer K in 2..8, got {body[key]!r}")
    if body["head"] != "bf16":
        raise ValueError("body.head: only 'bf16' (0xSero's loader has no EXL3 lm_head)")


def read_plan(path: str) -> dict:
    with open(path) as f:
        plan = json.load(f)
    validate(plan)
    return plan


def write_plan(plan: dict, path: str) -> None:
    validate(plan)
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(plan, f)
    os.replace(tmp, path)


def exl3_linear_bytes(n_in: int, n_out: int, K: int, tp: int = 1, split: str | None = None) -> int:
    """One EXL3 linear on one of tp ranks. split "out" slices trellis columns and svh, "in" trellis rows and suh."""
    if split not in (None, "in", "out"):
        raise ValueError(f"split {split!r}")
    i = n_in // tp if split == "in" else n_in
    o = n_out // tp if split == "out" else n_out
    return i * o * K // 8 + BF16 * (i + o) + MARKER


def expert_bytes(K: int, tp: int = 2, expert_tp: bool = True) -> int:
    """One routed expert's EXL3 bytes on one node (expert_tp: every node holds a 1/tp slice of every expert;
    otherwise the expert lives whole on one node)."""
    if not expert_tp:
        tp = 1
    gu = exl3_linear_bytes(HIDDEN, EXPERT_INTER, K, tp, "out")
    return 2 * gu + exl3_linear_bytes(EXPERT_INTER, HIDDEN, K, tp, "in")


def _experts_per_node(plan: dict, tp: int, expert_tp: bool) -> int:
    if expert_tp:
        return sum(expert_bytes(k, tp) for ent in plan["layers"].values() for k in ent["k"])
    per_rank = N_EXPERTS // tp
    return max(sum(expert_bytes(k, tp, False) for ent in plan["layers"].values()
                   for k in ent["k"][r * per_rank:(r + 1) * per_rank]) for r in range(tp))


def _body_per_node(body: dict, tp: int) -> dict[str, int]:
    q, kv = HEADS * HEAD_DIM, KV_HEADS * HEAD_DIM
    ka, ks, km = body["attn"], body["shared"], body["mlp"]
    attn = (exl3_linear_bytes(HIDDEN, q, ka, tp, "out") + 2 * exl3_linear_bytes(HIDDEN, kv, ka, tp, "out")
            + exl3_linear_bytes(q, HIDDEN, ka, tp, "in"))
    shared = 2 * exl3_linear_bytes(HIDDEN, EXPERT_INTER, ks, tp, "out") + exl3_linear_bytes(
        EXPERT_INTER, HIDDEN, ks, tp, "in")
    mlp = 2 * exl3_linear_bytes(HIDDEN, DENSE_INTER, km, tp, "out") + exl3_linear_bytes(
        DENSE_INTER, HIDDEN, km, tp, "in")
    return {"attn": attn * len(MAIN_LAYERS), "shared": shared * len(MOE_LAYERS), "mlp": mlp * len(DENSE_LAYERS)}


def _bf16_per_node(tp: int) -> dict[str, int]:
    q, kv = HEADS * HEAD_DIM, KV_HEADS * HEAD_DIM
    mtp_layer = 2 * HIDDEN * q + 2 * HIDDEN * kv + 3 * HIDDEN * DENSE_INTER + 2 * HIDDEN * HIDDEN   # + eh_proj
    return {"embed_head": 2 * VOCAB * HIDDEN * BF16 // tp,
            "router": len(MOE_LAYERS) * N_EXPERTS * HIDDEN * FP32,
            "mtp": MTP_LAYERS * mtp_layer * BF16 // tp,
            "mtp_heads": MTP_LAYERS * VOCAB * HIDDEN * BF16 // tp}


def bytes_per_node(plan: dict, expert_tp: bool = True, tp: int = 2) -> dict[str, int]:
    """Weight bytes on the busiest node by component, plus "total". Routed experts at K: params * K / 8 trellis plus
    the suh/svh fp16 vectors each node keeps (with expert_tp the unsplit side is replicated)."""
    validate(plan)
    out = {"experts": _experts_per_node(plan, tp, expert_tp)}
    out.update(_body_per_node(plan["body"], tp))
    out.update(_bf16_per_node(tp))
    out["total"] = sum(out.values())
    return out


def expert_bpw(plan: dict) -> float:
    """Average routed-expert K over layers 3-90 (the trellis bits per weight, scales excluded)."""
    ks = [k for ent in plan["layers"].values() for k in ent["k"]]
    return sum(ks) / len(ks)
