#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Synthetic Step 5 serving checkpoint for the memory / speed probe (plan, stage 2 GPU check 2). Pure Python, no torch.

Same layout, names, shapes and dtypes as p5_assemble.py's output for 0xSero's step5_exl3 loader, with random EXL3
tensors at a chosen K mix, so TP2 boot, MemAvailable, the KV pool and decode speed can be measured before any
quantization runs. The values mean nothing; the output is never a model.

  exl3/experts/L{LL}.safetensors  layers 3-90, 352 experts x {gate,up,down}_proj.{trellis,suh,svh,mul1}; one K per
                                  expert (gate/up/down alike), K from --k-mix or --plan (at most 3 distinct K per layer)
  exl3/body/L{LL}.safetensors     layers 0-91, self_attn.{q,k,v,o}_proj + share_expert.* (MoE layers) or mlp.* (0-2,
                                  91) at --body-k A/S/M (attention / shared expert / dense MLP K, default 4/4/4; a
                                  single number sets all three); q/k/v share one suh, gate/up share one suh (as the
                                  real encoder's shared Hessians do), so the plugin's fused GEMVs run
  BF16 shards + index             --from-official DIR: the official BF16 tensors themselves, hardlinked or rewritten
                                  byte for byte by p5_assemble.write_bf16 (no disk cost when hardlinked). Without it:
                                  random BF16 (moe.gate and router_bias F32) at shapes computed from config.json, text
                                  tensors only (vision tower omitted: unverified that the VL wrapper boots without it)
  config.json                     p5_assemble.transform_config (model_type step5 / step5_text, no sparse_config,
                                  quantization_config step5_exl3 body_format exl3); tokenizer files, chat_template.jinja

EXL3 tensors (exllamav3 1.6.2 source modules/quant/exl3.py:44-47, 67; exl3_lib/quantize.py:18-39, 114-128, 1849-1862; plugin
exl3_quant.py:165, 296-309): trellis int16 [in/16, out/16, 16*K], suh fp16 [in], svh fp16 [out], mul1 int32 [] =
0x83DCD12D (only its presence matters to the kernels). Trellis bytes are random (any bit pattern decodes to a codebook
value); one random block per (layer, projection, K) is shared by the experts of that layer. suh = +-1, svh = +-0.02.

  python3 serve/synth_ckpt.py OUT --meta /model [--from-official /model] [--k-mix 2:0.55,3:0.35,4:0.10 | --plan P]
                              [--body-k 4/4/4] [--seed 0] [--experts-only | --body-only] [--dry-run]
--experts-only rewrites only exl3/experts (bucket-count comparison), --body-only only exl3/body (body-K comparison);
the other half of exl3_plan.json is kept from the existing OUT. Every file is replaced through a new inode, so a
hardlinked copy of OUT is not touched. With --plan the body K comes from the plan. Prints the expected weight bytes per node (step5q.plan.bytes_per_node,
TP2, STEP5_EXPERT_TP=1) and the bytes written.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random
import struct
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from p5_assemble import bf16_groups, copy_tokenizer, transform_config, verify_dir, write_bf16  # noqa: E402
from p_util import atomic_json, body_modules, drop_page_cache, model_layers, write_raw  # noqa: E402
from step5q import plan as P  # noqa: E402
from step5q.names import sero_body_file, sero_body_key, sero_expert_file, sero_expert_key  # noqa: E402

MUL1_MARKER = struct.pack("<I", 0x83DCD12D)       # exllamav3 quantize.py:19 codebook_mul1_mult, stored as int32 []
SUH_SCALE, SVH_SCALE, BF16_STD = 1.0, 0.02, 0.02
SHARD_BYTES = 2 << 30


# ------------------------------------------------------------------------------------------------ shape math
def trellis_shape(n_in: int, n_out: int, K: int) -> list[int]:
    if n_in % 16 or n_out % 16:
        raise ValueError(f"EXL3 needs 16-multiples, got {n_in} x {n_out}")
    return [n_in // 16, n_out // 16, 16 * K]


def exl3_nbytes(n_in: int, n_out: int, K: int) -> int:
    """Bytes of one stored EXL3 linear: trellis + suh + svh + mul1 marker (= step5q.plan.exl3_linear_bytes, tp 1)."""
    return n_in * n_out * K // 8 + 2 * (n_in + n_out) + len(MUL1_MARKER)


def expert_dims() -> dict[str, tuple[int, int]]:
    """(in, out) per routed-expert projection."""
    return {"gate_proj": (P.HIDDEN, P.EXPERT_INTER), "up_proj": (P.HIDDEN, P.EXPERT_INTER),
            "down_proj": (P.EXPERT_INTER, P.HIDDEN)}


def body_dims(module: str) -> tuple[int, int]:
    """(in, out) of an EXL3 body module (self_attn.q_proj, share_expert.down_proj, mlp.gate_proj, ...)."""
    q, kv = P.HEADS * P.HEAD_DIM, P.KV_HEADS * P.HEAD_DIM
    base, proj = module.split(".")
    if base == "self_attn":
        return {"q_proj": (P.HIDDEN, q), "k_proj": (P.HIDDEN, kv), "v_proj": (P.HIDDEN, kv),
                "o_proj": (q, P.HIDDEN)}[proj]
    inter = P.EXPERT_INTER if base == "share_expert" else P.DENSE_INTER
    return (inter, P.HIDDEN) if proj == "down_proj" else (P.HIDDEN, inter)


def parse_k_mix(spec: str) -> list[tuple[int, float]]:
    """'2:0.55,3:0.35,4:0.10' -> [(2, .55), (3, .35), (4, .10)]; K in {2,3,4,5}, fractions sum to 1, <= 3 distinct K."""
    mix = []
    for part in spec.split(","):
        k, _, f = part.partition(":")
        mix.append((int(k), float(f)))
    ks = [k for k, _ in mix]
    if len(set(ks)) != len(ks) or any(k not in P.EXPERT_K for k in ks) or len(ks) > P.MAX_K_PER_LAYER:
        raise ValueError(f"--k-mix {spec}: distinct K from {P.EXPERT_K}, at most {P.MAX_K_PER_LAYER}")
    if any(f < 0 for _, f in mix) or abs(sum(f for _, f in mix) - 1) > 1e-6:
        raise ValueError(f"--k-mix {spec}: fractions must be >= 0 and sum to 1")
    return mix


def layer_ks(mix: list[tuple[int, float]], n_experts: int, rng: random.Random) -> list[int]:
    """Per-expert K for one layer: counts by largest remainder, then shuffled over the expert ids."""
    raw = [f * n_experts for _, f in mix]
    counts = [int(x) for x in raw]
    for i in sorted(range(len(mix)), key=lambda i: raw[i] - counts[i], reverse=True)[: n_experts - sum(counts)]:
        counts[i] += 1
    ks = [k for (k, _), c in zip(mix, counts) for _ in range(c)]
    rng.shuffle(ks)
    return ks


BODY_PARTS = ("attn", "shared", "mlp")


def parse_body_k(spec: str) -> dict:
    """'4' -> 4/4/4; '6/8/8' -> attention 6, shared expert 8, dense MLP 8 (step5q.plan body keys). K in 2..8."""
    parts = [int(x) for x in str(spec).split("/")]
    if len(parts) == 1:
        parts *= 3
    if len(parts) != 3 or any(not 2 <= k <= 8 for k in parts):
        raise ValueError(f"--body-k {spec}: one K or attn/shared/mlp, each in 2..8")
    return {**dict(zip(BODY_PARTS, parts)), "head": "bf16"}


def body_k_of(module: str, body: dict) -> int:
    """The plan's body K for an EXL3 body module (self_attn.* -> attn, share_expert.* -> shared, mlp.* -> mlp)."""
    return body[{"self_attn": "attn", "share_expert": "shared", "mlp": "mlp"}[module.split(".")[0]]]


def make_plan(mix: list[tuple[int, float]], body: dict, seed: int) -> dict:
    rng = random.Random(seed)
    avg = sum(k * f for k, f in mix)
    plan = {"bpw_target": round(avg, 4), "layers": {str(L): {"k": layer_ks(mix, P.N_EXPERTS, rng)} for L in P.MOE_LAYERS},
            "body": dict(body)}
    P.validate(plan)
    return plan


def text_shape(name: str, t: dict) -> tuple[str, list[int]] | None:
    """(dtype, shape) of an official BF16 text tensor computed from text_config (random mode); None if not a text
    tensor this table knows (vision, or something new: the caller fails on it)."""
    H, V, hd, nh = t["hidden_size"], t["vocab_size"], t["head_dim"], t["num_attention_heads"]
    q, kv, I, S, E = nh * hd, t["num_attention_groups"] * hd, t["intermediate_size"], t["share_expert_dim"], \
        t["moe_num_experts"]
    top = {"model.embed_tokens.weight": [V, H], "lm_head.weight": [V, H], "model.norm.weight": [H]}
    if name in top:
        return "BF16", top[name]
    if not name.startswith("model.layers."):
        return None
    rest = name.split(".", 3)[3]
    table = {"input_layernorm.weight": [H], "post_attention_layernorm.weight": [H],
             "self_attn.q_proj.weight": [q, H], "self_attn.k_proj.weight": [kv, H], "self_attn.v_proj.weight": [kv, H],
             "self_attn.o_proj.weight": [H, q], "self_attn.g_proj.weight": [nh, H],
             "self_attn.q_norm.weight": [hd], "self_attn.k_norm.weight": [hd],
             "mlp.gate_proj.weight": [I, H], "mlp.up_proj.weight": [I, H], "mlp.down_proj.weight": [H, I],
             "share_expert.gate_proj.weight": [S, H], "share_expert.up_proj.weight": [S, H],
             "share_expert.down_proj.weight": [H, S], "moe.gate.weight": [E, H], "moe.router_bias": [E],
             "eh_proj.weight": [H, 2 * H], "enorm.weight": [H], "hnorm.weight": [H],
             "transformer.shared_head.norm.weight": [H], "transformer.shared_head.output.weight": [V, H]}
    if rest not in table:
        return None
    return ("F32" if rest in ("moe.gate.weight", "moe.router_bias") else "BF16"), table[rest]


# ------------------------------------------------------------------------------------------------ random bytes
def sign_vector(n: int, scale: float, rng: random.Random) -> bytes:
    """fp16 [n] of +-scale."""
    pos, neg = struct.pack("<e", scale), struct.pack("<e", -scale)
    bits = rng.getrandbits(n)
    return b"".join(pos if (bits >> i) & 1 else neg for i in range(n))


def value_block(dtype: str, rng: random.Random, n: int = 1 << 19) -> bytes:
    """n random normal values (std BF16_STD) as BF16 or F32 bytes, tiled by fill()."""
    f32 = struct.pack(f"<{n}f", *(rng.gauss(0.0, BF16_STD) for _ in range(n)))
    if dtype == "F32":
        return f32
    return b"".join(f32[i + 2:i + 4] for i in range(0, len(f32), 4))   # bf16 = upper half of the fp32 word


def fill(block: bytes, nbytes: int) -> bytes:
    reps = -(-nbytes // len(block))
    return (block * reps)[:nbytes]


def exl3_entries(key_of, n_in: int, n_out: int, K: int, trellis: bytes, suh: bytes, svh: bytes) -> list:
    """write_raw entries of one EXL3 linear; key_of(piece) -> tensor name."""
    return [(key_of("trellis"), "I16", trellis_shape(n_in, n_out, K), trellis),
            (key_of("suh"), "F16", [n_in], suh), (key_of("svh"), "F16", [n_out], svh),
            (key_of("mul1"), "I32", [], MUL1_MARKER)]


# ------------------------------------------------------------------------------------------------ writers
def write_experts(out: str, L: int, ks: list[int], rng: random.Random) -> int:
    dims = expert_dims()
    blocks = {(p, K): rng.randbytes(n_in * n_out * K // 8) for p, (n_in, n_out) in dims.items() for K in set(ks)}
    signs = {p: (sign_vector(n_in, SUH_SCALE, rng), sign_vector(n_out, SVH_SCALE, rng)) for p, (n_in, n_out) in
             dims.items()}
    entries = []
    for e, K in enumerate(ks):
        for p, (n_in, n_out) in dims.items():
            entries += exl3_entries(lambda piece, e=e, p=p: sero_expert_key(L, e, p, piece), n_in, n_out, K,
                                    blocks[(p, K)], *signs[p])
    path = os.path.join(out, sero_expert_file(L))
    n = write_raw(path, entries, {"format": "pt", "synthetic": "random trellis"})
    drop_page_cache(path)
    return n


def write_body(out: str, L: int, is_moe: bool, body: dict, rng: random.Random) -> int:
    entries = []
    shared_su = {}                       # one su per input group: q/k/v share one, gate/up share one
    for module in body_modules(is_moe):
        n_in, n_out = body_dims(module)
        base, proj = module.split(".")
        group = (base, "in" if proj in ("q_proj", "k_proj", "v_proj", "gate_proj", "up_proj") else proj)
        if group not in shared_su:
            shared_su[group] = sign_vector(n_in, SUH_SCALE, rng)
        K = body_k_of(module, body)
        entries += exl3_entries(lambda piece, m=module: sero_body_key(L, m, piece), n_in, n_out, K,
                                rng.randbytes(n_in * n_out * K // 8), shared_su[group],
                                sign_vector(n_out, SVH_SCALE, rng))
    path = os.path.join(out, sero_body_file(L))
    n = write_raw(path, entries, {"format": "pt", "synthetic": "random trellis"})
    drop_page_cache(path)
    return n


def write_random_bf16(meta: str, out: str, cfg: dict, expert_layers: list[int], body_layers: list[int],
                      rng: random.Random) -> tuple[dict, int]:
    """Random BF16 shards for the official index's kept text tensors (shapes from config); returns (weight_map, size)."""
    from p5_assemble import keep_bf16
    t = cfg["text_config"]
    wm = json.load(open(os.path.join(meta, "model.safetensors.index.json")))["weight_map"]
    el, bl = set(expert_layers), set(body_layers)
    names = sorted(k for k in wm if keep_bf16(k, el, bl) and not k.startswith(("vision_model.", "vit_large_")))
    blocks = {dt: value_block(dt, rng) for dt in ("BF16", "F32")}
    one = {"BF16": struct.pack("<f", 1.0)[2:], "F32": struct.pack("<f", 1.0)}
    shards: list[list] = [[]]
    used = 0
    for name in names:
        ds = text_shape(name, t)
        if ds is None:
            raise SystemExit(f"random mode: no shape rule for {name}; use --from-official")
        dt, shape = ds
        nbytes = (4 if dt == "F32" else 2) * math.prod(shape)
        is_norm = name.endswith(("norm.weight", "layernorm.weight"))
        if used + nbytes > SHARD_BYTES and shards[-1]:
            shards.append([])
            used = 0
        shards[-1].append((name, dt, shape, None if not is_norm else one[dt], nbytes))
        used += nbytes
    weight_map, size = {}, 0
    for i, shard in enumerate(shards, 1):
        fname = f"model-{i:05d}-of-{len(shards):05d}.safetensors"
        entries = [(name, dt, shape, fill(one_v or blocks[dt], nb)) for name, dt, shape, one_v, nb in shard]
        size += write_raw(os.path.join(out, fname), entries, {"format": "pt", "synthetic": "random bf16"})
        drop_page_cache(os.path.join(out, fname))
        weight_map.update({name: fname for name, *_ in shard})
    return weight_map, size


def copy_meta(meta: str, out: str) -> None:
    copy_tokenizer(meta, out)   # also rewrites tokenizer_class (see p5_assemble.copy_tokenizer)


# ------------------------------------------------------------------------------------------------ main
def merge_partial(plan: dict, plan_path: str, experts_only: bool, body_only: bool, dry_run: bool) -> dict:
    """A partial rewrite keeps the other half of the existing exl3_plan.json, so the file describes what is on disk."""
    if not (experts_only or body_only) or dry_run:
        return plan
    if not os.path.exists(plan_path):
        raise SystemExit(f"{plan_path} missing: run a full synth_ckpt.py first")
    with open(plan_path) as f:
        old = json.load(f)
    merged = {**plan, "body": old["body"]} if experts_only else {**plan, "layers": old["layers"],
                                                                  "bpw_target": old["bpw_target"]}
    P.validate(merged)
    return merged


def report_expected(plan: dict) -> None:
    b = P.bytes_per_node(plan, expert_tp=True, tp=2)
    parts = ", ".join(f"{k} {v / P.GIB:.2f}" for k, v in b.items() if k != "total")
    print(f"expected weight bytes per node (TP2, EXPERT_TP=1): {b['total'] / P.GIB:.2f} GiB ({parts}); "
          f"expert avg K {P.expert_bpw(plan):.3f}", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("out")
    ap.add_argument("--meta", required=True, help="official dir: config.json, model.safetensors.index.json, tokenizer")
    ap.add_argument("--from-official", default="", help="official checkpoint dir: copy its real BF16 tensors")
    src = ap.add_mutually_exclusive_group()
    src.add_argument("--k-mix", default="2:0.55,3:0.35,4:0.10")
    src.add_argument("--plan", default="", help="plan.json (step5q.plan format) instead of --k-mix")
    ap.add_argument("--body-k", default="4/4/4", help="attention/shared/mlp K (or one K for all), 2..8")
    ap.add_argument("--seed", type=int, default=0)
    part = ap.add_mutually_exclusive_group()
    part.add_argument("--experts-only", action="store_true")
    part.add_argument("--body-only", action="store_true")
    ap.add_argument("--dry-run", action="store_true", help="print the expected bytes and exit")
    a = ap.parse_args()

    plan = P.read_plan(a.plan) if a.plan else make_plan(parse_k_mix(a.k_mix), parse_body_k(a.body_k), a.seed)
    plan = merge_partial(plan, os.path.join(a.out, "exl3_plan.json"), a.experts_only, a.body_only, a.dry_run)
    report_expected(plan)
    if a.dry_run:
        return
    cfg_off = json.load(open(os.path.join(a.meta, "config.json")))
    expert_layers, body_layers, n_experts = model_layers(cfg_off)
    if expert_layers != list(P.MOE_LAYERS) or n_experts != P.N_EXPERTS:
        raise SystemExit(f"config has expert layers {expert_layers[:2]}..{expert_layers[-1:]} x {n_experts}, "
                         f"step5q.plan expects {P.MOE_LAYERS} x {P.N_EXPERTS}")
    for d in ("exl3/experts", "exl3/body"):
        os.makedirs(os.path.join(a.out, d), exist_ok=True)
    rng = random.Random(a.seed)
    t0, total = time.time(), 0
    for L in ([] if a.body_only else expert_layers):
        ks = plan["layers"][str(L)]["k"]
        n = write_experts(a.out, L, ks, rng)
        total += n
        counts = {f"K{k}": ks.count(k) for k in sorted(set(ks))}
        print(f"L{L:02d} experts {counts} {n / P.GIB:.2f} GiB, {time.time() - t0:.0f}s", flush=True)
    if not a.experts_only:
        for L in body_layers:
            total += write_body(a.out, L, L in expert_layers, plan["body"], rng)
        print(f"body layers {body_layers[0]}-{body_layers[-1]} at attn/shared/mlp K "
              f"{'/'.join(str(plan['body'][k]) for k in BODY_PARTS)}", flush=True)
    if not (a.experts_only or a.body_only):
        if a.from_official:
            keep, counts = bf16_groups(a.from_official, expert_layers, body_layers)
            weight_map, size = write_bf16(a.from_official, a.out, keep, counts, lambda m: print(m, flush=True))
        else:
            weight_map, size = write_random_bf16(a.meta, a.out, cfg_off, expert_layers, body_layers, rng)
        total += size
        atomic_json(os.path.join(a.out, "model.safetensors.index.json"),
                    {"metadata": {"total_size": size}, "weight_map": weight_map}, indent=1)
        atomic_json(os.path.join(a.out, "config.json"),
                    transform_config(cfg_off, expert_layers, body_layers, n_experts), indent=2)
        copy_meta(a.meta, a.out)
    atomic_json(os.path.join(a.out, "exl3_plan.json"), plan)
    bad = verify_dir(a.out, "exl3")
    for b in bad:
        print("FAIL", b)
    print(f"{'OK  ' if not bad else 'FAIL'} {a.out}: wrote {total / P.GIB:.1f} GiB in {time.time() - t0:.0f}s "
          f"(verify.py logic, body format exl3)", flush=True)
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
