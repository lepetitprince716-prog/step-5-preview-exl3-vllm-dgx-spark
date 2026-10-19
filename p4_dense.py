# SPDX-License-Identifier: Apache-2.0
"""Dense pass: EXL3 encoding of Step 5's body linears on the calibration stream through pass 1's EXL3 experts (ported
from sudoingx dsv41_recipe scripts/quantize/p4_dense.py, Apache-2.0).

Per decoder layer 0..91 (MTP 92-94 stay BF16), one forward of the whole calibration set with pass 1's EXL3 routed
experts and the native body (not sequential for the body, as in sudoingx: body errors at K4-K5 are far below the
experts'), summing input Hessians over all tokens:
  self_attn.q/k/v_proj      the input_layernorm output (bf16), ONE Hessian; q, k, v are encoded as one concatenated
                            matrix and split, so their suh match (the plugin fuses q/k/v into one GEMV only then)
  self_attn.o_proj          the gated attention output (o_proj's input)
  share_expert.gate/up_proj the MoE input, one Hessian, encoded concatenated (shared suh), MoE layers 3-90
  share_expert.down_proj    its clamped-swiglu activation
  mlp.gate/up_proj, mlp.down_proj   the same for the dense layers 0-2 and 91
K: the plan's body block ({"attn": 4, "shared": 4, "mlp": 4}), --attn-k overrides attention (e.g. 5).
Two boxes: --part i/2 encodes the layers with layer % 2 == i (both stream every layer).
--variants 5/6/6,6/8/8 (attn/shared/mlp K) encodes extra body K sets from the same Hessians in the same stream into
/out/body_5-6-6/ etc. (frontier review section 2 row 2a: pick the body K by end-to-end KLD, not by a guess).

  python3 p4_dense.py /model /cal/cal.safetensors /pass1_out /plan/plan.json /out [--part 0/2] [--attn-k 5] \
      [--variants 5/6/6,6/8/8]
Outputs: /out/body_LL.safetensors (0xSero body keys) + /out/body_LL.json (Hessian-weighted relative error per matrix),
the same under /out/body_<a-s-m>/ per variant.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from p_torch import (ExpertCache, HAcc, encode_fused, files_for, h_err, load_exl3, load_seqs, logger,  # noqa: E402
                     mem_gib, quant_routed, resume_stream, save_stream)
from p_util import (atomic_json, drop_page_cache, model_layers, parse_layers, parse_part, parse_variants,  # noqa: E402
                    sero_body_key)


def install_capture(layer, acc: HAcc, is_moe: bool) -> list:
    """Pre-hook on the FFN down projection (returns None: a pre-hook's return value would replace the input)."""
    ffn = layer.share_expert if is_moe else layer.mlp
    return [ffn.down_proj.register_forward_pre_hook(lambda m, i: acc.add("down", i[0]))]


def attn_capture(acc: HAcc):
    """Streamer attn_hook(l, seq, h, o_in): h = q/k/v input (input_layernorm output, BF16), o_in = o_proj input."""
    def hook(_l: int, _i: int, h, o_in) -> None:
        acc.add("qkv", h)
        acc.add("o", o_in)
    return hook


def body_moe(s, cache: ExpertCache | None, acc: HAcc | None):
    def moe(x: torch.Tensor, layer) -> torch.Tensor:
        if acc is not None:
            acc.add("gu", x)
        if getattr(layer, "moe", None) is None:
            return layer.mlp(x).float()
        w, ids = s.route(layer, x)
        return quant_routed(x, w, ids, cache).to(x.dtype).float() + layer.share_expert(x).float()
    return moe


def write_body(layer, l: int, is_moe: bool, acc: HAcc, ks: dict, odir: str, work: str, seed: int, fwd_s: float,
               log) -> None:
    """Encode one body K set for layer l and write odir/body_LL.safetensors + .json."""
    from safetensors.torch import save_file
    t1 = time.time()
    kq = {"attn": ks["attn"], "ffn": ks["shared"] if is_moe else ks["mlp"]}
    out, metrics = encode_layer(layer, l, is_moe, acc, kq, work, seed)
    dst = os.path.join(odir, f"body_{l:02d}.safetensors")
    save_file(out, dst + ".tmp", metadata={"format": "pt", "encoder": "exllamav3 1.6.0 mul1"})
    os.chmod(dst + ".tmp", 0o644)
    os.replace(dst + ".tmp", dst)
    drop_page_cache(dst)
    metrics.update(k_set=ks, forward_s=round(fwd_s, 1), encode_s=round(time.time() - t1, 1), bytes=os.path.getsize(dst))
    atomic_json(os.path.join(odir, f"body_{l:02d}.json"), metrics, indent=1)
    worst = max((v["rel_err"], k) for k, v in metrics.items() if isinstance(v, dict) and "rel_err" in v)
    log(f"layer {l:2d} -> {odir}: K {ks}, encode {time.time() - t1:.0f}s, {metrics['bytes'] / 2 ** 20:.0f} MiB, "
        f"worst rel err {worst[0]:.4f} ({worst[1]}), {mem_gib()}")


def encode_layer(layer, l: int, is_moe: bool, acc: HAcc, kq: dict, work: str, seed: int) -> tuple[dict, dict]:
    from step5q.exl3 import h_data, reconstruct
    from p_torch import dims
    out, metrics = {}, {}
    H0 = {k: (v / max(acc.n[k], 1)).clone() for k, v in acc.H.items()}   # exllamav3 finalizes H in place
    ffn_name = "share_expert" if is_moe else "mlp"
    ffn = layer.share_expert if is_moe else layer.mlp
    att = layer.self_attn
    groups = [  # (Hessian key, [(module name, weight)], K)
        ("qkv", [("self_attn.q_proj", att.q_proj.weight), ("self_attn.k_proj", att.k_proj.weight),
                 ("self_attn.v_proj", att.v_proj.weight)], kq["attn"]),
        ("o", [("self_attn.o_proj", att.o_proj.weight)], kq["attn"]),
        ("gu", [(f"{ffn_name}.gate_proj", ffn.gate_proj.weight), (f"{ffn_name}.up_proj", ffn.up_proj.weight)],
         kq["ffn"]),
        ("down", [(f"{ffn_name}.down_proj", ffn.down_proj.weight)], kq["ffn"]),
    ]
    for j, (hk, mods, K) in enumerate(groups):
        if acc.n.get(hk, 0) == 0:
            raise SystemExit(f"layer {l}: no Hessian rows for {hk}: a capture hook never fired")
        ws = [w.float() for _, w in mods]
        H = h_data(acc.H[hk], acc.n[hk], f"L{l}.{hk}")
        parts = encode_fused(ws, H, int(K), seed + 10 * j, work)
        for (name, _), w, t in zip(mods, ws, parts):
            for piece, v in t.items():
                out[sero_body_key(l, name, piece)] = v.contiguous().cpu()
            metrics[name] = {"k": int(K), "rel_err": h_err(w, reconstruct(t, *dims(t)), H0[hk]), "rows": acc.n[hk]}
        del ws, parts
    return out, metrics


@torch.inference_mode()
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("model")
    ap.add_argument("cal")
    ap.add_argument("experts", help="pass 1 output dir (layer_LL.part*.safetensors)")
    ap.add_argument("plan")
    ap.add_argument("out")
    ap.add_argument("--part", default="0/1")
    ap.add_argument("--layers", default="", help="layers to encode (default every main-model layer)")
    ap.add_argument("--attn-k", type=int, default=0, help="override the plan's attention K")
    ap.add_argument("--max-seqs", type=int, default=0)
    ap.add_argument("--seed", type=int, default=914)
    ap.add_argument("--stream-dir", default="", help="save / resume the stream (default none: recompute)")
    ap.add_argument("--group-tokens", type=int, default=131072)
    ap.add_argument("--variants", default="", help="extra body K sets attn/shared/mlp, e.g. 5/6/6,6/8/8: encoded from "
                    "the same Hessians into <out>/body_<a-s-m>/ (one stream for all)")
    a = ap.parse_args()
    from step5q.stream import Streamer
    part, nparts = parse_part(a.part)
    plan = json.load(open(a.plan))
    body = plan.get("body", {})
    k_attn = a.attn_k or int(body.get("attn", 4))
    moe_layers, body_layers, n_exp = model_layers(json.load(open(os.path.join(a.model, "config.json"))))
    todo = parse_layers(a.layers, body_layers)
    os.makedirs(a.out, exist_ok=True)
    work = os.path.join(a.out, "work")
    os.makedirs(work, exist_ok=True)
    log = logger(os.path.join(a.out, f"dense.part{part}.log"))
    seqs = load_seqs(a.cal, a.max_seqs)
    s = Streamer(a.model, seqs, "cuda", max_group_tokens=a.group_tokens)
    resume_stream(s, a.stream_dir, log)
    targets = [(a.out, {"attn": k_attn, "shared": int(body.get("shared", 4)), "mlp": int(body.get("mlp", 4))})]
    targets += [(os.path.join(a.out, f"body_{name}"), ks) for name, ks in parse_variants(a.variants)]
    for d, _ in targets:
        os.makedirs(d, exist_ok=True)
    log(f"dense pass part {part}/{nparts}: {len(seqs)} sequences, {sum(map(len, seqs)):,} tokens, "
        f"layers {min(todo)}-{max(todo)}, K sets {[ks for _, ks in targets]}")
    t_all = time.time()
    for l in range(s.layer, max(todo) + 1):
        t0 = time.time()
        is_moe = l in moe_layers
        pending = [(d, ks) for d, ks in targets if not os.path.exists(os.path.join(d, f"body_{l:02d}.safetensors"))]
        mine = l in todo and l % nparts == part and bool(pending)
        layer = s.load_layer(l, experts=False)
        cache = None
        if is_moe:
            files, _ = files_for(l, experts_dir=a.experts)
            cache = ExpertCache(l, load_exl3(files, "cuda"), n_exp)
        acc = HAcc() if mine else None
        hooks = install_capture(layer, acc, is_moe) if mine else []
        try:
            s.run_layer(layer, moe=body_moe(s, cache, acc), attn_hook=attn_capture(acc) if mine else None)
        finally:
            for h in hooks:
                h.remove()
        del cache
        torch.cuda.empty_cache()
        t1 = time.time()
        if mine:
            for i, (odir, ks) in enumerate(pending):
                use = acc if i == len(pending) - 1 else acc.clone()    # exllamav3 finalizes Hessians in place
                write_body(layer, l, is_moe, use, ks, odir, work, a.seed + 100 * l, t1 - t0, log)
                del use
        else:
            log(f"layer {l:2d}: forward {t1 - t0:.0f}s (stream only)")
        del acc, layer
        torch.cuda.empty_cache()
        save_stream(s, a.stream_dir)
    log(f"DONE in {time.time() - t_all:.0f}s")


if __name__ == "__main__":
    main()
