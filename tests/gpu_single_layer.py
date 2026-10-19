#!/usr/bin/env python3
"""Single-layer GPU checks of step5q against the official reference, on a Spark with the FP8 checkpoint.

  docker run --rm --gpus all --entrypoint python3 -v ~/step5:/q -v ~/models/Step-5-Preview-FP8:/model:ro -w /q \
      dsv41-quant:dev tests/gpu_single_layer.py /model [--sliding 4] [--full 3] [--tokens 2048]

One line per check, `check=<name> ... OK|CHECK|ERROR`; exit status 1 if any line is not OK. Checks:
  attn    dense_attention vs the official Step5Attention.forward (sliding: _sliding_attention, full: the sparse
          indexer path) with a Step5Cache, batch 1, random tokens through the embedding and input_layernorm
  qknorm  official _project (fused Triton qk-norm + RoPE on CUDA) vs its torch fallback
  route   step5q.stream.route_topk vs Step5MoE.route (ids must be identical)
  fp8     kernel.linear_fp8_or_bf16 with the FP8 scale (act_quant + fp8_gemm) vs BF16 dequantized F.linear, 8 experts;
          plus linear_fp8_or_bf16(x, w_bf16, None) == F.linear (the BF16 branch the teacher takes)
  ffn     official layer._feed_forward (Triton clamped_swiglu + weighted_topk_gather) on BF16 experts vs a torch
          re-implementation of its CPU fallback
  encode  encode_expert at K=3 on expert 0 with a random-input Hessian, reconstruct, weight and output rel error
Thresholds are coarse (stated per line); the numbers are what matters.
"""
from __future__ import annotations

import argparse
import functools
import os
import sys
import tempfile
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

from step5q.ckpt import Checkpoint  # noqa: E402
from step5q.exl3 import encode_expert, expert_forward_dense, h_data, reconstruct  # noqa: E402
from step5q.refmodel import dense_attention, fp8_block_dequant, load_config, load_layer, load_reference  # noqa: E402
from step5q.stream import route_topk  # noqa: E402

FAILS: list[str] = []


def report(name: str, got: torch.Tensor, want: torch.Tensor, tol: float, **tags) -> None:
    d = got.float() - want.float()
    rel = (d.norm() / want.float().norm().clamp_min(1e-30)).item()
    verdict = "OK" if rel <= tol else "CHECK"
    if verdict != "OK":
        FAILS.append(name)
    extra = " ".join(f"{k}={v}" for k, v in tags.items())
    print(f"check={name} {extra} max_abs={d.abs().max().item():.3e} rel={rel:.3e} tol={tol:g} {verdict}", flush=True)


def run(name: str, fn, *args) -> None:
    try:
        fn(*args)
    except Exception as e:  # noqa: BLE001  (a diagnostic script: report and continue with the next check)
        FAILS.append(name)
        print(f"check={name} ERROR {type(e).__name__}: {e}", flush=True)


def hidden_for(ckpt: Checkpoint, layer, T: int, dev) -> tuple[torch.Tensor, torch.Tensor]:
    g = torch.Generator(device="cpu").manual_seed(1234)
    ids = torch.randint(0, 128000, (T,), generator=g).to(dev)
    resid = ckpt.get("model.embed_tokens.weight", dev)[ids].float()
    return resid, layer.input_layernorm(resid).to(torch.bfloat16)


@torch.inference_mode()
def check_attn(ref, cfg, ckpt, l: int, T: int, dev) -> None:
    layer = load_layer(ckpt, cfg, l, dev, experts=False)
    attn = layer.self_attn
    _, h = hidden_for(ckpt, layer, T, dev)
    pos = torch.arange(T, device=dev, dtype=torch.int32)
    cache = ref.Step5Cache(cfg, batch=1, max_tokens=T, device=dev, dtype=torch.bfloat16)
    seq_lens = torch.tensor([T], device=dev, dtype=torch.int32)
    past = cache.advance(seq_lens)
    # The official sparse prefill kernel tiles block_regions=16 regions (128 keys x BLOCK_D 256 x K+V) = 140 KB of
    # shared memory, above GB10's 99 KB; 8 regions fit and the reduction is the same (tile order only).
    block_regions = int(os.environ.get("STEP5_SPARSE_BLOCK_REGIONS", "8"))
    official = ref.sparse_attention_prefill
    ref.sparse_attention_prefill = functools.partial(official, block_regions=block_regions)
    try:
        want = attn(h, pos, cache, seq_lens, past, is_decode=False)
    finally:
        ref.sparse_attention_prefill = official
    kind = f"sparse(block_regions={block_regions})" if attn.is_sparse else f"window{attn.sliding_window}"
    report("attn", dense_attention(layer, h, pos), want, 1e-2, layer=l, path=kind, tokens=T)
    q1, k1, _ = attn._project(h, pos)
    q, k, _ = attn._split_packed_qkv(attn._project_packed_qkv(h))
    report("qknorm", q1, attn._apply_rope(attn.q_norm(q), pos), 1e-2, layer=l, tensor="q")
    report("qknorm", k1, attn._apply_rope(attn.k_norm(k), pos), 1e-2, layer=l, tensor="k")


@torch.inference_mode()
def check_fp8(ckpt, l: int, x: torch.Tensor, dev) -> None:
    kernel = sys.modules["kernel"]
    wn = f"model.layers.{l}.moe.gate_proj.weight"
    t = ckpt.get_many([wn, wn + "_scale_inv"], dev)
    w, s = t[wn], t[wn + "_scale_inv"]
    for e in range(8):
        wb = fp8_block_dequant(w[e], s[e])
        report("fp8", kernel.linear_fp8_or_bf16(x, w[e], s[e]), F.linear(x, wb), 5e-2, layer=l, expert=e,
               proj="gate_proj", tokens=x.shape[0])
    same = torch.equal(kernel.linear_fp8_or_bf16(x, wb, None), F.linear(x, wb))
    if not same:
        FAILS.append("fp8")
    print(f"check=fp8 layer={l} bf16_branch_equals_F.linear={same} {'OK' if same else 'CHECK'}", flush=True)


def _swiglu(gate: torch.Tensor, up: torch.Tensor, limit: float) -> torch.Tensor:
    return (F.silu(gate.float()).clamp(max=limit) * up.float().clamp(-limit, limit)).to(gate.dtype)


def ffn_reference(layer, x: torch.Tensor) -> torch.Tensor:
    """model.py Step5MoE.forward CPU fallback + Step5DecoderLayer._feed_forward, in torch on the GPU."""
    m, se = layer.moe, layer.share_expert
    w, ids = m.route(x)
    slot = torch.zeros((x.shape[0], m.top_k, x.shape[1]), dtype=x.dtype, device=x.device)
    for e in range(m.n_local_experts):
        rows, slots = (ids == e).nonzero(as_tuple=True)
        if rows.numel():
            t = x[rows]
            a = _swiglu(F.linear(t, m.gate_proj.weight[e]), F.linear(t, m.up_proj.weight[e]), m.limit)
            slot[rows, slots] = F.linear(a, m.down_proj.weight[e])
    routed = torch.zeros(x.shape, dtype=torch.float32, device=x.device)
    for s in range(m.top_k):
        routed = routed + slot[:, s].float() * w[:, s].unsqueeze(-1)
    shared = se.down_proj(_swiglu(se.gate_proj(x), se.up_proj(x), se.limit))
    return routed.to(x.dtype).float() + shared.float()


@torch.inference_mode()
def moe_input(ckpt, layer, T: int, dev) -> torch.Tensor:
    resid, h = hidden_for(ckpt, layer, T, dev)
    pos = torch.arange(T, device=dev, dtype=torch.int32)
    resid = resid + dense_attention(layer, h, pos).float()
    return layer.post_attention_layernorm(resid).to(torch.bfloat16)


@torch.inference_mode()
def check_route_ffn(layer, x: torch.Tensor) -> None:
    w0, i0 = layer.moe.route(x)
    m = layer.moe
    w1, i1 = route_topk(x, m.gate.weight, m.router_bias, m.top_k, m.scaling_factor, m.renormalize)
    same = torch.equal(i0, i1)
    if not same:
        FAILS.append("route")
    print(f"check=route layer={layer.layer_idx} ids_identical={same} "
          f"w_max_abs={(w0 - w1).abs().max().item():.3e} {'OK' if same else 'CHECK'}", flush=True)
    report("ffn", layer._feed_forward(x), ffn_reference(layer, x), 1e-2, layer=layer.layer_idx, tokens=x.shape[0])


@torch.no_grad()
def check_encode(layer, work: str) -> None:
    m, dev = layer.moe, layer.moe.gate_proj.weight.device
    w1, w3, w2 = (p.weight[0].float() for p in (m.gate_proj, m.up_proj, m.down_proj))
    g = torch.Generator(device=dev).manual_seed(0)
    x = torch.randn(8192, w1.shape[1], device=dev, generator=g)
    a = _swiglu(x @ w1.T, x @ w3.T, m.limit).float()
    t0 = time.time()
    t, pe = encode_expert(w1, w3, w2, h_data(x.T @ x, x.shape[0], "gu"), h_data(a.T @ a, a.shape[0], "d"), 3, 0, work)
    secs = time.time() - t0
    q = {p: reconstruct(t[p], *dims) for p, dims in (("w1", (4096, 1536)), ("w3", (4096, 1536)),
                                                     ("w2", (1536, 4096)))}
    # 3-bit trellis quantization of a Gaussian-like matrix: rate-distortion bound 2^-3 = 0.125 relative RMS error,
    # practical codes about 1.1-1.2x that (measured 0.152 on layer 4 expert 0); the output error compounds through
    # three projections and the gated activation (measured 0.24).
    for p, w in (("w1", w1), ("w3", w3), ("w2", w2)):
        report("encode", q[p], w, 0.20, layer=layer.layer_idx, expert=0, K=3, proj=p, proxy=f"{pe[p]:.3e}",
               seconds=f"{secs:.1f}")
    xt = torch.randn(2048, w1.shape[1], device=dev, generator=g)
    ones = torch.ones(xt.shape[0], device=dev)
    report("encode", expert_forward_dense(xt, ones, q["w1"], q["w3"], q["w2"]),
           expert_forward_dense(xt, ones, w1, w3, w2), 0.30, layer=layer.layer_idx, expert=0, K=3, proj="output")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("model")
    ap.add_argument("--sliding", type=int, default=4, help="a sliding-attention MoE layer with FP8 experts (3..87)")
    ap.add_argument("--full", type=int, default=3, help="a full-attention layer")
    ap.add_argument("--tokens", type=int, default=2048)
    ap.add_argument("--work", default="", help="exllamav3 debug dir (default: a fresh temp dir)")
    a = ap.parse_args()
    dev = torch.device("cuda")
    torch.cuda.set_per_process_memory_fraction(float(os.environ.get("STEP5_CUDA_MEM_FRACTION", "0.70")))
    ref, cfg, ckpt = load_reference(a.model), load_config(a.model), Checkpoint(a.model)
    for l in (a.sliding, a.full, 88):
        pre = f"model.layers.{l}."
        dts = sorted({(n[len(pre):], ckpt.meta(n)["dtype"]) for n in ckpt.names(pre)})
        print(f"info=dtypes layer={l} " + " ".join(f"{n}:{d}" for n, d in dts), flush=True)
    for l in (a.sliding, a.full):
        run("attn", check_attn, ref, cfg, ckpt, l, a.tokens, dev)
        torch.cuda.empty_cache()
    layer = load_layer(ckpt, cfg, a.sliding, dev, experts=True)
    x = moe_input(ckpt, layer, a.tokens, dev)
    run("fp8", check_fp8, ckpt, a.sliding, x[:512], dev)
    run("route", check_route_ffn, layer, x)
    run("encode", check_encode, layer, a.work or tempfile.mkdtemp(prefix="step5_x3_"))
    print(f"done: {len(FAILS)} failing checks {sorted(set(FAILS))}", flush=True)
    return 1 if FAILS else 0


if __name__ == "__main__":
    sys.exit(main())
