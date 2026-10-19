# SPDX-License-Identifier: Apache-2.0
"""StepFun's official Step 5 reference (`<model_dir>/inference/model.py`, Apache-2.0) run one decoder layer at a time;
the loading scheme is ported from sudoingx dsv41q (refmodel.load_reference / load_block).

Nothing from model.py is copied: the module is imported from the checkpoint directory. Adaptations for streaming:

1. A layer is built on the meta device and every parameter is replaced by the checkpoint tensor (no double
   allocation). Routed experts are dequantized from FP8 to BF16 (`fp8_block_dequant`) and their `weight_scale_inv`
   is set to None, so `kernel.linear_fp8_or_bf16(x, w, None)` takes its BF16 branch (`F.linear`, kernel.py:2515-2547)
   and the teacher runs BF16 weights with BF16 activations (no `act_quant`). Layers 88-90 ship BF16 experts.
2. RoPE tables are built with max_position=4096 (MAX_POSITION) instead of the config's 1M positions.
3. `dense_attention` replaces the official cache-bound attention: official `_project` (q/k/v GEMM, q/k RMSNorm,
   partial RoPE; the fused Triton kernel on CUDA, the official torch fallback on CPU), then SDPA (causal; on sliding
   layers additionally the 512 window, mirroring `_sliding_prefill_chunked`, model.py:1794-1823), then the official
   `_gate_and_project` (per-head sigmoid gate, o_proj). No SSMax (`ssmax_s` is unused by the official kernels,
   model.py:1449-1457). For full-attention layers and T <= 4096 this equals the official sparse path
   (kernel.py:711-713, 924-980, 1102-1106).
"""
from __future__ import annotations

import importlib
import json
import logging
import os
import sys
from types import ModuleType

import torch
import torch.nn.functional as F
from torch import nn

from .ckpt import Checkpoint

log = logging.getLogger(__name__)

MAX_POSITION = 4096          # every calibration / panel sequence is <= 4096 tokens
FP8_BLOCK = 128              # kernel.fp8_gemm: BLOCK_N = BLOCK_K = 128 weight blocks
SLIDING_PREFILL_CHUNK = 2048  # model.py:952, queries per chunk of the windowed SDPA
EXPERT_PROJS = ("gate_proj", "up_proj", "down_proj")

_REF: dict[str, ModuleType] = {}


def load_reference(model_dir: str) -> ModuleType:
    """Import `<model_dir>/inference/model.py` (and through it kernel.py, vision_encoder.py) once."""
    inf = os.path.join(os.path.abspath(model_dir), "inference")
    if inf in _REF:
        return _REF[inf]
    if inf not in sys.path:
        sys.path.insert(0, inf)
    model = importlib.import_module("model")
    if os.path.dirname(os.path.abspath(model.__file__)) != inf:
        raise ImportError(f"imported {model.__file__}, expected the reference in {inf}")
    _REF[inf] = model
    return model


def load_config(model_dir: str):
    """The official text config (Step5Config) from config.json, tp=1. The checkpoint config is the VL envelope;
    config_from_dict builds Step5VLConfig and its text_config carries quantization_config (model.py:503-545)."""
    ref = load_reference(model_dir)
    with open(os.path.join(model_dir, "config.json")) as f:
        cfg = ref.config_from_dict(json.load(f))
    cfg = getattr(cfg, "text_config", cfg)
    if cfg.tp != 1:
        raise ValueError(f"expected tp=1, config has tp_size={cfg.tp}")
    return cfg


def fp8_block_dequant(w: torch.Tensor, scale_inv: torch.Tensor, block: int = FP8_BLOCK) -> torch.Tensor:
    """FP8 e4m3 weight [..., O, I] with inverse block scales [..., ceil(O/128), ceil(I/128)] -> BF16 [..., O, I]:
    w[o, i] * scale_inv[o // 128, i // 128] in fp32, rounded once to BF16. Orientation as kernel.fp8_gemm (weight
    [N, K], scale [ceil(N/128), ceil(K/128)], `real_weight = w_fp8 * w_scale_inv[block]`, kernel.py:2438-2465, the
    kernel reads scale[n_block, k_idx], :2393-2412). Leading dims (experts) are done one slice at a time so the fp32
    temporary stays one expert wide."""
    O, I = w.shape[-2:]
    want = (-(-O // block), -(-I // block))
    if tuple(scale_inv.shape[-2:]) != want or scale_inv.shape[:-2] != w.shape[:-2]:
        raise ValueError(f"scale {tuple(scale_inv.shape)} does not match weight {tuple(w.shape)} / block {block}")
    if w.dim() == 2:
        s = scale_inv.float().repeat_interleave(block, 0)[:O].repeat_interleave(block, 1)[:, :I]
        return (w.float() * s).to(torch.bfloat16)
    out = torch.empty(w.shape, dtype=torch.bfloat16, device=w.device)
    flat_w, flat_s, flat_o = w.reshape(-1, O, I), scale_inv.reshape(-1, *want), out.view(-1, O, I)
    for e in range(flat_w.shape[0]):
        flat_o[e] = fp8_block_dequant(flat_w[e], flat_s[e], block)
    return out


def expert_weights(ckpt: Checkpoint, l: int, device: str | torch.device) -> tuple[torch.Tensor, ...]:
    """Routed experts of layer l as BF16 on `device`: (w1 gate [352,1536,4096], w3 up [352,1536,4096],
    w2 down [352,4096,1536]). One projection at a time keeps the FP8 copy and pinned buffer one projection wide."""
    out = []
    for proj in EXPERT_PROJS:
        wn = f"model.layers.{l}.moe.{proj}.weight"
        sn = wn + "_scale_inv"
        t = ckpt.get_many([wn, sn] if sn in ckpt.weight_map else [wn], device)
        w = t.pop(wn)
        if sn in t:
            if w.dtype != torch.float8_e4m3fn:
                raise TypeError(f"{wn}: scale present but dtype {w.dtype}")
            w = fp8_block_dequant(w, t.pop(sn))
        out.append(w.to(torch.bfloat16))
        del t, w
    return tuple(out)


def _is_routed_expert(name: str) -> bool:
    return any(f".moe.{p}." in name for p in EXPERT_PROJS)


def _set_tensor(root: nn.Module, key: str, value: torch.Tensor | None, keep_dtype: bool = False) -> None:
    """Replace parameter/buffer `key` (dotted path below root) by `value`, cast to the declared dtype unless
    keep_dtype (the BF16 experts replace FP8-declared parameters). None clears it."""
    path, attr = key.rsplit(".", 1)
    mod = root.get_submodule(path)
    table = mod._parameters if attr in mod._parameters else mod._buffers if attr in mod._buffers else None
    if table is None:
        raise KeyError(f"checkpoint tensor {key!r} has no parameter or buffer in the reference layer")
    old = table[attr]
    if value is None:
        table[attr] = None
        return
    if old is not None and tuple(old.shape) != tuple(value.shape):
        raise ValueError(f"{key}: reference {tuple(old.shape)} vs checkpoint {tuple(value.shape)}")
    if old is not None and not keep_dtype:
        value = value.to(old.dtype)
    table[attr] = nn.Parameter(value, requires_grad=False) if table is mod._parameters else value


def _dequant_non_expert(sd: dict[str, torch.Tensor]) -> None:
    """Any non-expert FP8 weight (none in revision 364b3b03: only moe.*_proj carry weight_scale_inv) -> BF16."""
    for k in [k for k in sd if k.endswith(".weight_scale_inv")]:
        base = k[: -len("_scale_inv")]
        sd[base] = fp8_block_dequant(sd[base], sd.pop(k))


@torch.no_grad()
def load_layer(ckpt: Checkpoint, cfg, l: int, device: str | torch.device, experts: bool = True) -> nn.Module:
    """Official Step5DecoderLayer `l` on `device`, every tensor from the FP8 checkpoint. Routed experts are BF16
    [352, out, in] on layer.moe.{gate,up,down}_proj.weight with weight_scale_inv None; with experts=False they are None
    (routing and attention still work, layer.moe.forward does not). Indexer tensors of full-attention layers are
    loaded too (the official sparse path needs them; dense_attention does not)."""
    ref = load_reference(ckpt.dir)
    dev = torch.device(device)
    layer = ref.Step5DecoderLayer(cfg, l, device="meta", dtype=torch.bfloat16)
    prefix = f"model.layers.{l}."
    names = [n for n in ckpt.names(prefix) if not _is_routed_expert(n)]
    sd = {n[len(prefix):]: t for n, t in ckpt.get_many(names, dev).items()}
    _dequant_non_expert(sd)
    for k, v in sd.items():
        # router weight: kept as stored (route() computes in fp32 either way; casting an fp32 tensor to the declared
        # bf16 would round it)
        _set_tensor(layer, k, v, keep_dtype=k == "moe.gate.weight")
    del sd
    if layer.moe is not None:
        ws = expert_weights(ckpt, l, dev) if experts else (None,) * 3
        for proj, w in zip(EXPERT_PROJS, ws):
            _set_tensor(layer, f"moe.{proj}.weight", w, keep_dtype=True)
            if getattr(layer.moe.get_submodule(proj), "weight_scale_inv", None) is not None:
                _set_tensor(layer, f"moe.{proj}.weight_scale_inv", None)
        del ws
    attn = layer.self_attn
    attn.build_rope(max_position=MAX_POSITION, device=dev)
    attn.pack_qkv_()
    left = [n for n, t in [*layer.named_parameters(), *layer.named_buffers()] if t.is_meta]
    if left:
        raise RuntimeError(f"layer {l}: not in the checkpoint: {left}")
    return layer.eval()


@torch.no_grad()
def load_head(ckpt: Checkpoint, cfg, device: str | torch.device) -> tuple[nn.Module, nn.Module]:
    """(official final norm Step5RMSNorm, lm_head nn.Linear BF16) for Streamer.final_logits."""
    ref = load_reference(ckpt.dir)
    t = ckpt.get_many(["model.norm.weight", "lm_head.weight"], device)
    norm = ref.Step5RMSNorm(cfg.hidden_size, eps=cfg.rms_norm_eps)
    norm.weight = nn.Parameter(t["model.norm.weight"].float(), requires_grad=False)
    head = nn.Linear(cfg.hidden_size, cfg.vocab_size, bias=False, device="meta", dtype=torch.bfloat16)
    head.weight = nn.Parameter(t["lm_head.weight"].to(torch.bfloat16), requires_grad=False)
    return norm.eval(), head.eval()


def window_attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, scale: float, window: int | None,
                     chunk: int = SLIDING_PREFILL_CHUNK) -> torch.Tensor:
    """Causal GQA attention, q [T, Hq, D], k/v [T, Hkv, D] -> [T, Hq, D]; q head h reads kv head h // (Hq / Hkv)
    (SDPA enable_gqa = repeat_interleave, the grouping of the official _split_packed_qkv).

    window None: plain causal SDPA. window w: token i attends to j with i - w < j <= i (official
    `_sliding_prefill_chunked`: mask `(distance >= 0) & (distance < window)`, distance = i - j, model.py:1811-1814),
    computed the same way (query chunks of `chunk`, explicit boolean mask) so sliding layers match the official
    arithmetic."""
    if window is None:
        out = F.scaled_dot_product_attention(q.transpose(0, 1)[None], k.transpose(0, 1)[None],
                                             v.transpose(0, 1)[None], is_causal=True, scale=scale, enable_gqa=True)
        # ponytail: one SDPA call; if no fused backend takes it, the math path holds [Hq, T, T] (4 GiB fp32 at T=4096)
        return out[0].transpose(0, 1)
    T = q.shape[0]
    out = torch.empty_like(q)
    pos = torch.arange(T, device=q.device)
    for start in range(0, T, chunk):
        stop = min(start + chunk, T)
        k0 = max(0, start - window + 1)
        dist = pos[start:stop].view(-1, 1) - pos[k0:stop].view(1, -1)
        mask = (dist >= 0) & (dist < window)
        o = F.scaled_dot_product_attention(
            q[start:stop].transpose(0, 1)[None], k[k0:stop].transpose(0, 1)[None], v[k0:stop].transpose(0, 1)[None],
            attn_mask=mask.view(1, 1, stop - start, -1), scale=scale, enable_gqa=True)
        out[start:stop] = o[0].transpose(0, 1)
    return out


def dense_attention(layer: nn.Module, hidden: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
    """Attention half of one decoder layer for one sequence: hidden BF16 [T, 4096] (post input_layernorm),
    positions int32 [T] -> attention output BF16 [T, 4096] (before the residual add)."""
    attn = layer.self_attn
    q, k, v = attn._project(hidden, positions)
    o = window_attention(q, k, v, attn.scaling, attn.sliding_window)
    return attn._gate_and_project(o, hidden)
