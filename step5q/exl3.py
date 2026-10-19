# SPDX-License-Identifier: Apache-2.0
"""EXL3 encoding of Step 5 linears with turboderp's exllamav3 (1.6.0 in the quant container, MIT) quantizer, ported
from sudoingx dsv41q (exl3.h_data / quant_args / encode_expert / reconstruct / expert_forward_dense); nothing from
exllamav3 is copied, its public functions are called.

Orientation: exllamav3 works on row-major (in_features, out_features) weights, the transpose of nn.Linear's
[out, in] (quantize_exl3 docstring; trellis (in/16, out/16, 16 K), suh (in,), svh (out,)). Everything here takes and
returns nn.Linear orientation and transposes at the exllamav3 boundary.

Step 5 routed expert (model.py:2032-2078, 1880-1899): gate/up [1536, 4096], down [4096, 1536];
a = silu(gate).clamp(max=7) * up.clamp(-7, 7), y = down(a) * routing weight (the weight multiplies the expert's
output after down_proj; DeepSeek-V4.1 applied it before w2). Gate and up see the same input, so they share one
Hessian and go through quantize_exl3_batch's shared-H path (one su for both); down gets its own Hessian.
"""
from __future__ import annotations

import os

import torch
import torch.nn.functional as F

EXPERT_K = (2, 3, 4, 5)       # README: integer K per expert (serving image 1.5.3 decodes half-integer K as K2)


def _int_k(K: float) -> int:
    """exllamav3's bindings reject 5.0 for K; half-integer K is excluded by the plan."""
    if not float(K).is_integer():
        raise ValueError(f"K must be an integer, got {K}")
    return int(K)


def h_data(H_sum: torch.Tensor, count: float, key: str) -> dict:
    """exllamav3's H_data dict. H_sum is a SUM of x^T x over `count` rows; finalize_capture_H divides by count in
    place, damps the diagonal by sigma_reg x its mean, draws su and moves H to the CPU, so the dict (and H_sum) is
    consumed: pass a clone to encode the same H twice."""
    return {"H": H_sum, "count": count, "finalized": False, "num_total": int(count) * H_sum.shape[0],
            "inf_nan": torch.zeros(2, dtype=torch.long, device=H_sum.device), "device": H_sum.device,
            "first_key": key}


def quant_args(K: float, seed: int, work_dir: str, codebook: str = "mul1", device: int = 0,
               no_refit: bool = False) -> dict:
    """quant_args for quantize_exl3[_batch]. no_refit: skip the post-LDLQ suh/svh refit where the installed
    exllamav3 has it (present in 1.6.2 source, quantize.py:1828-1839; unverified for 1.6.0). A refit gives each linear
    its own suh, which turns off 0xSero's fused q/k/v and gate/up GEMV (exl3_quant.py: fuse needs equal suh)."""
    qa = {"K": _int_k(K), "seed": seed, "devices": [device], "device_ratios": None, "apply_out_scales": None,
          "debug_dir": os.path.join(work_dir, "x3_debug")}
    qa[codebook] = True
    if no_refit:
        qa["no_refit"] = True
    return qa


def encode_expert(w1: torch.Tensor, w3: torch.Tensor, w2: torch.Tensor, H_gu: dict, H_d: dict, K: int, seed: int,
                  work_dir: str) -> tuple[dict, dict]:
    """One routed expert at bitrate K (same K for gate/up/down, as 0xSero's loader requires).
    w1/w3: gate/up [1536, 4096], w2: down [4096, 1536] (nn.Linear orientation, any float dtype).
    H_gu: h_data over the expert's input (4096 x 4096), shared by gate and up; H_d: over the down input (1536 x 1536).
    Returns ({"w1"|"w3"|"w2": {trellis, suh, svh, mul1}}, {"w1"|"w3"|"w2": proxy_err})."""
    if _int_k(K) not in EXPERT_K:
        raise ValueError(f"expert K must be one of {EXPERT_K}, got {K}")
    from exllamav3.modules.quant.exl3_lib.quantize import quantize_exl3, quantize_exl3_batch
    qa_g, qa_u = quant_args(K, seed, work_dir), quant_args(K, seed + 1, work_dir)
    (pe1, t1), (pe3, t3) = quantize_exl3_batch([w1.float().t().contiguous(), w3.float().t().contiguous()],
                                               [H_gu, H_gu], [qa_g, qa_u])
    _, pe2, t2 = quantize_exl3(w2.float().t().contiguous(), H_d, quant_args(K, seed + 2, work_dir),
                               return_weight_q=False)
    return {"w1": t1, "w3": t3, "w2": t2}, {"w1": float(pe1), "w3": float(pe3), "w2": float(pe2)}


def encode_linear(w: torch.Tensor, H: dict, K: int, seed: int, work_dir: str,
                  no_refit: bool = False) -> tuple[dict, float]:
    """One dense linear (attention, shared expert, dense MLP), w [out, in] nn.Linear orientation, H = h_data over its
    input. Linears that share an input (q/k/v; gate/up) pass the same H dict: finalize draws su once per dict.
    Returns ({trellis, suh, svh, mul1}, proxy_err)."""
    from exllamav3.modules.quant.exl3_lib.quantize import quantize_exl3
    _, pe, t = quantize_exl3(w.float().t().contiguous(), H, quant_args(K, seed, work_dir, no_refit=no_refit),
                             return_weight_q=False)
    return t, float(pe)


def reconstruct(t: dict, in_features: int, out_features: int) -> torch.Tensor:
    """Stored EXL3 pieces -> fp16 [out, in] (nn.Linear orientation): LinearEXL3.get_weight_tensor, the
    reconstruction the serving kernels compute (exl3.py:239-249, returns [in, out])."""
    from exllamav3.modules.quant.exl3 import LinearEXL3
    lin = LinearEXL3(None, in_features, out_features, suh=t["suh"], svh=t["svh"], trellis=t["trellis"],
                     mcg=t.get("mcg"), mul1=t.get("mul1"))
    return lin.get_weight_tensor().t()


def expert_forward_dense(x: torch.Tensor, g: torch.Tensor, w1: torch.Tensor, w3: torch.Tensor, w2: torch.Tensor,
                         limit: float = 7.0) -> torch.Tensor:
    """One routed expert as the EXL3 serving path computes it (0xSero exl3_prefill.heavy_recon, exllamav3 silu_mul):
    fp16 activations and weights, gate/up GEMMs out in fp16, a = silu(gate).clamp(max=limit) * up.clamp(-limit,
    limit) rounded to fp16 (exllamav3 does this in half precision, here in fp32 then rounded), down GEMM, then
    y = down.float() * g with g rounded to fp16 first (the plugin passes topk_weights.to(half)).
    x [n, 4096], g [n] routing weights, w1/w3 [1536, 4096], w2 [4096, 1536] -> fp32 [n, 4096]."""
    xh = x.to(torch.float16)
    gate = F.linear(xh, w1.to(torch.float16)).float()
    up = F.linear(xh, w3.to(torch.float16)).float()
    a = (F.silu(gate).clamp(max=limit) * up.clamp(-limit, limit)).to(torch.float16)
    y = F.linear(a, w2.to(torch.float16)).float()
    return y * g.to(torch.float16).float().unsqueeze(-1)
