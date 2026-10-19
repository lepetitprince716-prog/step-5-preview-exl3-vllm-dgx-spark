# SPDX-License-Identifier: MIT
"""Skinny GEMM for decode-sized inputs (at most MAX_ROWS rows): out[M, N] = x[M, K] @ w[N, K]^T, fp32 accumulation.

cuBLAS picks kernels for these shapes on GB10 that use a fraction of the DRAM bandwidth: the fp32 router (4096 -> 352,
gemmSN ~45 us a layer for 5.8 MB), the head-wise attention gate g_proj (4096 -> 32 a rank, a 2-CTA wmma kernel
~23 us for 256 KB) and the MTP draft heads (one row, gemvx ~2.96 ms for 528 MB a rank). fp16/bf16 inputs: each program
streams BLOCK_N rows of w over one K slice into a 16-row tensor-core tile (rows past M are zero); SPLIT_K > 1 writes
fp32 partials that a second launch sums in a fixed order. fp32 inputs (the router): IEEE fp32 multiply-adds up to
F32_FMA_ROWS rows (bf16-stored router weights widened in registers give the fp32 router's products exactly), the split3
tensor-core form above that. Every result is deterministic for a given row count; inputs above MAX_ROWS go to torch.
int8 weights with one fp32 scale a row (draft heads only, STEP5_SKINNY=draft8) halve the bytes the heads stream.
"""
from __future__ import annotations

import os

import torch
import triton
import triton.language as tl

MAX_ROWS = 16


@triton.jit
def _skinny_kernel(x_ptr, w_ptr, o_ptr, M, N, sxm, swn, som, k_per_split,
                   BN: tl.constexpr, BK: tl.constexpr, PARTIAL: tl.constexpr, F32: tl.constexpr):
    BM: tl.constexpr = 16
    pid_n = tl.program_id(0)
    pid_k = tl.program_id(1)
    rn = pid_n * BN + tl.arange(0, BN)
    rm = tl.arange(0, BM)
    m_ok = rm < M
    n_ok = rn < N
    acc = tl.zeros((BM, BN), dtype=tl.float32)
    k_lo = pid_k * k_per_split
    for k0 in range(k_lo, k_lo + k_per_split, BK):
        rk = k0 + tl.arange(0, BK)
        x = tl.load(x_ptr + rm[:, None] * sxm + rk[None, :], mask=m_ok[:, None], other=0.0)
        w = tl.load(w_ptr + rn[:, None] * swn + rk[None, :], mask=n_ok[:, None], other=0.0,
                    eviction_policy="evict_first")
        if F32:
            # fp32 x times bf16 w on tensor cores: x = hi + mid + lo (three bf16 terms, residual ~2^-24 |x|); each
            # bf16 x bf16 product is exact in fp32 and the mma sums in fp32, so this matches an fp32 dot to rounding
            xf = x.to(tl.float32)
            hi = xf.to(tl.bfloat16)
            r = xf - hi.to(tl.float32)
            mid = r.to(tl.bfloat16)
            lo = (r - mid.to(tl.float32)).to(tl.bfloat16)
            wt = tl.trans(w)
            acc = tl.dot(lo, wt, acc)
            acc = tl.dot(mid, wt, acc)
            acc = tl.dot(hi, wt, acc)
        else:
            acc = tl.dot(x, tl.trans(w), acc)
    if PARTIAL:
        base = o_ptr + pid_k * M * N
        tl.store(base + rm[:, None] * N + rn[None, :], acc, mask=m_ok[:, None] & n_ok[None, :])
    else:
        tl.store(o_ptr + rm[:, None] * som + rn[None, :], acc.to(o_ptr.dtype.element_ty),
                 mask=m_ok[:, None] & n_ok[None, :])


@triton.jit
def _fma_kernel(x_ptr, w_ptr, o_ptr, M, N, K, sxm, swn, som,
                BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr):
    """fp32 x, any w: IEEE fp32 multiply-adds on CUDA cores, one program a BN-row slab of w over all of K."""
    pid_n = tl.program_id(0)
    rn = pid_n * BN + tl.arange(0, BN)
    rm = tl.arange(0, BM)
    m_ok = rm < M
    n_ok = rn < N
    acc = tl.zeros((BM, BN), dtype=tl.float32)
    for k0 in range(0, K, BK):
        rk = k0 + tl.arange(0, BK)
        x = tl.load(x_ptr + rm[:, None] * sxm + rk[None, :], mask=m_ok[:, None], other=0.0).to(tl.float32)
        w = tl.load(w_ptr + rn[:, None] * swn + rk[None, :], mask=n_ok[:, None], other=0.0,
                    eviction_policy="evict_first").to(tl.float32)
        acc += tl.sum(x[:, None, :] * w[None, :, :], axis=2)
    tl.store(o_ptr + rm[:, None] * som + rn[None, :], acc.to(o_ptr.dtype.element_ty),
             mask=m_ok[:, None] & n_ok[None, :])


@triton.jit
def _skinny_w8_kernel(x_ptr, w_ptr, s_ptr, o_ptr, M, N, K, sxm, swn, som, k_per_split,
                      BN: tl.constexpr, BK: tl.constexpr, PARTIAL: tl.constexpr):
    """int8 w with one fp32 scale a row: out = (x @ w^T) * scale. int8 widens exactly to x's 16-bit type in
    registers, so the tensor-core products and the fp32 sums are the same as for the dequantized matrix. Program
    (n, k) covers BN rows over K[k * k_per_split, (k + 1) * k_per_split); PARTIAL writes fp32 partials."""
    BM: tl.constexpr = 16
    pid_n = tl.program_id(0)
    pid_k = tl.program_id(1)
    rn = pid_n * BN + tl.arange(0, BN)
    rm = tl.arange(0, BM)
    m_ok = rm < M
    n_ok = rn < N
    acc = tl.zeros((BM, BN), dtype=tl.float32)
    k_lo = pid_k * k_per_split
    k_hi = tl.minimum(k_lo + k_per_split, K)
    for k0 in range(k_lo, k_hi, BK):
        rk = k0 + tl.arange(0, BK)
        k_ok = rk < k_hi
        x = tl.load(x_ptr + rm[:, None] * sxm + rk[None, :], mask=m_ok[:, None] & k_ok[None, :], other=0.0)
        w = tl.load(w_ptr + rn[:, None] * swn + rk[None, :], mask=n_ok[:, None] & k_ok[None, :], other=0,
                    eviction_policy="evict_first")
        acc = tl.dot(x, tl.trans(w.to(x.dtype)), acc)
    acc = acc * tl.load(s_ptr + rn, mask=n_ok, other=0.0)[None, :]
    if PARTIAL:
        tl.store(o_ptr + pid_k * M * N + rm[:, None] * N + rn[None, :], acc, mask=m_ok[:, None] & n_ok[None, :])
    else:
        tl.store(o_ptr + rm[:, None] * som + rn[None, :], acc.to(o_ptr.dtype.element_ty),
                 mask=m_ok[:, None] & n_ok[None, :])


@triton.jit
def _sum_splits_kernel(p_ptr, o_ptr, MN, N, som, SPLIT: tl.constexpr, BLOCK: tl.constexpr):
    i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    ok = i < MN
    acc = tl.zeros((BLOCK,), dtype=tl.float32)
    for s in tl.static_range(SPLIT):
        acc += tl.load(p_ptr + s * MN + i, mask=ok, other=0.0)
    m = i // N
    n = i - m * N
    tl.store(o_ptr + m * som + n, acc.to(o_ptr.dtype.element_ty), mask=ok)


BK = 128
# fp32 inputs (the router) up to F32_FMA_ROWS rows: CUDA-core fp32 multiply-adds (any weight dtype), plan (BLOCK_N,
# BLOCK_K, warps); measured on GB10 for 4096 -> 352 with bf16 weights: 15-17 us at 1-4 rows against cuBLAS's 26-43 us
# on fp32 weights, error 1.1e-7 of the largest output (cuBLAS 1.3-1.8e-7). More rows with bf16 weights: x split into
# three bf16 terms on tensor cores ("split3", 16-18 us at any row count, error 5.5e-7 with 16 K splits).
# STEP5_SKINNY_F32 / STEP5_SKINNY_F32_ROWS override the plan and the row limit (tuning only).
F32_PLAN = tuple(int(v) for v in (os.environ.get("STEP5_SKINNY_F32") or "4,512,4").split(","))
F32_FMA_ROWS = int(os.environ.get("STEP5_SKINNY_F32_ROWS") or "4")


def plan(N: int, K: int, f32: bool = False) -> tuple[int, int]:
    """(BLOCK_N, SPLIT_K) from the shape alone, so a row's bits never depend on how many rows came with it."""
    if N >= 4096:                       # heads: a thousand programs without a K split
        return 64, 1
    bn = 16
    programs = triton.cdiv(N, bn)
    split, target = 1, (256 if f32 else 64)   # split3 sums shorter tensor-core chains (accuracy), others fill the GPU
    while programs * split < target and K % (2 * split * BK) == 0:
        split *= 2
    return bn, split


_WS: dict[tuple, torch.Tensor] = {}


def _workspace(dev: torch.device, n: int) -> torch.Tensor:
    key = (dev.index, n)
    ws = _WS.get(key)
    if ws is None:
        ws = _WS[key] = torch.empty((n,), dtype=torch.float32, device=dev)
    return ws


def skinny_mm(x: torch.Tensor, w: torch.Tensor, out_dtype: torch.dtype) -> torch.Tensor:
    """x [M, K] (M <= MAX_ROWS, unit stride along K), w [N, K] contiguous -> [M, N] in out_dtype.

    fp16/bf16 x must match w's dtype (tensor-core products, fp32 sums); fp32 x takes fp32 or bf16 w."""
    M, K = x.shape
    N = w.shape[0]
    if M > MAX_ROWS or w.shape[1] != K or x.stride(1) != 1 or not w.is_contiguous() or K % BK:
        raise ValueError(f"skinny_mm: x {tuple(x.shape)} stride {x.stride()}, w {tuple(w.shape)}")
    f32 = x.dtype == torch.float32
    if not f32 and x.dtype != w.dtype:
        raise ValueError(f"skinny_mm: x {x.dtype} with w {w.dtype}")
    out = torch.empty((M, N), dtype=out_dtype, device=x.device)
    if M == 0:
        return out
    if f32 and (M <= F32_FMA_ROWS or w.dtype != torch.bfloat16):
        bn, bk, warps = F32_PLAN
        bm = triton.next_power_of_2(M)
        _fma_kernel[(triton.cdiv(N, bn),)](x, w, out, M, N, K, x.stride(0), w.stride(0), out.stride(0),
                                           BM=bm, BN=bn, BK=bk, num_warps=warps, num_stages=3)
        return out
    bn, split = plan(N, K, f32)
    grid = (triton.cdiv(N, bn), split)
    if split == 1:
        _skinny_kernel[grid](x, w, out, M, N, x.stride(0), w.stride(0), out.stride(0), K,
                             BN=bn, BK=BK, PARTIAL=False, F32=f32, num_warps=4, num_stages=3)
        return out
    # one workspace a device, sized for MAX_ROWS: CUDA graphs of every decode size share it (they never overlap)
    part = _workspace(x.device, split * MAX_ROWS * N)
    _skinny_kernel[grid](x, w, part, M, N, x.stride(0), w.stride(0), out.stride(0), K // split,
                         BN=bn, BK=BK, PARTIAL=True, F32=f32, num_warps=4, num_stages=3)
    block = 256
    _sum_splits_kernel[(triton.cdiv(M * N, block),)](part, out, M * N, N, out.stride(0), SPLIT=split, BLOCK=block,
                                                     num_warps=4)
    return out


# int8 weights (the MTP draft heads and blocks): (BLOCK_N, BLOCK_K, warps, stages, target programs); K is split
# until the grid reaches the target. STEP5_SKINNY_W8 overrides it (tuning only).
W8_PLAN = tuple(int(v) for v in (os.environ.get("STEP5_SKINNY_W8") or "64,256,4,3,384").split(","))


def plan_w8(N: int, K: int) -> tuple[int, int]:
    """(SPLIT_K, K per split) from the shape alone; each split covers whole BLOCK_K steps."""
    bn, bk, _, _, target = W8_PLAN
    programs, split = triton.cdiv(N, bn), 1
    while programs * split < target and K // (2 * split) >= 2 * bk:
        split *= 2
    per = triton.cdiv(triton.cdiv(K, split), bk) * bk
    return triton.cdiv(K, per), per


def quantize_rows_int8(w: torch.Tensor, chunk: int = 8192) -> tuple[torch.Tensor, torch.Tensor]:
    """Symmetric int8 with one fp32 scale a row, w ~= q * scale[:, None]; works in row chunks to bound memory."""
    q = torch.empty(w.shape, dtype=torch.int8, device=w.device)
    scale = torch.empty((w.shape[0],), dtype=torch.float32, device=w.device)
    for i in range(0, w.shape[0], chunk):
        wf = w[i:i + chunk].float()
        s = wf.abs().amax(dim=1).clamp_min(1e-30) / 127.0
        q[i:i + chunk] = torch.round(wf / s[:, None]).clamp_(-127, 127).to(torch.int8)
        scale[i:i + chunk] = s
    return q, scale


def skinny_mm_w8(x: torch.Tensor, w: torch.Tensor, scale: torch.Tensor, out_dtype: torch.dtype) -> torch.Tensor:
    """x [M, K] fp16/bf16 (M <= MAX_ROWS, unit stride along K), w [N, K] int8 contiguous, scale [N] fp32 -> [M, N]."""
    M, K = x.shape
    N = w.shape[0]
    bn, bk, warps, stages, _ = W8_PLAN
    if (M > MAX_ROWS or w.dtype != torch.int8 or w.shape[1] != K or x.stride(1) != 1 or not w.is_contiguous()
            or x.dtype not in (torch.float16, torch.bfloat16) or scale.dtype != torch.float32
            or scale.shape != (N,)):
        raise ValueError(f"skinny_mm_w8: x {tuple(x.shape)} {x.dtype} stride {x.stride()}, w {tuple(w.shape)} "
                         f"{w.dtype}, scale {tuple(scale.shape)} {scale.dtype}")
    out = torch.empty((M, N), dtype=out_dtype, device=x.device)
    if M == 0:
        return out
    split, per = plan_w8(N, K)
    grid = (triton.cdiv(N, bn), split)
    if split == 1:
        _skinny_w8_kernel[grid](x, w, scale, out, M, N, K, x.stride(0), w.stride(0), out.stride(0), per,
                                BN=bn, BK=bk, PARTIAL=False, num_warps=warps, num_stages=stages)
        return out
    part = _workspace(x.device, split * MAX_ROWS * N)
    _skinny_w8_kernel[grid](x, w, scale, part, M, N, K, x.stride(0), w.stride(0), out.stride(0), per,
                            BN=bn, BK=bk, PARTIAL=True, num_warps=warps, num_stages=stages)
    block = 256
    _sum_splits_kernel[(triton.cdiv(M * N, block),)](part, out, M * N, N, out.stride(0), SPLIT=split, BLOCK=block,
                                                     num_warps=4)
    return out


@torch.library.custom_op("step5_spark::skinny_linear_w8", mutates_args=())
def skinny_linear_w8(x: torch.Tensor, w: torch.Tensor, scale: torch.Tensor, out_dtype: torch.dtype,
                     full: torch.Tensor | None = None) -> torch.Tensor:
    """x @ (w * scale[:, None])^T for int8 w up to MAX_ROWS rows. Above that: F.linear with `full` (the 16-bit
    matrix the int8 copy was made from) when given, else the int8 matrix widened to fp32 in row chunks."""
    x2 = x.reshape(-1, x.shape[-1])
    if x2.shape[0] <= MAX_ROWS:
        y = skinny_mm_w8(x2 if x2.stride(1) == 1 else x2.contiguous(), w, scale, out_dtype)
    elif full is not None:
        y = torch.nn.functional.linear(x2, full if full.dtype == x2.dtype else full.to(x2.dtype)).to(out_dtype)
    else:
        y = torch.empty((x2.shape[0], w.shape[0]), dtype=out_dtype, device=x.device)
        xf = x2.float()
        for i in range(0, w.shape[0], 8192):
            y[:, i:i + 8192] = (torch.nn.functional.linear(xf, w[i:i + 8192].float()) * scale[i:i + 8192]).to(out_dtype)
    return y.reshape(*x.shape[:-1], w.shape[0])


@skinny_linear_w8.register_fake
def _(x, w, scale, out_dtype, full=None):
    return x.new_empty((*x.shape[:-1], w.shape[0]), dtype=out_dtype)


@torch.library.custom_op("step5_spark::skinny_linear", mutates_args=())
def skinny_linear(x: torch.Tensor, w: torch.Tensor, out_dtype: torch.dtype) -> torch.Tensor:
    """x @ w^T for any number of rows: skinny_mm up to MAX_ROWS rows, F.linear (w widened to x's dtype) above."""
    x2 = x.reshape(-1, x.shape[-1])
    if x2.shape[0] <= MAX_ROWS:
        y = skinny_mm(x2 if x2.stride(1) == 1 else x2.contiguous(), w, out_dtype)
    else:
        y = torch.nn.functional.linear(x2, w if w.dtype == x2.dtype else w.to(x2.dtype)).to(out_dtype)
    return y.reshape(*x.shape[:-1], w.shape[0])


@skinny_linear.register_fake
def _(x, w, out_dtype):
    return x.new_empty((*x.shape[:-1], w.shape[0]), dtype=out_dtype)
