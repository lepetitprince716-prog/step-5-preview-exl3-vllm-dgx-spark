# SPDX-License-Identifier: MIT
"""One-launch routing prep for the decode-sized exl3_moe path.

For every K bucket the per-bucket torch prep (index the bucket map, argsort, bincount via zeros + scatter_add, two
gathers) costs about seven small kernels; with four buckets that is ~30 launches a layer. This kernel runs one program
a bucket and writes, for each, what exl3_moe takes: the bincount over the bucket's E + 1 bins (bin E holds the routes
to other buckets), and the token index (int64) and routing weight (fp16) of every route ordered by bin. The order is a
stable counting sort, so it is a function of the inputs alone. Up to MAX_ROUTES routes (tokens x top_k).
"""
from __future__ import annotations

import torch
import triton
import triton.language as tl

MAX_ROUTES = 128
_EBLOCK = 64


@triton.jit
def _prep_kernel(ids_ptr, w_ptr, maps_ptr, nexp_ptr, cnt_ptr, tok_ptr, ws_ptr, n, top_k, num_experts, cnt_stride,
                 list_stride, BLOCK: tl.constexpr, EBLOCK: tl.constexpr):
    b = tl.program_id(0)
    E = tl.load(nexp_ptr + b)
    i = tl.arange(0, BLOCK)
    ok = i < n
    ids = tl.load(ids_ptr + i, mask=ok, other=0)
    loc = tl.load(maps_ptr + b * num_experts + ids, mask=ok, other=0)
    loc = tl.where(ok, loc, E + 1)           # padding sorts after bin E and lands in no bin
    li = loc[:, None]
    lj = loc[None, :]
    before = (lj < li) | ((lj == li) & (i[None, :] < i[:, None]))
    pos = tl.sum(before.to(tl.int32), axis=1)
    tl.store(tok_ptr + b * list_stride + pos, (i // top_k).to(tl.int64), mask=ok)
    wt = tl.load(w_ptr + i, mask=ok, other=0.0)
    tl.store(ws_ptr + b * list_stride + pos, wt.to(tl.float16), mask=ok)
    for e0 in range(0, E + 1, EBLOCK):
        e = e0 + tl.arange(0, EBLOCK)
        c = tl.sum((lj == e[:, None]).to(tl.int64), axis=1)
        tl.store(cnt_ptr + b * cnt_stride + e, c, mask=e <= E)


class BucketPrep:
    """Bucket maps of one layer, stacked for the kernel: maps[b, expert] = local index in bucket b, or E_b."""

    def __init__(self, maps: list[torch.Tensor], sizes: list[int]) -> None:
        self.maps = torch.stack([m.to(torch.int64) for m in maps]).contiguous()
        self.sizes = sizes
        self.nexp = torch.tensor(sizes, dtype=torch.int64, device=self.maps.device)
        self.cnt_stride = max(sizes) + 1

    def __call__(self, topk_ids: torch.Tensor, topk_weights: torch.Tensor):
        """-> [(count [E_b + 1] int64, token_sorted [n] int64, weight_sorted [n] fp16)] for each bucket."""
        ids = topk_ids.reshape(-1)
        w = topk_weights.reshape(-1)
        n = ids.numel()
        if n > MAX_ROUTES or not ids.is_contiguous() or not w.is_contiguous():
            raise ValueError(f"BucketPrep: {n} routes (max {MAX_ROUTES}), contiguous {ids.is_contiguous()} "
                             f"{w.is_contiguous()}")
        nb = len(self.sizes)
        dev = ids.device
        cnt = torch.empty((nb, self.cnt_stride), dtype=torch.int64, device=dev)
        tok = torch.empty((nb, max(n, 1)), dtype=torch.int64, device=dev)
        ws = torch.empty((nb, max(n, 1)), dtype=torch.float16, device=dev)
        _prep_kernel[(nb,)](ids, w, self.maps, self.nexp, cnt, tok, ws, n, topk_ids.shape[-1], self.maps.shape[1],
                            self.cnt_stride, tok.shape[1], BLOCK=max(16, triton.next_power_of_2(n)), EBLOCK=_EBLOCK,
                            num_warps=4)
        return [(cnt[b, : e + 1], tok[b, :n], ws[b, :n]) for b, e in enumerate(self.sizes)]


def reference(maps: list[torch.Tensor], sizes: list[int], topk_ids: torch.Tensor, topk_weights: torch.Tensor):
    """The same outputs from torch ops (stable sort), for tests."""
    flat_ids = topk_ids.reshape(-1).long()
    flat_w = topk_weights.reshape(-1).to(torch.half)
    T, top_k = topk_ids.shape
    flat_tok = torch.arange(T, device=topk_ids.device).repeat_interleave(top_k)
    out = []
    for m, E in zip(maps, sizes):
        loc = m[flat_ids]
        order = torch.sort(loc, stable=True).indices
        cnt = torch.bincount(loc, minlength=E + 1)
        out.append((cnt, flat_tok[order], flat_w[order]))
    return out
