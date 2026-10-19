#!/usr/bin/env python3
"""moe_prep.BucketPrep against the torch reference (stable sort) on random routings, plus graph replay and time
against the plugin's per-bucket torch prep (one GPU, no server).

    python3 moe_prep_check.py PATCH_DIR
"""
from __future__ import annotations

import importlib.util
import sys

import torch

EXPERTS, TOP_K = 352, 8


def load(patch_dir: str):
    spec = importlib.util.spec_from_file_location("moe_prep", f"{patch_dir}/step5_spark/moe_prep.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def buckets(gen: torch.Generator, dev: torch.device, sizes: list[int]):
    perm = torch.randperm(EXPERTS, generator=gen, device="cpu")
    maps, start = [], 0
    for e in sizes:
        m = torch.full((EXPERTS,), e, dtype=torch.long)
        m[perm[start:start + e]] = torch.arange(e)
        maps.append(m.to(dev))
        start += e
    return maps


def torch_prep(maps, sizes, topk_ids, topk_weights):
    """The plugin's decode prep (exl3_quant apply, all_fused branch)."""
    T, top_k = topk_ids.shape
    flat_ids = topk_ids.reshape(-1).long()
    flat_w = topk_weights.reshape(-1).to(torch.half)
    flat_tok = torch.arange(T, device=topk_ids.device).repeat_interleave(top_k)
    out = []
    for m, E in zip(maps, sizes):
        loc = m[flat_ids]
        order = loc.argsort()
        cnt = torch.zeros(E + 1, dtype=torch.long, device=loc.device).scatter_add_(0, loc, torch.ones_like(loc))
        out.append((cnt, flat_tok[order], flat_w[order]))
    return out


def timed(fn, iters: int = 50) -> float:
    fn()
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        for _ in range(iters):
            fn()
    ts = []
    for _ in range(5):
        a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        a.record()
        g.replay()
        b.record()
        torch.cuda.synchronize()
        ts.append(a.elapsed_time(b) * 1000 / iters)
    return sorted(ts)[2]


def main() -> int:
    mp = load(sys.argv[1])
    dev = torch.device("cuda")
    gen = torch.Generator(device="cpu").manual_seed(0)
    sizes = [205, 117, 25, 5]        # K2..K5 bucket sizes of a typical layer (sum 352)
    maps = buckets(gen, dev, sizes)
    prep = mp.BucketPrep(maps, sizes)
    ok_all = True
    for T in (1, 2, 3, 4, 8, 12, 16):
        ids = torch.stack([torch.randperm(EXPERTS, generator=gen)[:TOP_K] for _ in range(T)]).to(torch.int32).to(dev)
        w = torch.rand((T, TOP_K), generator=gen).to(dev)
        got = prep(ids, w)
        ref = mp.reference(maps, sizes, ids, w)
        same = all(torch.equal(a, c) and torch.equal(b, d) and torch.equal(x, y)
                   for (a, b, x), (c, d, y) in zip(got, ref))
        # the plugin's argsort order may differ inside a bin; counts and per-bin multisets must not
        tp = torch_prep(maps, sizes, ids, w)
        same_counts = all(torch.equal(a, c) for (a, _, _), (c, _, _) in zip(got, tp))
        out_g = [torch.empty_like(t) for trip in got for t in trip]
        g = torch.cuda.CUDAGraph()
        torch.cuda.synchronize()
        with torch.cuda.graph(g):
            res = prep(ids, w)
            for o, t in zip(out_g, [t for trip in res for t in trip]):
                o.copy_(t)
        g.replay()
        torch.cuda.synchronize()
        graph_ok = all(torch.equal(o, t) for o, t in zip(out_g, [t for trip in got for t in trip]))
        t_k = timed(lambda: prep(ids, w))
        t_t = timed(lambda: torch_prep(maps, sizes, ids, w))
        ok_all &= same and same_counts and graph_ok
        print(f"T={T:2d} routes {T * TOP_K:3d}: equal to stable reference {same}, counts equal to plugin prep "
              f"{same_counts}, graph==eager {graph_ok} | kernel {t_k:6.1f} us, torch prep {t_t:6.1f} us", flush=True)
    print("PASS" if ok_all else "FAIL", flush=True)
    return 0 if ok_all else 1


if __name__ == "__main__":
    raise SystemExit(main())
