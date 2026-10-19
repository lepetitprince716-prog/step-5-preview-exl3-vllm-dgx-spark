#!/usr/bin/env python3
"""Correctness and speed of step5_spark/skinny.py against torch on the Step-5 decode shapes (one GPU, no server).

    python3 skinny_check.py PATCH_DIR
    python3 skinny_check.py PATCH_DIR w8 [BN,BK,WARPS,STAGES ...]     (int8 head only, one line a plan)

Shapes per TP2 rank: router 4096 -> 352 (fp32 x, bf16 w), g_proj 4096 -> 32 (fp16), draft head 4096 -> 64448 (fp16,
and int8 with one fp32 scale a row).
Prints max relative error against an fp64 reference, cuBLAS and skinny times (CUDA graph of back-to-back calls over
enough weight copies to exceed L2), and whether graph replay equals eager.
"""
from __future__ import annotations

import importlib.util
import sys

import torch

L2_BYTES = 64 << 20
# MTP block linears per TP2 rank (name, N, K): hidden 4096, 32 q heads + 2 kv heads of 192, dense intermediate 6912
MTP_BLOCK = [("eh_proj", 4096, 8192), ("qkv", 6912, 4096), ("o_proj", 4096, 6144), ("gate_up", 13824, 4096),
             ("down", 4096, 6912)]


def load_skinny(patch_dir: str):
    spec = importlib.util.spec_from_file_location("skinny", f"{patch_dir}/step5_spark/skinny.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def timed(fn, iters: int = 20) -> float:
    """us per call: CUDA graph of `iters` calls, median of 5 replays."""
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


def case(sk, name: str, N: int, K: int, x_dtype, w_dtype, rows_list) -> None:
    dev = torch.device("cuda")
    gen = torch.Generator(device=dev).manual_seed(0)
    w0 = (torch.randn((N, K), generator=gen, device=dev) * 0.02).to(w_dtype)
    copies = max(1, -(-L2_BYTES // (w0.numel() * w0.element_size())) + 1)
    ws = [w0] + [w0.clone() for _ in range(copies - 1)]
    for M in rows_list:
        x = torch.randn((M, K), generator=gen, device=dev).to(x_dtype)
        ref = (x.double() @ w0.double().t())
        got = sk.skinny_mm(x, w0, torch.float32)
        cub = torch.nn.functional.linear(x, w0 if w0.dtype == x.dtype else w0.to(x.dtype)).float()
        scale = ref.abs().max().item()
        e_sk = (got.double() - ref).abs().max().item() / scale
        e_cb = (cub.double() - ref).abs().max().item() / scale
        # graph replay equals eager
        out_g = torch.empty_like(got)
        g = torch.cuda.CUDAGraph()
        sk.skinny_mm(x, w0, torch.float32)
        torch.cuda.synchronize()
        with torch.cuda.graph(g):
            out_g.copy_(sk.skinny_mm(x, w0, torch.float32))
        g.replay()
        torch.cuda.synchronize()
        same = torch.equal(out_g, got)
        # row invariance: row 0 alone equals row 0 of the batch
        alone = sk.skinny_mm(x[:1].contiguous(), w0, torch.float32)
        inv = torch.equal(alone[0], got[0])
        i = [0]

        def run_sk():
            sk.skinny_mm(x, ws[i[0] % copies], torch.float32)
            i[0] += 1

        j = [0]
        wt = [w if w.dtype == x.dtype else w.to(x.dtype) for w in ws] if w0.dtype != x.dtype else ws

        def run_cb():
            torch.nn.functional.linear(x, wt[j[0] % copies])
            j[0] += 1

        t_sk, t_cb = timed(run_sk), timed(run_cb)
        mb = w0.numel() * w0.element_size() / 1e6
        cb_mb = wt[0].numel() * wt[0].element_size() / 1e6
        print(f"{name:8s} M={M:2d} N={N:6d} K={K}: err skinny {e_sk:.2e} cublas {e_cb:.2e} | graph==eager {same} "
              f"row-invariant {inv} | skinny {t_sk:8.1f} us ({mb / t_sk * 1e3:.0f} GB/s over {mb:.1f} MB) "
              f"cublas {t_cb:8.1f} us ({cb_mb / t_cb * 1e3:.0f} GB/s over {cb_mb:.1f} MB)", flush=True)


def case_w8(sk, N: int, K: int, rows_list, name: str = "head-w8", base: str = "skinny") -> None:
    """int8 weights: kernel against the dequantized matrix in fp64, quantization against the fp16 matrix (error and
    top-1 agreement), graph replay, and time against the fp16 path it replaces (skinny for heads, cuBLAS else)."""
    dev = torch.device("cuda")
    gen = torch.Generator(device=dev).manual_seed(0)
    w0 = (torch.randn((N, K), generator=gen, device=dev) * 0.02).half()
    q, s = sk.quantize_rows_int8(w0)
    deq = q.double() * s.double()[:, None]
    w0d = w0.double()
    copies = max(1, -(-L2_BYTES // q.numel()) + 1)
    qs = [q] + [q.clone() for _ in range(copies - 1)]
    hs = [w0] + [w0.clone() for _ in range(max(1, -(-L2_BYTES // (w0.numel() * 2)) + 1) - 1)]
    for M in rows_list:
        x = torch.randn((M, K), generator=gen, device=dev).half()
        got = sk.skinny_mm_w8(x, q, s, torch.float32)
        ref_q = x.double() @ deq.t()
        ref_h = x.double() @ w0d.t()
        e_kernel = ((got.double() - ref_q).abs().max() / ref_q.abs().max()).item()
        e_quant = ((got.double() - ref_h).abs().max() / ref_h.abs().max()).item()
        top1 = (got.argmax(-1) == ref_h.argmax(-1)).float().mean().item()
        out_g = torch.empty_like(got)
        g = torch.cuda.CUDAGraph()
        torch.cuda.synchronize()
        with torch.cuda.graph(g):
            out_g.copy_(sk.skinny_mm_w8(x, q, s, torch.float32))
        g.replay()
        torch.cuda.synchronize()
        same = torch.equal(out_g, got)
        i, j = [0], [0]

        def run_w8():
            sk.skinny_mm_w8(x, qs[i[0] % len(qs)], s, torch.float32)
            i[0] += 1

        def run_h():
            if base == "skinny":
                sk.skinny_mm(x, hs[j[0] % len(hs)], torch.float32)
            else:
                torch.nn.functional.linear(x, hs[j[0] % len(hs)])
            j[0] += 1

        t8, th = timed(run_w8), timed(run_h)
        mb = q.numel() / 1e6
        print(f"{name:8s} M={M:2d} N={N} K={K}: err vs dequantized {e_kernel:.2e}, vs fp16 {e_quant:.2e}, top-1 "
              f"agreement {top1:.2f} | graph==eager {same} | w8 {t8:8.1f} us ({mb / t8 * 1e3:.0f} GB/s over "
              f"{mb:.0f} MB) fp16 {base} {th:8.1f} us", flush=True)


def main() -> int:
    sk = load_skinny(sys.argv[1])
    print(torch.cuda.get_device_name(), torch.__version__, flush=True)
    if len(sys.argv) > 2 and sys.argv[2] == "w8":
        plans = [tuple(int(v) for v in p.split(",")) for p in sys.argv[3:]] or [sk.W8_PLAN]
        for p in plans:
            sk.W8_PLAN = p
            print("w8 plan (BN, BK, warps, stages)", p, flush=True)
            try:
                case_w8(sk, 64448, 4096, [1, 4])
                for name, n, k in MTP_BLOCK:
                    case_w8(sk, n, k, [1, 4], name=name, base="cublas")
            except Exception as exc:          # a plan Triton cannot compile: report and try the next
                print("  failed:", type(exc).__name__, str(exc)[:300], flush=True)
        return 0
    case(sk, "router", 352, 4096, torch.float32, torch.bfloat16, [1, 3, 4, 12, 16])
    case(sk, "g_proj", 32, 4096, torch.float16, torch.float16, [1, 3, 12, 16])
    case(sk, "head", 64448, 4096, torch.float16, torch.float16, [1, 2, 3, 4])
    case_w8(sk, 64448, 4096, [1, 2, 3, 4, 16])
    for name, n, k in MTP_BLOCK:
        case_w8(sk, n, k, [1, 3, 12, 16], name=name, base="cublas")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
