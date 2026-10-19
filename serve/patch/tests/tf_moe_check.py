#!/usr/bin/env python3
"""ST_EXL3_TF (TensorFold's grouped EXL3 experts, step5_spark/tf_exl3) against the plugin's exllamav3 exl3_moe buckets
and a float64 reference, on real Step-5 layers: expert-TP rank 0's slice on one GPU, no server.

    python3 tf_moe_check.py PATCH_DIR MODEL_DIR [LAYER ...]

Per layer, through Exl3MoEMethod.apply (fp32 inputs return the fp32 sums; bf16 is the serving dtype):
- TF's decode against this image's ExLlamaV3 reconstruct, one matrix per bit width (bit-exact);
- both paths' error against float64 from the decoded W_q (Hadamard, scales and the SwiGLU limit in float64) at two input
  scales, the larger one driving gate/up values past the limit;
- TF with bf16 inputs against fp16 inputs of the same values, rows alone against the same rows inside a window, and
  CUDA-graph replay against the eager call (all bit-exact);
- a call's time at decode sizes: a CUDA graph of 50 calls on different routings, median of 5 replays.
"""
from __future__ import annotations

import copy
import json
import math
import os
import sys

# the serving settings under test (the image's environment carries its own ST_EXL3_INTER)
os.environ.update({"STEP5_EXPERT_TP": "1", "ST_EXL3_INTER": "768", "ST_EXL3_TF": "1", "ST_EXL3_PREP": "1"})

import torch

HAD = 128
ROWS = 12                     # the accuracy window (4 streams x MTP k=2 + 1)
SCALES = (1.0, 6.0)
GRAPH_T = (1, 3, 4, 12)
TIMED_T = (1, 2, 3, 4, 6, 8, 12, 16, 32, 63)
ITERS = 50
LAYERS = (3, 30, 47, 90)


def plugin(patch_dir: str, model_dir: str):
    os.environ["STEP5_MODEL_DIR"] = model_dir
    import step5_spark
    step5_spark.__path__.insert(0, f"{patch_dir}/step5_spark")
    from step5_spark import exl3_quant as q
    from step5_spark.tf_exl3 import experts as tfx
    assert q.__file__.startswith(patch_dir) and q.TF and q.EXPERT_TP and q.INTER == 768, (q.__file__, q.TF, q.INTER)
    return q, tfx


def layer(q, L: int, N: int, top_k: int, dev):
    """(old-path method, TF method, linears, experts by K) for layer L, as process_weights_after_loading builds them."""
    lin, byk, stacked, _ = q.load_experts(L, 0, N, 0, dev, stack_scales=True)
    m = object.__new__(q.Exl3MoEMethod)
    m.layer_id, m.n_experts = L, N
    m.buckets = [q._Bucket(K, ids, lin, dev) for K, ids in sorted(byk.items())]
    m.maps = []
    for b in m.buckets:
        mp = torch.full((N,), b.E, dtype=torch.long)
        mp[torch.tensor(b.experts)] = torch.arange(b.E)
        m.maps.append(mp.to(dev))
    m.prep = q.BucketPrep(m.maps, [b.E for b in m.buckets])
    m.st_tables = None
    q._buffers(dev)
    q._prime(dev, m.buckets)
    m.tf = q.tf_layer(lin, list(range(N)), stacked, dev)
    m.tf_scratch = q.tf_scratch(m.tf, top_k, dev)
    old = copy.copy(m)
    old.tf = old.tf_scratch = None
    return old, m, lin, byk


def hadamard(dev) -> torch.Tensor:
    i = torch.arange(HAD)
    par = torch.tensor([bin(v).count("1") & 1 for v in range(HAD)])
    return (torch.where(par[i[:, None] & i[None, :]] == 1, -1.0, 1.0).double() / math.sqrt(HAD)).to(dev)


def rot(v: torch.Tensor, H: torch.Tensor) -> torch.Tensor:
    s = v.shape
    return (v.reshape(*s[:-1], s[-1] // HAD, HAD) @ H).reshape(s)


def reference(tfx, lin, cb: int, x: torch.Tensor, ids: torch.Tensor, w: torch.Tensor, limit: float):
    """float64 routed sum from the decoded W_q (ported from TensorFold's test_exl3_experts), Step-5's SwiGLU; also the
    fraction of activations the limit clamps."""
    H = hadamard(x.device)
    xd = x.double()
    out = torch.zeros_like(xd)
    clamped = total = 0

    def lin_ref(v, m):
        return rot(rot(v * m.suh.double(), H) @ tfx.dequant(m.trellis, cb).double(), H) * m.svh.double()

    for r in range(x.shape[0]):
        for j in range(ids.shape[1]):
            g_, u_, d_ = lin[int(ids[r, j])]
            gg, uu = lin_ref(xd[r], g_), lin_ref(xd[r], u_)
            sg = torch.nn.functional.silu(gg)
            clamped += int(((sg > limit) | (uu.abs() > limit)).sum())
            total += sg.numel()
            a = sg.clamp(max=limit) * uu.clamp(-limit, limit)
            out[r] += float(w[r, j]) * lin_ref(a, d_)
    return out, clamped / total


def errs(got: torch.Tensor, ref: torch.Tensor) -> tuple[float, float]:
    d = got.double() - ref
    return float(d.abs().max() / ref.abs().max()), float(d.norm() / ref.norm())


def routing(gen: torch.Generator, n: int, T: int, N: int, top_k: int, dev):
    ids = torch.stack([torch.stack([torch.randperm(N, generator=gen)[:top_k] for _ in range(T)]) for _ in range(n)])
    w = torch.rand((n, T, top_k), generator=gen) + 0.05
    w = w / w.sum(-1, keepdim=True)
    return ids.to(torch.int32).to(dev).contiguous(), w.float().to(dev).contiguous()


def inputs(gen: torch.Generator, T: int, scale: float, dev) -> torch.Tensor:
    """bf16 rows whose values are exact in fp16 as well (no values under fp16's normal range)."""
    x = torch.randn((T, 4096), generator=gen) * scale
    return torch.where(x.abs() < 2.0 ** -14, 0.0, x).to(torch.bfloat16).to(dev)


def timed(fn, iters: int = ITERS) -> float:
    fn(0)
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        for i in range(iters):
            fn(i)
    ts = []
    for _ in range(5):
        a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        a.record()
        g.replay()
        b.record()
        torch.cuda.synchronize()
        ts.append(a.elapsed_time(b) * 1000 / iters)
    return sorted(ts)[2]


def check_decode(q, tfx, lin, byk, cb: int) -> bool:
    ext = q._ext()
    ok = True
    for K, experts in sorted(byk.items()):
        for w in range(3):
            t = lin[experts[0]][w].trellis
            ours = tfx.dequant(t, cb)
            ref = torch.empty_like(ours)
            ext.reconstruct(ref, t, K, cb == tfx.CB_MCG, cb == tfx.CB_MUL1)
            torch.cuda.synchronize()
            ok &= torch.equal(ours.view(torch.int16), ref.view(torch.int16))
    return ok


def run_layer(q, tfx, L: int, N: int, top_k: int, dev, gen: torch.Generator) -> bool:
    old, tf, lin, byk = layer(q, L, N, top_k, dev)
    cb, limit = tf.tf.cb, q._pf.ACT_LIMIT
    ok = True
    dec = check_decode(q, tfx, lin, byk, cb)
    ok &= dec
    print(f"layer {L}: experts by K {{{', '.join(f'K{K}: {len(v)}' for K, v in sorted(byk.items()))}}}, codebook {cb}, "
          f"aligned16 {tf.tf.aligned16}; decode == ExLlamaV3 reconstruct {dec}", flush=True)
    for scale in SCALES:
        ids, w = routing(gen, 1, ROWS, N, top_k, dev)
        ids, w = ids[0], w[0]
        x = inputs(gen, ROWS, scale, dev)
        x32 = x.float()
        ref, frac = reference(tfx, lin, cb, x32, ids, w, limit)
        o32 = old.apply(None, x32, w, ids)
        e_old = errs(o32, ref)
        # exl3_moe rows alone and inside a 48-row window (the KL capture's prefill chunks) against the 12-row window
        ids48, w48 = routing(gen, 1, 48, N, top_k, dev)
        ids48[0, :ROWS], w48[0, :ROWS] = ids, w
        x48 = torch.cat([x32, inputs(gen, 48 - ROWS, scale, dev).float()])
        o48 = old.apply(None, x48, w48[0], ids48[0])[:ROWS]
        o1 = torch.cat([old.apply(None, x32[r:r + 1], w[r:r + 1], ids[r:r + 1]) for r in range(ROWS)])
        spread = max(float((o - o32).abs().max()) for o in (o1, o48)) / float(o32.abs().max())
        t32 = tf.apply(None, x32, w, ids)
        e_tf = errs(t32, ref)
        bf_same = torch.equal(tf.apply(None, x, w, ids), t32.to(torch.bfloat16))
        alone = all(torch.equal(tf.apply(None, x32[r:r + 1], w[r:r + 1], ids[r:r + 1]), t32[r:r + 1])
                    for r in (0, ROWS - 1))
        good = e_tf[1] < 1e-2 and e_tf[1] <= 1.5 * e_old[1] + 1e-6 and bf_same and alone
        ok &= good
        print(f"  x scale {scale}: clamped {frac:.4%} | rel max / rms err vs float64: old {e_old[0]:.2e} / "
              f"{e_old[1]:.2e}, TF {e_tf[0]:.2e} / {e_tf[1]:.2e} | TF bf16 in == fp16 in {bf_same}, "
              f"rows alone == in window {alone} | old rows alone / in 48 vs in 12: rel max diff {spread:.2e}",
              flush=True)
    for T in GRAPH_T:
        ids, w = routing(gen, 1, T, N, top_k, dev)
        x = inputs(gen, T, 1.0, dev)
        eager = tf.apply(None, x, w[0], ids[0]).clone()
        torch.cuda.synchronize()
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            out = tf.apply(None, x, w[0], ids[0])
        same = True
        for _ in range(3):
            g.replay()
            torch.cuda.synchronize()
            same &= torch.equal(out, eager)
        ok &= same
        print(f"  T={T:2d}: TF graph replay == eager {same}", flush=True)
    for T in TIMED_T:
        ids, w = routing(gen, ITERS, T, N, top_k, dev)
        x = inputs(gen, T, 1.0, dev)
        t_old = timed(lambda i: old.apply(None, x, w[i], ids[i]))
        t_tf = timed(lambda i: tf.apply(None, x, w[i], ids[i]))
        nbytes = sum(tf.tf.nbytes_read(sorted(set(ids[i].reshape(-1).tolist()))) for i in range(ITERS)) / ITERS
        print(f"  T={T:2d}: old {t_old:7.1f} us ({nbytes / t_old / 1e3:5.1f} GB/s), TF {t_tf:7.1f} us "
              f"({nbytes / t_tf / 1e3:5.1f} GB/s), TF/old {t_tf / t_old:.3f}", flush=True)
    del old, tf, lin
    torch.cuda.empty_cache()
    return ok


def main() -> int:
    patch_dir, model_dir = sys.argv[1], sys.argv[2]
    layers = [int(a) for a in sys.argv[3:]] or list(LAYERS)
    q, tfx = plugin(patch_dir, model_dir)
    cfg = json.load(open(f"{model_dir}/config.json"))
    text = cfg.get("text_config", cfg)
    N = int(text.get("moe_num_experts") or text.get("num_experts") or 352)
    top_k = int(text.get("moe_top_k") or text.get("num_experts_per_tok") or 8)
    dev = torch.device("cuda", torch.cuda.current_device())     # the plugin keys its buffers by device index
    print(f"experts {N}, top-k {top_k}, limit {q._pf.ACT_LIMIT}, TF decode {tfx.DECODE} pdl {tfx.DECODE_PDL} "
          f"ready {tfx.DECODE_READY}", flush=True)
    gen = torch.Generator().manual_seed(0)
    ok = all([run_layer(q, tfx, L, N, top_k, dev, gen) for L in layers])
    print("PASS" if ok else "FAIL", flush=True)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
