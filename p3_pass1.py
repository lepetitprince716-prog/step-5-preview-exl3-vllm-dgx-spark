# SPDX-License-Identifier: Apache-2.0
"""Pass 1: sequential EXL3 encoding of Step 5's routed experts, layer by layer (ported from sudoingx dsv41_recipe
scripts/quantize/p4_pass1.py, Apache-2.0).

Per MoE layer l (the stream entering l already went through the EXL3 experts of every layer < l):
  1. capture: Streamer.run_layer with a moe that records the MoE input x [N, 4096] (bf16, every calibration token) and
     returns 0, so the stream holds resid + attn afterwards; routing (top-8, weights) from Streamer.route.
  2. per expert e of this part (--part i/n, experts e % n == i): routing-weighted Hessians from e's routed tokens
       H_gu = sum (g x)(g x)^T + tau * mean(g^2) * H_all / N           (all-token prior: cold experts stay well posed)
       H_d  = sum (g a)(g a)^T, a = clamped_swiglu(x W1^T, x W3^T) of the native (FP8 -> BF16) expert
              + tau * prior from 1024 sampled tokens at g = sqrt(mean g^2)
     (Step 5 applies g after down_proj: the expert output is g * W2 a, so both Hessians weight a token by g^2.)
     step5q.exl3.encode_expert at the plan's K (same K for gate / up / down), reconstruct, and the routed-output error
     against the native expert (both through expert_forward_dense) per expert.
  3. part file /out/layer_LL.partI.safetensors (0xSero keys model.layers.L.moe.experts.E.{gate,up,down}_proj.*) + .json
     metrics (written after the tensors: the marker the peer waits for); wait for the other parts (exchange_sync.sh
     pushes them over the ConnectX link); the stream advances through ALL parts' EXL3 experts: resid += routed (EXL3,
     slot-order fp32 sum, rounded to BF16) + native shared expert, the same on every box.
  Dense layers (0-2, 91) run natively. After every layer the stream is saved (--stream-dir, Streamer.save) and a
  restart resumes from the newest saved stream entering a layer <= the first layer this part has not encoded yet
  (no layer_LL.partI.json in /out). --keep-stream-at 87 keeps the stream after layer 87 (stream_L88.safetensors), so a
  later `--layers 88-90 --reuse ... --old-plan ...` into a fresh /out with --stream-dir pointing at a copy of that file
  starts at layer 88. Layers below the start with no saved stream are replayed from their part files.

Partial re-encode (--reuse DIR --old-plan PLAN): an expert whose K is unchanged takes its EXL3 tensors from the assembled
checkpoint DIR (exl3/experts/LLL.safetensors) and only its error is measured; changed experts are dealt over the parts
by K (highest first), the rest round-robin; part files hold only the re-encoded experts.

Memory: one layer's x (N x 8 KiB), this part's native experts (bf16), one expert's Hessians at a time; the stream
advance holds every expert's reconstruction (352 x 3 x 12 MiB fp16) instead.

  python3 p3_pass1.py /model /cal/cal.safetensors /plan/plan.json /out [--layers 3-90] [--part 0/2] [--tau 512]
"""
from __future__ import annotations

import argparse
import functools
import json
import os
import sys
import time

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from p_torch import (ExpertCache, dims, load_exl3, load_seqs, logger, make_quant_moe, mem_gib, quant_routed,  # noqa: E402
                     resume_stream, save_stream, swiglu)
from p_util import (W_TO_PROJ, atomic_json, drop_page_cache, model_layers, parse_layers, parse_part,  # noqa: E402
                    plan_k, sero_expert_key)

CHUNK = int(os.environ.get("STEP5_EXPERT_CHUNK", "32768"))   # routed rows per Hessian / error chunk (hot experts)
ROUTE_CHUNK = 65536


class Reuse:
    """EXL3 experts of an assembled checkpoint (p5_assemble.py layout) and the plan they were encoded with."""

    def __init__(self, ckpt: str, old_plan: str):
        self.ckpt, self.old = ckpt, json.load(open(old_plan))

    def changed(self, plan: dict, l: int) -> list[int]:
        o, n = plan_k(self.old, l), plan_k(plan, l)
        return [e for e in range(len(n)) if n[e] != o[e]]

    def tensors(self, l: int, experts: list[int]) -> dict[str, torch.Tensor]:
        from safetensors import safe_open
        p = os.path.join(self.ckpt, "exl3", "experts", f"L{l:02d}.safetensors")
        want = set(experts)
        with safe_open(p, "pt", device="cuda") as f:      # keys model.layers.L.moe.experts.E.proj.piece
            out = {k: f.get_tensor(k) for k in f.keys() if int(k.split(".")[5]) in want}
        drop_page_cache(p)
        return out


def assign_parts(ks: list[int], changed: list[int], nparts: int) -> dict[int, int]:
    """Expert -> part for a partial re-encode: changed experts dealt by K (highest first), then the rest round-robin."""
    order = sorted(changed, key=lambda e: (-ks[e], e))
    part_of = {e: i % nparts for i, e in enumerate(order)}
    rest = [e for e in range(len(ks)) if e not in part_of]
    part_of.update({e: i % nparts for i, e in enumerate(rest)})
    return part_of


def wait_parts(out_dir: str, l: int, part: int, nparts: int) -> None:
    """Block until every other part published layer l: its metrics json (written after its tensor file), then the
    tensor files the json lists (exchange_sync.sh pushes both; rsync renames into place, so present = complete)."""
    for p in range(nparts):
        if p == part:
            continue
        mpath = os.path.join(out_dir, f"layer_{l:02d}.part{p}.json")
        while not os.path.exists(mpath):
            time.sleep(2)
        for name in json.load(open(mpath)).get("files", []):
            while not os.path.exists(os.path.join(out_dir, name)):
                time.sleep(2)


def capture(s, layer) -> torch.Tensor:
    """Run the layer's attention half for every sequence; return the MoE input x [N, H] bf16 in stream order. The moe
    returns 0, so afterwards resid = resid + attn and s.layer has advanced (advance() adds the MoE output)."""
    xs = []

    def moe(x: torch.Tensor, _layer) -> torch.Tensor:
        xs.append(x.clone())
        return torch.zeros(x.shape, dtype=torch.float32, device=x.device)
    s.run_layer(layer, moe=moe)
    return torch.cat(xs)


def advance(s, flat: torch.Tensor, fn) -> None:
    """resid += fn(x) for every token, in stream order (Streamer phase C is resid += y)."""
    si, so = 0, 0
    for a in range(0, flat.shape[0], ROUTE_CHUNK):
        y = fn(flat[a: a + ROUTE_CHUNK])
        r = 0
        while r < y.shape[0]:
            st = s.states[si]
            n = min(st.resid.shape[0] - so, y.shape[0] - r)
            st.resid[so: so + n] += y[r: r + n]
            r, so = r + n, so + n
            if so == st.resid.shape[0]:
                si, so = si + 1, 0
    if si != len(s.states):
        raise SystemExit(f"advance: {si} of {len(s.states)} sequences covered")


def chunked_err(flat, Wr, tok, slot, wq, wn) -> tuple[float, float]:
    """(sum |yq - yn|^2, sum |yn|^2) of one expert over its routed tokens."""
    from step5q.exl3 import expert_forward_dense
    err = energy = 0.0
    for c in range(0, tok.numel(), CHUNK):
        t_c, s_c = tok[c: c + CHUNK], slot[c: c + CHUNK]
        x_c, g_c = flat[t_c], Wr[t_c, s_c]
        yq = expert_forward_dense(x_c, g_c, *wq, limit=7.0)
        yn = expert_forward_dense(x_c, g_c, *wn, limit=7.0)
        err += (yq - yn).square().sum().item()
        energy += yn.square().sum().item()
    return err, energy


def route_all(s, layer, flat: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Routing of every captured token: (weights fp32 [N, 8], ids [N, 8])."""
    W_, I_ = [], []
    for c in range(0, flat.shape[0], ROUTE_CHUNK):
        w, i = s.route(layer, flat[c: c + ROUTE_CHUNK])
        W_.append(w.float())
        I_.append(i)
    return torch.cat(W_), torch.cat(I_)


def layer_prior(flat: torch.Tensor, Wr: torch.Tensor, l: int) -> tuple[torch.Tensor, float, torch.Tensor]:
    """The all-token prior of expert_hessians: (H_all = sum x^T x fp32, mean g^2, 1024 sampled tokens seeded by l)."""
    H_all = torch.zeros(flat.shape[1], flat.shape[1], dtype=torch.float32, device=flat.device)
    for c in range(0, flat.shape[0], CHUNK):
        xc = flat[c: c + CHUNK].float()
        H_all.addmm_(xc.t(), xc)
    gs = torch.Generator(device="cpu").manual_seed(1000 + l)
    xs = flat[torch.randperm(flat.shape[0], generator=gs)[:1024].to(flat.device)]
    return H_all, Wr.square().mean().item(), xs


def expert_hessians(flat, Wr, tok, slot, W1, W3, H_all, N, xs, mean_g2, tau) -> tuple[torch.Tensor, torch.Tensor]:
    H_gu = torch.zeros(flat.shape[1], flat.shape[1], dtype=torch.float32, device=flat.device)
    H_d = torch.zeros(W1.shape[0], W1.shape[0], dtype=torch.float32, device=flat.device)
    for c in range(0, tok.numel(), CHUNK):
        t_c, s_c = tok[c: c + CHUNK], slot[c: c + CHUNK]
        x_c, g_c = flat[t_c], Wr[t_c, s_c].float()[:, None]
        xg = x_c.float() * g_c
        H_gu.addmm_(xg.t(), xg)
        ag = swiglu(F.linear(x_c, W1), F.linear(x_c, W3)).float() * g_c
        H_d.addmm_(ag.t(), ag)
        del xg, ag
    H_gu.add_(H_all, alpha=tau * mean_g2 / N)
    ap = swiglu(F.linear(xs, W1), F.linear(xs, W3)).float() * mean_g2 ** 0.5
    H_d.add_(ap.t() @ ap, alpha=tau / xs.shape[0])
    return H_gu, H_d


@torch.inference_mode()
def quantize_layer(s, layer, l: int, plan: dict, part: int, nparts: int, a, log, reuse: Reuse | None) -> None:
    from step5q.exl3 import encode_expert, h_data, reconstruct
    from step5q.refmodel import expert_weights
    t0 = time.time()
    torch.cuda.reset_peak_memory_stats()
    flat = capture(s, layer)
    N = flat.shape[0]
    Wr, Ir = route_all(s, layer, flat)
    ks = plan_k(plan, l)
    n_exp = len(ks)
    tname = f"layer_{l:02d}.part{part}.safetensors"
    mpath = os.path.join(a.out, f"layer_{l:02d}.part{part}.json")
    t_a = time.time()
    changed = None
    if reuse is None:
        mine = [e for e in range(n_exp) if e % nparts == part]
    else:
        changed = set(reuse.changed(plan, l))
        part_of = assign_parts(ks, sorted(changed), nparts)
        mine = [e for e in range(n_exp) if part_of[e] == part]
    if os.path.exists(mpath):
        log(f"layer {l:2d}: part {part} already encoded, advancing")
    else:
        H_all, mean_g2, xs = layer_prior(flat, Wr, l)
        w1, w3, w2 = expert_weights(s.ckpt, l, "cuda")                  # bf16 [352, out, in]
        idx = torch.tensor(mine, device=w1.device)
        w1, w3, w2 = w1[idx].contiguous(), w3[idx].contiguous(), w2[idx].contiguous()   # this part's only
        torch.cuda.empty_cache()
        old = reuse.tensors(l, [e for e in mine if e not in changed]) if reuse is not None else {}
        tensors, metrics = {}, {"layer": l, "part": part, "experts": {}}
        t_h = t_enc = t_fwd = 0.0
        work = os.path.join(a.out, "work")
        os.makedirs(work, exist_ok=True)
        for j, e in enumerate(mine):
            t1 = time.time()
            tok, slot = (Ir == e).nonzero(as_tuple=True)
            W1, W3, W2 = w1[j], w3[j], w2[j]
            native = (W1.half(), W3.half(), W2.half())
            rec = {"n": tok.numel(), "K": ks[e]}
            if changed is not None and e not in changed:
                pre = f"model.layers.{l}.moe.experts.{e}."
                tq = {p: {k[len(pre) + len(p) + 1:]: v for k, v in old.items() if k.startswith(pre + p + ".")}
                      for p in ("gate_proj", "up_proj", "down_proj")}
                rec["reused"] = True
            else:
                H_gu, H_d = expert_hessians(flat, Wr, tok, slot, W1, W3, H_all, N, xs, mean_g2, a.tau)
                t2 = time.time()
                t_h += t2 - t1
                tens, perr = encode_expert(W1.float(), W3.float(), W2.float(), h_data(H_gu, tok.numel() + a.tau,
                                           f"L{l}.e{e}.gu"), h_data(H_d, tok.numel() + a.tau, f"L{l}.e{e}.d"),
                                           ks[e], l * 1000 + e, work)
                t_enc += time.time() - t2
                tq = {W_TO_PROJ[w]: v for w, v in tens.items()}
                for p, pieces in tq.items():
                    for piece, v in pieces.items():
                        tensors[sero_expert_key(l, e, p, piece)] = v.contiguous().cpu()
                rec["proxy"] = perr
                del H_gu, H_d, tens
            t3 = time.time()
            wq = tuple(reconstruct(tq[p], *dims(tq[p])) for p in ("gate_proj", "up_proj", "down_proj"))
            rec["out_err"], rec["out_energy"] = chunked_err(flat, Wr, tok, slot, wq, native)
            t_fwd += time.time() - t3
            metrics["experts"][str(e)] = rec
            del wq, native, tq
            if (j + 1) % 32 == 0:
                ee = sum(m["out_err"] for m in metrics["experts"].values())
                en = sum(m["out_energy"] for m in metrics["experts"].values())
                log(f"  layer {l:2d}: {j + 1}/{len(mine)} experts, {time.time() - t0:.0f}s, "
                    f"routed rel err so far {(ee / max(en, 1e-30)) ** 0.5:.4f}, {mem_gib()}")
        files, nbytes = [], 0
        if tensors or reuse is None:
            from safetensors.torch import save_file
            tpath = os.path.join(a.out, tname)
            save_file(tensors, tpath + ".tmp", metadata={"format": "pt", "encoder": "exllamav3 1.6.0 mul1"})
            os.chmod(tpath + ".tmp", 0o644)
            os.replace(tpath + ".tmp", tpath)
            drop_page_cache(tpath)
            files, nbytes = [tname], os.path.getsize(tpath)
        mem_enc = torch.cuda.max_memory_allocated() / 2 ** 30
        del w1, w3, w2, H_all, xs, tensors, old
        torch.cuda.empty_cache()
        te = sum(m["out_err"] for m in metrics["experts"].values())
        tn = sum(m["out_energy"] for m in metrics["experts"].values())
        metrics.update(tokens=N, files=files, bytes=nbytes, routed_out_rel_err=(te / max(tn, 1e-30)) ** 0.5,
                       max_mem_gib_encode=round(mem_enc, 1),
                       seconds={"capture": round(t_a - t0, 1), "hessians": round(t_h, 1), "encode": round(t_enc, 1),
                                "err_fwd": round(t_fwd, 1), "total_encode": round(time.time() - t0, 1)})
        atomic_json(mpath, metrics)
        log(f"layer {l:2d} part {part}/{nparts}: {N:,} tokens, {len(mine)} experts, {nbytes / 2 ** 30:.2f} GiB, "
            f"routed rel err {metrics['routed_out_rel_err']:.4f}, {metrics['seconds']}, max mem {mem_enc:.1f} GiB")
    t_w = time.time()
    if nparts > 1:
        wait_parts(a.out, l, part, nparts)
    t_s = time.time()
    cache = layer_cache(l, a.out, n_exp, reuse)
    advance(s, flat, functools.partial(stream_moe, s, layer, cache=cache))
    del flat, Wr, Ir, cache
    torch.cuda.empty_cache()
    log(f"layer {l:2d}: wait {t_s - t_w:.0f}s, advance {time.time() - t_s:.0f}s, total {time.time() - t0:.0f}s, "
        f"{mem_gib()}")


def stream_moe(s, layer, x: torch.Tensor, cache: ExpertCache) -> torch.Tensor:
    w, ids = s.route(layer, x)
    return quant_routed(x, w, ids, cache).to(x.dtype).float() + layer.share_expert(x).float()


def layer_cache(l: int, out_dir: str, n_exp: int, reuse: Reuse | None) -> ExpertCache:
    """Every expert of layer l from the part files (all parts), the reuse checkpoint filling the rest."""
    import glob
    files = sorted(glob.glob(os.path.join(out_dir, f"layer_{l:02d}.part*.safetensors")))
    tens = load_exl3(files, "cuda")
    if reuse is not None:
        have = {int(k.split(".")[5]) for k in tens}
        missing = [e for e in range(n_exp) if e not in have]
        if missing:
            tens.update(reuse.tensors(l, missing))
    cache = ExpertCache(l, tens, n_exp)
    del tens
    return cache


@torch.inference_mode()
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("model")
    ap.add_argument("cal")
    ap.add_argument("plan")
    ap.add_argument("out")
    ap.add_argument("--layers", default="", help="MoE layers to encode, lo-hi (default all MoE layers)")
    ap.add_argument("--part", default="0/1")
    ap.add_argument("--tau", type=float, default=512.0)
    ap.add_argument("--max-seqs", type=int, default=0)
    ap.add_argument("--reuse", default="", help="assembled checkpoint whose experts are kept where K is unchanged")
    ap.add_argument("--old-plan", default="", help="the plan --reuse was encoded with")
    ap.add_argument("--stream-dir", default="", help="default <out>/stream.partI ('none': never save)")
    ap.add_argument("--keep-stream-at", type=int, default=-1, help="keep the stream saved after this layer (e.g. 87: "
                    "a later --reuse --old-plan re-encode of 88-90 resumes there instead of replaying from layer 3)")
    ap.add_argument("--group-tokens", type=int, default=65536)
    a = ap.parse_args()
    from step5q.stream import Streamer
    part, nparts = parse_part(a.part)
    if bool(a.reuse) != bool(a.old_plan):
        raise SystemExit("--reuse and --old-plan go together")
    os.makedirs(a.out, exist_ok=True)
    log = logger(os.path.join(a.out, f"pass1.part{part}.log"))
    plan = json.load(open(a.plan))
    moe_layers, _, _ = model_layers(json.load(open(os.path.join(a.model, "config.json"))))
    todo = parse_layers(a.layers, moe_layers)
    lo, hi = min(todo), max(todo)
    stream_dir = "" if a.stream_dir == "none" else (a.stream_dir or os.path.join(a.out, f"stream.part{part}"))
    reuse = Reuse(a.reuse, a.old_plan) if a.reuse else None
    seqs = load_seqs(a.cal, a.max_seqs)
    s = Streamer(a.model, seqs, "cuda", max_group_tokens=a.group_tokens)
    done = lambda l: os.path.exists(os.path.join(a.out, f"layer_{l:02d}.part{part}.json"))  # noqa: E731
    first = next((l for l in todo if not done(l)), hi + 1)
    resume_stream(s, stream_dir, log, at=first)       # never past the first layer this part still has to encode
    log(f"pass 1 part {part}/{nparts}: {len(seqs)} sequences, {sum(map(len, seqs)):,} tokens, layers {lo}-{hi}, "
        f"stream at layer {s.layer}" + (f", re-encoding K changes vs {a.old_plan}" if reuse else ""))
    for l in range(s.layer, hi + 1):
        t0 = time.time()
        layer = s.load_layer(l, experts=False)
        if l not in moe_layers:
            s.run_layer(layer)
            log(f"layer {l:2d}: dense, native, {time.time() - t0:.0f}s")
        elif l < lo:                                    # resume without a saved stream: replay from the part files
            s.run_layer(layer, moe=make_quant_moe(s, layer_cache(l, a.out, len(plan_k(plan, l)), reuse)))
            log(f"layer {l:2d}: replayed from part files, {time.time() - t0:.0f}s")
        else:
            quantize_layer(s, layer, l, plan, part, nparts, a, log, reuse)
        del layer
        torch.cuda.empty_cache()
        save_stream(s, stream_dir, {a.keep_stream_at + 1} if a.keep_stream_at >= 0 else set())
    log("DONE")


if __name__ == "__main__":
    main()
