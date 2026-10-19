# SPDX-License-Identifier: Apache-2.0
"""Measured per-layer sensitivity for the allocator (ported from sudoingx dsv41_recipe
scripts/quantize/p4_sensitivity.py): how much the final distribution moves (KL vs the clean teacher) per unit of
relative noise energy in MoE layer l's routed-expert output.

One layer-streamed pass carries the clean stream plus one copy per MoE layer l, branched off the clean stream at l's
input; copy l gets r + sigma * rms(r per token) * N(0, 1) in layer l's routed output only (r = the routed sum before the
shared expert is added), then runs clean to the end. alpha_l = mean KL(clean || copy_l) / sigma^2.

Cost: layer j runs (1 + number of copies branched so far) streams, so the 88 MoE layers cost ~88^2 / 2 layer-forwards
of the probe set (quadratic; sudoingx's 40 layers took ~27 min, expect ~4-5x). Resumable per layer: after every layer
all streams (clean + copies) go to <work>/state_LL (bf16 on disk via Streamer.save) and a restart continues from the
newest complete one; the noise of copy l is seeded by l, so a resumed run draws the same noise.

Calibration tokens only (never the KLD panel). Default probe: 16 sequences x 512 tokens, spread over the set.
  python3 p1_sensitivity.py /model /cal/cal.safetensors /out/alpha.json [--seqs 16 --len 512 --sigma 0.1]
                            [--work /out/sens_work]
"""
from __future__ import annotations

import argparse
import copy
import json
import os
import shutil
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from p_torch import kl_sums, load_seqs, logger, native_slots  # noqa: E402
from p_util import atomic_json, model_layers  # noqa: E402


def clone_states(states: list) -> list:
    """Independent copies of the per-sequence states (only resid changes between layers)."""
    out = []
    for st in states:
        c = copy.copy(st)
        c.resid = st.resid.clone()
        out.append(c)
    return out


def noisy_moe(s, sigma: float, seed: int):
    gen = torch.Generator(device="cuda").manual_seed(seed)

    def moe(x: torch.Tensor, layer) -> torch.Tensor:
        w, ids = s.route(layer, x)
        m = layer.moe
        _, r = native_slots(x, w, ids, m.gate_proj.weight, m.up_proj.weight, m.down_proj.weight)
        r = r.float()
        rms = r.square().mean(-1, keepdim=True).sqrt()
        r = r + sigma * rms * torch.randn(r.shape, generator=gen, device=r.device)
        return r.to(x.dtype).float() + layer.share_expert(x).float()
    return moe


def save_all(s, streams: dict[int, list], clean: list, work: str, layer: int) -> None:
    """Every stream after `layer` into work/state_LL/{clean,cLL}; then drop older states."""
    d = os.path.join(work, f"state_{layer:02d}")
    tmp = d + ".tmp"
    shutil.rmtree(tmp, ignore_errors=True)
    for key, states in [("clean", clean)] + [(f"c{k:02d}", v) for k, v in sorted(streams.items())]:
        s.states = states
        s.save(os.path.join(tmp, key))
    os.replace(tmp, d)
    for f in os.listdir(work):
        if f.startswith("state_") and f != os.path.basename(d) and not f.endswith(".tmp"):
            shutil.rmtree(os.path.join(work, f), ignore_errors=True)


def load_all(s, work: str, log) -> tuple[list, dict[int, list]] | None:
    done = sorted(f for f in os.listdir(work) if f.startswith("state_") and not f.endswith(".tmp")) \
        if os.path.isdir(work) else []
    if not done:
        return None
    d = os.path.join(work, done[-1])
    streams = {}
    for key in sorted(os.listdir(d)):
        s.load(os.path.join(d, key))
        states = clone_states(s.states)
        if key == "clean":
            clean, nxt = states, s.layer
        else:
            streams[int(key[1:])] = states
    s.states, s.layer = clean, nxt
    log(f"resumed from {d}: {len(streams)} copies, next layer {nxt}")
    return clean, streams


@torch.inference_mode()
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("model")
    ap.add_argument("cal")
    ap.add_argument("out", help="alpha.json")
    ap.add_argument("--seqs", type=int, default=16)
    ap.add_argument("--len", type=int, default=512)
    ap.add_argument("--sigma", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=4242)
    ap.add_argument("--work", default="", help="per-layer state checkpoints (default <out dir>/sens_work)")
    a = ap.parse_args()
    from step5q.stream import Streamer
    work = a.work or os.path.join(os.path.dirname(os.path.abspath(a.out)), "sens_work")
    os.makedirs(work, exist_ok=True)
    log = logger(os.path.join(work, "p1.log"))
    moe_layers, body_layers, _ = model_layers(json.load(open(os.path.join(a.model, "config.json"))))
    allseqs = load_seqs(a.cal)
    step = max(1, len(allseqs) // a.seqs)
    seqs = [x[: a.len] for x in allseqs[::step][: a.seqs]]          # spread over the (domain-ordered) set
    s = Streamer(a.model, seqs, "cuda")
    got = load_all(s, work, log)
    clean, streams = got if got else (s.states, {})
    t0 = time.time()
    for j in range(s.layer, len(body_layers)):
        layer = s.load_layer(j, experts=j in moe_layers)
        if j in moe_layers:
            streams[j] = clone_states(clean)                        # copy j branches at layer j's input
        for key in [None] + sorted(streams):
            s.states = clean if key is None else streams[key]
            s.layer = j
            s.run_layer(layer, moe=noisy_moe(s, a.sigma, a.seed + j) if key == j else None)
        s.states = clean
        del layer
        torch.cuda.empty_cache()
        save_all(s, streams, clean, work, j)
        s.states, s.layer = clean, j + 1
        log(f"layer {j:2d}: {len(streams) + 1} streams, {time.time() - t0:.0f}s")
    norm, head = s.load_head()                                 # official final norm + lm_head modules
    kl, n = kl_sums(s, clean, streams, norm, head)
    alpha = {str(l): kl[l] / n / a.sigma ** 2 for l in sorted(kl)}
    for l in sorted(kl):
        log(f"alpha[{l:2d}] = {alpha[str(l)]:.5f}  (KL {kl[l] / n:.6f})")
    atomic_json(a.out, {"sigma": a.sigma, "tokens": n, "seqs": len(seqs), "alpha": alpha}, indent=1)
    log(f"DONE {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
