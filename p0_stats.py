# SPDX-License-Identifier: Apache-2.0
"""Pass 0 (ported from sudoingx dsv41_recipe scripts/quantize/p4_pass0.py): stream the teacher (FP8 experts dequantized
to BF16, BF16 activations) over the calibration set and record, per MoE layer, what the bit allocator needs. No weights
change.

Per routed expert e:  count (routed slots = usage), top1 (tokens whose first slot is e), sum_g, sum_g2 (routing weights),
                      out_energy = sum ||g * y_e(x)||^2 (energy e writes into the routed output),
                      tshare = sum_t ||g y_e(x_t)||^2 / ||r_t||^2 (per-token share of the routed output r_t; FINDINGS 13:
                      energy share misprices super experts, the allocator uses tshare),
                      out_max = max |y_e(x_t)| over every routed token (down_proj output before g: Su et al.'s super-
                      expert criterion, arXiv 2507.23279 Eq. 6, read by p2_allocate.py --super-max)
Per layer:            tokens, routed_out_energy, shared_out_energy, resid_{in,out}_energy, resid_{in,out}_max, seconds
--index (build.py's per-sequence index) adds the same per-expert arrays and a token count per domain under "domains";
domains are coarsened as sudoingx does (p_util.coarse_domain: code_debug -> code, long_context -> long, multilingual_*
-> multilingual, every exllamav3 standard-text row -> std), the names p0b_merge_stats.py weights.

The MoE is run explicitly (p_torch.native_slots: official Step5MoE semantics, weight after down_proj, fp32 slot-order
sum) so each slot's contribution is known; Streamer.run_layer then adds exactly the official output to the stream.

  python3 p0_stats.py /model /cal/cal.safetensors /out/stats [--layers 3-90] [--max-seqs 0] [--index cal_index.json]
                      [--stream-dir /out/stats/stream --keep-stream-layers 3,30,60,89]
--keep-stream-layers keeps stream_LL.safetensors (the stream entering those layers) for p1b_relerr.py.
Writes /out/stats/layer_LL.json (finished layers are not rewritten; with --stream-dir the residual stream is saved after
every layer and a restart continues from it, otherwise a restart recomputes the stream from layer 0).
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from p_torch import load_seqs, logger, mem_gib, native_slots, resume_stream, save_stream  # noqa: E402
from p_util import atomic_json, coarse_domain, model_layers, parse_layers  # noqa: E402

FIELDS = ("count", "top1", "sum_g", "sum_g2", "out_energy", "tshare")


class LayerStats:
    def __init__(self, n_experts: int, dev, domains: list[str]):
        self.E, self.domains = n_experts, domains
        z = lambda n: torch.zeros(n, dtype=torch.float64, device=dev)  # noqa: E731
        self.v = {k: z(n_experts) for k in FIELDS}
        self.d = {k: z(len(domains) * n_experts) for k in FIELDS}             # [domain * E + e]
        self.d_tokens = z(len(domains))
        self.out_max = z(n_experts)               # max |down_proj output| per expert, all routed tokens (Su et al.)
        self.scalars = {"tokens": 0, "routed_out_energy": 0.0, "shared_out_energy": 0.0,
                        "resid_in_energy": 0.0, "resid_out_energy": 0.0, "resid_in_max": 0.0, "resid_out_max": 0.0}

    def add_max(self, idx: torch.Tensor, val: torch.Tensor) -> None:
        self.out_max.scatter_reduce_(0, idx.flatten(), val.flatten().double(), "amax")

    def add(self, field: str, idx: torch.Tensor, val: torch.Tensor, dom: torch.Tensor | None) -> None:
        idx, val = idx.flatten(), val.flatten().double()
        self.v[field].index_add_(0, idx, val)
        if dom is not None:
            self.d[field].index_add_(0, dom.flatten() * self.E + idx, val)

    def to_json(self) -> dict:
        rec = {**self.scalars, **{k: v.tolist() for k, v in self.v.items()}, "out_max": self.out_max.tolist()}
        if self.domains:
            E = self.E
            rec["domains"] = {n: {"tokens": self.d_tokens[i].item(), **{k: v[i * E:(i + 1) * E].tolist()
                                                                         for k, v in self.d.items()}}
                              for i, n in enumerate(self.domains)}
        return rec


def domain_ids(seqs: list[list[int]], index: str, dev) -> tuple[list[str], torch.Tensor | None]:
    if not index:
        return [], None
    ix = json.load(open(index))[: len(seqs)]
    if len(ix) != len(seqs) or any(int(r.get("length", len(x))) != len(x) for r, x in zip(ix, seqs)):
        raise SystemExit(f"{index} does not describe these {len(seqs)} sequences")
    labels = [coarse_domain(r) for r in ix]
    names = sorted(set(labels))
    did = {d: i for i, d in enumerate(names)}
    return names, torch.cat([torch.full((len(x),), did[d], dtype=torch.long) for d, x in zip(labels, seqs)]).to(dev)


def resid_energy(s) -> float:
    return sum(st.resid.double().square().sum().item() for st in s.states)


def resid_max(s) -> float:
    return max(st.resid.abs().max().item() for st in s.states)


def stats_moe(s, st: LayerStats, dom_tok: torch.Tensor | None, off: list[int]):
    """moe(x, layer) that records the statistics and returns the official MoE output (fp32)."""
    def moe(x: torch.Tensor, layer) -> torch.Tensor:
        n = x.shape[0]
        d = dom_tok[off[0]: off[0] + n] if dom_tok is not None else None
        off[0] += n
        w, ids = s.route(layer, x)
        m = layer.moe
        slot, routed = native_slots(x, w, ids, m.gate_proj.weight, m.up_proj.weight, m.down_proj.weight)
        st.add_max(ids, slot.float().abs().amax(-1))                                  # [N, k]: max |y_e(x_t)|
        en = (slot.float() * w[..., None].float()).square().sum(-1)                  # [N, k]: ||g y_e(x_t)||^2
        rn = routed.float().square().sum(-1, keepdim=True).clamp_min(1e-30)          # [N, 1]: ||r_t||^2
        dk = d[:, None].expand_as(ids) if d is not None else None
        st.add("count", ids, torch.ones_like(w, dtype=torch.float64), dk)
        st.add("top1", ids[:, 0], torch.ones(n, dtype=torch.float64, device=x.device), d)
        st.add("sum_g", ids, w, dk)
        st.add("sum_g2", ids, w.double().square(), dk)
        st.add("out_energy", ids, en, dk)
        st.add("tshare", ids, en / rn, dk)
        if d is not None:
            st.d_tokens.index_add_(0, d, torch.ones(n, dtype=torch.float64, device=x.device))
        shared = layer.share_expert(x).float()
        st.scalars["routed_out_energy"] += routed.float().square().sum().item()
        st.scalars["shared_out_energy"] += shared.square().sum().item()
        st.scalars["tokens"] += n
        del slot, en
        return routed.float() + shared
    return moe


@torch.inference_mode()
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("model")
    ap.add_argument("cal")
    ap.add_argument("out")
    ap.add_argument("--layers", default="", help="MoE layers to record (default all MoE layers)")
    ap.add_argument("--max-seqs", type=int, default=0)
    ap.add_argument("--index", default="", help="build.py's per-sequence index: per-domain statistics too")
    ap.add_argument("--stream-dir", default="", help="save / resume the residual stream here")
    ap.add_argument("--keep-stream-layers", default="", help="keep the saved streams entering these layers (e.g. "
                    "3,30,60,89 for p1b_relerr.py) instead of pruning them; needs --stream-dir")
    ap.add_argument("--group-tokens", type=int, default=65536)
    a = ap.parse_args()
    from step5q.stream import Streamer
    os.makedirs(a.out, exist_ok=True)
    log = logger(os.path.join(a.out, "p0.log"))
    cfg_json = json.load(open(os.path.join(a.model, "config.json")))
    moe_layers, body_layers, n_exp = model_layers(cfg_json)
    want = parse_layers(a.layers, moe_layers)
    keep = set(parse_layers(a.keep_stream_layers, []))
    if keep and not a.stream_dir:
        raise SystemExit("--keep-stream-layers needs --stream-dir")
    seqs = load_seqs(a.cal, a.max_seqs)
    t0 = time.time()
    s = Streamer(a.model, seqs, "cuda", max_group_tokens=a.group_tokens)
    resume_stream(s, a.stream_dir, log)
    n_tok = sum(map(len, seqs))
    dom_names, dom_tok = domain_ids(seqs, a.index, s.device if hasattr(s, "device") else "cuda")
    log(f"p0: {len(seqs)} sequences, {n_tok:,} tokens, layers {min(want)}-{max(want)}"
        + (f", domains {dom_names}" if dom_names else ""))
    for l in range(s.layer, max(want) + 1):
        path = os.path.join(a.out, f"layer_{l:02d}.json")
        record = l in want and l in moe_layers and not os.path.exists(path)
        t1 = time.time()
        layer = s.load_layer(l, experts=l in moe_layers)
        t2 = time.time()
        if not record:
            s.run_layer(layer)
            log(f"layer {l:2d}: stream only, {time.time() - t1:.0f}s")
        else:
            st = LayerStats(n_exp, "cuda", dom_names)
            off = [0]
            st.scalars["resid_in_energy"], st.scalars["resid_in_max"] = resid_energy(s), resid_max(s)
            s.run_layer(layer, moe=stats_moe(s, st, dom_tok, off))
            torch.cuda.synchronize()
            if off[0] != n_tok:
                raise SystemExit(f"layer {l}: the moe saw {off[0]} tokens, the set has {n_tok}")
            st.scalars["resid_out_energy"], st.scalars["resid_out_max"] = resid_energy(s), resid_max(s)
            rec = st.to_json()
            rec.update(layer=l, n_experts=n_exp, load_s=round(t2 - t1, 1), run_s=round(time.time() - t2, 1))
            atomic_json(path, rec)
            c, oe = st.v["count"], st.v["out_energy"]
            log(f"layer {l:2d}: load {t2 - t1:4.0f}s run {time.time() - t2:5.0f}s | used {(c > 0).sum().item()}/{n_exp}"
                f" min/med/max count {c.min().item():.0f}/{c.median().item():.0f}/{c.max().item():.0f}"
                f" | top energy share {(oe.max() / oe.sum().clamp_min(1e-30)).item():.3f}"
                f" top tshare/token {(st.v['tshare'].max() / max(n_tok, 1)).item():.4f} | {mem_gib()}")
        del layer
        torch.cuda.empty_cache()
        save_stream(s, a.stream_dir, keep)
    log(f"DONE in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
