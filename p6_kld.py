# SPDX-License-Identifier: Apache-2.0
"""Fidelity on the panel (ported from sudoingx dsv41_recipe scripts/quantize/p4_kld.py, Apache-2.0): KL(teacher ||
ours), top-1 agreement and NLL at every scored (assistant) position of every panel conversation, per domain. Teacher and
quantized model run through the same layer-streamed reference (step5q.stream), so the number measures the weights.

  --mode ref      teacher (FP8 experts dequantized to BF16, BF16 activations); writes <out>/ref.safetensors: per
                  scored position the teacher's top-128 token ids + log-probs and the NLL of the true next token
  --mode quant    routed experts from EXL3 (--experts pass-1 dir, or --assembled serving dir) and, with --body /
                  --assembled, the body linears from their EXL3 reconstruction; layers without EXL3 tensors stay native;
                  reads ref.safetensors, writes <out>/kld.json
  --mode serving  no model run: --serving-capture DIR holds the serving engine's log-probs (0xSero plugin
                  STEP5_CAPTURE_DIR: cap_NNNNN.npy fp16 [rows, vocab], step5.py:79-110) captured over the same panel
                  sequences in order (one prompt-logprobs request per sequence, STEP5_CAPTURE_MIN_ROWS=1,
                  STEP5_CAPTURE_STRIDE=1); rows must add up to the panel's token count; writes <out>/kld_serving.json
KL uses the teacher's top-128 support plus one tail bucket (exact when the top-128 holds all the mass).
Panel file: safetensors {"tokens", "offsets", "score"} (score 1 on assistant tokens); index: [{"domain": ...}, ...].

  python3 p6_kld.py /model /cal/panel.safetensors /cal/panel_index.json /out/kld --mode ref
  python3 p6_kld.py /model /cal/panel.safetensors /cal/panel_index.json /out/kld --mode quant --experts /p1 --body /dense
  python3 p6_kld.py /model /cal/panel.safetensors /cal/panel_index.json /out/kld --mode serving --serving-capture /cap
"""
from __future__ import annotations

import argparse
import glob
import json
import math
import os
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from p_torch import (ExpertCache, files_for, load_exl3, logger, make_quant_moe, overlay_body,  # noqa: E402
                     seq_logits)
from p_util import model_layers  # noqa: E402

TOPK = 128


def kl_vs_ref(lq: torch.Tensor, pid: torch.Tensor, plp: torch.Tensor) -> torch.Tensor:
    """Per position KL(P || Q) on P's top-k support + one tail bucket. lq: Q log-probs [n, vocab]."""
    P, Q = plp.exp(), lq.gather(-1, pid).exp()
    p_rest = (1 - P.sum(-1)).clamp_min(1e-12)
    q_rest = (1 - Q.sum(-1)).clamp_min(1e-12)
    return (P * (plp - Q.clamp_min(1e-30).log())).sum(-1) + p_rest * (p_rest.log() - q_rest.log())


def scored(score: torch.Tensor, a: int, b: int) -> torch.Tensor:
    """Positions t of sequence [a, b) whose next token t+1 is scored (position t predicts token t+1)."""
    return score[a:b][1:].bool().nonzero().squeeze(-1)


class Scorer:
    """Accumulates KL / top-1 / NLL per sequence against ref.safetensors (sequence order)."""

    def __init__(self, ref: dict, seqs: list[list[int]], index: list[dict]):
        self.ref, self.seqs, self.index, self.o = ref, seqs, index, 0
        self.kl, self.agree, self.nll, self.by = [], [], [], {}

    def add(self, i: int, lq: torch.Tensor, pos: torch.Tensor) -> None:
        k = int(self.ref["lens"][i])
        if k != pos.numel():
            raise SystemExit(f"sequence {i}: {pos.numel()} scored positions, ref has {k}")
        pid = self.ref["ids"][self.o: self.o + k].to(lq.device).long()
        plp = self.ref["logprobs"][self.o: self.o + k].to(lq.device).float()
        self.o += k
        kl = kl_vs_ref(lq, pid, plp)
        ag = (lq.argmax(-1) == pid[:, 0]).float()
        tgt = torch.tensor(self.seqs[i], device=lq.device)[1:][pos.to(lq.device)]
        nl = -lq.gather(-1, tgt[:, None]).squeeze(-1)
        self.kl.append(kl.cpu())
        self.agree.append(ag.cpu())
        self.nll.append(nl.float().cpu())
        dom = str(self.index[i].get("domain", "all")) if i < len(self.index) else "all"
        nref = self.ref["nll"][self.o - k: self.o]
        d = self.by.setdefault(dom, [0.0, 0.0, 0.0, 0.0, 0])
        d[0] += kl.sum().item()
        d[1] += ag.sum().item()
        d[2] += nl.sum().item()
        d[3] += nref.sum().item()
        d[4] += k

    def result(self) -> dict:
        kl, ag, nll = torch.cat(self.kl), torch.cat(self.agree), torch.cat(self.nll)
        nref = self.ref["nll"][: kl.numel()].float()
        res = {"positions": int(kl.numel()), "kld_mean": kl.mean().item(), "kld_p99": kl.quantile(0.99).item(),
               "top1_agreement": ag.mean().item(), "nll_teacher": nref.mean().item(), "nll_ours": nll.mean().item(),
               "by_domain": {k: {"kld": v[0] / v[4], "top1": v[1] / v[4], "nll_ours": v[2] / v[4],
                                 "nll_teacher": v[3] / v[4], "positions": v[4]} for k, v in sorted(self.by.items())}}
        res["ppl_teacher"], res["ppl_ours"] = math.exp(res["nll_teacher"]), math.exp(res["nll_ours"])
        return res


def serving_rows(cap_dir: str):
    """The capture files (memory-mapped) and their first global row."""
    import numpy as np
    files = sorted(glob.glob(os.path.join(cap_dir, "cap_*.npy")))
    if not files:
        raise SystemExit(f"no cap_*.npy in {cap_dir}")
    caps = [np.load(f, mmap_mode="r") for f in files]
    starts = [0]
    for c in caps:
        starts.append(starts[-1] + c.shape[0])
    return caps, starts


def rows_of(caps, starts, a: int, b: int) -> torch.Tensor:
    """Global capture rows [a, b) as fp32, read across file boundaries."""
    import numpy as np
    out = []
    for c, s0, s1 in zip(caps, starts, starts[1:]):
        lo, hi = max(a, s0), min(b, s1)
        if lo < hi:
            out.append(np.asarray(c[lo - s0: hi - s0]))
    return torch.from_numpy(np.concatenate(out, 0)).float()


def run_serving(a, seqs, offs, score, index, ref, log) -> dict:
    caps, starts = serving_rows(a.serving_capture)
    if starts[-1] != offs[-1]:
        raise SystemExit(f"capture holds {starts[-1]} rows, the panel {offs[-1]} tokens: capture one request per "
                         "sequence with STEP5_CAPTURE_MIN_ROWS=1 STEP5_CAPTURE_STRIDE=1, in panel order")
    sc = Scorer(ref, seqs, index)
    for i in range(len(seqs)):
        pos = scored(score, offs[i], offs[i + 1])
        rows = rows_of(caps, starts, offs[i], offs[i + 1])[:, : a.vocab]
        sc.add(i, torch.log_softmax(rows[pos], -1), pos)      # renormalise over the real vocabulary
    log(f"serving capture: {len(caps)} files, {starts[-1]:,} rows")
    return sc.result()


@torch.inference_mode()
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("model")
    ap.add_argument("panel")
    ap.add_argument("index")
    ap.add_argument("out")
    ap.add_argument("--mode", choices=["ref", "quant", "serving"], required=True)
    ap.add_argument("--experts", default="", help="pass-1 dir (layer_LL.part*.safetensors)")
    ap.add_argument("--body", default="", help="dense-pass dir (body_LL.safetensors)")
    ap.add_argument("--assembled", default="", help="serving dir (exl3/experts, exl3/body) instead of the two above")
    ap.add_argument("--serving-capture", default="")
    ap.add_argument("--vocab", type=int, default=128896, help="columns of the capture that are real vocabulary")
    ap.add_argument("--max-seqs", type=int, default=0)
    ap.add_argument("--max-layers", type=int, default=0, help="debug: stop after N layers")
    a = ap.parse_args()
    from safetensors.torch import load_file, save_file
    os.makedirs(a.out, exist_ok=True)
    log = logger(os.path.join(a.out, f"kld.{a.mode}.log"))
    d = load_file(a.panel)
    toks, offs, score = d["tokens"].tolist(), d["offsets"].tolist(), d["score"]
    n = len(offs) - 1 if not a.max_seqs else min(a.max_seqs, len(offs) - 1)
    offs = offs[: n + 1]
    seqs = [toks[offs[i]: offs[i + 1]] for i in range(n)]
    index = json.load(open(a.index))[:n]
    ref_path = os.path.join(a.out, "ref.safetensors")
    if a.mode == "serving":
        res = run_serving(a, seqs, offs, score, index, load_file(ref_path), log)
        res["serving_capture"] = a.serving_capture
        json.dump(res, open(os.path.join(a.out, "kld_serving.json"), "w"), indent=1)
        log(json.dumps(res, indent=1))
        return
    from step5q.stream import Streamer
    t0 = time.time()
    moe_layers, body_layers, n_exp = model_layers(json.load(open(os.path.join(a.model, "config.json"))))
    s = Streamer(a.model, seqs, "cuda")
    for l in body_layers[: a.max_layers or None]:
        t1 = time.time()
        ex, bd = files_for(l, a.experts, a.body, a.assembled) if a.mode == "quant" else ([], [])
        has_ex = bool(ex) and l in moe_layers
        layer = s.load_layer(l, experts=l in moe_layers and not has_ex)
        done = overlay_body(layer, l, load_exl3(bd, "cuda")) if bd else []
        cache = ExpertCache(l, load_exl3(ex, "cuda"), n_exp) if has_ex else None
        s.run_layer(layer, moe=make_quant_moe(s, cache) if a.mode == "quant" else None)
        log(f"layer {l:2d} {'EXL3' if has_ex else 'native'} experts" + (f", EXL3 body {','.join(done)}" if done else "")
            + f": {time.time() - t1:.0f}s")
        del layer, cache
        torch.cuda.empty_cache()
    norm, head = s.load_head()                                 # official final norm + lm_head modules
    if a.mode == "ref":
        ids, lps, nll, lens = [], [], [], []
        for i in range(n):
            pos = scored(score, offs[i], offs[i + 1]).cuda()
            lp = torch.log_softmax(seq_logits(s, i, norm, head)[pos].float(), -1)
            top = torch.topk(lp, TOPK, -1)
            ids.append(top.indices.int().cpu())
            lps.append(top.values.half().cpu())
            tgt = torch.tensor(seqs[i], device=lp.device)[1:][pos]
            nll.append((-lp.gather(-1, tgt[:, None]).squeeze(-1)).float().cpu())
            lens.append(pos.numel())
        save_file({"ids": torch.cat(ids), "logprobs": torch.cat(lps), "nll": torch.cat(nll),
                   "lens": torch.tensor(lens, dtype=torch.int64)}, ref_path)
        log(f"ref saved: {sum(lens):,} scored positions, mean NLL {torch.cat(nll).mean().item():.4f}, "
            f"{time.time() - t0:.0f}s")
        return
    sc = Scorer(load_file(ref_path), seqs, index)
    for i in range(n):
        pos = scored(score, offs[i], offs[i + 1]).cuda()
        sc.add(i, torch.log_softmax(seq_logits(s, i, norm, head)[pos].float(), -1), pos)
    res = sc.result()
    res.update(experts=a.experts, body=a.body, assembled=a.assembled, seconds=round(time.time() - t0))
    json.dump(res, open(os.path.join(a.out, "kld.json"), "w"), indent=1)
    log(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
