#!/usr/bin/env python3
"""Serving-engine fidelity from kld_capture.py captures, inside the serving image (numpy, torch).

    python3 kld_eval.py pack CAPDIR OUT STEP5_DIR  rows placed by position: OUT/cap_NNNNN.npy fp16 [tokens, vocab] a
                                                   sequence
    python3 kld_eval.py pair STEP5_DIR A B OUT     KL / top-1 / NLL against the teacher (p6_kld's Scorer) for packed A
                                                   and B, and the paired per-position KL difference B - A

pack: a sequence's request wrote, per engine step, a one-row sampling call and then the step's prompt-logprob rows;
kld_capture.placement (STEP5_DIR/serve/kld_capture.py) checks that structure against the server's own prompt
log-probs and places every block; the final sampling row becomes position n-1 (never scored).
pair: p6_kld.py's kl_vs_ref, scored and Scorer from STEP5_DIR, the panel at STEP5_DIR/cal and the teacher reference at
STEP5_DIR/runs/p6/ref/ref.safetensors, on the sequences the captures hold.
"""
from __future__ import annotations

import json
import math
import os
import sys

import numpy as np
import torch

VOCAB = 128896                      # columns of the capture that are real vocabulary (p6_kld --vocab)


def manifest(capdir: str) -> tuple[dict, list[dict]]:
    lines = [json.loads(x) for x in open(os.path.join(capdir, "manifest.jsonl"))]
    return lines[0], lines[1:]


def pack_one(kc, capdir: str, rec: dict, toks: list[int]) -> np.ndarray:
    blocks, final, err = kc.placement(capdir, rec, toks)
    if err:
        raise SystemExit(f"sequence {rec['seq']}: {err}")
    last = np.load(os.path.join(capdir, final), mmap_mode="r")
    out = np.empty((rec["tokens"], last.shape[1]), dtype=np.float16)
    for name, start in blocks:
        rows = np.load(os.path.join(capdir, name), mmap_mode="r")
        out[start:start + rows.shape[0]] = rows
    out[-1] = last[0]
    return out


def cmd_pack(capdir: str, out_dir: str, step5: str) -> int:
    sys.path.insert(0, step5)
    sys.path.insert(0, os.path.join(step5, "serve"))
    import kld_capture as kc
    from safetensors.torch import load_file
    head, recs = manifest(capdir)
    d = load_file(os.path.join(step5, "cal", "panel.safetensors"))
    toks, offs = d["tokens"].tolist(), d["offsets"].tolist()
    os.makedirs(out_dir, exist_ok=False)
    for k, rec in enumerate(recs):
        i = rec["seq"]
        seq = toks[offs[i]:offs[i + 1]]
        if len(seq) != rec["tokens"]:
            raise SystemExit(f"sequence {i}: {rec['tokens']} tokens captured, the panel has {len(seq)}")
        np.save(os.path.join(out_dir, f"cap_{k:05d}.npy"), pack_one(kc, capdir, rec, seq))
        print(f"sequence {i}: {rec['tokens']} rows from {len(rec['files'])} files", flush=True)
    json.dump({"sequences": [r["seq"] for r in recs], "from": capdir}, open(os.path.join(out_dir, "packed.json"), "w"))
    print(f"packed {len(recs)} sequences ({len(head['ignored'])} earlier capture files ignored)", flush=True)
    return 0


def scorer_for(p6, step5: str, packed: str):
    """p6_kld's Scorer over the packed sequences: the reference sliced to them, in their order."""
    from safetensors.torch import load_file
    seqs_idx = json.load(open(os.path.join(packed, "packed.json")))["sequences"]
    d = load_file(os.path.join(step5, "cal", "panel.safetensors"))
    toks, offs, score = d["tokens"].tolist(), d["offsets"].tolist(), d["score"]
    index = json.load(open(os.path.join(step5, "cal", "panel_index.json")))
    ref = load_file(os.path.join(step5, "runs", "p6", "ref", "ref.safetensors"))
    lens = ref["lens"].tolist()
    starts = [0]
    for x in lens:
        starts.append(starts[-1] + x)
    cut = lambda t: torch.cat([t[starts[i]:starts[i + 1]] for i in seqs_idx])
    sub = {"ids": cut(ref["ids"]), "logprobs": cut(ref["logprobs"]), "nll": cut(ref["nll"]),
           "lens": torch.tensor([lens[i] for i in seqs_idx], dtype=torch.int64)}
    seqs = [toks[offs[i]:offs[i + 1]] for i in seqs_idx]
    sc = p6.Scorer(sub, seqs, [index[i] for i in seqs_idx])
    argmax = []
    for k, i in enumerate(seqs_idx):
        pos = p6.scored(score, offs[i], offs[i + 1])
        rows = torch.from_numpy(np.load(os.path.join(packed, f"cap_{k:05d}.npy"))).float()[:, :VOCAB]
        lq = torch.log_softmax(rows[pos], -1)
        sc.add(k, lq, pos)
        argmax.append(lq.argmax(-1))
    return sc, torch.cat(argmax), seqs_idx


def cmd_pair(step5: str, a: str, b: str, out: str) -> int:
    sys.path.insert(0, step5)
    import p6_kld as p6
    sa, ama, ia = scorer_for(p6, step5, a)
    sb, amb, ib = scorer_for(p6, step5, b)
    if ia != ib:
        raise SystemExit("the two captures hold different sequences")
    ka, kb = torch.cat(sa.kl).double(), torch.cat(sb.kl).double()
    d = kb - ka
    # positions inside a sequence are correlated: the standard error of the position-weighted mean difference with the
    # sequences as clusters, and the mean of the per-sequence mean differences
    per = [(y.double() - x.double()) for x, y in zip(sa.kl, sb.kl)]
    S, N, D = len(per), d.numel(), d.mean()
    se_cluster = math.sqrt(S / (S - 1) * sum(float((p - D).sum()) ** 2 for p in per)) / N
    seq_means = torch.tensor([float(p.mean()) for p in per], dtype=torch.float64)
    res = {"a": a, "b": b, "sequences": ia, "a_result": sa.result(), "b_result": sb.result(),
           "kl_diff_mean": D.item(), "kl_diff_se_positions": (d.std() / math.sqrt(N)).item(),
           "kl_diff_se_clustered": se_cluster, "kl_diff_rel": (D / ka.mean()).item(),
           "kl_diff_seq_mean": seq_means.mean().item(), "kl_diff_seq_se": (seq_means.std() / math.sqrt(S)).item(),
           "seqs_b_lower": int((seq_means < 0).sum()), "argmax_agreement_a_b": (ama == amb).float().mean().item()}
    os.makedirs(out, exist_ok=True)
    json.dump(res, open(os.path.join(out, "kld_pair.json"), "w"), indent=1)
    for k in ("a", "b"):
        r = res[f"{k}_result"]
        print(f"{k}: positions {r['positions']:,}, KL mean {r['kld_mean']:.5f}, p99 {r['kld_p99']:.4f}, top-1 "
              f"{r['top1_agreement']:.4f}, NLL {r['nll_ours']:.4f} (teacher {r['nll_teacher']:.4f})")
    print(f"KL b - a: {res['kl_diff_mean']:+.6f} ({res['kl_diff_rel']:+.2%} of a), SE {res['kl_diff_se_clustered']:.6f} "
          f"with sequences as clusters ({res['kl_diff_se_positions']:.6f} over positions); per-sequence means "
          f"{res['kl_diff_seq_mean']:+.6f} +- {res['kl_diff_seq_se']:.6f}, b lower in {res['seqs_b_lower']} of {S}; "
          f"argmax agreement a/b {res['argmax_agreement_a_b']:.4f}")
    return 0


def main() -> int:
    cmd = sys.argv[1] if len(sys.argv) > 1 else ""
    if cmd == "pack" and len(sys.argv) == 5:
        return cmd_pack(sys.argv[2], sys.argv[3], sys.argv[4])
    if cmd == "pair" and len(sys.argv) == 6:
        return cmd_pair(*sys.argv[2:6])
    raise SystemExit("usage: kld_eval.py pack CAPDIR OUT STEP5_DIR | pair STEP5_DIR A B OUT")


if __name__ == "__main__":
    raise SystemExit(main())
