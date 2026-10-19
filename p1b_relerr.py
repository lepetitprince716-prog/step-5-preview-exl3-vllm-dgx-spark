# SPDX-License-Identifier: Apache-2.0
"""Per-K relative error table of Step 5's routed experts, measured (frontier review section 2 row 1a, stage 3 probe).
p2_allocate.py's default REL_ERR {2: 0.162, 3: 0.065, 4: 0.024, 5: 0.010} is sudoingx's DeepSeek-V4.1 layer-0 number
(5120 x 2304, FP8); Step 5's experts are 4096 x 1536.

Per probe layer L (default 3, 30, 60, 89; each needs the native stream entering L: p0_stats.py --stream-dir
--keep-stream-layers 3,30,60,89, run on the same --cal / --max-seqs):
  capture the MoE input of every calibration token, route, the same routing-weighted Hessians + all-token prior as
  p3_pass1.py (--tau), ~--experts experts spread over the usage range; per expert and K in --ks: encode_expert,
  reconstruct, and sqrt(out_err / out_energy) of the routed output (expert_forward_dense vs the native expert, routing
  weights applied) with gate+up quantized only ("gate_up"), down only ("down"), all three ("all").
Output JSON: p_util.relerr_table (pooled medians "rel_err", per layer band "band", per layer "detail") + "records";
p2_allocate.py --rel-err reads it directly. Rewritten after every layer; a restart skips the layers already in it.

--swap-kl-layer L (L must be a probe layer): all experts of L at K2 swapped into the native model on --kl-seqs x
--kl-len calibration tokens (the same spread as p1_sensitivity.py), run natively to the end: KL(native || swapped),
the mean per-token relative routed-noise energy |r_q - r_n|^2 / |r_n|^2 at L, alpha_swap = KL / noise, and with
--alpha (p1 output) p1's alpha_L and its prediction alpha_L x noise. Checks that p1's Gaussian injection does not
misplace the layer's sensitivity. The K2 tensors go to <out dir>/relerr_work/swap_LL.safetensors (reused on restart).

  python3 p1b_relerr.py /model /cal/cal.safetensors /out/stats/stream /out/rel_err.json [--layers 3,30,60,89]
      [--experts 64] [--ks 2,3,4,5] [--swap-kl-layer 30 --alpha /out/alpha.json]
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from p1_sensitivity import clone_states  # noqa: E402
from p3_pass1 import capture, chunked_err, expert_hessians, layer_prior, route_all  # noqa: E402
from p_torch import (ExpertCache, dims, kl_sums, load_exl3, load_seqs, logger, mem_gib, native_slots,  # noqa: E402
                     quant_routed, stream_files)
from p_util import (atomic_json, model_layers, parse_layers, relerr_table, sero_expert_key, spread_pick,  # noqa: E402
                    W_TO_PROJ)


def enter_layer(s, path: str, l: int) -> None:
    from step5q.stream import load_stream
    resids, at = load_stream(path, s.lengths, s.device)
    if at != l:
        raise SystemExit(f"{path}: stream enters layer {at}, expected {l}")
    for st, r in zip(s.states, resids):
        st.resid = r
    s.layer = l


def encode(W1, W3, W2, H_gu, H_d, n: int, tau: int, K: int, seed: int, work: str, tag: str) -> dict:
    """encode_expert on clones of the Hessians (h_data consumes them) -> {"w1"|"w3"|"w2": pieces}."""
    from step5q.exl3 import encode_expert, h_data
    tens, _ = encode_expert(W1.float(), W3.float(), W2.float(), h_data(H_gu.clone(), n + tau, f"{tag}.gu"),
                            h_data(H_d.clone(), n + tau, f"{tag}.d"), K, seed, work)
    return tens


@torch.inference_mode()
def probe_layer(s, l: int, path: str, n_exp: int, a, work: str, swap_path: str, log) -> list[dict]:
    from step5q.exl3 import reconstruct
    from step5q.refmodel import expert_weights
    t0 = time.time()
    enter_layer(s, path, l)
    layer = s.load_layer(l, experts=False)
    flat = capture(s, layer)
    N = flat.shape[0]
    Wr, Ir = route_all(s, layer, flat)
    H_all, mean_g2, xs = layer_prior(flat, Wr, l)
    counts = torch.bincount(Ir.flatten(), minlength=n_exp).tolist()
    pick = spread_pick(counts, a.experts)
    do_swap = bool(swap_path) and not os.path.exists(swap_path)
    todo = list(range(n_exp)) if do_swap else pick
    w1, w3, w2 = expert_weights(s.ckpt, l, "cuda")
    idx = torch.tensor(todo, device=w1.device)
    w1, w3, w2 = w1[idx].contiguous(), w3[idx].contiguous(), w2[idx].contiguous()
    torch.cuda.empty_cache()
    recs, swap = [], {}
    for j, e in enumerate(todo):
        tok, slot = (Ir == e).nonzero(as_tuple=True)
        W1, W3, W2 = w1[j], w3[j], w2[j]
        H_gu, H_d = expert_hessians(flat, Wr, tok, slot, W1, W3, H_all, N, xs, mean_g2, a.tau)
        if do_swap:   # ponytail: picked experts are encoded at K2 twice (swap + table); ~64 extra encodes per run
            for w, pieces in encode(W1, W3, W2, H_gu, H_d, tok.numel(), a.tau, 2, l * 1000 + e, work,
                                    f"L{l}.e{e}").items():
                for piece, v in pieces.items():
                    swap[sero_expert_key(l, e, W_TO_PROJ[w], piece)] = v.contiguous().cpu()
        if e in pick:
            native = (W1.half(), W3.half(), W2.half())
            for K in a.k_list:
                t = encode(W1, W3, W2, H_gu, H_d, tok.numel(), a.tau, K, l * 1000 + e, work, f"L{l}.e{e}")
                q = tuple(reconstruct(t[w], *dims(t[w])) for w in ("w1", "w3", "w2"))
                rec = {"layer": l, "expert": e, "n": tok.numel(), "K": K}
                for key, wq in (("gate_up", (q[0], q[1], native[2])), ("down", (native[0], native[1], q[2])),
                                ("all", q)):
                    err, energy = chunked_err(flat, Wr, tok, slot, wq, native)
                    rec[key] = (err / max(energy, 1e-30)) ** 0.5
                recs.append(rec)
                del t, q
            del native
        del H_gu, H_d
        if (j + 1) % 16 == 0:
            log(f"  layer {l}: {j + 1}/{len(todo)} experts, {time.time() - t0:.0f}s, {mem_gib()}")
    if do_swap:
        from safetensors.torch import save_file
        save_file(swap, swap_path + ".tmp", metadata={"format": "pt", "encoder": "exllamav3 1.6.0 mul1"})
        os.replace(swap_path + ".tmp", swap_path)
    del flat, Wr, Ir, H_all, xs, w1, w3, w2, layer, swap
    torch.cuda.empty_cache()
    log(f"layer {l}: {len(pick)} experts x K {a.k_list}, {N:,} tokens, {time.time() - t0:.0f}s")
    return recs


@torch.inference_mode()
def swap_kl(a, l: int, path: str, swap_path: str, moe_layers: list[int], n_layers: int, n_exp: int, log) -> dict:
    """KL(native || L's experts at K2) on a spread subset of the calibration set, prefixes of the saved stream (causal:
    the stream at a prefix does not depend on later tokens)."""
    from safetensors import safe_open
    from step5q.stream import Streamer
    t0 = time.time()
    allseqs = load_seqs(a.cal, a.max_seqs)
    ids = list(range(0, len(allseqs), max(1, len(allseqs) // a.kl_seqs)))[: a.kl_seqs]
    s = Streamer(a.model, [allseqs[i][: a.kl_len] for i in ids], "cuda")
    with safe_open(path, framework="pt", device="cpu") as f:
        if json.loads(f.metadata()["lengths"]) != [len(x) for x in allseqs]:
            raise SystemExit(f"{path}: saved for other sequences (same --cal / --max-seqs as p0?)")
        for st, i, n in zip(s.states, ids, s.lengths):
            st.resid = f.get_tensor(f"s{i}")[:n].float().to(s.device)
    clean, swapped = s.states, clone_states(s.states)
    cache = ExpertCache(l, load_exl3([swap_path], "cuda"), n_exp)
    noise = [0.0, 0]

    def swap_moe(x: torch.Tensor, layer) -> torch.Tensor:
        w, e_ids = s.route(layer, x)
        m = layer.moe
        _, rn = native_slots(x, w, e_ids, m.gate_proj.weight, m.up_proj.weight, m.down_proj.weight)
        rq = quant_routed(x, w, e_ids, cache).to(x.dtype)
        rel = (rq.float() - rn.float()).square().sum(-1) / rn.float().square().sum(-1).clamp_min(1e-30)
        noise[0] += rel.sum().item()
        noise[1] += x.shape[0]
        return rq.float() + layer.share_expert(x).float()
    for j in range(l, n_layers):
        layer = s.load_layer(j, experts=j in moe_layers)
        for states, moe in ((clean, None), (swapped, swap_moe if j == l else None)):
            s.states, s.layer = states, j
            s.run_layer(layer, moe=moe)
        del layer
        if j == l:
            cache = None                                    # the swapped experts are only needed at layer l
        torch.cuda.empty_cache()
    norm, head = s.load_head()
    kl, n = kl_sums(s, clean, {"swap": swapped}, norm, head)
    out = {"layer": l, "K": 2, "seqs": len(ids), "tokens": n, "kl": kl["swap"] / n, "noise": noise[0] / max(noise[1], 1)}
    out["alpha_swap"] = out["kl"] / max(out["noise"], 1e-30)
    if a.alpha:
        al = json.load(open(a.alpha))["alpha"].get(str(l))
        if al is not None:
            out.update(alpha_p1=al, kl_pred_p1=al * out["noise"], ratio_swap_over_p1=out["alpha_swap"] / max(al, 1e-30))
    log(f"swap layer {l} at K2: KL {out['kl']:.6f}, noise {out['noise']:.5f}, alpha_swap {out['alpha_swap']:.5f}"
        f"{', alpha_p1 %.5f' % out['alpha_p1'] if 'alpha_p1' in out else ''}, {time.time() - t0:.0f}s")
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("model")
    ap.add_argument("cal")
    ap.add_argument("stream_dir", help="p0_stats.py --stream-dir with the kept stream_LL.safetensors")
    ap.add_argument("out", help="rel_err.json")
    ap.add_argument("--layers", default="3,30,60,89")
    ap.add_argument("--experts", type=int, default=64)
    ap.add_argument("--ks", default="2,3,4,5")
    ap.add_argument("--max-seqs", type=int, default=0, help="as given to p0_stats.py")
    ap.add_argument("--tau", type=int, default=512)
    ap.add_argument("--group-tokens", type=int, default=131072)
    ap.add_argument("--swap-kl-layer", type=int, default=-1)
    ap.add_argument("--kl-seqs", type=int, default=16)
    ap.add_argument("--kl-len", type=int, default=512)
    ap.add_argument("--alpha", default="", help="p1_sensitivity.py alpha.json, for the swap comparison")
    a = ap.parse_args()
    a.k_list = [int(k) for k in a.ks.split(",")]
    from step5q.stream import Streamer
    moe_layers, body_layers, n_exp = model_layers(json.load(open(os.path.join(a.model, "config.json"))))
    probes = parse_layers(a.layers, moe_layers)
    if any(l not in moe_layers for l in probes):
        raise SystemExit(f"--layers {a.layers}: probe layers must be MoE layers {moe_layers[0]}-{moe_layers[-1]}")
    if a.swap_kl_layer >= 0 and a.swap_kl_layer not in probes:
        raise SystemExit("--swap-kl-layer must be one of --layers (its Hessians come from the probe pass)")
    files = stream_files(a.stream_dir)
    missing = [l for l in probes if l not in files]
    if missing:
        raise SystemExit(f"no stream_LL.safetensors for layers {missing} in {a.stream_dir} (p0 --keep-stream-layers)")
    work = os.path.join(os.path.dirname(os.path.abspath(a.out)), "relerr_work")
    os.makedirs(work, exist_ok=True)
    log = logger(os.path.join(work, "p1b.log"))
    swap_path = os.path.join(work, f"swap_{a.swap_kl_layer:02d}.safetensors") if a.swap_kl_layer >= 0 else ""
    res = json.load(open(a.out)) if os.path.exists(a.out) else {}
    recs = res.get("records", [])
    done = {r["layer"] for r in recs}
    todo = [l for l in probes if l not in done or (l == a.swap_kl_layer and not os.path.exists(swap_path))]
    first, last = moe_layers[0], moe_layers[-1]
    if todo:
        s = Streamer(a.model, load_seqs(a.cal, a.max_seqs), "cuda", max_group_tokens=a.group_tokens)
        for l in todo:
            new = probe_layer(s, l, files[l], n_exp, a, work, swap_path if l == a.swap_kl_layer else "", log)
            recs = [r for r in recs if r["layer"] != l] + new
            res = relerr_table(recs, first, last) | {"records": recs, "swap": res.get("swap")}
            atomic_json(a.out, res, indent=1)
        del s
        torch.cuda.empty_cache()
    if a.swap_kl_layer >= 0 and not res.get("swap"):
        res["swap"] = swap_kl(a, a.swap_kl_layer, files[a.swap_kl_layer], swap_path, moe_layers, len(body_layers),
                              n_exp, log)
        atomic_json(a.out, res, indent=1)
    log(f"rel_err (pooled median of 'all'): {res.get('rel_err')}")
    log(f"bands: {res.get('band')}")


if __name__ == "__main__":
    main()
