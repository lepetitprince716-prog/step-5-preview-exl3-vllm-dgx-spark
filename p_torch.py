# SPDX-License-Identifier: Apache-2.0
"""Torch helpers shared by the Step 5 passes (p0, p1, p3, p4, p6). Ported in part from sudoingx dsv41_recipe
(scripts/quantize/p4_pass1.py, dsv41q/exl3.py, Apache-2.0), adapted to Step 5:

  - routing weight AFTER down_proj, top-k slot outputs summed in slot order in fp32 (inference/model.py Step5MoE.forward
    and its CPU fallback), the routed sum rounded to BF16 before the shared expert is added in fp32
    (Step5DecoderLayer._feed_forward);
  - EXL3 experts run as step5q.exl3.expert_forward_dense (fp16 weights / activations, weight after down_proj);
  - q/k/v (and gate/up) are encoded as ONE concatenated matrix and split along the output tiles, so the parts share a
    single su (the plugin's fused GEMV requires equal suh, exl3_quant.py:181-182; its TP slicing at 128-multiples of the
    output tiles is lossless, exl3_quant.py:170-173, which is what the split relies on).

Only the documented step5q names are imported (README "Layout and contracts").
"""
from __future__ import annotations

import glob
import os
import sys
from typing import Callable

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from p_util import PIECES, drop_page_cache  # noqa: E402

LIMIT = 7.0          # swiglu clamp of every Step 5 layer (config swiglu_limits / swiglu_limits_shared, all 7.0)
HID, INTER = 4096, 1536


# ------------------------------------------------------------------------------------------------- data
def load_seqs(path: str, max_seqs: int = 0) -> list[list[int]]:
    """Token set written by caldata/build.py: safetensors {"tokens": int [T], "offsets": int [n + 1]} (sudoingx's
    format; ASSUMED for lane C's build.py)."""
    from safetensors.torch import load_file
    d = load_file(path)
    t, o = d["tokens"].tolist(), d["offsets"].tolist()
    seqs = [t[o[i]: o[i + 1]] for i in range(len(o) - 1)]
    return seqs[:max_seqs] if max_seqs else seqs


def load_exl3(files: list[str], device) -> dict[str, torch.Tensor]:
    from safetensors.torch import load_file
    t: dict[str, torch.Tensor] = {}
    for f in files:
        t.update(load_file(f, device=str(device)))
        drop_page_cache(f)
    return t


def modules_of(tensors: dict[str, torch.Tensor], prefix: str) -> dict[str, dict[str, torch.Tensor]]:
    """{module path after prefix: {piece: tensor}} for EXL3 pieces under prefix."""
    mods: dict[str, dict[str, torch.Tensor]] = {}
    for k, v in tensors.items():
        if k.startswith(prefix):
            path, piece = k[len(prefix):].rsplit(".", 1)
            if piece in PIECES:
                mods.setdefault(path, {})[piece] = v
    return mods


def dims(t: dict[str, torch.Tensor]) -> tuple[int, int]:
    """(in_features, out_features) from trellis [in/16, out/16, 16 K]."""
    tr = t["trellis"]
    return tr.shape[0] * 16, tr.shape[1] * 16


# ------------------------------------------------------------------------------------------------- Hessians
class HAcc:
    """Sum of x^T x per key (fp32) and the row count (sudoingx p4_dense.HAcc)."""

    def __init__(self) -> None:
        self.H: dict[str, torch.Tensor] = {}
        self.n: dict[str, int] = {}

    def add(self, key: str, x: torch.Tensor) -> None:
        x = x.reshape(-1, x.shape[-1]).float()
        if key not in self.H:
            self.H[key] = torch.zeros(x.shape[-1], x.shape[-1], dtype=torch.float32, device=x.device)
            self.n[key] = 0
        self.H[key].addmm_(x.t(), x)
        self.n[key] += x.shape[0]

    def clone(self) -> "HAcc":
        c = HAcc()
        c.H = {k: v.clone() for k, v in self.H.items()}
        c.n = dict(self.n)
        return c


def h_err(w: torch.Tensor, wq: torch.Tensor, H: torch.Tensor) -> float:
    """Relative output error under the calibration inputs: sqrt(tr(D H D^T) / tr(W H W^T)), D = Wq - W ([out, in])."""
    d = wq.float() - w
    return ((d @ H) * d).sum().div(((w @ H) * w).sum().clamp_min(1e-30)).sqrt().item()


# ------------------------------------------------------------------------------------------------- EXL3 tensors
def split_out(t: dict[str, torch.Tensor], sizes: list[int]) -> list[dict[str, torch.Tensor]]:
    """Split one EXL3 matrix encoded as concat(outputs) into parts of `sizes` output channels: trellis along its
    output-tile dim, svh along outputs; suh / mul1 / mcg are shared (copied). Every size must be a multiple of 128
    (Hadamard blocks stay whole, as in the plugin's TP slicing)."""
    if sum(sizes) != t["trellis"].shape[1] * 16:
        raise ValueError(f"split {sizes} of {t['trellis'].shape[1] * 16} outputs")
    out, a = [], 0
    for n in sizes:
        if n % 128:
            raise ValueError(f"split size {n} is not a multiple of 128")
        p = {"trellis": t["trellis"][:, a // 16:(a + n) // 16].contiguous(), "svh": t["svh"][a:a + n].contiguous(),
             "suh": t["suh"].clone()}
        for k in ("mul1", "mcg"):
            if k in t:
                p[k] = t[k].clone()
        out.append(p)
        a += n
    return out


def encode_fused(ws: list[torch.Tensor], H: dict, K: int, seed: int, work_dir: str) -> list[dict[str, torch.Tensor]]:
    """EXL3-encode several [out_i, in] matrices that read the same input as one concatenated matrix (one Hessian, one
    su), then split per matrix. step5q.exl3.encode_linear does the encode."""
    from step5q.exl3 import encode_linear
    w = torch.cat([x.float() for x in ws], 0)
    res = encode_linear(w, H, K, seed, work_dir)
    t = res[0] if isinstance(res, tuple) else res
    return split_out(t, [x.shape[0] for x in ws]) if len(ws) > 1 else [t]


# ------------------------------------------------------------------------------------------------- MoE forwards
def swiglu(gate: torch.Tensor, up: torch.Tensor, limit: float = LIMIT) -> torch.Tensor:
    """inference/model.py clamped_swiglu: silu(gate).clamp(max) * up.clamp(-limit, limit) in fp32, rounded to input."""
    return (F.silu(gate.float()).clamp(max=limit) * up.float().clamp(-limit, limit)).to(gate.dtype)


def native_slots(x: torch.Tensor, w: torch.Tensor, ids: torch.Tensor, W1: torch.Tensor, W3: torch.Tensor,
                 W2: torch.Tensor, limit: float = LIMIT) -> tuple[torch.Tensor, torch.Tensor]:
    """Official Step5MoE.forward on BF16 experts (W* [352, out, in]): per-slot BF16 contributions [N, k, H] and the
    routed output (fp32 slot-order weighted sum, rounded to BF16) [N, H]."""
    N, k = ids.shape
    slot = torch.zeros(N, k, x.shape[-1], dtype=x.dtype, device=x.device)
    for e in ids.unique().tolist():
        rows, slots = (ids == e).nonzero(as_tuple=True)
        t = x[rows]
        slot[rows, slots] = F.linear(swiglu(F.linear(t, W1[e]), F.linear(t, W3[e]), limit), W2[e]).to(x.dtype)
    out = torch.zeros(N, x.shape[-1], dtype=torch.float32, device=x.device)
    for s in range(k):
        out += slot[:, s].float() * w[:, s, None].float()
    return slot, out.to(x.dtype)


class ExpertCache:
    """Reconstructed fp16 weights of one layer's EXL3 experts ({e: (w1, w3, w2)}, nn.Linear orientation), built once
    per layer: the per-slot forward below visits every expert up to k times per group."""

    def __init__(self, layer: int, tensors: dict[str, torch.Tensor], n_experts: int):
        from step5q.exl3 import reconstruct
        self.w: dict[int, tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = {}
        mods = modules_of(tensors, f"model.layers.{layer}.moe.experts.")
        for e in range(n_experts):
            try:
                g, u, d = (mods[f"{e}.{p}"] for p in ("gate_proj", "up_proj", "down_proj"))
            except KeyError:
                raise SystemExit(f"layer {layer}: expert {e} missing from the EXL3 tensors") from None
            self.w[e] = (reconstruct(g, *dims(g)), reconstruct(u, *dims(u)), reconstruct(d, *dims(d)))


def quant_routed(x: torch.Tensor, w: torch.Tensor, ids: torch.Tensor, cache: ExpertCache) -> torch.Tensor:
    """EXL3 routed output, fp32 [N, H]: expert_forward_dense per (slot, expert), accumulated slot by slot so each token
    sums its k contributions in slot order (rows are unique within one index_add_, so the order is deterministic)."""
    from step5q.exl3 import expert_forward_dense
    out = torch.zeros(x.shape[0], x.shape[-1], dtype=torch.float32, device=x.device)
    for s in range(ids.shape[1]):
        col = ids[:, s]
        for e in col.unique().tolist():
            rows = (col == e).nonzero().squeeze(1)
            out.index_add_(0, rows, expert_forward_dense(x[rows], w[rows, s].float(), *cache.w[e], limit=LIMIT))
    return out


def make_quant_moe(s, cache: ExpertCache | None) -> Callable:
    """A Streamer moe(x, layer) -> fp32: EXL3 routed experts (cache) or the native experts on the layer (cache None),
    routed sum rounded to BF16, plus the layer's native (or overlaid) shared expert; dense layers run layer.mlp."""
    def moe(x: torch.Tensor, layer) -> torch.Tensor:
        if getattr(layer, "moe", None) is None:
            return layer.mlp(x).float()
        w, ids = s.route(layer, x)
        if cache is None:
            m = layer.moe
            _, routed = native_slots(x, w, ids, m.gate_proj.weight, m.up_proj.weight, m.down_proj.weight)
        else:
            routed = quant_routed(x, w, ids, cache).to(x.dtype)
        return routed.float() + layer.share_expert(x).float()
    return moe


# ------------------------------------------------------------------------------------------------- body overlay
def overlay_body(layer, layer_idx: int, tensors: dict[str, torch.Tensor]) -> list[str]:
    """Copy reconstructed EXL3 body weights (sero body keys) into the official layer's BF16 linears in place (in-place
    keeps the packed-QKV views of Step5Attention valid). fp16 -> bf16 rounding as in sudoingx's overlay.py."""
    from step5q.exl3 import reconstruct
    done = []
    for path, t in sorted(modules_of(tensors, f"model.layers.{layer_idx}.").items()):
        if path.startswith("moe."):
            continue
        mod = layer.get_submodule(path)
        w = reconstruct(t, *dims(t))
        if tuple(w.shape) != tuple(mod.weight.shape):
            raise SystemExit(f"layer {layer_idx} {path}: EXL3 {tuple(w.shape)} vs module {tuple(mod.weight.shape)}")
        with torch.no_grad():
            mod.weight.data.copy_(w.to(mod.weight.dtype))
        done.append(path)
    return done


# ------------------------------------------------------------------------------------------------- stream helpers
def seq_rows(s, i: int) -> torch.Tensor:
    """Global token rows of sequence i (sequences concatenated in order)."""
    return torch.arange(s.offsets[i], s.offsets[i] + s.lengths[i], device=s.device)


def seq_logits(s, i: int, norm, head) -> "torch.Tensor":
    """fp32 logits [L_i, vocab] of sequence i: Streamer.final_logits with the modules from Streamer.load_head()."""
    return s.final_logits(norm, head, seq_rows(s, i))


def kl_sums(s, ref_states: list, others: dict, norm, head) -> tuple[dict, int]:
    """Summed per-token KL(ref || other) for each {key: states} and the token count, sequence by sequence (the
    reference log-softmax once per sequence)."""
    kl = {k: 0.0 for k in others}
    n = 0
    for i in range(len(s.lengths)):
        s.states = ref_states
        ref = torch.log_softmax(seq_logits(s, i, norm, head).float(), -1)
        p = ref.exp()
        for k, states in others.items():
            s.states = states
            lq = torch.log_softmax(seq_logits(s, i, norm, head).float(), -1)
            kl[k] += (p * (ref - lq)).sum().item()
        n += ref.shape[0]
        del ref, p
    s.states = ref_states
    return kl, n


def stream_files(stream_dir: str) -> dict[int, str]:
    """{layer: path} of the saved streams (Streamer.save names: stream_L{layer:02d}.safetensors = entering layer)."""
    out = {}
    for f in glob.glob(os.path.join(stream_dir, "stream_L*.safetensors")) if stream_dir else []:
        out[int(os.path.basename(f)[8:-12])] = f
    return out


def resume_stream(s, stream_dir: str, log: Callable[[str], None], at: int | None = None) -> None:
    """Load the newest saved stream (with `at`: the newest entering a layer <= at, e.g. a kept snapshot)."""
    files = {l: f for l, f in stream_files(stream_dir).items() if at is None or l <= at}
    if not files:
        return
    from step5q.stream import load_stream
    l = max(files)
    resids, s.layer = load_stream(files[l], s.lengths, s.device)
    for st, r in zip(s.states, resids):
        st.resid = r
    log(f"stream resumed from {files[l]}: next layer {s.layer}")


def save_stream(s, stream_dir: str, keep: set[int] | frozenset = frozenset()) -> None:
    """Streamer.save without its pruning of older files, then prune here except the layers in `keep` (snapshots for
    p1b_relerr.py or a later partial re-encode)."""
    if not stream_dir:
        return
    path = s.save(stream_dir, prune=False)
    for l, f in stream_files(stream_dir).items():
        if f != path and l not in keep:
            os.remove(f)


def files_for(layer: int, experts_dir: str = "", body_dir: str = "", assembled: str = "") -> tuple[list[str], list[str]]:
    """EXL3 files of one layer: (expert files, body files) from pass outputs or an assembled serving dir."""
    if assembled:
        e = os.path.join(assembled, "exl3", "experts", f"L{layer:02d}.safetensors")
        b = os.path.join(assembled, "exl3", "body", f"L{layer:02d}.safetensors")
        return [e] if os.path.exists(e) else [], [b] if os.path.exists(b) else []
    ex = sorted(glob.glob(os.path.join(experts_dir, f"layer_{layer:02d}.part*.safetensors"))) if experts_dir else []
    b = os.path.join(body_dir, f"body_{layer:02d}.safetensors") if body_dir else ""
    return ex, [b] if b and os.path.exists(b) else []


def mem_gib() -> str:
    return (f"alloc {torch.cuda.memory_allocated() / 2 ** 30:.1f} GiB, "
            f"peak {torch.cuda.max_memory_allocated() / 2 ** 30:.1f} GiB") if torch.cuda.is_available() else "cpu"


def logger(path: str) -> Callable[[str], None]:
    f = open(path, "a")

    def log(msg: str) -> None:
        print(msg, flush=True)
        f.write(msg + "\n")
        f.flush()
    return log
