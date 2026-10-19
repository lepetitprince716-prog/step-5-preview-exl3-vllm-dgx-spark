# SPDX-License-Identifier: Apache-2.0
"""Layer-streaming forward of the official Step 5 reference over a fixed token set, ported from sudoingx dsv41q
(stream.Streamer): every sequence goes through layer l before any goes through l+1, so one decoder layer is resident.

What crosses layers is the official fp32 residual stream (`Step5Model.forward`, model.py:2182-2205), one tensor per
sequence. One layer is `Step5DecoderLayer.forward` (model.py:2125-2143) split in three:

  A (per sequence): h = input_layernorm(resid).bf16; resid += dense_attention(layer, h, positions).float()
  B (per group of consecutive sequences, <= max_group_tokens tokens): x = post_attention_layernorm(resid).bf16 over
    the group's tokens; y = moe(x, layer) -> fp32 [N, 4096] (default: the official layer._feed_forward, i.e. routed
    experts summed in top-k slot order in fp32, rounded to BF16, plus the shared expert, added in fp32)
  C (per sequence): resid += y

Memory (GB10, CUDA draws from system RAM): 853K tokens of fp32 residual are 14 GB; only one group's intermediates
are alive at a time (x bf16 + y fp32 = 12 KiB/token, plus the official MoE's [N, 8, 4096] BF16 slot buffer =
64 KiB/token: ~4 GiB at the default 65536-token group). The process is capped with
torch.cuda.set_per_process_memory_fraction(STEP5_CUDA_MEM_FRACTION, default 0.70) so a runaway allocation fails here
instead of wedging the box.
"""
from __future__ import annotations

import glob
import json
import logging
import os
from collections.abc import Callable
from dataclasses import dataclass

import torch
from torch import nn

from .ckpt import Checkpoint, drop_page_cache
from .refmodel import MAX_POSITION, dense_attention, load_config, load_head, load_layer

log = logging.getLogger(__name__)

ROUTER_RENORM_EPS = 1e-20     # model.py:1831, the deployed router kernel's renormalisation guard
MoEFn = Callable[[torch.Tensor, nn.Module], torch.Tensor]
Hook = Callable[[int, torch.Tensor, "torch.Tensor | None", "torch.Tensor | None", int], None]
AttnHook = Callable[[int, int, torch.Tensor, torch.Tensor], None]


@dataclass
class SeqState:
    resid: torch.Tensor           # fp32 [L, 4096]


def route_topk(x: torch.Tensor, gate_w: torch.Tensor, router_bias: torch.Tensor, top_k: int, scaling: float,
               renormalize: bool = True) -> tuple[torch.Tensor, torch.Tensor]:
    """Step5MoE.route (model.py:2020-2030), op for op: fp32 logits, sigmoid, + router_bias for selection only,
    weights = selected scores - bias, renormalised (+ ROUTER_RENORM_EPS), x scaling. -> (w fp32 [N, k], ids int64)."""
    prob = torch.sigmoid(torch.nn.functional.linear(x.float(), gate_w.float()))
    top, ids = torch.topk(prob + router_bias.view(1, -1), top_k, dim=-1)
    w = top - router_bias[ids]
    if renormalize:
        w = w / (w.sum(dim=-1, keepdim=True) + ROUTER_RENORM_EPS)
    return w * scaling, ids


def group_sequences(lengths: list[int], max_tokens: int) -> list[list[int]]:
    """Consecutive runs of sequence indices holding at most max_tokens tokens (a longer sequence is its own group)."""
    out: list[list[int]] = []
    n = 0
    for i, L in enumerate(lengths):
        if not out or n + L > max_tokens:
            out.append([])
            n = 0
        out[-1].append(i)
        n += L
    return out


def save_stream(path: str, resids: list[torch.Tensor], layer: int) -> None:
    """Residual stream entering `layer`: one BF16 tensor per sequence ("s{i}") + metadata, written atomically.
    BF16 on disk (7 GB at 853K tokens instead of 14): a resumed run differs from an uninterrupted one by that
    rounding."""
    from safetensors.torch import save_file
    tensors = {f"s{i}": r.to("cpu", torch.bfloat16).contiguous() for i, r in enumerate(resids)}
    meta = {"layer": str(layer), "lengths": json.dumps([int(r.shape[0]) for r in resids])}
    tmp = path + ".tmp"
    save_file(tensors, tmp, metadata=meta)
    os.replace(tmp, path)
    drop_page_cache(path)


def load_stream(path: str, lengths: list[int], device: str | torch.device) -> tuple[list[torch.Tensor], int]:
    """(fp32 residuals on device, layer) from save_stream; the sequence lengths must match."""
    from safetensors import safe_open
    with safe_open(path, framework="pt", device=str(device)) as f:
        meta = f.metadata()
        if json.loads(meta["lengths"]) != list(lengths):
            raise ValueError(f"{path}: saved for other sequences")
        resids = [f.get_tensor(f"s{i}").float() for i in range(len(lengths))]
    drop_page_cache(path)
    return resids, int(meta["layer"])


def _official_ffn(x: torch.Tensor, layer: nn.Module) -> torch.Tensor:
    return layer._feed_forward(x).float()


class Streamer:
    """seqs: token id lists (each 1..4096 tokens, prefilled alone from position 0, as the official forward does)."""

    def __init__(self, model_dir: str, seqs: list[list[int]], device: str | torch.device = "cuda",
                 max_group_tokens: int = 65536):
        self.device = torch.device(device)
        if self.device.type == "cuda":
            frac = float(os.environ.get("STEP5_CUDA_MEM_FRACTION", "0.70"))
            torch.cuda.set_per_process_memory_fraction(frac, self.device.index)   # None: current device
        bad = [i for i, s in enumerate(seqs) if not 1 <= len(s) <= MAX_POSITION]
        if bad:
            raise ValueError(f"sequences {bad[:8]} are empty or longer than {MAX_POSITION} tokens")
        self.ckpt = Checkpoint(model_dir)
        self.cfg = load_config(model_dir)
        self.max_group_tokens = max_group_tokens
        self.lengths = [len(s) for s in seqs]
        self.offsets = [0]
        for L in self.lengths[:-1]:
            self.offsets.append(self.offsets[-1] + L)
        self.n_tokens = self.offsets[-1] + self.lengths[-1]
        embed = self.ckpt.get("model.embed_tokens.weight", self.device)          # [vocab, 4096] bf16
        with torch.inference_mode():
            self.states = [SeqState(embed[torch.tensor(s, device=self.device)].float()) for s in seqs]
        del embed
        self.layer = 0

    # -- layers -------------------------------------------------------------------------------------------------------
    def load_layer(self, l: int, experts: bool = True) -> nn.Module:
        return load_layer(self.ckpt, self.cfg, l, self.device, experts=experts)

    def load_head(self) -> tuple[nn.Module, nn.Module]:
        return load_head(self.ckpt, self.cfg, self.device)

    def groups(self) -> list[list[int]]:
        return group_sequences(self.lengths, self.max_group_tokens)

    @torch.inference_mode()
    def route(self, layer: nn.Module, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """(topk_w fp32 [N, 8], topk_ids int64 [N, 8]) exactly as layer.moe.route (checked in gpu_single_layer.py)."""
        m = layer.moe
        return route_topk(x, m.gate.weight, m.router_bias, m.top_k, m.scaling_factor, m.renormalize)

    @torch.inference_mode()
    def run_layer(self, layer: nn.Module, moe: MoEFn | None = None, hook: Hook | None = None,
                  attn_hook: AttnHook | None = None) -> None:
        """Layer self.layer for every sequence (phases A, B, C above), then self.layer += 1.

        moe(x bf16 [N, 4096], layer) -> fp32 [N, 4096]: routed + shared expert (dense layers: the MLP).
        hook(l, x, topk_ids, topk_w, token_base): once per group, before moe; token_base is the global row of the
          group's first token; topk_ids / topk_w are None on dense layers (0-2, 91).
        attn_hook(l, seq_idx, h, o_in): per sequence; h = attention input BF16 [L, 4096] (q/k/v input), o_in =
          gated attention output BF16 [L, 64 * 192] (o_proj input)."""
        l = self.layer
        if layer.layer_idx != l:
            raise ValueError(f"stream is at layer {l}, got layer {layer.layer_idx}")
        moe = moe or _official_ffn
        for grp in self.groups():
            for i in grp:
                self._phase_a(layer, i, attn_hook)
            x = torch.cat([layer.post_attention_layernorm(self.states[i].resid).to(torch.bfloat16) for i in grp])
            if hook is not None:
                w, ids = self.route(layer, x) if layer.moe is not None else (None, None)
                hook(l, x, ids, w, self.offsets[grp[0]])
                del w, ids
            y = moe(x, layer)
            del x
            off = 0
            for i in grp:
                n = self.lengths[i]
                self.states[i].resid += y[off: off + n].float()
                off += n
            del y
        self.layer += 1

    def _phase_a(self, layer: nn.Module, i: int, attn_hook: AttnHook | None) -> None:
        st = self.states[i]
        h = layer.input_layernorm(st.resid).to(torch.bfloat16)
        pos = torch.arange(st.resid.shape[0], device=st.resid.device, dtype=torch.int32)
        seen: list[torch.Tensor] = []
        handle = (layer.self_attn.o_proj.register_forward_pre_hook(lambda m, args: seen.append(args[0]))
                  if attn_hook is not None else None)
        try:
            a = dense_attention(layer, h, pos)
        finally:
            if handle is not None:
                handle.remove()
        if attn_hook is not None:
            attn_hook(self.layer, i, h, seen[0])
        st.resid += a.float()

    # -- output -------------------------------------------------------------------------------------------------------
    @torch.inference_mode()
    def final_logits(self, norm: nn.Module, lm_head: nn.Module, rows) -> torch.Tensor:
        """fp32 logits [len(rows), vocab] for global token rows (sequences concatenated in order), after the last
        layer: the official tail (model.py:2198-2205, 2264-2269): resid -> BF16 (fp32_residual_connection) -> final
        norm -> lm_head in BF16 -> float. Callers chunk rows (a row is 0.5 MB of logits)."""
        rows = torch.as_tensor(rows, dtype=torch.long, device=self.device).view(-1)
        if rows.numel() and (int(rows.min()) < 0 or int(rows.max()) >= self.n_tokens):
            raise IndexError(f"rows outside [0, {self.n_tokens})")
        starts = torch.tensor(self.offsets, dtype=torch.long, device=self.device)
        seq = torch.searchsorted(starts, rows, right=True) - 1
        h = torch.empty((rows.numel(), self.cfg.hidden_size), dtype=torch.float32, device=self.device)
        for s in torch.unique(seq).tolist():
            m = seq == s
            h[m] = self.states[s].resid[rows[m] - self.offsets[s]]
        x = h.to(torch.bfloat16) if self.cfg.fp32_residual_connection else h   # embed_tokens dtype
        x = norm(x)
        return lm_head(x.to(lm_head.weight.dtype)).float()

    # -- resume -------------------------------------------------------------------------------------------------------
    def save(self, out_dir: str, prune: bool = True) -> str:
        """Write <out_dir>/stream_L{layer:02d}.safetensors (the stream entering self.layer); with prune, older
        stream_L*.safetensors in out_dir are removed after the new file is in place (7 GB each)."""
        os.makedirs(out_dir, exist_ok=True)
        path = os.path.join(out_dir, f"stream_L{self.layer:02d}.safetensors")
        save_stream(path, [st.resid for st in self.states], self.layer)
        if prune:
            for old in glob.glob(os.path.join(out_dir, "stream_L*.safetensors")):
                if old != path:
                    os.remove(old)
        log.info("stream saved at layer %d: %s", self.layer, path)
        return path

    def load(self, in_dir: str) -> int:
        """Resume from the newest stream_L*.safetensors in in_dir; returns the layer the stream is at."""
        files = sorted(glob.glob(os.path.join(in_dir, "stream_L*.safetensors")))
        if not files:
            raise FileNotFoundError(f"no stream_L*.safetensors in {in_dir}")
        resids, self.layer = load_stream(files[-1], self.lengths, self.device)
        for st, r in zip(self.states, resids):
            st.resid = r
        log.info("stream restored at layer %d from %s", self.layer, files[-1])
        return self.layer
