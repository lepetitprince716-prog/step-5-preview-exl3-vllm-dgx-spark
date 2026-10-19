"""Tensor names: the official checkpoint, 0xSero's step5_exl3 loader (sero keys), and exllamav3's naming.

Sero layout (plugin/step5_spark/exl3_quant.py module docstring, :155-160, :296-297):
  exl3/experts/L{LL}.safetensors  model.layers.{L}.moe.experts.{E}.{gate,up,down}_proj.{trellis,suh,svh,mul1}
  exl3/body/L{LL}.safetensors     model.layers.{L}.{self_attn.{q,k,v,o}_proj | share_expert.{gate,up,down}_proj |
                                   mlp.{gate,up,down}_proj}.{trellis,suh,svh,mul1}   (share_expert without "moe.")
exllamav3 has no Step 5 architecture; its Step 3.5 one (architecture/step3_5.py:130-254) names the same modules the
same way (moe.experts.{E}.{gate,up,down}_proj, share_expert.*, self_attn.*, mlp.*), so exllamav3_key equals the sero
key. Whether TensorFold reads that layout is unverified (it has no step5 family).
"""
from __future__ import annotations

import re

PIECES = ("trellis", "suh", "svh", "mul1", "mcg")
PROJ = {"w1": "gate_proj", "w3": "up_proj", "w2": "down_proj"}       # encode_expert keys -> module names
BODY_MODULES = tuple(f"self_attn.{p}_proj" for p in "qkvo") + tuple(
    f"{m}.{p}_proj" for m in ("share_expert", "mlp") for p in ("gate", "up", "down"))

_EXPERT_RE = re.compile(r"model\.layers\.(\d+)\.moe\.experts\.(\d+)\.(gate_proj|up_proj|down_proj)\.(\w+)")
_BODY_RE = re.compile(r"model\.layers\.(\d+)\.((?:self_attn|share_expert|mlp)\.\w+_proj)\.(\w+)")


def official_name(layer: int, module: str, param: str = "weight") -> str:
    """model.layers.{L}.{module}.{param}, e.g. (3, "moe.gate_proj") -> model.layers.3.moe.gate_proj.weight."""
    return f"model.layers.{layer}.{module}.{param}"


def _piece(piece: str) -> str:
    if piece not in PIECES:
        raise ValueError(f"unknown EXL3 piece {piece!r}")
    return piece


def sero_expert_key(layer: int, expert: int, proj: str, piece: str) -> str:
    """proj: gate_proj / up_proj / down_proj, or encode_expert's w1 / w3 / w2."""
    proj = PROJ.get(proj, proj)
    if proj not in PROJ.values():
        raise ValueError(f"unknown expert projection {proj!r}")
    return f"model.layers.{layer}.moe.experts.{expert}.{proj}.{_piece(piece)}"


def sero_body_key(layer: int, module: str, piece: str) -> str:
    """module: one of BODY_MODULES (self_attn.q_proj, share_expert.gate_proj, mlp.down_proj, ...)."""
    if module not in BODY_MODULES:
        raise ValueError(f"unknown body module {module!r}")
    return f"model.layers.{layer}.{module}.{_piece(piece)}"


def parse_sero_key(key: str) -> tuple[int, int | None, str, str]:
    """Inverse of sero_expert_key / sero_body_key: (layer, expert or None, proj or module, piece)."""
    m = _EXPERT_RE.fullmatch(key)
    if m:
        return int(m[1]), int(m[2]), m[3], m[4]
    m = _BODY_RE.fullmatch(key)
    if m and m[2] in BODY_MODULES:
        return int(m[1]), None, m[2], m[3]
    raise ValueError(f"not a sero EXL3 key: {key!r}")


def sero_expert_file(layer: int) -> str:
    return f"exl3/experts/L{layer:02d}.safetensors"


def sero_body_file(layer: int) -> str:
    return f"exl3/body/L{layer:02d}.safetensors"


def exllamav3_key(layer: int, module: str, piece: str, expert: int | None = None) -> str:
    """exllamav3 naming (Step 3.5 architecture): identical to the sero keys. module is a BODY_MODULES entry, or an
    expert projection when expert is given."""
    if expert is None:
        return sero_body_key(layer, module, piece)
    return sero_expert_key(layer, expert, module, piece)
