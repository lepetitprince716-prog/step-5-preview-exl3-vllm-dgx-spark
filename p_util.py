# SPDX-License-Identifier: Apache-2.0
"""Pure-Python helpers shared by the Step 5 passes (no torch import): raw safetensors I/O, the 0xSero serving key
names, plan / layer-range parsing. Parts are ported from sudoingx dsv41_recipe (scripts/quantize, Apache-2.0).

Raw safetensors copy: tensors are moved byte for byte (header parsed with json, data ranges copied), so the
assembler handles BF16 / FP8 / int16 trellis tensors without torch and its tests run on any machine.
"""
from __future__ import annotations

import json
import os
import struct
import sys
from typing import Iterable

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from step5q.names import PIECES, sero_body_key, sero_expert_key  # noqa: E402,F401  (lane A; pure Python)
from step5q.names import PROJ as W_TO_PROJ  # noqa: E402,F401  encode_expert w1/w3/w2 -> gate/up/down_proj

EXPERT_PROJS = ("gate_proj", "up_proj", "down_proj")
ATTN_PROJS = ("q_proj", "k_proj", "v_proj", "o_proj")
MLP_PROJS = ("gate_proj", "up_proj", "down_proj")


# ------------------------------------------------------------------------------------------------- key names
def body_modules(is_moe: bool) -> list[str]:
    """The EXL3 body modules of one decoder layer (attention always; shared expert on MoE layers, dense MLP else)."""
    ffn = [f"share_expert.{p}" for p in MLP_PROJS] if is_moe else [f"mlp.{p}" for p in MLP_PROJS]
    return [f"self_attn.{p}" for p in ATTN_PROJS] + ffn


# ------------------------------------------------------------------------------------------------- config / plan
def text_config(cfg: dict) -> dict:
    return cfg.get("text_config", cfg)


def model_layers(cfg: dict) -> tuple[list[int], list[int], int]:
    """(expert_layers, body_layers, n_experts) of the main model from an official config (MTP layers excluded)."""
    t = text_config(cfg)
    n = int(t["num_hidden_layers"])
    moe = t.get("moe_layer_list")
    if moe is None:
        moe = [int(x) for x in str(t.get("moe_layers_enum", "")).split(",") if x.strip()]
    expert_layers = sorted(int(x) for x in moe if int(x) < n)
    return expert_layers, list(range(n)), int(t["moe_num_experts"])


def parse_layers(spec: str, default: Iterable[int]) -> list[int]:
    """'3-90', '5', '3,7,11' or '' (default)."""
    if not spec:
        return list(default)
    out: list[int] = []
    for part in spec.split(","):
        a, _, b = part.partition("-")
        out += list(range(int(a), int(b or a) + 1))
    return out


def parse_part(spec: str) -> tuple[int, int]:
    i, n = (int(x) for x in spec.split("/"))
    if not 0 <= i < n:
        raise SystemExit(f"--part {spec}: need 0 <= i < n")
    return i, n


def plan_k(plan: dict, layer: int) -> list[int]:
    ks = plan["layers"][str(layer)]["k"]
    return [int(k) for k in ks]


def k_of_trellis_shape(shape: list[int]) -> int:
    """EXL3 trellis [in/16, out/16, 16 K] (uint16) -> K (integer K only)."""
    return int(shape[-1]) // 16


# ------------------------------------------------------------------------------------------------- safetensors raw
_DT_BYTES = {"F64": 8, "F32": 4, "F16": 2, "BF16": 2, "I64": 8, "I32": 4, "I16": 2, "I8": 1, "U8": 1, "U16": 2,
             "U32": 4, "U64": 8, "BOOL": 1, "F8_E4M3": 1, "F8_E5M2": 1}


def read_header(path: str) -> tuple[dict, int]:
    """(header without __metadata__, byte offset of the data section)."""
    with open(path, "rb") as f:
        (n,) = struct.unpack("<Q", f.read(8))
        hdr = json.loads(f.read(n))
    hdr.pop("__metadata__", None)
    return hdr, 8 + n


def read_metadata(path: str) -> dict:
    with open(path, "rb") as f:
        (n,) = struct.unpack("<Q", f.read(8))
        return json.loads(f.read(n)).get("__metadata__", {}) or {}


def write_raw(path: str, tensors: list[tuple[str, str, list[int], bytes | tuple[str, int, int]]],
              metadata: dict | None = None) -> int:
    """Write a safetensors file from (key, dtype, shape, data) where data is bytes or (src_path, start, end) to copy.
    Keys are sorted (safetensors' own order). Atomic (tmp + rename). Returns the file size."""
    items = sorted(tensors, key=lambda t: t[0])
    hdr: dict = {}
    off = 0
    for key, dt, shape, data in items:
        n = len(data) if isinstance(data, (bytes, bytearray)) else data[2] - data[1]
        want = _DT_BYTES[dt]
        for s in shape:
            want *= int(s)
        if n != want:
            raise ValueError(f"{key}: {n} bytes for {dt}{shape} (want {want})")
        hdr[key] = {"dtype": dt, "shape": [int(s) for s in shape], "data_offsets": [off, off + n]}
        off += n
    if metadata:
        hdr["__metadata__"] = {str(k): str(v) for k, v in metadata.items()}
    hb = json.dumps(hdr, separators=(",", ":")).encode()
    hb += b" " * (-len(hb) % 8)
    tmp = path + ".tmp"
    with open(tmp, "wb") as out:
        out.write(struct.pack("<Q", len(hb)))
        out.write(hb)
        for _, _, _, data in items:
            if isinstance(data, (bytes, bytearray)):
                out.write(data)
                continue
            src, a, b = data
            with open(src, "rb") as f:
                f.seek(a)
                left = b - a
                while left:
                    chunk = f.read(min(left, 64 << 20))
                    if not chunk:
                        raise IOError(f"{src}: short read")
                    out.write(chunk)
                    left -= len(chunk)
    os.chmod(tmp, 0o644)
    os.replace(tmp, path)
    return os.path.getsize(path)


def raw_entries(path: str, rename=None, keep=None) -> list[tuple[str, str, list[int], tuple[str, int, int]]]:
    """write_raw entries copying tensors of `path` (optionally filtered by keep(key) and renamed by rename(key))."""
    hdr, base = read_header(path)
    out = []
    for k, v in hdr.items():
        if keep is not None and not keep(k):
            continue
        a, b = v["data_offsets"]
        out.append((rename(k) if rename else k, v["dtype"], v["shape"], (path, base + a, base + b)))
    return out


def check_safetensors(path: str) -> str | None:
    """verify.py's size check (0xSero Step-5-Preview-Four-Sparks scripts/verify.py:48-61): header parses and the file
    is exactly as long as its header implies."""
    try:
        size = os.path.getsize(path)
        with open(path, "rb") as f:
            (hlen,) = struct.unpack("<Q", f.read(8))
            if hlen <= 0 or hlen > 100 * 2 ** 20:
                return f"bad header length {hlen}"
            hdr = json.loads(f.read(hlen))
        end = max((v["data_offsets"][1] for k, v in hdr.items() if k != "__metadata__"), default=0)
        if 8 + hlen + end != size:
            return f"size {size} != {8 + hlen + end} implied by its header (truncated?)"
    except Exception as e:  # noqa: BLE001
        return f"unreadable: {e}"
    return None


def drop_page_cache(path: str) -> None:
    """Evict a file's pages (GB10: CUDA allocations cannot reclaim page cache; sudoingx FINDINGS 1)."""
    if not hasattr(os, "posix_fadvise") or not os.path.exists(path):
        return
    paths = [os.path.join(path, f) for f in os.listdir(path)] if os.path.isdir(path) else [path]
    for p in paths:
        if os.path.isfile(p):
            fd = os.open(p, os.O_RDONLY)
            try:
                os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
            finally:
                os.close(fd)


def atomic_json(path: str, obj, indent: int | None = None) -> None:
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(obj, f, indent=indent)
    os.chmod(tmp, 0o644)
    os.replace(tmp, path)


# ------------------------------------------------------------------------------------------------- domains
COARSE = {"code_debug": "code", "long_context": "long"}
# sudoingx's eight panel domains at 1, the instruct supplement at 1.5, exllamav3's standard text at 0.5
DEFAULT_DOMAIN_WEIGHTS = {"code": 1.0, "long": 1.0, "agentic": 1.0, "chat": 1.0, "writing": 1.0, "knowledge": 1.0,
                          "math": 1.0, "multilingual": 1.0, "instruct": 1.5, "std": 0.5}


def coarse_domain(rec: dict) -> str:
    """build.py index row -> allocation domain (sudoingx p4_pass0.coarse): every exllamav3 standard-text row is "std"
    (its code.utf8 would otherwise merge into the prompt domain "code"), multilingual_* is one group."""
    d = str(rec.get("domain", "all"))
    if rec.get("source") == "exllamav3_std":
        return "std"
    return "multilingual" if d.startswith("multilingual") else COARSE.get(d, d)


# ------------------------------------------------------------------------------------------------- statistics
def percentile(xs: list[float], q: float) -> float:
    """Linear-interpolation percentile (numpy's default), q in [0, 100]."""
    s = sorted(xs)
    if not s:
        raise ValueError("percentile of nothing")
    pos = (len(s) - 1) * q / 100.0
    lo = int(pos)
    hi = min(lo + 1, len(s) - 1)
    return s[lo] + (s[hi] - s[lo]) * (pos - lo)


def median(xs: list[float]) -> float:
    return percentile(xs, 50.0)


def bands_from_layers(probe: list[int], first: int, last: int) -> dict[str, int]:
    """Each probe layer stands for the layers closer to it than to its neighbours: {"a-b": probe layer}."""
    p = sorted(probe)
    out = {}
    for i, l in enumerate(p):
        a = first if i == 0 else int((p[i - 1] + l) // 2) + 1
        b = last if i == len(p) - 1 else int((l + p[i + 1]) // 2)
        out[f"{a}-{b}"] = l
    return out


def relerr_table(records: list[dict], first: int, last: int) -> dict:
    """p1b_relerr.py records ({"layer", "K", "gate_up", "down", "all"}: relative output errors sqrt(err / energy) of
    one expert) -> the table p2_allocate.py --rel-err reads: medians pooled ("rel_err"), per band ("band", the probe
    layer standing for its neighbours) and per probe layer split gate/up vs down ("detail")."""
    layers = sorted({r["layer"] for r in records})
    ks = sorted({int(r["K"]) for r in records})

    def med(rs: list[dict], key: str) -> float:
        return median([r[key] for r in rs])
    table = {"rel_err": {str(k): med([r for r in records if r["K"] == k], "all") for k in ks}, "band": {},
             "detail": {}}
    for band, l in bands_from_layers(layers, first, last).items():
        table["band"][band] = {str(k): med([r for r in records if r["layer"] == l and r["K"] == k], "all") for k in ks}
    for l in layers:
        table["detail"][str(l)] = {str(k): {key: med([r for r in records if r["layer"] == l and r["K"] == k], key)
                                            for key in ("gate_up", "down", "all")}
                                   | {"n": sum(1 for r in records if r["layer"] == l and r["K"] == k)} for k in ks}
    return table


def parse_variants(spec: str) -> list[tuple[str, dict[str, int]]]:
    """p4_dense.py --variants "5/6/6,6/8/8" (attn/shared/mlp K) -> [("5-6-6", {"attn": 5, "shared": 6, "mlp": 6}), ...]."""
    out = []
    for v in filter(None, spec.split(",")):
        ks = [int(k) for k in v.split("/")]
        if len(ks) != 3 or not all(2 <= k <= 8 for k in ks):
            raise SystemExit(f"--variants {v!r}: need attn/shared/mlp K, each 2..8")
        out.append(("-".join(map(str, ks)), dict(zip(("attn", "shared", "mlp"), ks))))
    return out


def spread_pick(counts: list[int], n: int) -> list[int]:
    """n experts spread over the usage range: the routed experts (count > 0) sorted by count, taken at evenly spaced
    ranks (both ends included); all of them if there are at most n. Sorted by expert id."""
    used = sorted((c, e) for e, c in enumerate(counts) if c > 0)
    if len(used) <= n:
        return sorted(e for _, e in used)
    if n <= 1:
        return [used[len(used) // 2][1]][:n]
    return sorted({used[round(i * (len(used) - 1) / (n - 1))][1] for i in range(n)})
