# SPDX-License-Identifier: Apache-2.0
"""Assemble the Step 5 serving checkpoint for 0xSero's step5_spark plugin (body_format exl3) from the official FP8
snapshot, pass 1's part files and the dense pass's body files. Pure Python: tensors are copied byte for byte (no torch).
Ported in structure from sudoingx dsv41_recipe scripts/quantize/p4_assemble.py (Apache-2.0).

Output (plan "装配器必须产出"; plugin exl3_quant.py:1-13, 296, 157-160; verify.py):
  config.json                      official config; model_type step5, text_config.model_type step5_text, architectures
                                   unchanged, text_config.sparse_config REMOVED (the plugin applies SSMax whenever it is
                                   present, step5.py:531; the official kernels never do, inference/model.py:1449-1457),
                                   quantization_config {quant_method step5_exl3, expert_layers, body_layers, n_experts,
                                   body_format exl3}
  exl3/experts/L{LL}.safetensors   the layer's part files merged (keys model.layers.L.moe.experts.E.{gate,up,down}_proj.*)
  exl3/body/L{LL}.safetensors      body_LL from p4_dense.py (self_attn.{q,k,v,o}_proj, share_expert.* or mlp.*)
  model-NNNNN-of-MMMMM.safetensors every official tensor except: routed experts (+ weight_scale_inv) of the expert
                                   layers, the EXL3 body projections of the body layers, sparse_indexer_*, ssmax_s.
                                   Kept: embed, lm_head, norms, moe.gate, router_bias, g_proj, q_norm / k_norm, all of
                                   MTP 92-94, vision_model.*, vit_large_projector. A shard whose tensors are all kept is
                                   hardlinked; the others are rewritten.
  model.safetensors.index.json     BF16 tensors only; tokenizer files, chat_template.jinja, exl3_plan.json,
                                   sha256-manifest.txt (verify.py --sha)
--naming exllamav3 also writes --exllamav3-out: the same tensors with exllamav3 / Step-3.5 key names (identical to the
plugin's: exllamav3 architecture/step3_5.py), every file flat and in one index, quantization_config exl3-style.
The result is checked with the logic of 0xSero's scripts/verify.py (body format exl3).

  python3 p5_assemble.py /model /pass1_out /dense_out /plan/plan.json /out/serve [--reuse DIR] [--no-hash]
                         [--naming exllamav3 --exllamav3-out /out/serve-exl3v]
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import re
import shutil
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from p_util import (EXPERT_PROJS, atomic_json, body_modules, check_safetensors, k_of_trellis_shape,  # noqa: E402
                    model_layers, read_header, raw_entries, write_raw)

TOKENIZER_FILES = ["tokenizer.json", "tokenizer_config.json", "special_tokens_map.json", "chat_template.jinja",
                   "generation_config.json", "tokenizer.model", "LICENSE.txt", "LICENSE-CN.txt", "LICENSE"]
TOKENIZER_CLASS = "PreTrainedTokenizerFast"


def copy_tokenizer(src_dir: str, dst_dir: str) -> None:
    """Copy the tokenizer files; tokenizer_config.json gets tokenizer_class PreTrainedTokenizerFast. The official
    value "LlamaTokenizerFast" makes transformers 5.x AutoTokenizer (vLLM's default loader) build a Python
    LlamaTokenizer that drops newline, space and CJK tokens (quant-container check 2026-10-15; the official
    inference/generate.py:378-387 avoids automatic class selection for the same reason). With the generic fast class
    AutoTokenizer is id-identical to tokenizer.json."""
    for name in TOKENIZER_FILES:
        src = os.path.join(src_dir, name)
        if not os.path.exists(src):
            continue
        if name == "tokenizer_config.json":
            cfg = json.load(open(src))
            cfg["tokenizer_class"] = TOKENIZER_CLASS
            atomic_json(os.path.join(dst_dir, name), cfg, indent=2)
        else:
            shutil.copyfile(src, os.path.join(dst_dir, name))
_LAYER = re.compile(r"^model\.layers\.(\d+)\.(.+)$")
_EXPERT = re.compile(r"^moe\.(gate|up|down)_proj\.(weight|weight_scale_inv)$")
_BODY = re.compile(r"^(self_attn\.[qkvo]_proj|share_expert\.(gate|up|down)_proj|mlp\.(gate|up|down)_proj)\.weight$")
_DROP = re.compile(r"^self_attn\.(sparse_indexer_[^.]+\.(weight|bias)|ssmax_s)$")


# ------------------------------------------------------------------------------------------------- config
def transform_config(cfg: dict, expert_layers: list[int], body_layers: list[int], n_experts: int) -> dict:
    out = copy.deepcopy(cfg)
    out["model_type"] = "step5"
    t = out["text_config"]
    t["model_type"] = "step5_text"
    t.pop("sparse_config", None)
    out["quantization_config"] = {"quant_method": "step5_exl3", "expert_layers": list(expert_layers),
                                  "body_layers": list(body_layers), "n_experts": int(n_experts), "body_format": "exl3"}
    return out


# ------------------------------------------------------------------------------------------------- BF16 tensors
def keep_bf16(name: str, expert_layers: set[int], body_layers: set[int]) -> bool:
    """Does an official tensor go into the BF16 shards?"""
    m = _LAYER.match(name)
    if not m:
        return True
    l, rest = int(m.group(1)), m.group(2)
    if _DROP.match(rest):
        return False
    if l in expert_layers and _EXPERT.match(rest):
        return False
    if l in body_layers and _BODY.match(rest):
        return False
    return True


def bf16_groups(official: str, expert_layers: list[int], body_layers: list[int]) -> tuple[dict, dict]:
    """({source file: [kept keys]}, {source file: number of keys})."""
    wm = json.load(open(os.path.join(official, "model.safetensors.index.json")))["weight_map"]
    el, bl = set(expert_layers), set(body_layers)
    keep: dict[str, list[str]] = {}
    total: dict[str, int] = {}
    for k, f in wm.items():
        total[f] = total.get(f, 0) + 1
        if keep_bf16(k, el, bl):
            keep.setdefault(f, []).append(k)
    return keep, total


def write_bf16(official: str, out: str, keep: dict, total: dict, log) -> tuple[dict, int]:
    files = sorted(keep)
    n = len(files)
    weight_map, size = {}, 0
    for i, f in enumerate(files, 1):
        src = os.path.join(official, f)
        name = f"model-{i:05d}-of-{n:05d}.safetensors"
        dst = os.path.join(out, name)
        hdr, _ = read_header(src)
        fp8 = [k for k in keep[f] if hdr[k]["dtype"].startswith("F8")]
        if fp8:
            raise SystemExit(f"{f}: kept tensors are FP8 ({fp8[:3]}...): the serving loader expects BF16 there")
        want = set(keep[f])
        if not (os.path.exists(dst) and set(read_header(dst)[0]) == want and not check_safetensors(dst)):
            if len(want) == total[f]:
                if os.path.exists(dst):
                    os.remove(dst)
                try:
                    os.link(src, dst)
                    how = "hardlinked"
                except OSError:
                    shutil.copyfile(src, dst)
                    how = "copied"
            else:
                write_raw(dst, raw_entries(src, keep=lambda k: k in want), {"format": "pt"})
                how = "rewritten"
            log(f"[{i}/{n}] {f}: {len(want)} of {total[f]} tensors {how} -> {name}")
        for k in want:
            weight_map[k] = name
        size += sum(hdr[k]["data_offsets"][1] - hdr[k]["data_offsets"][0] for k in want)
    return weight_map, size


# ------------------------------------------------------------------------------------------------- EXL3 files
def _k_by_expert(hdr: dict, layer: int, n_experts: int) -> dict[int, set[int]]:
    ks: dict[int, set[int]] = {}
    for e in range(n_experts):
        for p in EXPERT_PROJS:
            pre = f"model.layers.{layer}.moe.experts.{e}.{p}."
            for piece in ("trellis", "suh", "svh"):
                if pre + piece not in hdr:
                    raise SystemExit(f"layer {layer}: missing {pre + piece}")
            ks.setdefault(e, set()).add(k_of_trellis_shape(hdr[pre + "trellis"]["shape"]))
    return ks


def merge_experts(parts: list[str], dst: str, layer: int, n_experts: int, plan_k: list[int] | None,
                  reuse_file: str = "") -> None:
    entries, seen = [], set()
    for p in parts:
        for ent in raw_entries(p):
            if ent[0] in seen:
                raise SystemExit(f"{p}: {ent[0]} also in another part")
            seen.add(ent[0])
            entries.append(ent)
    if reuse_file:                      # --reuse: experts the parts do not hold come from the old checkpoint
        have = {int(k.split(".")[5]) for k in seen}
        entries += raw_entries(reuse_file, keep=lambda k: int(k.split(".")[5]) not in have)
    hdr = {k: {"shape": s} for k, _, s, _ in entries}
    bad = [k for k in hdr if not k.startswith(f"model.layers.{layer}.moe.experts.")]
    if bad:
        raise SystemExit(f"layer {layer}: foreign keys in the parts: {bad[:3]}")
    for e, ks in _k_by_expert(hdr, layer, n_experts).items():
        if len(ks) != 1:
            raise SystemExit(f"layer {layer} expert {e}: mixed K {sorted(ks)} (the plugin needs one K per expert)")
        if plan_k is not None and ks != {plan_k[e]}:
            raise SystemExit(f"layer {layer} expert {e}: K {ks} but the plan says {plan_k[e]}")
    write_raw(dst, entries, {"format": "pt", "quant": "exl3", "encoder": "exllamav3 1.6.0"})


def copy_body(src: str, dst: str, layer: int, is_moe: bool) -> list[str]:
    """Copy one layer's body file; returns warnings (suh mismatch = the plugin will not fuse that group)."""
    hdr, base = read_header(src)
    for m in body_modules(is_moe):
        for piece in ("trellis", "suh", "svh"):
            if f"model.layers.{layer}.{m}.{piece}" not in hdr:
                raise SystemExit(f"{src}: missing model.layers.{layer}.{m}.{piece}")
    extra = [k for k in hdr if not k.startswith(f"model.layers.{layer}.")]
    if extra:
        raise SystemExit(f"{src}: foreign keys {extra[:3]}")
    warn = []
    ffn = "share_expert" if is_moe else "mlp"
    for group in (("self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj"), (f"{ffn}.gate_proj", f"{ffn}.up_proj")):
        blobs = []
        with open(src, "rb") as f:
            for m in group:
                a, b = hdr[f"model.layers.{layer}.{m}.suh"]["data_offsets"]
                f.seek(base + a)
                blobs.append(f.read(b - a))
        if any(x != blobs[0] for x in blobs[1:]):
            warn.append(f"layer {layer}: {group} suh differ (no fused GEMV)")
    write_raw(dst, raw_entries(src), {"format": "pt", "quant": "exl3", "encoder": "exllamav3 1.6.0"})
    return warn


# ------------------------------------------------------------------------------------------------- verify
def verify_dir(d: str, body_format: str = "exl3", sha: bool = False) -> list[str]:
    """0xSero scripts/verify.py's checks (Step-5-Preview-Four-Sparks, read in full), as a function."""
    bad = []
    p = lambda *x: os.path.join(d, *x)  # noqa: E731
    try:
        cfg = json.load(open(p("config.json")))
    except Exception as e:  # noqa: BLE001
        return [f"config.json: {e}"]
    qc = cfg.get("quantization_config") or {}
    if cfg.get("model_type") != "step5" or qc.get("quant_method") != "step5_exl3":
        bad.append(f"config.json: expected model_type step5 / quant_method step5_exl3, got "
                   f"{cfg.get('model_type')} / {qc.get('quant_method')}")
    try:
        wm = json.load(open(p("model.safetensors.index.json")))["weight_map"]
        shards = sorted(set(wm.values()))
    except Exception as e:  # noqa: BLE001
        bad.append(f"model.safetensors.index.json: {e}")
        wm, shards = {}, []
    expected = list(shards)
    expected += [f"exl3/experts/L{L:02d}.safetensors" for L in qc.get("expert_layers", [])]
    expected += [f"exl3/body/L{L:02d}.safetensors" for L in qc.get("body_layers", [])]
    if not qc.get("expert_layers") or not qc.get("body_layers"):
        bad.append("config.json: quantization_config lists no expert_layers / body_layers")
    bf16_body = sorted({f for f in shards if os.path.basename(f).startswith("body-bf16-")})
    if body_format in ("hybrid", "bf16", "fp8") and not bf16_body:
        bad.append(f"no body-bf16-*.safetensors: --body-format {body_format} needs the BF16 body files")
    for f in sorted(os.listdir(d)):
        if f.startswith("body-bf16-") and f.endswith(".safetensors") and f not in bf16_body:
            bad.append(f"{f} is on disk but not in model.safetensors.index.json")
    for name in ("tokenizer_config.json", "chat_template.jinja"):
        if not os.path.exists(p(name)):
            bad.append(f"missing {name}")
    if not (os.path.exists(p("tokenizer.json")) or os.path.exists(p("tokenizer.model"))):
        bad.append("missing tokenizer.json / tokenizer.model")
    for rel in expected:
        if not os.path.exists(p(rel)):
            bad.append(f"missing {rel}")
            continue
        err = check_safetensors(p(rel))
        if err:
            bad.append(f"{rel}: {err}")
    if sha:
        man = p("sha256-manifest.txt")
        if not os.path.exists(man):
            bad.append("no manifest: checksums NOT verified")
        else:
            for line in filter(str.strip, open(man)):
                want, rel = line.split(maxsplit=1)
                rel = rel.strip().lstrip("*")
                if not os.path.exists(p(rel)):
                    bad.append(f"sha256: missing {rel}")
                elif sha256(p(rel)) != want:
                    bad.append(f"sha256 mismatch: {rel}")
    return bad


def sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for blk in iter(lambda: f.read(64 << 20), b""):
            h.update(blk)
    return h.hexdigest()


def write_manifest(d: str) -> None:
    lines = []
    for root, _, files in os.walk(d):
        for f in sorted(files):
            rel = os.path.relpath(os.path.join(root, f), d)
            if rel != "sha256-manifest.txt" and not f.endswith(".tmp"):
                lines.append(f"{sha256(os.path.join(d, rel))}  {rel}")
    with open(os.path.join(d, "sha256-manifest.txt"), "w") as fh:
        fh.write("\n".join(sorted(lines, key=lambda x: x.split("  ", 1)[1])) + "\n")


# ------------------------------------------------------------------------------------------------- exllamav3 layout
def exllamav3_layout(out: str, dst: str, official_cfg: dict, weight_map: dict, plan: dict, log) -> None:
    """Every tensor in one index, flat files (hardlinks into `out`); keys unchanged (exllamav3's Step-3.5 names =
    the plugin's). Config: official, quantization_config exl3-style (exllamav3 conversion/compile.py:196-242)."""
    os.makedirs(dst, exist_ok=True)

    def link(src: str, name: str) -> None:
        d = os.path.join(dst, name)
        if os.path.exists(d):
            os.remove(d)
        try:
            os.link(src, d)
        except OSError:
            shutil.copyfile(src, d)
    wm = {}
    for f in sorted(set(weight_map.values())):
        link(os.path.join(out, f), f)
    wm.update(weight_map)
    for sub in ("experts", "body"):
        for f in sorted(os.listdir(os.path.join(out, "exl3", sub))):
            name = f"model-exl3-{sub}-{f}"
            link(os.path.join(out, "exl3", sub, f), name)
            for k in read_header(os.path.join(dst, name))[0]:
                wm[k] = name
    ks = [k for v in plan["layers"].values() for k in v["k"]]
    cfg = copy.deepcopy(official_cfg)
    cfg["quantization_config"] = {"quant_method": "exl3", "version": "1.6.0", "bits": round(sum(ks) / len(ks), 4),
                                  "head_bits": 16, "codebook": "mul1",
                                  "original_quantization_config": official_cfg.get("quantization_config")}
    atomic_json(os.path.join(dst, "config.json"), cfg, indent=2)
    atomic_json(os.path.join(dst, "model.safetensors.index.json"),
                {"metadata": {"format": "exllamav3-style names (step5 p5_assemble.py)"},
                 "weight_map": dict(sorted(wm.items()))}, indent=1)
    for f in TOKENIZER_FILES:
        if os.path.exists(os.path.join(out, f)):
            link(os.path.join(out, f), f)
    log(f"exllamav3 layout: {len(wm)} tensors in {len(set(wm.values()))} files -> {dst}")


# ------------------------------------------------------------------------------------------------- main
def assemble(official: str, pass1: str, dense: str, plan_path: str, out: str, reuse: str = "",
             no_hash: bool = False, naming: str = "sero", exl3v_out: str = "", log=print) -> list[str]:
    os.makedirs(os.path.join(out, "exl3", "experts"), exist_ok=True)
    os.makedirs(os.path.join(out, "exl3", "body"), exist_ok=True)
    cfg = json.load(open(os.path.join(official, "config.json")))
    plan = json.load(open(plan_path))
    expert_layers, body_layers, n_experts = model_layers(cfg)
    keep, total = bf16_groups(official, expert_layers, body_layers)
    weight_map, size = write_bf16(official, out, keep, total, log)
    atomic_json(os.path.join(out, "model.safetensors.index.json"),
                {"metadata": {"total_size": size, "format": "step5_exl3 BF16 tensors (p5_assemble.py)"},
                 "weight_map": dict(sorted(weight_map.items()))}, indent=1)
    for l in expert_layers:
        parts = sorted(os.path.join(pass1, f) for f in os.listdir(pass1)
                       if re.fullmatch(rf"layer_{l:02d}\.part\d+\.safetensors", f))
        rf = os.path.join(reuse, "exl3", "experts", f"L{l:02d}.safetensors") if reuse else ""
        if not parts and not rf:
            raise SystemExit(f"layer {l}: no pass-1 part files in {pass1}")
        merge_experts(parts, os.path.join(out, "exl3", "experts", f"L{l:02d}.safetensors"), l, n_experts,
                      [int(k) for k in plan["layers"][str(l)]["k"]] if str(l) in plan.get("layers", {}) else None, rf)
        log(f"experts L{l:02d}: {len(parts)} part file(s)" + (" + reuse" if rf else ""))
    warns = []
    for l in body_layers:
        src = os.path.join(dense, f"body_{l:02d}.safetensors")
        if not os.path.exists(src):
            raise SystemExit(f"layer {l}: no {src}")
        warns += copy_body(src, os.path.join(out, "exl3", "body", f"L{l:02d}.safetensors"), l, l in expert_layers)
    atomic_json(os.path.join(out, "config.json"), transform_config(cfg, expert_layers, body_layers, n_experts),
                indent=2)
    copy_tokenizer(official, out)
    shutil.copyfile(plan_path, os.path.join(out, "exl3_plan.json"))
    if not no_hash:
        write_manifest(out)
    bad = verify_dir(out, "exl3")
    if naming == "exllamav3":
        exllamav3_layout(out, exl3v_out or out.rstrip("/") + "-exllamav3", cfg, weight_map, plan, log)
    for w in warns:
        log("WARN " + w)
    return bad


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("official")
    ap.add_argument("pass1", help="p3_pass1.py output dir (layer_LL.partI.safetensors)")
    ap.add_argument("dense", help="p4_dense.py output dir (body_LL.safetensors)")
    ap.add_argument("plan")
    ap.add_argument("out")
    ap.add_argument("--reuse", default="", help="assembled dir supplying experts the part files do not hold")
    ap.add_argument("--no-hash", action="store_true")
    ap.add_argument("--naming", choices=["sero", "exllamav3"], default="sero")
    ap.add_argument("--exllamav3-out", default="")
    a = ap.parse_args(argv)
    bad = assemble(a.official, a.pass1, a.dense, a.plan, a.out, a.reuse, a.no_hash, a.naming, a.exllamav3_out)
    for b in bad:
        print("FAIL", b)
    print(("OK   " if not bad else "FAIL ") + a.out)
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
