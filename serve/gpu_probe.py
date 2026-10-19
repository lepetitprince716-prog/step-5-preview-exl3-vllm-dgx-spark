#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Helpers for serve/gpu_checks.sh (one subcommand per measurement; each prints PASS/FAIL lines with the numbers).

  sigs [--label serve|quant]   exllamav3 version, exl3_moe / exl3_moe_coop binding docstrings, EXL3_INT8_GEMV and the
                               int8 GEMV max K, whether quantize.refit_scales exists
  decode URL MODEL [--tokens 256] [--reps 3] [--key-file F]
                               (host, stdlib) greedy /v1/completions stream with ignore_eos: TTFT and decode tok/s
  kvlog                        (host, stdlib) KV-pool lines of a vLLM log on stdin
  ssmax /model                 (quant image, CPU) the 23 self_attn.ssmax_s tensors: min / max / mean per layer
  xver-encode OUT [--ks 2,3,4,5,6,8]  (quant image, exllamav3 1.6.0, GPU) encode one gate-shaped expert linear per K with
                               step5q.exl3 (README contract: h_data, encode_linear, reconstruct), save pieces + the 1.6.0
                               reconstruction to OUT/xver_K{K}.safetensors
  xver-decode OUT              (serving image, exllamav3 1.5.3, GPU) rebuild each with LinearEXL3 the way the plugin
                               builds it (exl3_quant.py:84-89) and compare with the 1.6.0 reconstruction
  tokcheck /model              (serving or quant image, CPU) ids from vLLM's get_tokenizer / AutoTokenizer vs
                               tokenizer.json on a newline + CJK + digits sample (check 6)
"""
from __future__ import annotations

import argparse
import glob
import json
import math
import os
import re
import statistics
import sys
import time
import urllib.request

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
XVER_IN, XVER_OUT = 4096, 1536          # a routed expert's gate_proj (in, out)
KV_RE = re.compile(r"GPU KV cache size: [0-9,]+ tokens|Available KV cache memory: [0-9.]+ GiB|"
                   r"Maximum concurrency for [0-9,]+ tokens per request: [0-9.]+x|"
                   r"KV cache memory[^\n]{0,80}|Loading weights took [0-9.]+ seconds")


def _version(pkg: str) -> str:
    import importlib.metadata as md
    try:
        return md.version(pkg)
    except md.PackageNotFoundError:
        return "absent"


def int8_gemv_info(ext) -> str:
    """The int8 GEMV gate: EXL3_INT8_GEMV as the container sees it (unset: exllamav3 1.6.2's C++ default is mode 2,
    exl3_gemv_int8.cu:23-31), and the highest K the int8 path takes on device 0 (1.6.2: exl3_gemv_int8_max_k,
    6 on Hopper/Blackwell, else 5; EXL3_INT8_GEMV_MAX_K overrides; above it m=1 falls to the cooperative kernel)."""
    mode = os.environ.get("EXL3_INT8_GEMV", "unset")
    fn = getattr(ext, "exl3_gemv_int8_max_k", None)
    if fn is None:
        names = sorted(n for n in dir(ext) if "int8" in n or "gemv" in n)
        return f"EXL3_INT8_GEMV={mode}; exl3_gemv_int8_max_k ABSENT (int8/gemv symbols: {names})"
    try:
        max_k = fn(0)
    except Exception as e:  # noqa: BLE001  (needs a CUDA device)
        max_k = f"error: {e}"
    return f"EXL3_INT8_GEMV={mode}; exl3_gemv_int8_max_k(0)={max_k}; " \
           f"EXL3_INT8_GEMV_MAX_K={os.environ.get('EXL3_INT8_GEMV_MAX_K', 'unset')}"


def cmd_sigs(a: argparse.Namespace) -> int:
    try:
        import exllamav3_ext as ext                    # how the plugin imports it (exl3_quant.py:49-51)
    except ImportError:
        from exllamav3.ext import exllamav3_ext as ext
    from exllamav3.modules.quant.exl3_lib import quantize as q
    ver = _version("exllamav3")
    ok = hasattr(ext, "exl3_moe")
    print(f"[{a.label}] exllamav3 {ver}; vllm {_version('vllm')}; step5-spark {_version('step5-spark')}")
    for name in ("exl3_moe", "exl3_moe_coop", "exl3_moe_max_concurrency", "reconstruct"):
        f = getattr(ext, name, None)
        print(f"[{a.label}] {name}: {(f.__doc__ or '').strip() if f else 'ABSENT'}")
    gemv = int8_gemv_info(ext)
    refit = "present" if hasattr(q, "refit_scales") else "absent"
    print(f"[{a.label}] int8 GEMV: {gemv}")
    print(f"[{a.label}] exllamav3.modules.quant.exl3_lib.quantize.refit_scales: {refit}")
    if a.label == "serve":
        print(f"CHECK 1 {'PASS' if ok else 'FAIL'}: exllamav3 {ver}, exl3_moe {'present' if ok else 'missing'}, "
              f"exl3_moe_coop {'present' if hasattr(ext, 'exl3_moe_coop') else 'absent'}; {gemv}; refit_scales {refit}")
    return 0 if ok else 1


def _stream_once(url: str, model: str, tokens: int, key: str) -> tuple[float, float, int]:
    body = {"model": model, "prompt": "The history of", "max_tokens": tokens, "temperature": 0, "ignore_eos": True,
            "stream": True, "stream_options": {"include_usage": True}}
    hdr = {"Content-Type": "application/json", **({"Authorization": f"Bearer {key}"} if key else {})}
    req = urllib.request.Request(url.rstrip("/") + "/v1/completions", data=json.dumps(body).encode(), headers=hdr)
    t0 = time.perf_counter()
    t_first = t_last = None
    n = 0
    with urllib.request.urlopen(req, timeout=900) as r:
        for raw in r:
            line = raw.decode().strip()
            if not line.startswith("data: ") or line == "data: [DONE]":
                continue
            ev = json.loads(line[6:])
            if ev.get("usage"):
                n = int(ev["usage"].get("completion_tokens") or n)
            if ev.get("choices") and ev["choices"][0].get("text"):
                t_last = time.perf_counter()
                t_first = t_first or t_last
    if t_first is None or n < 2:
        raise RuntimeError(f"no tokens streamed (completion_tokens {n})")
    return t_first - t0, (n - 1) / max(t_last - t_first, 1e-9), n


def cmd_decode(a: argparse.Namespace) -> int:
    key = open(a.key_file).read().strip() if a.key_file else ""
    _stream_once(a.url, a.model, 16, key)                                          # warm-up
    runs = [_stream_once(a.url, a.model, a.tokens, key) for _ in range(a.reps)]
    ttft = statistics.median(r[0] for r in runs)
    tps = statistics.median(r[1] for r in runs)
    print(json.dumps({"decode_tok_s": round(tps, 2), "ttft_s": round(ttft, 3), "tokens": runs[0][2],
                      "runs_tok_s": [round(r[1], 2) for r in runs]}))
    return 0


def cmd_kvlog(_: argparse.Namespace) -> int:
    hits = KV_RE.findall(sys.stdin.read())
    for h in dict.fromkeys(hits):
        print(h)
    return 0 if hits else 1


def cmd_ssmax(a: argparse.Namespace) -> int:
    import torch
    from safetensors import safe_open
    wm = json.load(open(os.path.join(a.model, "model.safetensors.index.json")))["weight_map"]
    names = sorted((k for k in wm if k.endswith(".self_attn.ssmax_s")), key=lambda k: int(k.split(".")[2]))
    bad = []
    for k in names:
        with safe_open(os.path.join(a.model, wm[k]), framework="pt", device="cpu") as f:
            t = f.get_tensor(k)
        x = t.float()
        finite = bool(torch.isfinite(x).all())
        print(f"layer {int(k.split('.')[2]):2d} {str(t.dtype):14s} {list(t.shape)} min {x.min().item():+.6f} "
              f"max {x.max().item():+.6f} mean {x.mean().item():+.6f} zeros {int((x == 0).sum())}/{x.numel()}")
        if not finite:
            bad.append(k)
    ok = len(names) == 23 and not bad
    print(f"CHECK 3 {'PASS' if ok else 'FAIL'}: {len(names)} ssmax_s tensors (23 expected), "
          f"{len(bad)} with non-finite values")
    return 0 if ok else 1


TOK_SAMPLE = ("<|im_start|>user\n第 1 行：请把 12345 和 6789 相加。\n\nline 2:  two  spaces\tand a tab<|im_end|>\n"
              "<|im_start|>assistant\n<think>想一想\n</think>答案是 19134。<|im_end|>\n")


def cmd_tokcheck(a: argparse.Namespace) -> int:
    """Ids from the serving-side loader (vLLM's get_tokenizer, else transformers AutoTokenizer) vs tokenizer.json.
    transformers 5.x turns the official tokenizer_class "LlamaTokenizerFast" into a Python LlamaTokenizer that drops
    newline, space and CJK tokens; p5_assemble.copy_tokenizer rewrites the class to PreTrainedTokenizerFast."""
    import warnings
    warnings.filterwarnings("ignore")
    from tokenizers import Tokenizer
    raw = Tokenizer.from_file(os.path.join(a.model, "tokenizer.json")).encode(TOK_SAMPLE, add_special_tokens=False).ids
    try:
        from vllm.transformers_utils.tokenizer import get_tokenizer
        tok, how = get_tokenizer(a.model), "vllm.get_tokenizer"
    except ImportError:
        from transformers import AutoTokenizer
        tok, how = AutoTokenizer.from_pretrained(a.model), "transformers.AutoTokenizer"
    got = tok(TOK_SAMPLE, add_special_tokens=False)["input_ids"]
    cls = json.load(open(os.path.join(a.model, "tokenizer_config.json"))).get("tokenizer_class")
    ok = list(got) == list(raw)
    print(f"tokenizer_class {cls}: {how} -> {type(tok).__name__}, {len(got)} ids vs tokenizer.json {len(raw)} ids, "
          f"{'MATCH' if ok else 'MISMATCH'}")
    if not ok:
        first = next((i for i, (x, y) in enumerate(zip(got, raw)) if x != y), min(len(got), len(raw)))
        print(f"  first difference at index {first}: loader {got[first:first + 6]} vs tokenizer.json {raw[first:first + 6]}")
    return 0 if ok else 1


def cmd_xver_encode(a: argparse.Namespace) -> int:
    import importlib.metadata as md
    import torch
    from safetensors.torch import save_file
    sys.path.insert(0, ROOT)
    from step5q.exl3 import encode_linear, h_data, reconstruct
    os.makedirs(a.out, exist_ok=True)
    g = torch.Generator(device="cuda").manual_seed(20261015)
    w = torch.randn(XVER_OUT, XVER_IN, generator=g, device="cuda") * 0.02              # [out, in], nn.Linear layout
    x = torch.randn(8192, XVER_IN, generator=g, device="cuda")
    H_sum = x.t() @ x
    ver = md.version("exllamav3")
    for K in [int(k) for k in a.ks.split(",")]:
        t, pe = encode_linear(w, h_data(H_sum.clone(), x.shape[0], f"xver.K{K}"), K, 1000 + K, a.out)
        ref = reconstruct(t, XVER_IN, XVER_OUT)
        out = {k: v.contiguous().cpu() for k, v in t.items() if isinstance(v, torch.Tensor)}
        out["ref"] = ref.contiguous().cpu()
        save_file(out, os.path.join(a.out, f"xver_K{K}.safetensors"), {"exllamav3": ver, "K": str(K)})
        rel = ((ref.float() - w).norm() / w.norm()).item()
        print(f"K{K}: encoded with exllamav3 {ver}, trellis {list(t['trellis'].shape)}, proxy_err {pe:.4g}, "
              f"weight rel err {rel:.4f}")
    return 0


def _rebuild(t: dict, n_in: int, n_out: int):
    import torch
    from exllamav3.modules.quant.exl3 import LinearEXL3
    dev = torch.device("cuda")
    g = lambda n: t[n].to(dev) if n in t else None  # noqa: E731
    lin = LinearEXL3(config=None, in_features=n_in, out_features=n_out, scale=None, su=None, sv=None,
                     suh=g("suh"), svh=g("svh"), trellis=g("trellis"), mcg=g("mcg"), mul1=g("mul1"), bias=None,
                     out_dtype=torch.half, key="xver")
    if hasattr(lin, "get_weight_tensor"):
        return lin.get_weight_tensor().t()
    eye = torch.eye(n_in, dtype=torch.half, device=dev)
    return lin.forward(eye, {"reconstruct": True}).t()


def cmd_xver_decode(a: argparse.Namespace) -> int:
    import importlib.metadata as md
    from safetensors import safe_open
    files = sorted(glob.glob(os.path.join(a.out, "xver_K*.safetensors")))
    if not files:
        print(f"CHECK 5 FAIL: no xver_K*.safetensors in {a.out} (run xver-encode first)")
        return 1
    ver, worst = md.version("exllamav3"), 0.0
    for p in files:
        with safe_open(p, framework="pt", device="cpu") as f:
            t = {k: f.get_tensor(k) for k in f.keys()}
            meta = f.metadata() or {}
        got = _rebuild(t, XVER_IN, XVER_OUT).float().cpu()
        ref = t["ref"].float()
        diff = (got - ref).abs().max().item()
        worst = max(worst, diff if math.isfinite(diff) else math.inf)
        print(f"K{meta.get('K')}: encoded {meta.get('exllamav3')}, decoded {ver}: max |diff| {diff:.3g}, "
              f"ref max |w| {ref.abs().max().item():.3g}, bit-exact {diff == 0}")
    ok = worst == 0
    print(f"CHECK 5 {'PASS' if ok else 'FAIL'}: {len(files)} K values, worst max |diff| {worst:.3g} "
          f"(1.6.0 encode -> {ver} decode)")
    return 0 if ok else 1


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    sg = sub.add_parser("sigs")
    sg.add_argument("--label", choices=["serve", "quant"], default="serve",
                    help="serve: the serving image (prints the CHECK 1 line); quant: the encoder image, info only")
    d = sub.add_parser("decode")
    d.add_argument("url")
    d.add_argument("model")
    d.add_argument("--tokens", type=int, default=256)
    d.add_argument("--reps", type=int, default=3)
    d.add_argument("--key-file", default="")
    sub.add_parser("kvlog")
    s = sub.add_parser("ssmax")
    s.add_argument("model")
    e = sub.add_parser("xver-encode")
    e.add_argument("out")
    e.add_argument("--ks", default="2,3,4,5,6,8", help="expert K 2-5 plus body candidates 6 and 8")
    x = sub.add_parser("xver-decode")
    x.add_argument("out")
    t = sub.add_parser("tokcheck")
    t.add_argument("model")
    a = ap.parse_args()
    fn = {"sigs": cmd_sigs, "decode": cmd_decode, "kvlog": cmd_kvlog, "ssmax": cmd_ssmax,
          "xver-encode": cmd_xver_encode, "xver-decode": cmd_xver_decode, "tokcheck": cmd_tokcheck}[a.cmd]
    sys.exit(fn(a))


if __name__ == "__main__":
    main()
