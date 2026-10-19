#!/usr/bin/env python3
# Verbatim copy of 0xSero/Step-5-Preview-Four-Sparks@662949a scripts/verify.py (MIT License, Copyright (c) 2026 0xSero).
"""Check a downloaded Step-5-Preview-Spark checkpoint directory before serving it. Stdlib only.

  python3 scripts/verify.py [--sha] [--body-format hybrid|bf16|exl3] MODEL_DIR
  python3 - [--sha] [--body-format F] MODEL_DIR < scripts/verify.py   (how copy-to-peers.sh / launch.sh run it remotely)

Checks:
  - config.json is the Step-5 Spark layout (model_type step5, quantization_config.quant_method step5_exl3)
  - every shard named in model.safetensors.index.json exists (BF16 shards model-*.safetensors and the BF16 body
    files body-bf16-*.safetensors)
  - --body-format hybrid (default, the release setting) or bf16: the BF16 body files are present and the index maps
    the body projections of every layer in quantization_config.body_layers to them (exl3: not required)
  - exl3/experts/LNN.safetensors for every layer in quantization_config.expert_layers,
    exl3/body/LNN.safetensors for every layer in quantization_config.body_layers
  - tokenizer files and chat_template.jinja are present
  - every .safetensors file has a parseable header and the size its header implies (catches truncated downloads)
  - with --sha: sha256 of every file listed in sha256-manifest.txt (slow: reads all 341.4 GB). Without the manifest
    this is a failure ("no manifest: checksums NOT verified"); download.sh / copy-to-peers.sh accept that only with
    --no-verify, which drops --sha. Without --sha the sha256 pass does not run.
Exit code 0 when everything passes; prints one line per problem otherwise.
"""
import hashlib, json, os, struct, sys

import re

SHA, BODY_FORMAT, args = False, "hybrid", []
argv = sys.argv[1:]
while argv:
    a = argv.pop(0)
    if a == "--sha":
        SHA = True
    elif a == "--body-format" and argv:
        BODY_FORMAT = argv.pop(0)
    else:
        args.append(a)
if len(args) != 1 or BODY_FORMAT not in ("hybrid", "bf16", "exl3", "fp8"):
    sys.exit(__doc__)
D = os.path.abspath(os.path.expanduser(args[0]))
bad = []
NO_MANIFEST = "no manifest: checksums NOT verified (sha256-manifest.txt is missing here; re-run scripts/download.sh " \
    "to fetch it, or pass --no-verify to download.sh / copy-to-peers.sh to accept header/size checks only)"


def p(*parts):
    return os.path.join(D, *parts)


def check_safetensors(path):
    try:
        size = os.path.getsize(path)
        with open(path, "rb") as f:
            (hlen,) = struct.unpack("<Q", f.read(8))
            if hlen <= 0 or hlen > 100 * 2**20:
                return f"bad header length {hlen}"
            hdr = json.loads(f.read(hlen))
        end = max((v["data_offsets"][1] for k, v in hdr.items() if k != "__metadata__"), default=0)
        if 8 + hlen + end != size:
            return f"size {size} != {8 + hlen + end} implied by its header (truncated?)"
    except Exception as e:  # noqa: BLE001
        return f"unreadable: {e}"
    return None


try:
    cfg = json.load(open(p("config.json")))
except Exception as e:  # noqa: BLE001
    sys.exit(f"FAIL {D}: config.json: {e}")
qc = cfg.get("quantization_config") or {}
if cfg.get("model_type") != "step5" or qc.get("quant_method") != "step5_exl3":
    bad.append(f"config.json: expected model_type step5 / quant_method step5_exl3, got "
               f"{cfg.get('model_type')} / {qc.get('quant_method')}")

try:
    wm = json.load(open(p("model.safetensors.index.json")))["weight_map"]
    shards = sorted(set(wm.values()))
except Exception as e:  # noqa: BLE001
    bad.append(f"model.safetensors.index.json: {e}"); wm, shards = {}, []

expected = [s for s in shards]
expected += [f"exl3/experts/L{L:02d}.safetensors" for L in qc.get("expert_layers", [])]
expected += [f"exl3/body/L{L:02d}.safetensors" for L in qc.get("body_layers", [])]
if not qc.get("expert_layers") or not qc.get("body_layers"):
    bad.append("config.json: quantization_config lists no expert_layers / body_layers")

# BF16 body files (hybrid body: BF16 weights for prefill-sized batches, EXL3 for decode).
BODY_RE = re.compile(r"^model\.layers\.(\d+)\.(self_attn\.[qkvo]_proj|share_expert\.(gate|up|down)_proj|"
                     r"mlp\.(gate|up|down)_proj)\.weight$")
bf16_body = sorted({f for f in shards if os.path.basename(f).startswith("body-bf16-")})
if BODY_FORMAT in ("hybrid", "bf16", "fp8"):
    if not bf16_body:
        bad.append(f"no body-bf16-*.safetensors in model.safetensors.index.json: --body-format {BODY_FORMAT} needs the "
                   "BF16 body files (download the full repo, or serve with BODY_FORMAT=exl3)")
    else:
        covered = {int(m.group(1)) for k, f in wm.items() if f in bf16_body and (m := BODY_RE.match(k))}
        missing = sorted(set(qc.get("body_layers", [])) - covered)
        if missing:
            bad.append(f"BF16 body files cover no projections for body layers {missing[:8]}{'...' if len(missing) > 8 else ''}")
for f in sorted(os.listdir(D)) if os.path.isdir(D) else []:
    if f.startswith("body-bf16-") and f.endswith(".safetensors") and f not in bf16_body:
        bad.append(f"{f} is on disk but not in model.safetensors.index.json (mixed checkpoint revisions?)")

for name in ["tokenizer_config.json", "chat_template.jinja"]:
    if not os.path.exists(p(name)):
        bad.append(f"missing {name}")
if not (os.path.exists(p("tokenizer.json")) or os.path.exists(p("tokenizer.model"))):
    bad.append("missing tokenizer.json / tokenizer.model")

total = 0
for rel in expected:
    f = p(rel)
    if not os.path.exists(f):
        bad.append(f"missing {rel}"); continue
    total += os.path.getsize(f)
    err = check_safetensors(f)
    if err:
        bad.append(f"{rel}: {err}")

if SHA:
    man = p("sha256-manifest.txt")
    if not os.path.exists(man):
        bad.append(NO_MANIFEST)
    else:
        n_sha = 0
        for line in open(man):
            if not line.strip():
                continue
            want, rel = line.split(maxsplit=1); rel = rel.strip().lstrip("*")
            if not os.path.exists(p(rel)):
                bad.append(f"sha256: missing {rel}"); continue
            h = hashlib.sha256()
            with open(p(rel), "rb") as fh:
                for blk in iter(lambda: fh.read(64 * 2**20), b""):
                    h.update(blk)
            n_sha += 1
            if h.hexdigest() != want:
                bad.append(f"sha256 mismatch: {rel}")
        if n_sha == 0:
            bad.append("sha256-manifest.txt lists no files: checksums NOT verified")
        print(f"sha256: {n_sha} files checked against sha256-manifest.txt")

for b in bad:
    print("FAIL", b)
n_exp, n_body = len(qc.get("expert_layers", [])), len(qc.get("body_layers", []))
print(f"{'OK  ' if not bad else 'FAIL'} {D}: {len(shards) - len(bf16_body)} BF16 shards, {len(bf16_body)} BF16 body files, "
      f"{n_exp} EXL3 expert files, {n_body} EXL3 body files, {total / 1e9:.1f} GB checked (body format {BODY_FORMAT})")
sys.exit(1 if bad else 0)
