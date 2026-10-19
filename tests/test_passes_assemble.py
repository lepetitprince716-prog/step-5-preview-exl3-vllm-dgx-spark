"""p5_assemble.py and the key names on a synthetic official checkpoint (pure Python, no torch, no checkpoint).
With the real official config / index present (STEP5_OFFICIAL_DIR, default the scratchpad clone), the config transform
and the BF16 keep-list are also checked against them; with 0xSero's verify.py present (STEP5_VERIFY_PY), the assembled
synthetic dir is also run through the original script."""
import json
import os
import re
import subprocess
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import p5_assemble as p5  # noqa: E402
from p_util import body_modules, model_layers, raw_entries, read_header, sero_body_key, sero_expert_key, write_raw  # noqa: E402

SCRATCH = os.environ.get("STEP5_SCRATCH", "/tmp/step5-test-repos")
OFFICIAL = os.environ.get("STEP5_OFFICIAL_DIR", os.path.join(SCRATCH, "Step-5-Preview-hf"))
VERIFY_PY = os.environ.get("STEP5_VERIFY_PY", os.path.join(SCRATCH, "Step-5-Preview-Four-Sparks/scripts/verify.py"))

N_LAYERS, MOE, FULL, MTP, NE, KS = 6, [3, 4], [3], [6], 4, {3: [2, 3, 4, 5], 4: [5, 5, 3, 3]}
IN, OUT = 32, 16           # tiny EXL3 matrices: trellis [IN/16, OUT/16, 16 K] int16


def t(dt, shape, fill=0):
    n = {"BF16": 2, "F16": 2, "F32": 4, "I16": 2, "I32": 4, "F8_E4M3": 1}[dt]
    for s in shape:
        n *= s
    return dt, list(shape), bytes([fill % 256]) * n


def official_names():
    """Synthetic official tensor names following the real index's patterns (scaled down)."""
    names = ["model.embed_tokens.weight", "model.norm.weight", "lm_head.weight", "vit_large_projector.weight",
             "vision_model.conv1.weight", "vision_model.ln_pre.weight"]
    for l in range(N_LAYERS + len(MTP)):
        p = f"model.layers.{l}."
        names += [p + "input_layernorm.weight", p + "post_attention_layernorm.weight"]
        names += [p + f"self_attn.{m}.weight" for m in ("q_proj", "k_proj", "v_proj", "o_proj", "g_proj",
                                                        "q_norm", "k_norm")]
        if l in MOE:
            names += [p + "moe.gate.weight", p + "moe.router_bias"]
            names += [p + f"moe.{m}.weight" for m in ("gate_proj", "up_proj", "down_proj")]
            if l == 3:                                   # FP8 layer: scales; layer 4 plays an 88-90 BF16 expert layer
                names += [p + f"moe.{m}.weight_scale_inv" for m in ("gate_proj", "up_proj", "down_proj")]
            names += [p + f"share_expert.{m}.weight" for m in ("gate_proj", "up_proj", "down_proj")]
        else:
            names += [p + f"mlp.{m}.weight" for m in ("gate_proj", "up_proj", "down_proj")]
        if l in FULL:
            names += [p + "self_attn.ssmax_s"] + [p + f"self_attn.sparse_indexer_{x}.weight"
                                                  for x in ("q", "k", "z", "w", "q_norm", "k_norm")]
            names += [p + "self_attn.sparse_indexer_k_norm.bias"]
        if l in MTP:
            names += [p + x for x in ("eh_proj.weight", "enorm.weight", "hnorm.weight",
                                      "transformer.shared_head.norm.weight", "transformer.shared_head.output.weight")]
    return names


def expected_bf16(names):
    """Independent statement of what the serving index must hold."""
    keep = []
    for n in names:
        m = re.match(r"model\.layers\.(\d+)\.(.*)", n)
        if not m:
            keep.append(n)
            continue
        l, rest = int(m.group(1)), m.group(2)
        if "sparse_indexer" in rest or rest.endswith("ssmax_s"):
            continue
        if l >= N_LAYERS:                                # MTP: everything
            keep.append(n)
            continue
        if rest in ("input_layernorm.weight", "post_attention_layernorm.weight", "moe.gate.weight",
                    "moe.router_bias", "self_attn.g_proj.weight", "self_attn.q_norm.weight", "self_attn.k_norm.weight"):
            keep.append(n)
    return set(keep)


def make_official(root):
    d = root / "official"
    d.mkdir()
    names = official_names()
    shard_of = {}
    for n in names:
        m = re.match(r"model\.layers\.(\d+)\.", n)
        if n.startswith("vision_model") or n.startswith("vit_"):
            shard_of[n] = "model-vit-00001.safetensors"
        elif m and int(m.group(1)) in MTP:
            shard_of[n] = "model-mtp3-full-00001.safetensors"
        elif m and int(m.group(1)) < 3:
            shard_of[n] = "model-00001.safetensors"
        else:
            shard_of[n] = "model-00002.safetensors"
    for f in sorted(set(shard_of.values())):
        ents = []
        for i, n in enumerate(k for k in names if shard_of[k] == f):
            dt = "F8_E4M3" if (".moe." in n and n.endswith("_proj.weight") and ".layers.3." in n) else (
                "F32" if n.endswith("scale_inv") or n.endswith("router_bias") else "BF16")
            ents.append((n, *t(dt, [4, 8], i)))
        write_raw(str(d / f), ents)
    (d / "model.safetensors.index.json").write_text(json.dumps({"metadata": {}, "weight_map": shard_of}))
    cfg = {"architectures": ["Step5ForConditionalGeneration"], "model_type": "step5_vl", "image_token_id": 128001,
           "vision_config": {"model_type": "perception_encoder", "width": 1536},
           "text_config": {"architectures": ["Step5ForCausalLM"], "model_type": "step5", "num_hidden_layers": N_LAYERS,
                           "moe_layer_list": MOE, "moe_num_experts": NE, "hidden_size": 4096,
                           "sparse_config": {"enabled": True}, "num_nextn_predict_layers": len(MTP)},
           "quantization_config": {"quant_method": "fp8", "weight_block_size": [128, 128]}}
    (d / "config.json").write_text(json.dumps(cfg))
    for f in ("tokenizer.json", "tokenizer_config.json", "chat_template.jinja", "special_tokens_map.json"):
        (d / f).write_text("{}")
    return d, names


def exl3_entries(key_fn, K, n_in=IN, n_out=OUT, suh_fill=7):
    return [(key_fn("trellis"), *t("I16", [n_in // 16, n_out // 16, 16 * K], K)),
            (key_fn("suh"), *t("F16", [n_in], suh_fill)), (key_fn("svh"), *t("F16", [n_out], 3)),
            (key_fn("mul1"), *t("I32", []))]


def make_passes(root, drop_expert=None, mixed=False):
    p1, dn = root / "p1", root / "dense"
    p1.mkdir()
    dn.mkdir()
    for l in MOE:
        for part in (0, 1):
            ents = []
            for e in range(NE):
                if e % 2 != part or (l, e) == drop_expert:
                    continue
                for j, p in enumerate(("gate_proj", "up_proj", "down_proj")):
                    K = KS[l][e] + (1 if mixed and j == 2 and e == 0 else 0)
                    ents += exl3_entries(lambda pc, p=p, e=e: sero_expert_key(l, e, p, pc), K)
            write_raw(str(p1 / f"layer_{l:02d}.part{part}.safetensors"), ents)
    for l in range(N_LAYERS):
        ents = []
        for m in body_modules(l in MOE):
            ents += exl3_entries(lambda pc, m=m: sero_body_key(l, m, pc), 4)
        write_raw(str(dn / f"body_{l:02d}.safetensors"), ents)
    plan = root / "plan.json"
    plan.write_text(json.dumps({"bpw_target": 3.5, "layers": {str(l): {"k": KS[l]} for l in MOE},
                                "body": {"attn": 4, "shared": 4, "mlp": 4, "head": "bf16"}}))
    return p1, dn, plan


@pytest.fixture
def built(tmp_path):
    off, names = make_official(tmp_path)
    p1, dn, plan = make_passes(tmp_path)
    out = tmp_path / "serve"
    bad = p5.assemble(str(off), str(p1), str(dn), str(plan), str(out), naming="exllamav3",
                      exl3v_out=str(tmp_path / "serve-exl3v"), log=lambda m: None)
    return tmp_path, off, names, out, bad


def test_config_transform():
    cfg = {"architectures": ["A"], "model_type": "step5_vl", "x": 1, "quantization_config": {"quant_method": "fp8"},
           "text_config": {"model_type": "step5", "sparse_config": {"enabled": True}, "num_hidden_layers": 92, "y": 2}}
    new = p5.transform_config(cfg, [3, 4], [0, 1, 2, 3, 4], 352)
    assert new["model_type"] == "step5" and new["text_config"]["model_type"] == "step5_text"
    assert "sparse_config" not in new["text_config"]
    assert new["quantization_config"] == {"quant_method": "step5_exl3", "expert_layers": [3, 4],
                                          "body_layers": [0, 1, 2, 3, 4], "n_experts": 352, "body_format": "exl3"}
    assert new["architectures"] == ["A"] and new["x"] == 1 and new["text_config"]["y"] == 2
    assert cfg["text_config"]["sparse_config"] == {"enabled": True}          # input untouched


@pytest.mark.skipif(not os.path.exists(os.path.join(OFFICIAL, "config.json")), reason="official config not present")
def test_real_official_config_and_keep_list():
    cfg = json.load(open(os.path.join(OFFICIAL, "config.json")))
    el, bl, ne = model_layers(cfg)
    assert el == list(range(3, 91)) and bl == list(range(92)) and ne == 352
    new = p5.transform_config(cfg, el, bl, ne)
    assert "sparse_config" not in new["text_config"] and new["architectures"] == cfg["architectures"]
    wm = json.load(open(os.path.join(OFFICIAL, "model.safetensors.index.json")))["weight_map"]
    kept = {k for k in wm if p5.keep_bf16(k, set(el), set(bl))}
    for k in wm:
        m = re.match(r"model\.layers\.(\d+)\.(.*)", k)
        if not m:
            assert k in kept, k                                   # embed, lm_head, norm, vision, projector
            continue
        l, rest = int(m.group(1)), m.group(2)
        dropped = bool("sparse_indexer" in rest or rest == "self_attn.ssmax_s"
                   or (3 <= l <= 90 and re.fullmatch(r"moe\.(gate|up|down)_proj\.(weight|weight_scale_inv)", rest))
                   or (l <= 91 and re.fullmatch(r"(self_attn\.[qkvo]_proj|share_expert\.\w+_proj|mlp\.\w+_proj)\.weight",
                                                rest)))
        assert (k not in kept) == dropped, k
    assert sum(1 for k in kept if re.match(r"model\.layers\.9[234]\.", k)) == 51     # MTP in full
    assert sum(1 for k in kept if k.startswith("vision_model.")) == 666


def test_assembled_layout(built):
    root, off, names, out, bad = built
    assert bad == []
    wm = json.load(open(out / "model.safetensors.index.json"))["weight_map"]
    assert set(wm) == expected_bf16(names)                        # BF16 index: exactly the expected names
    cfg = json.load(open(out / "config.json"))
    assert cfg["quantization_config"]["expert_layers"] == MOE
    assert cfg["quantization_config"]["body_layers"] == list(range(N_LAYERS))
    for l in MOE:
        hdr, _ = read_header(str(out / "exl3" / "experts" / f"L{l:02d}.safetensors"))
        want = {sero_expert_key(l, e, p, pc) for e in range(NE) for p in ("gate_proj", "up_proj", "down_proj")
                for pc in ("trellis", "suh", "svh", "mul1")}
        assert set(hdr) == want
        assert hdr[sero_expert_key(l, 0, "gate_proj", "trellis")]["shape"][-1] == 16 * KS[l][0]
    for l in range(N_LAYERS):
        hdr, _ = read_header(str(out / "exl3" / "body" / f"L{l:02d}.safetensors"))
        assert {k.rsplit(".", 1)[0] for k in hdr} == {f"model.layers.{l}.{m}" for m in body_modules(l in MOE)}
    src = {f: os.stat(off / f).st_ino for f in ("model-vit-00001.safetensors", "model-mtp3-full-00001.safetensors")}
    linked = {os.stat(out / f).st_ino for f in set(wm.values())}
    assert set(src.values()) <= linked                            # fully kept shards are hardlinked
    assert (out / "sha256-manifest.txt").exists() and (out / "exl3_plan.json").exists()
    assert p5.verify_dir(str(out), "exl3", sha=True) == []


def test_exllamav3_layout(built):
    root, _, names, out, _ = built
    d = root / "serve-exl3v"
    wm = json.load(open(d / "model.safetensors.index.json"))["weight_map"]
    exl3 = {k for f in os.listdir(out / "exl3" / "experts") for k in read_header(str(out / "exl3" / "experts" / f))[0]}
    exl3 |= {k for f in os.listdir(out / "exl3" / "body") for k in read_header(str(out / "exl3" / "body" / f))[0]}
    assert set(wm) == expected_bf16(names) | exl3                 # one index over everything, same key names
    for f in set(wm.values()):
        assert (d / f).exists() and "/" not in f
    assert json.load(open(d / "config.json"))["quantization_config"]["quant_method"] == "exl3"


@pytest.mark.skipif(not os.path.exists(VERIFY_PY), reason="0xSero verify.py not present")
def test_original_verify_py(built):
    _, _, _, out, _ = built
    r = subprocess.run([sys.executable, VERIFY_PY, "--sha", "--body-format", "exl3", str(out)], capture_output=True,
                       text=True)
    assert r.returncode == 0, r.stdout + r.stderr


def test_verify_detects_problems(built):
    _, _, _, out, _ = built
    f = out / "exl3" / "body" / "L02.safetensors"
    f.write_bytes(f.read_bytes()[:-1])                            # truncated download
    os.remove(out / "chat_template.jinja")
    bad = p5.verify_dir(str(out), "exl3")
    assert any("L02" in b and "truncated" in b for b in bad) and any("chat_template" in b for b in bad)


def test_missing_expert_fails(tmp_path):
    off, _ = make_official(tmp_path)
    p1, dn, plan = make_passes(tmp_path, drop_expert=(4, 1))
    with pytest.raises(SystemExit, match="missing"):
        p5.assemble(str(off), str(p1), str(dn), str(plan), str(tmp_path / "o"), log=lambda m: None)


def test_mixed_k_fails(tmp_path):
    off, _ = make_official(tmp_path)
    p1, dn, plan = make_passes(tmp_path, mixed=True)
    with pytest.raises(SystemExit, match="mixed K"):
        p5.assemble(str(off), str(p1), str(dn), str(plan), str(tmp_path / "o"), log=lambda m: None)


def test_body_suh_mismatch_warns(tmp_path):
    src = tmp_path / "b.safetensors"
    ents = []
    for i, m in enumerate(body_modules(False)):
        ents += exl3_entries(lambda pc, m=m: sero_body_key(0, m, pc), 4, suh_fill=i)
    write_raw(str(src), ents)
    warn = p5.copy_body(str(src), str(tmp_path / "L00.safetensors"), 0, False)
    assert len(warn) == 2


def test_key_names_match_plugin():
    """The keys the plugin reads (exl3_quant.py:156-160 body, :296 experts; read, not imported: needs vllm)."""
    module = "moe.share_expert.gate_up_proj"                     # vLLM prefix of the fused shared-expert linear
    base, leaf = module.rsplit(".", 1)
    base = base.replace("moe.share_expert", "share_expert")
    parts = {"qkv_proj": ("q_proj", "k_proj", "v_proj"), "gate_up_proj": ("gate_proj", "up_proj")}[leaf]
    plugin = [f"model.layers.7.{base}.{p}.trellis" for p in parts]
    assert plugin == [sero_body_key(7, f"share_expert.{p}", "trellis") for p in parts]
    assert sero_expert_key(5, 351, "down_proj", "svh") == "model.layers.5.moe.experts.351.down_proj.svh"
    assert body_modules(True)[:4] == [f"self_attn.{p}_proj" for p in "qkvo"]


def test_raw_roundtrip(tmp_path):
    a = tmp_path / "a.safetensors"
    write_raw(str(a), [("x", *t("F16", [2, 3], 1)), ("y", *t("I32", []))], {"format": "pt"})   # numpy has no bf16
    b = tmp_path / "b.safetensors"
    write_raw(str(b), raw_entries(str(a), rename=lambda k: "z." + k))
    hdr, base = read_header(str(b))
    assert set(hdr) == {"z.x", "z.y"} and hdr["z.x"]["shape"] == [2, 3] and hdr["z.y"]["shape"] == []
    assert hdr["z.y"]["dtype"] == "I32" and hdr["z.y"]["data_offsets"] == [12, 16] and base % 8 == 0
    try:
        from safetensors.numpy import load_file
    except ImportError:
        return
    assert load_file(str(b))["z.y"].shape == ()
