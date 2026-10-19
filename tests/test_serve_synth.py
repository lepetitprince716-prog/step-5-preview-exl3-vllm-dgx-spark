"""serve/synth_ckpt.py: EXL3 shape math, the tensors it writes, and a tiny end-to-end checkpoint that passes 0xSero's
verify.py with --body-format exl3 (pure python)."""
import json
import os
import random
import struct
import subprocess
import sys

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "serve"))
import synth_ckpt as S  # noqa: E402
from p_util import check_safetensors, k_of_trellis_shape, read_header  # noqa: E402
from step5q import plan as P  # noqa: E402

TINY_EXPERT = {"gate_proj": (32, 16), "up_proj": (32, 16), "down_proj": (16, 32)}


def tiny_body(module):
    base, proj = module.split(".")
    return {"q_proj": (32, 32), "k_proj": (32, 16), "v_proj": (32, 16), "o_proj": (32, 32)}.get(proj) or \
        ((48, 32) if proj == "down_proj" else (32, 48))


def test_trellis_shape_and_bytes_match_plan_accounting():
    assert S.trellis_shape(4096, 1536, 3) == [256, 96, 48]
    assert S.trellis_shape(1536, 4096, 2) == [96, 256, 32]
    with pytest.raises(ValueError):
        S.trellis_shape(100, 16, 2)
    for K in P.EXPERT_K:
        for n_in, n_out in S.expert_dims().values():
            assert S.exl3_nbytes(n_in, n_out, K) == P.exl3_linear_bytes(n_in, n_out, K)
            assert S.trellis_shape(n_in, n_out, K)[0] * S.trellis_shape(n_in, n_out, K)[1] * 16 * K * 2 == \
                n_in * n_out * K // 8                                          # int16 words = K bits per weight


def test_body_dims():
    assert S.body_dims("self_attn.q_proj") == (4096, 12288) and S.body_dims("self_attn.k_proj") == (4096, 768)
    assert S.body_dims("self_attn.o_proj") == (12288, 4096)
    assert S.body_dims("share_expert.gate_proj") == (4096, 1536) and S.body_dims("share_expert.down_proj") == (1536, 4096)
    assert S.body_dims("mlp.up_proj") == (4096, 13824) and S.body_dims("mlp.down_proj") == (13824, 4096)
    for m in ("self_attn.q_proj", "self_attn.k_proj", "share_expert.gate_proj", "mlp.gate_proj"):
        assert (S.body_dims(m)[1] // 2) % 128 == 0                             # TP2 output split at 128-multiples


def test_k_mix_parse_and_counts():
    mix = S.parse_k_mix("2:0.55,3:0.35,4:0.10")
    ks = S.layer_ks(mix, 352, random.Random(0))
    assert len(ks) == 352 and {k: ks.count(k) for k in (2, 3, 4)} == {2: 194, 3: 123, 4: 35}
    assert ks != sorted(ks)                                                   # shuffled over expert ids
    assert S.layer_ks(S.parse_k_mix("2:1.0"), 352, random.Random(0)) == [2] * 352
    three = S.layer_ks(S.parse_k_mix("2:0.98,3:0.01,4:0.01"), 352, random.Random(0))
    assert len(set(three)) == 3
    for bad in ("2:0.5,3:0.4", "1:1.0", "2:0.25,3:0.25,4:0.25,5:0.25", "2:0.5,2:0.5", "2:1.5,3:-0.5"):
        with pytest.raises(ValueError):
            S.parse_k_mix(bad)


def test_make_plan_is_valid_and_deterministic():
    a = S.make_plan(S.parse_k_mix("2:0.6,3:0.4"), S.parse_body_k("4"), 7)
    assert a == S.make_plan(S.parse_k_mix("2:0.6,3:0.4"), S.parse_body_k("4/4/4"), 7)
    assert sorted(a["layers"], key=int) == [str(L) for L in P.MOE_LAYERS]
    assert abs(P.expert_bpw(a) - 2.4) < 0.01


def test_text_shape_table():
    t = {"hidden_size": 4096, "vocab_size": 128896, "head_dim": 192, "num_attention_heads": 64,
         "num_attention_groups": 4, "intermediate_size": 13824, "share_expert_dim": 1536, "moe_num_experts": 352}
    assert S.text_shape("model.embed_tokens.weight", t) == ("BF16", [128896, 4096])
    assert S.text_shape("model.layers.3.moe.gate.weight", t) == ("F32", [352, 4096])
    assert S.text_shape("model.layers.3.moe.router_bias", t) == ("F32", [352])
    assert S.text_shape("model.layers.7.self_attn.g_proj.weight", t) == ("BF16", [64, 4096])
    assert S.text_shape("model.layers.92.eh_proj.weight", t) == ("BF16", [4096, 8192])
    assert S.text_shape("model.layers.93.transformer.shared_head.output.weight", t) == ("BF16", [128896, 4096])
    assert S.text_shape("vision_model.x.weight", t) is None


def test_write_experts_tensors(tmp_path, monkeypatch):
    monkeypatch.setattr(S, "expert_dims", lambda: dict(TINY_EXPERT))
    os.makedirs(tmp_path / "exl3" / "experts")
    ks = [2, 3, 3, 4]
    S.write_experts(str(tmp_path), 5, ks, random.Random(0))
    path = str(tmp_path / "exl3" / "experts" / "L05.safetensors")
    assert check_safetensors(path) is None
    hdr, base = read_header(path)
    assert len(hdr) == len(ks) * 3 * 4
    for e, K in enumerate(ks):
        for p, (n_in, n_out) in TINY_EXPERT.items():
            pre = f"model.layers.5.moe.experts.{e}.{p}"
            assert hdr[pre + ".trellis"]["dtype"] == "I16" and hdr[pre + ".trellis"]["shape"] == [n_in // 16, n_out // 16, 16 * K]
            assert k_of_trellis_shape(hdr[pre + ".trellis"]["shape"]) == K
            assert hdr[pre + ".suh"] == {**hdr[pre + ".suh"], "dtype": "F16", "shape": [n_in]}
            assert hdr[pre + ".svh"]["dtype"] == "F16" and hdr[pre + ".svh"]["shape"] == [n_out]
            assert hdr[pre + ".mul1"]["dtype"] == "I32" and hdr[pre + ".mul1"]["shape"] == []
    a, b = hdr["model.layers.5.moe.experts.0.gate_proj.mul1"]["data_offsets"]
    with open(path, "rb") as f:
        f.seek(base + a)
        assert struct.unpack("<I", f.read(b - a))[0] == 0x83DCD12D
    a, b = hdr["model.layers.5.moe.experts.0.gate_proj.suh"]["data_offsets"]
    with open(path, "rb") as f:
        f.seek(base + a)
        vals = struct.unpack(f"<{(b - a) // 2}e", f.read(b - a))
    assert set(vals) <= {1.0, -1.0} and len(set(vals)) == 2


def test_write_body_shares_su_within_input_groups(tmp_path, monkeypatch):
    monkeypatch.setattr(S, "body_dims", tiny_body)
    os.makedirs(tmp_path / "exl3" / "body")
    for L, is_moe in ((2, False), (3, True)):
        S.write_body(str(tmp_path), L, is_moe, S.parse_body_k("4"), random.Random(L))
        path = str(tmp_path / "exl3" / "body" / f"L{L:02d}.safetensors")
        hdr, base = read_header(path)

        def raw(k):
            a, b = hdr[k]["data_offsets"]
            with open(path, "rb") as f:
                f.seek(base + a)
                return f.read(b - a)
        ffn = "share_expert" if is_moe else "mlp"
        pre = f"model.layers.{L}."
        assert raw(pre + "self_attn.q_proj.suh") == raw(pre + "self_attn.k_proj.suh") == raw(pre + "self_attn.v_proj.suh")
        assert raw(pre + f"{ffn}.gate_proj.suh") == raw(pre + f"{ffn}.up_proj.suh")
        assert hdr[pre + "self_attn.o_proj.trellis"]["shape"] == [2, 2, 64]
        assert not any(k.startswith(pre + ("mlp" if is_moe else "share_expert")) for k in hdr)


def tiny_meta(d):
    t = {"model_type": "step5", "hidden_size": 32, "vocab_size": 64, "head_dim": 16, "num_attention_heads": 2,
         "num_attention_groups": 1, "intermediate_size": 48, "share_expert_dim": 16, "moe_num_experts": 352,
         "num_hidden_layers": 92, "moe_layer_list": list(range(3, 91)), "sparse_config": {"enabled": True}}
    cfg = {"architectures": ["Step5ForConditionalGeneration"], "model_type": "step5_vl", "text_config": t,
           "vision_config": {"model_type": "perception_encoder"}, "quantization_config": {"quant_method": "fp8"}}
    names = ["model.embed_tokens.weight", "lm_head.weight", "model.norm.weight", "vision_model.x.weight"]
    for L in range(95):
        names += [f"model.layers.{L}.{n}" for n in ("input_layernorm.weight", "post_attention_layernorm.weight",
                                                     "self_attn.q_proj.weight", "self_attn.k_proj.weight",
                                                     "self_attn.v_proj.weight", "self_attn.o_proj.weight",
                                                     "self_attn.g_proj.weight", "self_attn.q_norm.weight",
                                                     "self_attn.k_norm.weight")]
        if 3 <= L <= 90:
            names += [f"model.layers.{L}.{n}" for n in ("moe.gate.weight", "moe.router_bias", "moe.gate_proj.weight",
                                                         "moe.up_proj.weight", "moe.down_proj.weight",
                                                         "moe.gate_proj.weight_scale_inv",
                                                         "share_expert.gate_proj.weight", "share_expert.up_proj.weight",
                                                         "share_expert.down_proj.weight")]
        else:
            names += [f"model.layers.{L}.mlp.{p}_proj.weight" for p in ("gate", "up", "down")]
        if L % 4 == 3 and L < 92:
            names += [f"model.layers.{L}.self_attn.ssmax_s", f"model.layers.{L}.self_attn.sparse_indexer_q.weight"]
        if L >= 92:
            names += [f"model.layers.{L}.{n}" for n in ("eh_proj.weight", "enorm.weight", "hnorm.weight",
                                                         "transformer.shared_head.norm.weight",
                                                         "transformer.shared_head.output.weight")]
    os.makedirs(d)
    json.dump(cfg, open(os.path.join(d, "config.json"), "w"))
    json.dump({"weight_map": {n: "model-00001.safetensors" for n in names}},
              open(os.path.join(d, "model.safetensors.index.json"), "w"))
    for f in ("tokenizer.json", "tokenizer_config.json", "special_tokens_map.json", "chat_template.jinja"):
        open(os.path.join(d, f), "w").write("{}")


def test_tiny_checkpoint_passes_0xsero_verify(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(S, "expert_dims", lambda: dict(TINY_EXPERT))
    monkeypatch.setattr(S, "body_dims", tiny_body)
    monkeypatch.setattr(S, "value_block", lambda dt, rng, n=64: S.fill(b"\x00\x3c" if dt == "BF16" else b"\0" * 4, n))
    meta, out = str(tmp_path / "meta"), str(tmp_path / "out")
    tiny_meta(meta)
    monkeypatch.setattr(sys, "argv", ["synth_ckpt.py", out, "--meta", meta, "--k-mix", "2:0.55,3:0.35,4:0.10"])
    with pytest.raises(SystemExit) as e:
        S.main()
    assert e.value.code == 0, capsys.readouterr().out
    cfg = json.load(open(os.path.join(out, "config.json")))
    assert cfg["model_type"] == "step5" and cfg["text_config"]["model_type"] == "step5_text"
    assert "sparse_config" not in cfg["text_config"]
    assert cfg["quantization_config"] == {"quant_method": "step5_exl3", "expert_layers": list(range(3, 91)),
                                          "body_layers": list(range(92)), "n_experts": 352, "body_format": "exl3"}
    wm = json.load(open(os.path.join(out, "model.safetensors.index.json")))["weight_map"]
    assert "model.layers.3.moe.gate.weight" in wm and "model.layers.92.self_attn.q_proj.weight" in wm
    assert not any(".moe.gate_proj." in k or "ssmax_s" in k or "sparse_indexer" in k or k.startswith("vision_model")
                   for k in wm)
    assert "model.layers.5.self_attn.q_proj.weight" not in wm                  # body layers are EXL3 only
    r = subprocess.run([sys.executable, os.path.join(ROOT, "serve", "verify.py"), "--body-format", "exl3", out],
                       capture_output=True, text=True)
    assert r.returncode == 0 and r.stdout.startswith("OK"), r.stdout
    plan = json.load(open(os.path.join(out, "exl3_plan.json")))
    hdr, _ = read_header(os.path.join(out, "exl3", "experts", "L03.safetensors"))
    ks = [k_of_trellis_shape(hdr[f"model.layers.3.moe.experts.{e}.gate_proj.trellis"]["shape"]) for e in range(352)]
    assert ks == plan["layers"]["3"]["k"]

    # --experts-only replaces the expert files through new inodes (a hardlinked copy keeps its own)
    link = str(tmp_path / "L03.link")
    os.link(os.path.join(out, "exl3", "experts", "L03.safetensors"), link)
    monkeypatch.setattr(sys, "argv", ["synth_ckpt.py", out, "--meta", meta, "--k-mix", "2:1.0", "--experts-only"])
    with pytest.raises(SystemExit) as e:
        S.main()
    assert e.value.code == 0
    hdr2, _ = read_header(os.path.join(out, "exl3", "experts", "L03.safetensors"))
    assert {k_of_trellis_shape(v["shape"]) for k, v in hdr2.items() if k.endswith("trellis")} == {2}
    assert read_header(link)[0] == hdr
    assert json.load(open(os.path.join(out, "exl3_plan.json")))["layers"]["3"]["k"] == [2] * 352


def test_parse_body_k_and_module_mapping():
    assert S.parse_body_k("4") == {"attn": 4, "shared": 4, "mlp": 4, "head": "bf16"}
    b = S.parse_body_k("6/8/8")
    assert b == {"attn": 6, "shared": 8, "mlp": 8, "head": "bf16"}
    assert [S.body_k_of(m, b) for m in ("self_attn.q_proj", "self_attn.o_proj", "share_expert.down_proj",
                                        "mlp.gate_proj")] == [6, 6, 8, 8]
    for bad in ("1", "9", "4/4", "4/4/4/4", "x"):
        with pytest.raises(ValueError):
            S.parse_body_k(bad)


def test_body_variant_bytes_per_node():
    """frontier review 2b: 6/8/8 costs about +1.8 GB per node and 8/8/8 about +3.0 GB over 4/4/4 (step5q.plan)."""
    mix = S.parse_k_mix("2:0.55,3:0.35,4:0.10")
    tot = {b: P.bytes_per_node(S.make_plan(mix, S.parse_body_k(b), 0))["total"] for b in ("4/4/4", "6/8/8", "8/8/8")}
    assert round((tot["6/8/8"] - tot["4/4/4"]) / 1e9, 1) == 1.8
    assert round((tot["8/8/8"] - tot["4/4/4"]) / 1e9, 1) == 3.0


def test_write_body_uses_each_modules_k(tmp_path, monkeypatch):
    monkeypatch.setattr(S, "body_dims", tiny_body)
    os.makedirs(tmp_path / "exl3" / "body")
    for L, is_moe in ((2, False), (3, True)):
        S.write_body(str(tmp_path), L, is_moe, S.parse_body_k("6/8/5"), random.Random(L))
        hdr, _ = read_header(str(tmp_path / "exl3" / "body" / f"L{L:02d}.safetensors"))
        pre = f"model.layers.{L}."
        ks = {k[len(pre):-len(".trellis")]: k_of_trellis_shape(v["shape"]) for k, v in hdr.items()
              if k.endswith(".trellis")}
        ffn = ("share_expert", 8) if is_moe else ("mlp", 5)
        assert ks == {**{f"self_attn.{p}_proj": 6 for p in "qkvo"},
                      **{f"{ffn[0]}.{p}_proj": ffn[1] for p in ("gate", "up", "down")}}


def test_merge_partial_keeps_the_other_half(tmp_path):
    mix = S.parse_k_mix("2:0.6,3:0.4")
    old = S.make_plan(mix, S.parse_body_k("4"), 0)
    path = str(tmp_path / "exl3_plan.json")
    with pytest.raises(SystemExit):
        S.merge_partial(old, path, True, False, False)
    json.dump(old, open(path, "w"))
    new = S.make_plan(S.parse_k_mix("2:1.0"), S.parse_body_k("8"), 0)
    e = S.merge_partial(new, path, True, False, False)
    assert e["layers"] == new["layers"] and e["body"] == old["body"]
    b = S.merge_partial(new, path, False, True, False)
    assert b["layers"] == old["layers"] and b["body"] == new["body"]
    assert S.merge_partial(new, path, False, False, False) is new


def test_body_only_rewrite_on_tiny_checkpoint(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(S, "expert_dims", lambda: dict(TINY_EXPERT))
    monkeypatch.setattr(S, "body_dims", tiny_body)
    monkeypatch.setattr(S, "value_block", lambda dt, rng, n=64: S.fill(b"\x00\x3c" if dt == "BF16" else b"\0" * 4, n))
    meta, out = str(tmp_path / "meta"), str(tmp_path / "out")
    tiny_meta(meta)
    monkeypatch.setattr(sys, "argv", ["synth_ckpt.py", out, "--meta", meta])
    with pytest.raises(SystemExit) as e:
        S.main()
    assert e.value.code == 0
    experts = os.path.join(out, "exl3", "experts", "L03.safetensors")
    ino = os.stat(experts).st_ino
    monkeypatch.setattr(sys, "argv", ["synth_ckpt.py", out, "--meta", meta, "--body-only", "--body-k", "6/8/8"])
    with pytest.raises(SystemExit) as e:
        S.main()
    assert e.value.code == 0, capsys.readouterr().out
    assert os.stat(experts).st_ino == ino                                      # experts untouched
    hdr, _ = read_header(os.path.join(out, "exl3", "body", "L03.safetensors"))
    assert k_of_trellis_shape(hdr["model.layers.3.self_attn.q_proj.trellis"]["shape"]) == 6
    assert k_of_trellis_shape(hdr["model.layers.3.share_expert.up_proj.trellis"]["shape"]) == 8
    plan = json.load(open(os.path.join(out, "exl3_plan.json")))
    assert plan["body"] == {"attn": 6, "shared": 8, "mlp": 8, "head": "bf16"}
    r = subprocess.run([sys.executable, os.path.join(ROOT, "serve", "verify.py"), "--body-format", "exl3", out],
                       capture_output=True, text=True)
    assert r.returncode == 0, r.stdout


def test_gpu_probe_int8_gemv_info_and_xver_default(monkeypatch):
    import gpu_probe as G

    class Ext:
        @staticmethod
        def exl3_gemv_int8_max_k(dev):
            return 6

    class Old:
        exl3_gemv = None
    monkeypatch.setenv("EXL3_INT8_GEMV", "1")
    monkeypatch.delenv("EXL3_INT8_GEMV_MAX_K", raising=False)
    assert G.int8_gemv_info(Ext()) == "EXL3_INT8_GEMV=1; exl3_gemv_int8_max_k(0)=6; EXL3_INT8_GEMV_MAX_K=unset"
    assert "ABSENT" in G.int8_gemv_info(Old()) and "exl3_gemv" in G.int8_gemv_info(Old())
    with open(G.__file__) as f:
        assert 'add_argument("--ks", default="2,3,4,5,6,8"' in f.read()
