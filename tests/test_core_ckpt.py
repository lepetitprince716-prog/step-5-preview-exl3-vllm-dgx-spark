"""ckpt.py: Checkpoint.get_many against safetensors' own reader on a tiny two-shard checkpoint (CPU)."""
import json

import pytest

torch = pytest.importorskip("torch")
st = pytest.importorskip("safetensors.torch")

from step5q.ckpt import Checkpoint  # noqa: E402


@pytest.fixture
def ckpt_dir(tmp_path):
    g = torch.Generator().manual_seed(0)
    shards = {
        "a.safetensors": {"model.layers.3.moe.gate_proj.weight": (torch.randn(2, 256, 128, generator=g) * 4).to(
                              torch.float8_e4m3fn),
                          "model.layers.3.moe.gate_proj.weight_scale_inv": torch.rand(2, 2, 1, generator=g),
                          "model.layers.3.input_layernorm.weight": torch.randn(64, generator=g).to(torch.bfloat16)},
        "b.safetensors": {"model.norm.weight": torch.randn(64, generator=g),
                          "model.layers.30.moe.router_bias": torch.randn(16, generator=g)},
    }
    wm = {}
    for f, ts in shards.items():
        st.save_file(ts, str(tmp_path / f))
        wm.update({k: f for k in ts})
    (tmp_path / "model.safetensors.index.json").write_text(json.dumps({"weight_map": wm}))
    return tmp_path, {k: v for ts in shards.values() for k, v in ts.items()}


@pytest.mark.parametrize("gap", [0, 64 << 20])
def test_get_many_matches(ckpt_dir, gap):
    path, want = ckpt_dir
    c = Checkpoint(str(path))
    got = c.get_many(list(want), "cpu", gap=gap)
    assert set(got) == set(want)
    for k, v in want.items():
        assert got[k].dtype == v.dtype and got[k].shape == v.shape
        assert torch.equal(got[k].view(torch.uint8), v.view(torch.uint8))


def test_names_and_meta(ckpt_dir):
    path, want = ckpt_dir
    c = Checkpoint(str(path))
    assert c.names("model.layers.3.") == sorted(k for k in want if k.startswith("model.layers.3."))
    assert c.meta("model.norm.weight")["shape"] == [64]
    assert torch.equal(c.get("model.norm.weight"), want["model.norm.weight"])
