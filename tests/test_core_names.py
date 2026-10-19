"""names.py: key formats and round trips (pure python)."""
import pytest

from step5q import names


def test_official_name():
    assert names.official_name(3, "moe.gate_proj") == "model.layers.3.moe.gate_proj.weight"
    want = "model.layers.3.moe.gate_proj.weight_scale_inv"
    assert names.official_name(3, "moe.gate_proj", "weight_scale_inv") == want


def test_sero_expert_key_round_trip():
    for proj, mod in names.PROJ.items():
        for piece in names.PIECES:
            k = names.sero_expert_key(87, 351, proj, piece)
            assert k == f"model.layers.87.moe.experts.351.{mod}.{piece}"
            assert names.parse_sero_key(k) == (87, 351, mod, piece)
            assert names.sero_expert_key(87, 351, mod, piece) == k


def test_sero_body_key_round_trip():
    for mod in names.BODY_MODULES:
        k = names.sero_body_key(5, mod, "trellis")
        assert k == f"model.layers.5.{mod}.trellis"
        assert names.parse_sero_key(k) == (5, None, mod, "trellis")
    assert names.sero_body_key(3, "share_expert.gate_proj", "suh") == "model.layers.3.share_expert.gate_proj.suh"


def test_exllamav3_key_equals_sero():
    assert names.exllamav3_key(4, "self_attn.q_proj", "svh") == names.sero_body_key(4, "self_attn.q_proj", "svh")
    assert names.exllamav3_key(4, "w2", "mul1", expert=7) == "model.layers.4.moe.experts.7.down_proj.mul1"


def test_files():
    assert names.sero_expert_file(3) == "exl3/experts/L03.safetensors"
    assert names.sero_body_file(91) == "exl3/body/L91.safetensors"


@pytest.mark.parametrize("bad", [
    lambda: names.sero_expert_key(3, 0, "w4", "trellis"),
    lambda: names.sero_expert_key(3, 0, "w1", "scale"),
    lambda: names.sero_body_key(3, "self_attn.g_proj", "trellis"),
    lambda: names.sero_body_key(3, "moe.share_expert.gate_proj", "trellis"),
    lambda: names.parse_sero_key("model.layers.3.self_attn.g_proj.trellis"),
])
def test_rejects(bad):
    with pytest.raises(ValueError):
        bad()
