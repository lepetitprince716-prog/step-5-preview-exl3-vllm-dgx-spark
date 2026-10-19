"""plan.py: validation and byte accounting (pure python)."""
import json

import pytest

from step5q import plan as P

BODY = {"attn": 4, "shared": 4, "mlp": 4, "head": "bf16"}


def make_plan(ks):
    return {"bpw_target": 2.5, "layers": {str(l): {"k": list(ks)} for l in P.MOE_LAYERS}, "body": dict(BODY)}


def test_exl3_linear_bytes_by_hand():
    # 256 in x 128 out at K3 on 2 ranks, output split: trellis 256 x 64 x 3 / 8, suh 256, svh 64 fp16, marker 4
    assert P.exl3_linear_bytes(256, 128, 3, 2, "out") == 256 * 64 * 3 // 8 + 2 * (256 + 64) + 4
    assert P.exl3_linear_bytes(256, 128, 3, 2, "in") == 128 * 128 * 3 // 8 + 2 * (128 + 128) + 4
    assert P.exl3_linear_bytes(256, 128, 3) == 256 * 128 * 3 // 8 + 2 * (256 + 128) + 4


def test_expert_bytes_by_hand():
    # expert TP over 2 nodes: gate/up [4096 -> 768], down [768 -> 4096]
    per_node = 3 * 4096 * 768 * 4 // 8 + 2 * 2 * (4096 + 768) + 2 * (768 + 4096) + 3 * 4
    assert P.expert_bytes(4) == per_node
    whole = 3 * 4096 * 1536 * 4 // 8 + 3 * 2 * (4096 + 1536) + 3 * 4
    assert P.expert_bytes(4, expert_tp=False) == whole


def test_expert_parallel_takes_busiest_node():
    plan = make_plan([5] * 176 + [2] * 176)      # node 0 holds every K5 expert
    got = P.bytes_per_node(plan, expert_tp=False)["experts"]
    assert got == len(P.MOE_LAYERS) * 176 * P.expert_bytes(5, expert_tp=False)
    assert P.bytes_per_node(plan, expert_tp=True)["experts"] < got


def test_matches_plan_file_budget():
    """research/2026-10-15-step5-2spark-plan.md, per node at 2.5 bpw: experts 85.1 GiB + suh/svh 0.84, attention K4
    2.29, shared + dense MLP 0.55, embed + lm_head 0.98, router 0.47, MTP 0.87 + heads 1.47, total ~92.5."""
    plan = make_plan([2] * 176 + [3] * 176)
    assert P.expert_bpw(plan) == 2.5
    b = {k: v / P.GIB for k, v in P.bytes_per_node(plan).items()}
    assert b["experts"] == pytest.approx(85.1 + 0.84, abs=0.03)
    assert b["attn"] == pytest.approx(2.29, abs=0.01)
    assert b["shared"] + b["mlp"] == pytest.approx(0.55, abs=0.01)
    assert b["embed_head"] == pytest.approx(0.98, abs=0.01)
    assert b["router"] == pytest.approx(0.47, abs=0.01)
    assert b["mtp"] == pytest.approx(0.87, abs=0.01)
    assert b["mtp_heads"] == pytest.approx(1.47, abs=0.01)
    assert b["total"] == pytest.approx(92.5, abs=0.1)


@pytest.mark.parametrize("mutate", [
    lambda p: p["layers"].pop("90"),
    lambda p: p["layers"]["3"]["k"].pop(),
    lambda p: p["layers"]["3"]["k"].__setitem__(0, 6),
    lambda p: p["layers"]["3"]["k"].__setitem__(0, 2.5),
    lambda p: p["layers"]["3"].__setitem__("k", [2] * 88 + [3] * 88 + [4] * 88 + [5] * 88),
    lambda p: p["body"].__setitem__("head", 6),
    lambda p: p["body"].pop("mlp"),
])
def test_validate_rejects(mutate):
    p = make_plan([3] * 352)
    mutate(p)
    with pytest.raises((ValueError, KeyError)):
        P.validate(p)


def test_read_write_round_trip(tmp_path):
    p = make_plan([2] * 100 + [3] * 200 + [4] * 52)
    path = str(tmp_path / "plan.json")
    P.write_plan(p, path)
    assert P.read_plan(path) == p
    assert json.load(open(path))["layers"]["3"]["k"][:2] == [2, 2]
