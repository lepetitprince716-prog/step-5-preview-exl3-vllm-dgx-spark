"""p2_allocate.py on synthetic pass-0 statistics (pure Python, no checkpoint)."""
import json
import os
import random
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import p2_allocate as p2  # noqa: E402

LAYERS = list(range(3, 11)) + [88, 89, 90]
E = 24


def make_stats(tmp_path, seed=0, super_layer=3):
    rng = random.Random(seed)
    d = tmp_path / "stats"
    d.mkdir()
    for l in LAYERS:
        tokens = 1000
        count = [rng.randint(5, 900) for _ in range(E)]
        tshare = [c * rng.uniform(0.05, 0.3) for c in count]        # spread so the greedy wants every K
        out_energy = [rng.uniform(1, 10) for _ in range(E)]
        out_max = [rng.uniform(1, 10) for _ in range(E)]
        if l == super_layer:
            out_energy[0] = 1e4                                      # one super expert (> 20% of the energy)
            tshare[0] = 0.5
        if l in (5, 89):
            out_max[7] = 1e4                                         # massive-activation expert (Su et al.)
        rec = {"tokens": tokens, "count": count, "tshare": tshare, "out_energy": out_energy, "out_max": out_max,
               "layer": l}
        (d / f"layer_{l:02d}.json").write_text(json.dumps(rec))
    alpha = {str(l): 4.0 / (1 + i) ** 2 for i, l in enumerate(LAYERS)}   # sensitivity falls through the stack
    ap = tmp_path / "alpha.json"
    ap.write_text(json.dumps({"alpha": alpha}))
    return str(d), str(ap)


def run(tmp_path, *extra, seed=0):
    stats, alpha = make_stats(tmp_path, seed)
    out = str(tmp_path / "plan.json")
    plan = p2.main([stats, "--alpha", alpha, "-o", out, *extra])
    assert json.load(open(out)) == json.loads(json.dumps(plan))
    return plan


def all_k(plan):
    return [k for l in LAYERS for k in plan["layers"][str(l)]["k"]]


@pytest.mark.parametrize("bpw", [2.4, 2.75, 3.3])
def test_budget_kinds_floors_super(tmp_path, bpw):
    plan = run(tmp_path, "--bpw", str(bpw))
    ks = all_k(plan)
    assert len(ks) == len(LAYERS) * E
    assert sum(ks) / len(ks) <= bpw + 1e-9                     # memory budget respected
    assert sum(ks) / len(ks) >= bpw - 1.0 / len(ks) * 8         # and (nearly) used
    assert set(ks) <= {2, 3, 4, 5}
    for l in LAYERS:
        row = plan["layers"][str(l)]["k"]
        assert len(set(row)) <= 3, (l, sorted(set(row)))     # at most 3 K kinds per layer
        if l in (88, 89, 90):
            assert min(row) >= 3                                # official BF16 layers not starved
    assert plan["layers"]["3"]["k"][0] == 5                     # super expert at K5
    assert plan["summary"]["super_experts"] == 1
    assert plan["body"] == {"attn": 4, "shared": 4, "mlp": 4, "head": "bf16"}


def test_bytes_report(tmp_path):
    plan = run(tmp_path, "--bpw", "2.4")
    rep = plan["bytes_per_node"]
    assert rep["experts"] == int(sum(p2.expert_bytes(k) for k in all_k(plan)))
    assert rep["tp"] == 2 and rep["expert_tp"] is True and rep["decode_expert_bytes_per_token"] > 0
    assert "components" not in rep                              # a subset of layers is not a serving plan
    # one K2 expert on one node: half the trellis, gate/up suh + down svh whole (plugin EXPERT_TP slicing)
    k2 = 3 * 4096 * 1536 * 2 / 8 / 2 + 2 * (4096 * 2 + 768 * 2 + 4) + (768 * 2 + 4096 * 2 + 4)
    assert p2.expert_bytes(2) == k2


def test_full_plan_validates(tmp_path):
    """88 MoE layers x 352 experts: the plan passes lane A's validate and gets its per-component bytes."""
    rng = random.Random(1)
    d = tmp_path / "stats"
    d.mkdir()
    for l in range(3, 91):
        count = [rng.randint(1, 5000) for _ in range(352)]
        rec = {"tokens": 10000, "count": count, "tshare": [c * rng.uniform(0.05, 0.3) for c in count],
               "out_energy": [rng.uniform(1, 10) for _ in range(352)]}
        (d / f"layer_{l:02d}.json").write_text(json.dumps(rec))
    ap = tmp_path / "alpha.json"
    ap.write_text(json.dumps({"alpha": {str(l): 1.0 / l for l in range(3, 91)}}))
    plan = p2.main([str(d), "--alpha", str(ap), "-o", str(tmp_path / "plan.json"), "--bpw", "2.4"])
    p2.validate(plan)
    comp = plan["bytes_per_node"]["components"]
    assert comp["experts"] == plan["bytes_per_node"]["experts"] and comp["total"] > comp["experts"]


def test_decode_cap(tmp_path):
    free = run(tmp_path, "--bpw", "3.0")
    cap = free["summary"]["decode_bpw"] - 0.2
    sub = tmp_path / "capped"
    sub.mkdir()
    capped = run(sub, "--bpw", "3.0", "--decode-bpw", str(cap))
    assert capped["summary"]["decode_bpw"] <= cap + 1e-6
    for l in LAYERS:
        assert len(set(capped["layers"][str(l)]["k"])) <= 3


def test_prune_cold_recorded_only(tmp_path):
    plan = run(tmp_path, "--bpw", "2.4", "--prune-cold", "3")
    stats = json.load(open(tmp_path / "stats" / "layer_04.json"))
    coldest = sorted(range(E), key=lambda e: (stats["count"][e], e))[:3]
    assert plan["prune"]["4"] == sorted(coldest)
    assert all(len(plan["prune"][str(l)]) == 3 for l in LAYERS)
    assert len(plan["layers"]["4"]["k"]) == E                   # not applied: every expert keeps a K


def test_enforce_kinds_respects_floors():
    ex = [{"layer": 3, "e": i, "imp": 1.0, "use": 0.1, "floor": 2, "super": False} for i in range(10)]
    ex[0]["floor"] = 5                                          # super expert alone in the K5 class
    K = [5, 2, 2, 2, 3, 3, 3, 4, 4, 4]
    p2.enforce_kinds(ex, K, 3)
    assert len(set(K)) == 3 and K[0] == 5                       # K5 (smallest class) cannot move down


def test_infeasible_floors(tmp_path):
    stats, alpha = make_stats(tmp_path)
    with pytest.raises(SystemExit):
        p2.main([stats, "--alpha", alpha, "-o", str(tmp_path / "p.json"), "--bpw", "2.0"])


def test_score_mode(tmp_path, capsys):
    plan = run(tmp_path, "--bpw", "2.4")
    p2.main([str(tmp_path / "stats"), "--alpha", str(tmp_path / "alpha.json"), "--score", str(tmp_path / "plan.json")])
    out = json.loads(capsys.readouterr().out.strip().splitlines()[-1])
    assert abs(out["bpw"] - plan["summary"]["bpw"]) < 1e-3


def test_super_max_first_75pct_only(tmp_path, capsys):
    plan = run(tmp_path, "--bpw", "2.4", "--super-max")
    assert plan["layers"]["5"]["k"][7] == 5                     # layer 5 is in the first 75% of the MoE layers
    assert plan["summary"]["super_by"]["max"] >= 1 and plan["summary"]["super_experts"] == 2
    assert plan["layers"]["3"]["k"][0] == 5                     # --super-share still triggers
    sub = tmp_path / "off"
    sub.mkdir()
    plain = run(sub, "--bpw", "2.4")
    assert plain["summary"]["super_experts"] == 1               # without --super-max only the energy share counts
    st = p2.load_stats(str(tmp_path / "stats"))
    flags = p2.super_max_flags(st)
    assert 89 not in flags and flags[5] == {7}                  # layer 89 is past the candidate layers


def test_layer_guard(tmp_path):
    plan = run(tmp_path, "--bpw", "2.75", "--layer-guard", "5,4")
    st = p2.load_stats(str(tmp_path / "stats"))
    alpha = {int(k): v for k, v in json.load(open(tmp_path / "alpha.json"))["alpha"].items()}
    for l in LAYERS:
        imp = [alpha[l] * t for t in st[l]["tshare"]]
        first, second = sorted(range(E), key=lambda e: -imp[e])[:2]
        row = plan["layers"][str(l)]["k"]
        assert row[first] == 5 and row[second] >= 4, (l, row[first], row[second])
        assert len(set(row)) <= 3


def test_cost_printed_before_and_after_kinds(tmp_path, capsys):
    plan = run(tmp_path, "--bpw", "2.4")
    out = capsys.readouterr().out
    assert "predicted_cost before enforce_kinds" in out
    assert plan["summary"]["cost_before_kinds"] <= plan["summary"]["predicted_cost"] + 1e-12


def test_rel_err_json_with_bands(tmp_path):
    t = {"rel_err": {"2": 0.2, "3": 0.08, "4": 0.03, "5": 0.012},
         "band": {"3-6": {"2": 0.3, "3": 0.1, "4": 0.04, "5": 0.015}}}
    path = tmp_path / "rel_err.json"
    path.write_text(json.dumps(t))
    pooled, bands = p2.load_rel_err(str(path))
    assert pooled[2] == 0.2 and bands == [(3, 6, {2: 0.3, 3: 0.1, 4: 0.04, 5: 0.015})]
    assert abs(p2.eps_of(4, pooled, bands)[2] - 0.09) < 1e-12 and abs(p2.eps_of(9, pooled, bands)[2] - 0.04) < 1e-12
    plan = run(tmp_path, "--bpw", "2.4", "--rel-err", str(path))
    assert plan["summary"]["rel_err"]["2"] == 0.2 and "3-6" in plan["summary"]["rel_err_bands"]
    assert p2.load_rel_err("2:0.1,3:0.05,4:0.02,5:0.01")[0][3] == 0.05
    with pytest.raises(SystemExit):
        p2.load_rel_err("2:0.1,3:0.05")


def test_body_k_written(tmp_path):
    plan = run(tmp_path, "--bpw", "2.4", "--attn-k", "5", "--shared-k", "6", "--mlp-k", "6")
    assert plan["body"] == {"attn": 5, "shared": 6, "mlp": 6, "head": "bf16"}
