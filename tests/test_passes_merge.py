"""p0b_merge_stats.py and the p1b_relerr.py table math (pure Python)."""
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import p0b_merge_stats as p0b  # noqa: E402
from p_util import bands_from_layers, coarse_domain, parse_variants, percentile, relerr_table, spread_pick  # noqa: E402

E = 3


def dom(tokens, count, tshare):
    return {"tokens": tokens, "count": count, "top1": count, "sum_g": count, "sum_g2": count, "out_energy": tshare,
            "tshare": tshare}


def write_run(d, out_max):
    d.mkdir()
    rec = {"layer": 3, "tokens": 300, "count": [0] * E, "tshare": [0] * E, "out_energy": [0] * E, "out_max": out_max,
           "routed_out_energy": 30.0, "shared_out_energy": 3.0, "resid_in_energy": 1.0, "resid_out_energy": 2.0,
           "resid_in_max": 5.0, "resid_out_max": 6.0,
           "domains": {"code": dom(200, [200, 0, 0], [20, 0, 0]), "chat": dom(100, [0, 100, 0], [0, 10, 0]),
                       "random": dom(0, [0, 0, 0], [0, 0, 0])}}
    (d / "layer_03.json").write_text(json.dumps(rec))


def test_merge_reweights_and_drops(tmp_path):
    write_run(tmp_path / "a", [1.0, 9.0, 2.0])
    write_run(tmp_path / "b", [4.0, 1.0, 2.0])
    out = tmp_path / "w"
    rep = p0b.main([str(out), str(tmp_path / "a"), str(tmp_path / "b"), "--weights", "code=1,chat=3,writing=2"])
    rec = json.load(open(out / "layer_03.json"))
    T = 600                                                     # code 400 + chat 200 pooled over the two runs
    assert rec["tokens"] == T
    # per-token averages are the weighted mix: code 1/4, chat 3/4 (writing absent, random 0 tokens -> dropped)
    assert abs(rec["count"][0] / T - 0.25 * 1.0) < 1e-12 and abs(rec["count"][1] / T - 0.75 * 1.0) < 1e-12
    assert abs(rec["tshare"][1] / T - 0.75 * 0.1) < 1e-12
    assert rec["out_max"] == [4.0, 9.0, 2.0] and rec["resid_out_max"] == 6.0
    assert rep["weights"] == {"chat": 0.75, "code": 0.25} and rep["dropped"] == ["random"]


def test_default_weights_cover_coarse_domains():
    assert coarse_domain({"domain": "code_debug", "source": "chat"}) == "code"
    assert coarse_domain({"domain": "code", "source": "exllamav3_std"}) == "std"   # code.utf8 is standard text
    assert coarse_domain({"domain": "multilingual_zh", "source": "instruct"}) == "multilingual"
    w = p0b.parse_weights("")
    assert w["instruct"] == 1.5 and w["std"] == 0.5 and sum(1 for v in w.values() if v == 1.0) == 8


def test_percentile_matches_numpy_linear():
    xs = [5.0, 1.0, 3.0, 2.0, 4.0]
    assert percentile(xs, 50) == 3.0 and percentile(xs, 0) == 1.0 and percentile(xs, 100) == 5.0
    assert abs(percentile(xs, 99.5) - 4.98) < 1e-12


def test_relerr_table_and_bands():
    assert bands_from_layers([3, 30, 60, 89], 3, 90) == {"3-16": 3, "17-45": 30, "46-74": 60, "75-90": 89}
    recs = [{"layer": l, "K": k, "gate_up": g * k, "down": 2 * g * k, "all": 3 * g * k}
            for l in (3, 30) for k in (2, 3) for g in (1.0, 2.0, 3.0)]
    t = relerr_table(recs, 3, 90)
    assert t["rel_err"] == {"2": 12.0, "3": 18.0}               # pooled median of "all"
    assert set(t["band"]) == {"3-16", "17-90"} and t["band"]["3-16"]["2"] == 12.0
    assert t["detail"]["30"]["3"] == {"gate_up": 6.0, "down": 12.0, "all": 18.0, "n": 3}


def test_spread_pick_covers_usage_range():
    counts = [0, 5, 1, 9, 0, 3, 7]                   # used by count: e2(1) e5(3) e1(5) e6(7) e3(9)
    assert spread_pick(counts, 3) == [1, 2, 3]                           # coldest, median, hottest
    assert spread_pick(counts, 10) == [1, 2, 3, 5, 6]                    # all routed experts, zeros skipped
    assert spread_pick(counts, 1) == [1]
    assert spread_pick([0, 0], 4) == []


def test_parse_variants():
    assert parse_variants("5/6/6,6/8/8") == [("5-6-6", {"attn": 5, "shared": 6, "mlp": 6}),
                                             ("6-8-8", {"attn": 6, "shared": 8, "mlp": 8})]
    assert parse_variants("") == []
    for bad in ("5/6", "1/4/4", "4/4/9"):
        try:
            parse_variants(bad)
        except SystemExit:
            continue
        raise AssertionError(bad)
