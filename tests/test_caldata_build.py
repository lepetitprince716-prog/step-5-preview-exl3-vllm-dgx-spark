"""caldata/build.py: split, truncation, standard-text windows, packing (pure python)."""
import json
import os
import random
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "caldata"))
import build as B  # noqa: E402


def rec(i, split):
    return {"id": i, "split": split, "domain": "code", "messages": []}


def test_cut_keeps_short_and_truncates_long_with_mask():
    ids, score, full = B.cut([1, 2, 3], [0, 1, 1], 4)
    assert (ids, score, full) == ([1, 2, 3], [0, 1, 1], None)
    ids, score, full = B.cut(list(range(10)), [0] * 3 + [1] * 7, 4)
    assert ids == [0, 1, 2, 3] and score == [0, 0, 0, 1] and full == 10
    ids, score, full = B.cut(list(range(5000)), None, 4096)
    assert len(ids) == 4096 and score is None and full == 5000


def test_split_is_disjoint_sorted_and_rejects_duplicates():
    cal, panel = B.split_records([rec("p0002", "cal"), rec("p0001", "panel"), rec("i0000", "cal"), rec("p0000", "cal")])
    assert [r["id"] for r in cal] == ["i0000", "p0000", "p0002"]
    assert [r["id"] for r in panel] == ["p0001"]
    with pytest.raises(ValueError, match="duplicate"):
        B.split_records([rec("p0001", "cal"), rec("p0001", "panel")])
    with pytest.raises(ValueError, match="unknown split"):
        B.split_records([rec("p0001", "test")])


def test_std_rows_follow_sudoingx_weights():
    rows = dict(B.std_rows_per_file(64))
    assert rows == {"c4.utf8": 11, "code.utf8": 11, "multilingual.utf8": 6, "technical.utf8": 6, "wiki.utf8": 28,
                    "tiny.utf8": 3}
    assert sum(rows.values()) * 2048 == 133_120          # RECIPE.md section 2: exllamav3 standard text 133,120 tokens


def test_std_windows_in_bounds_and_seeded():
    w1 = B.std_windows(10_000, 20, 2048, random.Random(B.SEED))
    w2 = B.std_windows(10_000, 20, 2048, random.Random(B.SEED))
    assert w1 == w2
    assert all(0 <= s and e - s == 2048 and e <= 10_000 for s, e in w1)
    short = B.std_windows(100, 2, 2048, random.Random(0))     # file shorter than a window: whole file
    assert all(s == 0 and e == 100 for s, e in short)


def test_pack_offsets():
    flat, offs = B.pack([[1, 2], [], [3, 4, 5]])
    assert flat == [1, 2, 3, 4, 5] and offs == [0, 2, 2, 5]


def test_for_template_parses_tool_arguments_without_touching_input():
    msgs = [{"role": "assistant", "content": "", "tool_calls": [
        {"id": "c1", "type": "function", "function": {"name": "get_weather", "arguments": '{"city": "Lisbon"}'}},
        {"id": "c2", "type": "function", "function": {"name": "x", "arguments": "not json"}}]}]
    out = B.for_template(msgs)
    assert out[0]["tool_calls"][0]["function"]["arguments"] == {"city": "Lisbon"}
    assert out[0]["tool_calls"][1]["function"]["arguments"] == {}
    assert msgs[0]["tool_calls"][0]["function"]["arguments"] == '{"city": "Lisbon"}'


def test_source_of_prefix():
    assert B.source_of("p0001") == "chat" and B.source_of("i0001") == "instruct"


def test_std_dir_of(tmp_path):
    d = tmp_path / "exllamav3" / "conversion" / "standard_cal_data"
    d.mkdir(parents=True)
    (d / "c4.utf8").write_text("x")
    assert B.std_dir_of(str(tmp_path)) == str(d)
    assert B.std_dir_of(str(d)) == str(d)
    with pytest.raises(SystemExit):
        B.std_dir_of(str(tmp_path / "nope"))


def test_sha256_file(tmp_path):
    p = tmp_path / "a"
    p.write_bytes(b"abc")
    assert B.sha256_file(str(p)) == "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"


def test_base_prompt_ids_never_collide_with_instruct_ids():
    import prompts
    import prompts_instruct
    base = {x["id"] for x in prompts.build_prompt_set(None)}
    inst = {x["id"] for x in prompts_instruct.build_prompt_set(None)}
    assert not base & inst
    assert all(x["split"] == "cal" for x in prompts_instruct.build_prompt_set(None))
    json.dumps(prompts.build_prompt_set(None))               # records are JSON-serialisable as generated


def test_turn_mask_scores_each_turn_between_header_and_end():
    # BOS sys | user U | head H | gen g g | user V | head H | gen h h
    full = [0, 9, 1, 1, 2, 7, 7, 1, 1, 2, 8, 8]
    turns = [(full[:5], full[:7]), (full[:10], full[:12])]
    assert B.turn_mask(full, turns) == [0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 1, 1]


def test_turn_mask_rejects_history_rendered_differently():
    full = [0, 1, 2, 7, 7, 1, 2, 8]
    with pytest.raises(AssertionError, match="turn 0"):
        B.turn_mask(full, [([0, 1, 2], [0, 1, 2, 6, 7])])        # turn not a prefix of the full ids
    with pytest.raises(AssertionError, match="turn 0"):
        B.turn_mask(full, [([0, 1, 3], [0, 1, 2, 7, 7])])        # generation header not a prefix of the turn
    with pytest.raises(AssertionError):
        B.turn_mask(full, [([0, 1, 2], [0, 1, 2])])              # empty generated span
