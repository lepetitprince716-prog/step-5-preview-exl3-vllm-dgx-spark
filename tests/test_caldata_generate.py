"""caldata/generate_stepfun.py: effort mapping, cost from usage, reply parsing, key handling (pure python, no network)."""
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "caldata"))
import generate_stepfun as G  # noqa: E402


def test_effort_map_default_and_validation():
    m = G.parse_effort_map("off=low,low=medium,high=high")
    assert m == {"off": "low", "low": "medium", "high": "high"}
    assert G.api_effort({"thinking": False, "effort": None}, m) == "low"
    assert G.api_effort({"thinking": True, "effort": "low"}, m) == "medium"
    assert G.api_effort({"thinking": True, "effort": "high"}, m) == "high"
    for bad in ("off=low,low=medium", "off=none,low=low,high=high", "max=high,off=low,low=low,high=high"):
        with pytest.raises(ValueError):
            G.parse_effort_map(bad)


def test_every_prompt_maps_to_an_api_effort():
    m = G.parse_effort_map("off=low,low=medium,high=high")
    for mod in G.SETS.values():
        for it in mod.build_prompt_set(None):
            assert G.api_effort(it, m) in G.API_EFFORTS


def test_call_cost_both_cached_field_shapes():
    usd = G.PLATFORMS["intl"][2:]
    assert G.call_cost({"prompt_tokens": 1_000_000, "completion_tokens": 1_000_000}, usd) == pytest.approx(3.70)
    cn_style = {"prompt_tokens": 1000, "completion_tokens": 0, "prompt_tokens_details": {"cached_tokens": 400}}
    intl_style = {"prompt_tokens": 1000, "completion_tokens": 0, "cached_tokens": 400}
    want = (600 * 1.00 + 400 * 0.10) / 1e6
    assert G.call_cost(cn_style, usd) == pytest.approx(want)
    assert G.call_cost(intl_style, usd) == pytest.approx(want)
    assert G.call_cost({}, usd) == 0


def test_assistant_message_keeps_reasoning_and_tool_calls():
    resp = {"choices": [{"finish_reason": "tool_calls", "message": {
        "role": "assistant", "content": None, "reasoning_content": "think",
        "tool_calls": [{"id": "c1", "type": "function", "index": 0,
                        "function": {"name": "get_weather", "arguments": "{\"city\":\"Lisbon\"}"}}]}}]}
    msg, fin = G.assistant_message(resp)
    assert fin == "tool_calls" and msg["content"] == "" and msg["reasoning_content"] == "think"
    assert msg["tool_calls"] == [{"id": "c1", "type": "function",
                                  "function": {"name": "get_weather", "arguments": "{\"city\":\"Lisbon\"}"}}]
    msg, _ = G.assistant_message({"choices": [{"message": {"content": "a", "reasoning": "r"}}]})
    assert msg["reasoning_content"] == "r"                    # "general" format field name


def test_history_strips_reasoning_only_when_asked():
    msgs = [{"role": "user", "content": "q"}, {"role": "assistant", "content": "a", "reasoning_content": "r"}]
    assert G.history(msgs, True) is msgs
    assert G.history(msgs, False)[1] == {"role": "assistant", "content": "a"}
    assert msgs[1]["reasoning_content"] == "r"


def test_read_key_from_file_never_printed(tmp_path, capsys, monkeypatch):
    p = tmp_path / "stepfun.env"
    p.write_text("OTHER=1\nSTEPFUN_API_KEY='sk-test-123'\n")
    os.chmod(p, 0o600)
    monkeypatch.delenv("STEPFUN_API_KEY", raising=False)
    assert G.read_key(str(p)) == "sk-test-123"
    p.chmod(0o644)
    assert G.read_key(str(p)) == "sk-test-123"
    out = capsys.readouterr()
    assert "sk-test-123" not in out.out + out.err and "chmod 600" in out.err
    p.write_text("NOPE=1\n")
    with pytest.raises(SystemExit) as e:
        G.read_key(str(p))
    assert "sk-" not in str(e.value)
