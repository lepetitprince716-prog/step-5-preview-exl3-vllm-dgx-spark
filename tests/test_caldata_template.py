"""The official chat template as build.py uses it.

Part 1 (pure python) reads chat_template.jinja and checks the facts build.py relies on. Part 2 (transformers) renders a
2-turn conversation and a tool round through build.Renderer and checks the exact text and the assistant mask.
Template location: $STEP5_META (default /model, the official checkpoint mount in the quant container); skipped when
absent. On the Mac: STEP5_META=<Step-5-Preview-hf clone> python3 -m pytest tests/test_caldata_template.py
"""
import os
import re
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "caldata"))
META = os.environ.get("STEP5_META", "/model")
TEMPLATE = os.path.join(META, "chat_template.jinja")
pytestmark = pytest.mark.skipif(not os.path.exists(TEMPLATE), reason=f"no {TEMPLATE} (set STEP5_META)")

BOS = "<｜begin▁of▁sentence｜>"


def template() -> str:
    with open(TEMPLATE, encoding="utf8") as f:
        return f.read()


def literal(src: str) -> str:
    return src.encode().decode("unicode_escape").encode("latin-1").decode("utf8")


# ------------------------------------------------------------------------------------------------ part 1: the text
def test_generation_prompt_ends_with_think_without_newline():
    m = re.search(r"if add_generation_prompt %\}\s*\{\{- '([^']*)' \}\}", template())
    assert m, "generation prompt literal not found"
    assert literal(m.group(1)) == "<|im_start|>assistant\n<think>"


def test_reasoning_system_line_and_roles():
    t = template()
    assert t.count("\"Reasoning: \" + reasoning_effort + '\\n\\n'") == 3        # tools / system / bare branches
    assert "reasoning_effort is defined" in t
    assert "'<|im_start|>tool_response\\n'" in t and "'<tool_response>'" in t
    assert "preserve_thinking | default(true)" in t
    assert "{%- generation %}" in t and "{%- endgeneration %}" in t
    assert "message.reasoning_content is string" in t
    assert "reasoning_content + '</think>' + content" in t
    assert "tool_call.arguments | fromjson" in t       # string branch: why build.py hands it a mapping


# ------------------------------------------------------------------------------------------------ part 2: rendering
@pytest.fixture(scope="module")
def renderer():
    pytest.importorskip("transformers")
    import build
    return build.Renderer(META)


TWO_TURN = {"id": "t0", "api_effort": "high", "tools": None, "messages": [
    {"role": "user", "content": "Hi"},
    {"role": "assistant", "content": "A1", "reasoning_content": "R1"},
    {"role": "user", "content": "Q2"},
    {"role": "assistant", "content": "A2", "reasoning_content": "R2"}]}


def test_two_turn_text_keeps_every_reasoning(renderer):
    want = (BOS + "<|im_start|>system\nReasoning: high\n\n<|im_end|>\n"
            "<|im_start|>user\nHi<|im_end|>\n<|im_start|>assistant\n<think>R1</think>A1<|im_end|>\n"
            "<|im_start|>user\nQ2<|im_end|>\n<|im_start|>assistant\n<think>R2</think>A2<|im_end|>\n")
    assert renderer.render_text(TWO_TURN) == want
    first = {**TWO_TURN, "messages": TWO_TURN["messages"][:1]}
    assert renderer.render_text(first, generation_prompt=True).endswith("<|im_start|>user\nHi<|im_end|>\n"
                                                                        "<|im_start|>assistant\n<think>")


def test_no_effort_means_no_system_line(renderer):
    assert "Reasoning:" not in renderer.render_text({**TWO_TURN, "api_effort": None})


def spans_of(ids, mask):
    spans, cur = [], []
    for t, m in zip(ids, mask):
        if m:
            cur.append(t)
        elif cur:
            spans.append(cur)
            cur = []
    return spans + ([cur] if cur else [])


def test_assistant_mask_covers_generated_spans_only(renderer):
    # chat_template.jinja:57-82: the generation block is reasoning + '</think>' + content + '<|im_end|>' + '\n'.
    # Compared as token ids (the expected text encoded alone: every piece sits between special tokens, so it
    # tokenizes the same in context), not through decode().
    ids, mask = renderer.encode(TWO_TURN)
    want = [renderer.tok(t, add_special_tokens=False)["input_ids"]
            for t in ("R1</think>A1<|im_end|>\n", "R2</think>A2<|im_end|>\n")]
    assert spans_of(ids, mask) == want
    assert ids[len(ids) - len(want[1]):] == want[1] and mask[-1] == 1     # the final newline is scored


def test_transformers_ids_match_tokenizer_json(renderer):
    """The ids build.py writes (transformers' tokenizer, as vLLM and the official generate.py use) equal the raw
    tokenizer.json pipeline's on a rendered conversation with newlines, CJK and digits."""
    tokenizers = pytest.importorskip("tokenizers")
    raw = tokenizers.Tokenizer.from_file(os.path.join(META, "tokenizer.json"))
    text = renderer.render_text({**TWO_TURN, "messages": TWO_TURN["messages"][:1] + [
        {"role": "assistant", "content": "第 1 行\n\nline 2: 12345", "reasoning_content": "想一想\n"}]})
    assert renderer.tok(text, add_special_tokens=False)["input_ids"] == raw.encode(text, add_special_tokens=False).ids


def test_tool_round_renders_xml_call_and_tool_response_role(renderer):
    tools = [{"type": "function", "function": {"name": "get_weather", "description": "Current weather for a city",
                                               "parameters": {"type": "object", "properties": {
                                                   "city": {"type": "string"}}, "required": ["city"]}}}]
    rec = {"id": "t1", "api_effort": "low", "tools": tools, "messages": [
        {"role": "user", "content": "Weather in Lisbon?"},
        {"role": "assistant", "content": "", "reasoning_content": "R",
         "tool_calls": [{"id": "c1", "type": "function",
                         "function": {"name": "get_weather", "arguments": "{\"city\": \"Lisbon\"}"}}]},
        {"role": "tool", "tool_call_id": "c1", "content": "{\"temp_c\": 23}"},
        {"role": "assistant", "content": "Rain.", "reasoning_content": "R2"}]}
    text = renderer.render_text(rec)
    assert text.startswith(BOS + "<|im_start|>system\nReasoning: low\n\n# Tools\n\n")
    assert ("<|im_start|>assistant\n<think>R</think><tool_call>\n<function=get_weather>\n<parameter=city>\nLisbon\n"
            "</parameter>\n</function>\n</tool_call><|im_end|>\n") in text
    assert "<|im_start|>tool_response\n<tool_response>{\"temp_c\": 23}</tool_response><|im_end|>\n" in text
    assert text.endswith("<|im_start|>assistant\n<think>R2</think>Rain.<|im_end|>\n")
    ids, mask = renderer.encode(rec)
    assert len(ids) == len(mask) and ids[0] == renderer.bos
    call = ("R</think><tool_call>\n<function=get_weather>\n<parameter=city>\nLisbon\n</parameter>\n</function>\n"
            "</tool_call><|im_end|>\n")
    want = [renderer.tok(t, add_special_tokens=False)["input_ids"] for t in (call, "R2</think>Rain.<|im_end|>\n")]
    assert spans_of(ids, mask) == want                   # the tool_response turn and both headers are not scored
