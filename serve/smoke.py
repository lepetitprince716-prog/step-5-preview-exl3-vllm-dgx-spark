#!/usr/bin/env python3
"""Smoke test for the served Step 5 (OpenAI-compatible API on the head, loopback). Standard library only.

    python3 serve/smoke.py [--base http://127.0.0.1:8888] [--model step5] [--max-tokens 256] [--long N]

Three requests, one line each: plain chat (greedy), a tool-call request (tools given, auto choice), and a reasoning
request (the step3p5 reasoning parser splits reasoning_content from content). --long N adds a prompt of about N tokens
(a repeated numbered list) and asks for the last line, checking long-context prefill. Prints finish reason, usage,
reasoning / content lengths, tool calls, and decode tok/s (completion tokens over wall time minus the first-token wait
is not available through the non-streaming API, so the figure is a lower bound). Exit status 1 if a request fails.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.error
import urllib.request

TOOLS = [{
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "Get the current weather for a city.",
        "parameters": {"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]},
    },
}]


def post(base: str, body: dict, timeout: int = 1800) -> tuple[dict, float]:
    req = urllib.request.Request(base + "/v1/chat/completions", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.load(r), time.time() - t0


def run(name: str, base: str, body: dict) -> bool:
    try:
        res, secs = post(base, body)
    except urllib.error.HTTPError as e:
        print(f"{name}: HTTP {e.code} {e.read()[:300]!r}", flush=True)
        return False
    except Exception as e:  # noqa: BLE001
        print(f"{name}: {type(e).__name__}: {e}", flush=True)
        return False
    ch = res["choices"][0]
    msg = ch["message"]
    u = res.get("usage", {})
    comp = u.get("completion_tokens", 0)
    reasoning = msg.get("reasoning_content") or msg.get("reasoning") or ""
    content = msg.get("content") or ""
    tools = msg.get("tool_calls") or []
    print(f"{name}: finish={ch.get('finish_reason')} prompt={u.get('prompt_tokens')} completion={comp} "
          f"reasoning_chars={len(reasoning)} content_chars={len(content)} tool_calls={len(tools)} "
          f"secs={secs:.1f} tok/s>={comp / max(secs, 1e-9):.1f}", flush=True)
    if tools:
        print(f"  tool: {json.dumps(tools[0].get('function', {}))[:200]}", flush=True)
    print(f"  content: {content[:200]!r}", flush=True)
    return True


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="http://127.0.0.1:8888")
    ap.add_argument("--model", default="step5")
    ap.add_argument("--max-tokens", type=int, default=256)
    ap.add_argument("--long", type=int, default=0, help="approximate prompt tokens for a long-context request")
    a = ap.parse_args()
    ok = True
    base = {"model": a.model, "temperature": 0, "max_tokens": a.max_tokens}
    ok &= run("chat", a.base, {**base, "messages": [{"role": "user", "content": "Reply with the single word OK."}]})
    ok &= run("tool", a.base, {**base, "tools": TOOLS, "tool_choice": "auto",
                               "messages": [{"role": "user", "content": "What is the weather in Paris right now?"}]})
    ok &= run("reason", a.base, {**base, "max_tokens": max(a.max_tokens, 1024),
                                 "messages": [{"role": "user", "content": "What is 17 * 23? Show the steps briefly."}]})
    if a.long:
        lines = [f"line {i}: item number {i}" for i in range(a.long // 8)]
        prompt = "\n".join(lines) + "\n\nWhich item number is on the last line? Answer with the number only."
        ok &= run(f"long{a.long}", a.base, {**base, "max_tokens": 32,
                                             "messages": [{"role": "user", "content": prompt}]})
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
