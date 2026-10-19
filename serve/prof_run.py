#!/usr/bin/env python3
"""Capture a torch-profiler trace of single-stream decode on a server started with `--profiler-config` (torch).

Sends a short warm-up, POST /start_profile, one streamed chat completion (greedy), POST /stop_profile, and prints the
measured decode rate. The workers write their traces under the profiler directory on each node.

    python3 serve/prof_run.py [--tokens 300] [--prompt TEXT]
"""
from __future__ import annotations

import argparse
import json
import time
import urllib.request

API = "http://127.0.0.1:8888"
PROMPT = ("Write a Python class LRUCache with get(key) and put(key, value), both O(1), using a dict and a doubly "
          "linked list. Then write unittest test cases for it.")


def post(path: str, body: dict | None = None, timeout: float = 600):
    data = json.dumps(body).encode() if body is not None else b""
    req = urllib.request.Request(API + path, data=data, headers={"Content-Type": "application/json"}, method="POST")
    return urllib.request.urlopen(req, timeout=timeout)


def chat(prompt: str, tokens: int) -> dict:
    body = {"model": "step5", "messages": [{"role": "user", "content": prompt}], "max_tokens": tokens,
            "temperature": 0, "stream": True, "stream_options": {"include_usage": True}}
    t0, first, last, n = time.perf_counter(), None, None, None
    with post("/v1/chat/completions", body) as resp:
        for raw in resp:
            line = raw.decode().strip()
            if not line.startswith("data:") or line == "data: [DONE]":
                continue
            chunk = json.loads(line[5:])
            if chunk.get("usage"):
                n = chunk["usage"]["completion_tokens"]
            for ch in chunk.get("choices", []):
                d = ch.get("delta") or {}
                if d.get("content") or d.get("reasoning") or d.get("reasoning_content"):
                    now = time.perf_counter()
                    first = first if first is not None else now
                    last = now
    return {"tokens": n, "ttft_s": (first - t0) if first else None,
            "decode_tps": (n - 1) / (last - first) if n and first and last and last > first else None}


def prompt_ids(n: int) -> list[int]:
    """n token ids of plain English filler (tokenized by the server), distinct per call so no prefix-cache hit."""
    words = ("the river carried small boats past old stone houses while workers counted grain and sang about "
             "distant harbors lanterns bells markets bridges").split()
    salt = str(time.time_ns())
    text = f"Notes {salt}. " + " ".join(words[i % len(words)] for i in range(int(n * 1.1) + 64))
    with post("/tokenize", {"model": "step5", "prompt": text, "add_special_tokens": True}) as r:
        ids = json.load(r)["tokens"]
    if len(ids) < n:
        raise SystemExit(f"filler too short: {len(ids)} < {n}")
    return ids[:n]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tokens", type=int, default=300)
    ap.add_argument("--prompt", default=PROMPT)
    ap.add_argument("--prefill", type=int, default=0,
                    help="profile one prompt of this many tokens instead of decode (a 32-token reply runs first so "
                         "the profiler's delay_iterations pass before the prompt)")
    a = ap.parse_args()
    chat("Write a haiku about rain.", 32)
    if a.prefill:
        ids = prompt_ids(a.prefill)
        post("/start_profile").read()
        chat("Count from 1 to 40.", 32)
        t0 = time.perf_counter()
        with post("/v1/completions", {"model": "step5", "prompt": ids, "max_tokens": 1, "temperature": 0}) as r:
            r.read()
        r = {"prefill_tokens": len(ids), "prefill_s": time.perf_counter() - t0}
    else:
        post("/start_profile").read()
        r = chat(a.prompt, a.tokens)
    post("/stop_profile", timeout=1800).read()
    print(json.dumps(r), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
