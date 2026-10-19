#!/usr/bin/env python3
"""Greedy outputs of a serving experiment, and how two experiments' outputs compare. Standard library only.

    python3 serve/greedy_cmp.py capture NAME [--base http://127.0.0.1:8888] [--max-tokens 384]
    python3 serve/greedy_cmp.py compare NAME_A NAME_B

capture sends a fixed prompt list one request at a time (temperature 0) and writes reasoning and content text to
runs/exp/NAME-greedy.json. compare prints, per prompt, whether the two outputs are identical and the length of their
common prefix in characters. Spec decoding verifies drafts with the target model, so draft-only changes should not
move these texts beyond the run-to-run floor of the kernels.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import urllib.request

EXP = os.path.join(os.path.expanduser("~"), "step5", "runs", "exp")
PROMPTS = [
    "Explain briefly how a hash map handles collisions.",
    "用三句话介绍长城的历史。",
    "Write a Python function that returns the n-th Fibonacci number iteratively, with a docstring.",
    "把这句话翻译成英文：今天的会议推迟到下午三点，请大家相互转告。",
    "What is 17 * 23? Show the steps.",
    "列出三种常见的排序算法，并说明它们的平均时间复杂度。",
]


def capture(a: argparse.Namespace) -> int:
    out = []
    for p in PROMPTS:
        body = {"model": "step5", "messages": [{"role": "user", "content": p}], "temperature": 0,
                "max_tokens": a.max_tokens}
        req = urllib.request.Request(a.base + "/v1/chat/completions", data=json.dumps(body).encode(),
                                     headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=1800) as r:
            msg = json.load(r)["choices"][0]["message"]
        text = (msg.get("reasoning_content") or msg.get("reasoning") or "") + "\n<content>\n" + (msg.get("content") or "")
        out.append({"prompt": p, "text": text})
        print(f"{len(text):6d} chars  {p[:40]}", flush=True)
    path = os.path.join(EXP, f"{a.name}-greedy.json")
    with open(path, "w") as f:
        json.dump(out, f, ensure_ascii=False, indent=1)
    print("wrote", path)
    return 0


def compare(a: argparse.Namespace) -> int:
    runs = []
    for name in (a.a, a.b):
        with open(os.path.join(EXP, f"{name}-greedy.json")) as f:
            runs.append(json.load(f))
    same = 0
    for x, y in zip(*runs):
        n = 0
        for cx, cy in zip(x["text"], y["text"]):
            if cx != cy:
                break
            n += 1
        ident = x["text"] == y["text"]
        same += ident
        print(f"{'same' if ident else 'DIFF'}  common prefix {n:5d} / {len(x['text']):5d} / {len(y['text']):5d} chars  "
              f"{x['prompt'][:40]}")
    print(f"{same} of {len(runs[0])} identical")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("capture")
    c.add_argument("name")
    c.add_argument("--base", default="http://127.0.0.1:8888")
    c.add_argument("--max-tokens", type=int, default=384)
    d = sub.add_parser("compare")
    d.add_argument("a")
    d.add_argument("b")
    a = ap.parse_args()
    return capture(a) if a.cmd == "capture" else compare(a)


if __name__ == "__main__":
    sys.exit(main())
