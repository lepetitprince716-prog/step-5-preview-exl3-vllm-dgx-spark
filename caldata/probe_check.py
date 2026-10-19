#!/usr/bin/env python3
"""Summarize a generate_stepfun.py JSONL without printing any secret or the full text.

    probe_check.py FILE.jsonl
"""
from __future__ import annotations

import json
import sys


def main() -> int:
    path = sys.argv[1]
    try:
        records = [json.loads(line) for line in open(path) if line.strip()]
    except (OSError, ValueError) as exc:
        print(f"no records: {exc}")
        return 1
    print(f"{len(records)} records")
    for rec in records[-2:]:
        print("keys:", sorted(rec.keys()))
        msgs = rec.get("messages") or []
        last = msgs[-1] if msgs else {}
        reasoning = last.get("reasoning_content") or last.get("reasoning") or ""
        print(
            "last role", last.get("role"),
            "content_len", len(last.get("content") or ""),
            "reasoning_len", len(reasoning),
            "tool_calls", len(last.get("tool_calls") or []),
        )
        for key in ("usage", "api_effort", "model", "cost_usd", "finish_reason", "error"):
            if key in rec:
                print(key, rec[key])
    return 0


if __name__ == "__main__":
    sys.exit(main())
