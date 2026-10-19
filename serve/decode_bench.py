#!/usr/bin/env python3
"""Decode speed of the served Step 5 with kit_bench.py's prompt sets and timing, counting reasoning deltas too.

kit_bench.Client.stream times only chunks whose delta carries `content`; Step 5 always thinks first and the image's
vLLM streams the thinking as `delta.reasoning` (the step3p5 reasoning parser; `reasoning_content` in older vLLM), so a
384-token completion often has no content chunk at all and kit_bench's decode command sees no samples. This wrapper
subclasses the client to count content, reasoning and reasoning_content deltas and runs
kit_bench's own cmd_decode (same prompts, same tok/s definition: (completion tokens - 1) / (last chunk - first chunk)).

    python3 serve/decode_bench.py --kit ~/dsv41-tensorfold/engine-tp2/tools/dsv41/kit_bench.py \
        --base http://127.0.0.1:8888 --model step5 --set b --tokens 384 --reps 2 --out decode.json

`--cmd concurrent` / `--cmd sustained` run kit_bench's commands of the same name with this client (`--streams`,
`--seconds`). Every run also records vLLM's speculative-decoding counters from /metrics before and after, so the
summary carries the mean acceptance length of the measured window.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import statistics
import sys
import time
import urllib.error
import urllib.request


def load_kit(path: str):
    spec = importlib.util.spec_from_file_location("kit_bench", path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["kit_bench"] = mod
    spec.loader.exec_module(mod)
    return mod


SPEC_COUNTERS = ("vllm:spec_decode_num_drafts", "vllm:spec_decode_num_draft_tokens",
                 "vllm:spec_decode_num_accepted_tokens")


def spec_counters(base: str) -> dict[str, float]:
    """Sum of each speculative-decoding counter over all label sets (missing counters read as 0)."""
    totals = dict.fromkeys(SPEC_COUNTERS, 0.0)
    try:
        with urllib.request.urlopen(base.rstrip("/") + "/metrics", timeout=10) as r:
            text = r.read().decode()
    except (urllib.error.URLError, OSError):
        return totals
    for line in text.splitlines():
        if line.startswith("#"):
            continue
        name = line.split("{", 1)[0].split(" ", 1)[0]
        for c in SPEC_COUNTERS:
            if name in (c, c + "_total"):
                totals[c] += float(line.rsplit(" ", 1)[1])
    return totals


def acceptance(before: dict[str, float], after: dict[str, float]) -> dict:
    d = {k: after[k] - before[k] for k in SPEC_COUNTERS}
    drafts = d["vllm:spec_decode_num_drafts"]
    return {"drafts": drafts, "draft_tokens": d["vllm:spec_decode_num_draft_tokens"],
            "accepted_tokens": d["vllm:spec_decode_num_accepted_tokens"],
            "mean_acceptance_length": (1 + d["vllm:spec_decode_num_accepted_tokens"] / drafts) if drafts else None}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--kit", required=True, help="path of kit_bench.py")
    ap.add_argument("--base", default="http://127.0.0.1:8888")
    ap.add_argument("--model", default="step5")
    ap.add_argument("--cmd", choices=("decode", "concurrent", "sustained"), default="decode")
    ap.add_argument("--set", default="b")
    ap.add_argument("--tokens", type=int, default=384)
    ap.add_argument("--reps", type=int, default=2)
    ap.add_argument("--streams", default="2,4", help="concurrent / sustained: comma-separated stream counts")
    ap.add_argument("--seconds", type=float, default=90, help="sustained: window per stream count")
    ap.add_argument("--out", default="")
    a = ap.parse_args()
    kit = load_kit(os.path.expanduser(a.kit))

    class Client(kit.Client):
        def stream(self, body: dict, path: str) -> dict:
            body = dict(body, model=self.model, stream=True, stream_options={"include_usage": True})
            start = time.perf_counter()
            first = last = None
            usage, pieces, n_chunks = None, [], 0
            with self._req(path, body) as resp:
                for raw in resp:
                    line = raw.decode().strip()
                    if not line.startswith("data:") or line == "data: [DONE]":
                        continue
                    chunk = json.loads(line[5:])
                    if chunk.get("usage"):
                        usage = chunk["usage"]
                    for choice in chunk.get("choices", []):
                        delta = choice.get("delta") or {}
                        piece = (choice.get("text") or delta.get("content") or delta.get("reasoning")
                                 or delta.get("reasoning_content") or "")
                        if piece:
                            now = time.perf_counter()
                            first = first if first is not None else now
                            last = now
                            n_chunks += 1
                            pieces.append(piece)
            end = time.perf_counter()
            n = int(usage["completion_tokens"]) if usage else None
            return {"start": start, "end": end, "ttft_s": (first - start) if first else None,
                    "decode_s": (last - first) if first else None, "tokens": n,
                    "prompt_tokens": int(usage["prompt_tokens"]) if usage else None,
                    "decode_tps": (n - 1) / (last - first) if n and first and last > first else None,
                    "chunks": n_chunks, "text": "".join(pieces)}

    c = Client(a.base, a.model, None)
    before = spec_counters(a.base)
    if a.cmd != "decode":
        rows = kit.cmd_concurrent(c, a) if a.cmd == "concurrent" else kit.cmd_sustained(c, a)
        summary = {"cmd": a.cmd, "set": a.set, "tokens": a.tokens, "streams": a.streams,
                   "spec": acceptance(before, spec_counters(a.base)), "rows": rows}
        print(f"{a.cmd}: spec {summary['spec']}", flush=True)
        if a.out:
            json.dump(summary, open(a.out, "w"), indent=1)
        return 0
    rows = kit.cmd_decode(c, a)
    med = [r["decode_tps_median"] for r in rows]
    summary = {"set": a.set, "tokens": a.tokens, "reps": a.reps, "prompts": len(rows),
               "decode_tps_median_of_medians": statistics.median(med), "decode_tps_min": min(med),
               "decode_tps_max": max(med), "spec": acceptance(before, spec_counters(a.base)), "rows": rows}
    for r in rows:
        print(f"{r['prompt']:5s} tok/s {r['decode_tps_median']:.2f} tokens {r['tokens_all']} ttft {r['ttft_s_median']:.2f}s",
              flush=True)
    print(f"set {a.set}: {len(rows)} prompts, median {summary['decode_tps_median_of_medians']:.2f} tok/s "
          f"(min {summary['decode_tps_min']:.2f}, max {summary['decode_tps_max']:.2f}), spec {summary['spec']}",
          flush=True)
    if a.out:
        json.dump(summary, open(a.out, "w"), indent=1)
    return 0


if __name__ == "__main__":
    sys.exit(main())
