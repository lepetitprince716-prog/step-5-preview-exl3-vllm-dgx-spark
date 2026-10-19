#!/usr/bin/env python3
"""Generate EvalPlus (HumanEval+ / MBPP+) solutions from an OpenAI-compatible chat endpoint, with concurrency and a
token budget large enough for a model that thinks before answering. EvalPlus's own `codegen --backend openai` is
sequential and caps completions at 768 tokens, which cuts Step 5's reasoning off.

    gen_evalplus.py --dataset humaneval --base-url http://127.0.0.1:8888/v1 --model step5 --out out/he-local
    gen_evalplus.py --dataset mbpp --base-url https://api.stepfun.com/step_plan/v1 --model step-5-preview \
        --api-key-file ~/step5/stepfun.env --out out/mbpp-api --workers 4

Prompting is EvalPlus's chat recipe verbatim (instruction prefix + the problem in a ```python block), temperature 0.
Writes <out>/samples.raw.jsonl ({"task_id", "solution": the assistant content}) for `evalplus.sanitize` /
`evalplus.evaluate`, and <out>/meta.jsonl (usage, finish reason, reasoning length, seconds per task). Re-running resumes
from the tasks already in samples.raw.jsonl. The API key is read from a file (KEY=value lines or a bare key) and never
printed.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import openai

INSTRUCTION_PREFIX = ("Please provide a self-contained Python script that solves the following problem in a markdown "
                      "code block:")


def load_key(path: str) -> str:
    for line in open(path).read().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        return line.split("=", 1)[1].strip().strip("'\"") if "=" in line else line
    raise SystemExit(f"no key in {path}")


def problems(dataset: str) -> dict:
    if dataset == "humaneval":
        from evalplus.data import get_human_eval_plus
        return get_human_eval_plus()
    if dataset == "mbpp":
        from evalplus.data import get_mbpp_plus
        return get_mbpp_plus()
    raise SystemExit(f"unknown dataset {dataset}")


def one(client: openai.OpenAI, model: str, task_id: str, prompt: str, max_tokens: int, timeout: float,
        retries: int) -> dict:
    message = INSTRUCTION_PREFIX + f"\n```python\n{prompt.strip()}\n```"
    last = ""
    for attempt in range(retries + 1):
        t0 = time.time()
        try:
            r = client.chat.completions.create(model=model, messages=[{"role": "user", "content": message}],
                                               temperature=0, max_tokens=max_tokens, timeout=timeout)
            ch = r.choices[0]
            msg = ch.message
            reasoning = getattr(msg, "reasoning_content", None) or getattr(msg, "reasoning", None) or ""
            if not reasoning and hasattr(msg, "model_extra") and msg.model_extra:
                reasoning = msg.model_extra.get("reasoning_content") or msg.model_extra.get("reasoning") or ""
            u = r.usage
            return {"task_id": task_id, "solution": msg.content or "", "finish_reason": ch.finish_reason,
                    "prompt_tokens": getattr(u, "prompt_tokens", None), "completion_tokens":
                    getattr(u, "completion_tokens", None), "reasoning_chars": len(reasoning),
                    "content_chars": len(msg.content or ""), "seconds": round(time.time() - t0, 1),
                    "attempt": attempt}
        except Exception as e:  # noqa: BLE001  (retry on any transport / API error, record the last one)
            last = f"{type(e).__name__}: {str(e)[:200]}"
            time.sleep(min(60, 5 * (attempt + 1)))
    return {"task_id": task_id, "solution": "", "error": last, "attempt": retries}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, choices=["humaneval", "mbpp"])
    ap.add_argument("--base-url", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--api-key-file", default="")
    ap.add_argument("--out", required=True)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--max-tokens", type=int, default=16384)
    ap.add_argument("--timeout", type=float, default=1800)
    ap.add_argument("--retries", type=int, default=3)
    ap.add_argument("--limit", type=int, default=0, help="first N tasks only (smoke)")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    raw_path, meta_path = os.path.join(a.out, "samples.raw.jsonl"), os.path.join(a.out, "meta.jsonl")
    done = set()
    if os.path.exists(raw_path):
        for line in open(raw_path):
            if line.strip():
                done.add(json.loads(line)["task_id"])
    key = load_key(os.path.expanduser(a.api_key_file)) if a.api_key_file else "none"
    client = openai.OpenAI(api_key=key, base_url=a.base_url)
    probs = problems(a.dataset)
    todo = [(tid, p["prompt"]) for tid, p in probs.items() if tid not in done]
    if a.limit:
        todo = todo[:a.limit]
    print(f"{a.dataset}: {len(probs)} tasks, {len(done)} done, {len(todo)} to go, {a.workers} workers, "
          f"max_tokens {a.max_tokens}", flush=True)
    lock = threading.Lock()
    t_all = time.time()
    n_ok = n_err = 0
    with open(raw_path, "a") as raw, open(meta_path, "a") as meta, ThreadPoolExecutor(a.workers) as ex:
        futs = {ex.submit(one, client, a.model, tid, pr, a.max_tokens, a.timeout, a.retries): tid for tid, pr in todo}
        for i, f in enumerate(as_completed(futs), 1):
            r = f.result()
            with lock:
                if "error" in r:
                    n_err += 1
                    print(f"[{i}/{len(todo)}] {r['task_id']}: ERROR {r['error']}", flush=True)
                else:
                    n_ok += 1
                    raw.write(json.dumps({"task_id": r["task_id"], "solution": r["solution"]}) + "\n")
                    raw.flush()
                    print(f"[{i}/{len(todo)}] {r['task_id']}: finish={r['finish_reason']} "
                          f"completion={r['completion_tokens']} reasoning_chars={r['reasoning_chars']} "
                          f"content_chars={r['content_chars']} {r['seconds']}s", flush=True)
                meta.write(json.dumps(r) + "\n")
                meta.flush()
    print(f"DONE ok={n_ok} err={n_err} in {time.time() - t_all:.0f}s -> {raw_path}", flush=True)
    return 1 if n_err else 0


if __name__ == "__main__":
    sys.exit(main())
