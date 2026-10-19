# SPDX-License-Identifier: Apache-2.0
# Derived from sudoingx dsv41q scripts/quantize/caldata/generate.py (Apache-2.0): same prompt sets, tool loop, follow-up
# turns, resume and budget stop; the endpoint is StepFun's official API instead of OpenRouter.
"""Generate native Step-5-Preview conversations for the calibration set and the KLD panel through StepFun's official
chat-completions API (model step-5-preview). Resumable: done ids in the output JSONL are skipped. Hard budget stop.

The prompts are sudoingx's published sets, unchanged (prompts.py, prompts_instruct.py); the canned tool results are
prompts.TOOL_RESULTS. Every answer, reasoning trace and tool call in the output comes from the API.

API format (verified 2026-10-15, see notes/laneC.md):
  POST {base}/v1/chat/completions, Bearer key; reasoning_effort low|medium|high; reasoning_format deepseek-style makes
  the reply carry message.reasoning_content (default "general" uses message.reasoning); tool_calls OpenAI-style with
  arguments as a JSON string; usage.prompt_tokens / completion_tokens, cached tokens in
  usage.prompt_tokens_details.cached_tokens (cn docs) or usage.cached_tokens (intl docs).
  --platform intl: https://api.stepfun.ai, USD 1.00 in / 0.10 cached / 2.70 out per M tokens.
  --platform cn:   https://api.stepfun.com, CNY 7 in / 0.70 cached / 20 out per M tokens (--cny-per-usd required).

Step 5 has no thinking-off mode (its chat template always opens <think>), so sudoingx's labels are mapped onto the
three API efforts with --effort-map (default off=low,low=medium,high=high); the effort sent is stored per record as
api_effort and build.py renders the system line "Reasoning: <api_effort>" from it.

The key comes from --key-file (a STEPFUN_API_KEY= line; default /q/stepfun.env, else ~/step5/stepfun.env) or the
STEPFUN_API_KEY environment variable. It is never printed or written.

  python3 generate_stepfun.py out.jsonl --platform intl [--wiki wiki.utf8] [--workers 8] [--budget-usd 5]
  python3 generate_stepfun.py instruct.jsonl --platform intl --set instruct
"""
from __future__ import annotations

import argparse
import json
import os
import stat
import sys
import threading
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import prompts  # noqa: E402
import prompts_instruct  # noqa: E402
from prompts import TOOL_RESULTS  # noqa: E402

SETS = {"base": prompts, "instruct": prompts_instruct}
MODEL = "step-5-preview"
API_EFFORTS = ("low", "medium", "high")
# (chat-completions URL, currency, price per M tokens: input cache miss, input cache hit, output)
PLATFORMS = {
    "intl": ("https://api.stepfun.ai/v1/chat/completions", "USD", 1.00, 0.10, 2.70),
    "cn": ("https://api.stepfun.com/v1/chat/completions", "CNY", 7.00, 0.70, 20.00),
    # Step Plan subscription: same host, dedicated path prefix; usage is deducted from the plan's monthly credits
    # (platform.stepfun.com/docs/zh/step-plan/integrations/reasoning-api). The CNY list prices are kept here only to
    # report an equivalent spend; the budget fuse then counts credits at list price.
    "plan": ("https://api.stepfun.com/step_plan/v1/chat/completions", "CNY", 7.00, 0.70, 20.00),
}
KEY_FILES = ("/q/stepfun.env", "~/step5/stepfun.env")
RETRY_STATUS = {408, 409, 425, 429, 500, 502, 503, 504}
FATAL_STATUS = {401, 402, 403}
BASE_SPLIT = {"cal": 299, "panel": 201}      # RECIPE.md section 2 (wiki.utf8 from exllamav3 v1.6.0 @151539c7)


class Fatal(RuntimeError):
    """Auth or billing failure: stop every worker."""


def parse_effort_map(text: str) -> dict[str, str]:
    """'off=low,low=medium,high=high' -> {"off": "low", ...}; every sudoingx label must map to an API effort."""
    out = {}
    for part in text.split(","):
        k, _, v = part.partition("=")
        k, v = k.strip(), v.strip()
        if k not in ("off", "low", "high") or v not in API_EFFORTS:
            raise ValueError(f"bad effort map entry {part!r}: keys off/low/high, values {API_EFFORTS}")
        out[k] = v
    if set(out) != {"off", "low", "high"}:
        raise ValueError(f"effort map must cover off, low and high, got {sorted(out)}")
    return out


def api_effort(item: dict, emap: dict[str, str]) -> str:
    return emap[item["effort"]] if item["thinking"] else emap["off"]


def read_key(path: str) -> str:
    """The API key from an env file (STEPFUN_API_KEY= line) or the environment. Never printed."""
    cands = [path] if path else [p for p in KEY_FILES if os.path.exists(os.path.expanduser(p))]
    for p in cands:
        p = os.path.expanduser(p)
        if os.stat(p).st_mode & (stat.S_IRWXG | stat.S_IRWXO):
            print(f"warning: {p} is readable by group/others; chmod 600 it", file=sys.stderr)
        with open(p) as f:
            for line in f:
                if line.startswith("STEPFUN_API_KEY="):
                    key = line.split("=", 1)[1].strip().strip('"').strip("'")
                    if key:
                        return key
        raise SystemExit(f"no STEPFUN_API_KEY= line in {p}")
    if os.environ.get("STEPFUN_API_KEY"):
        return os.environ["STEPFUN_API_KEY"].strip()
    raise SystemExit("set STEPFUN_API_KEY or pass --key-file <env file with a STEPFUN_API_KEY= line>")


def call_cost(usage: dict, prices: tuple[float, float, float]) -> float:
    """Cost of one call in the platform's currency from the documented usage fields."""
    p_in, p_cache, p_out = prices
    prompt = int(usage.get("prompt_tokens") or 0)
    cached = int((usage.get("prompt_tokens_details") or {}).get("cached_tokens") or usage.get("cached_tokens") or 0)
    out = int(usage.get("completion_tokens") or 0)
    return ((prompt - cached) * p_in + cached * p_cache + out * p_out) / 1e6


def assistant_message(resp: dict) -> tuple[dict, str | None]:
    """The reply as an OpenAI-format assistant message that keeps the reasoning and the tool calls."""
    m = resp["choices"][0]["message"]
    msg = {"role": "assistant", "content": m.get("content") or ""}
    reasoning = m.get("reasoning_content") or m.get("reasoning")
    if reasoning:
        msg["reasoning_content"] = reasoning
    if m.get("tool_calls"):
        msg["tool_calls"] = [{"id": t.get("id"), "type": "function",
                              "function": {"name": t["function"]["name"], "arguments": t["function"]["arguments"]}}
                             for t in m["tool_calls"]]
    return msg, resp["choices"][0].get("finish_reason")


def history(msgs: list[dict], send_reasoning: bool) -> list[dict]:
    """Messages as sent back to the API: reasoning_content kept on earlier assistant turns unless disabled."""
    if send_reasoning:
        return msgs
    return [{k: v for k, v in m.items() if k != "reasoning_content"} for m in msgs]


class Generator:
    def __init__(self, a: argparse.Namespace, key: str):
        self.url, self.currency, *prices = PLATFORMS[a.platform]
        self.prices = tuple(prices)
        self.to_usd = 1.0 if self.currency == "USD" else 1.0 / a.cny_per_usd
        self.key, self.a = key, a
        self.emap = parse_effort_map(a.effort_map)
        self.followups = SETS[a.set].FOLLOWUPS
        self.lock = threading.Lock()
        self.spent_usd = 0.0
        self.dropped_usd = 0.0          # spend on items that failed part-way (counted in spent_usd too)
        self.stop = threading.Event()

    def post(self, body: dict) -> dict:
        data = json.dumps(body).encode()
        for attempt in range(self.a.tries):
            req = urllib.request.Request(self.url, data=data, headers={"Authorization": f"Bearer {self.key}",
                                                                       "Content-Type": "application/json"})
            try:
                with urllib.request.urlopen(req, timeout=self.a.timeout) as r:
                    out = json.load(r)
                if "choices" not in out:
                    raise RuntimeError(str(out)[:300])
                return out
            except urllib.error.HTTPError as e:
                msg = f"HTTP {e.code}: {e.read().decode(errors='replace')[:300]}"
                if e.code in FATAL_STATUS:
                    self.stop.set()
                    raise Fatal(msg) from None
                if e.code not in RETRY_STATUS or attempt == self.a.tries - 1:
                    raise RuntimeError(msg) from None
            except (urllib.error.URLError, TimeoutError, ConnectionError, RuntimeError, json.JSONDecodeError) as e:
                if attempt == self.a.tries - 1:
                    raise RuntimeError(f"failed after {self.a.tries}: {str(e)[:300]}") from None
            time.sleep(min(120, 4 * 2 ** attempt))
        raise AssertionError("unreachable")

    def over_budget(self) -> bool:
        with self.lock:
            return self.spent_usd >= self.a.budget_usd

    def run_item(self, it: dict) -> dict | None:
        if self.stop.is_set() or self.over_budget():
            return None
        msgs = list(it["messages"])
        effort = api_effort(it, self.emap)
        meta = {"calls": 0, "cost_usd": 0.0, "finish": [], "usage": [], "model": [], "system_fingerprint": [],
                "platform": self.a.platform, "api_effort": effort, "reasoning_sent_back": self.a.send_reasoning}

        def step() -> dict:
            if self.stop.is_set() or self.over_budget():      # checked before every call, not only per item
                raise RuntimeError("budget or fatal stop reached mid-conversation")
            body = {"model": MODEL, "messages": history(msgs, self.a.send_reasoning), "max_tokens": self.a.max_tokens,
                    "reasoning_effort": effort, "reasoning_format": "deepseek-style",
                    "temperature": self.a.temperature, "top_p": self.a.top_p}
            if it["tools"]:
                body["tools"] = it["tools"]
            resp = self.post(body)
            u = resp.get("usage") or {}
            cost = call_cost(u, self.prices) * self.to_usd
            with self.lock:
                self.spent_usd += cost
            meta["calls"] += 1
            meta["cost_usd"] += cost
            meta["usage"].append(u)
            meta["model"].append(resp.get("model"))
            meta["system_fingerprint"].append(resp.get("system_fingerprint"))
            a, fin = assistant_message(resp)
            meta["finish"].append(fin)
            msgs.append(a)
            return a

        try:
            a = step()
            rounds = 0
            while a.get("tool_calls") and rounds < 3:          # sudoingx: canned results for every call, 3 rounds
                for tc in a["tool_calls"]:
                    try:
                        args = json.loads(tc["function"]["arguments"] or "{}")
                    except json.JSONDecodeError:
                        args = {}
                    fn = TOOL_RESULTS.get(tc["function"]["name"], lambda _: {"error": "unknown tool"})
                    msgs.append({"role": "tool", "tool_call_id": tc["id"], "content": json.dumps(fn(args))})
                a = step()
                rounds += 1
            if it.get("followup") and not a.get("tool_calls"):
                msgs.append({"role": "user", "content": self.followups[int(it["id"][1:]) % len(self.followups)]})
                step()
        except Exception:
            with self.lock:                  # spend of a dropped item still counts, across restarts too
                self.dropped_usd += meta["cost_usd"]
                with open(self.a.out + ".failed.jsonl", "a") as f:
                    f.write(json.dumps({"id": it["id"], "cost_usd": meta["cost_usd"]}) + "\n")
            raise
        keep = ("id", "domain", "split", "thinking", "effort", "tools", "followup")
        return {**{k: it.get(k) for k in keep}, "api_effort": effort, "messages": msgs, "meta": meta}


def load_done(path: str) -> tuple[set[str], float]:
    """Ids already in the output and the USD spent so far (finished items plus failed ones in <out>.failed.jsonl)."""
    done, spent = set(), 0.0
    for p, is_out in ((path, True), (path + ".failed.jsonl", False)):
        if os.path.exists(p):
            with open(p) as f:
                for line in f:
                    r = json.loads(line)
                    if is_out:
                        done.add(r["id"])
                    spent += float(r["meta"]["cost_usd"] if is_out else r["cost_usd"])
    return done, spent


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("out")
    ap.add_argument("--platform", choices=sorted(PLATFORMS), required=True,
                    help="intl = api.stepfun.ai (USD), cn = api.stepfun.com (CNY), plan = api.stepfun.com/step_plan "
                         "(Step Plan credits, CNY-equivalent); must match where the key was issued")
    ap.add_argument("--cny-per-usd", type=float, default=0.0, help="required with --platform cn (budget is in USD)")
    ap.add_argument("--set", choices=sorted(SETS), default="base")
    ap.add_argument("--wiki", help="exllamav3 v1.6.0 conversion/standard_cal_data/wiki.utf8 (long-context prompts)")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--budget-usd", type=float, default=5.0)
    ap.add_argument("--key-file", default="")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--max-tokens", type=int, default=6000)
    ap.add_argument("--temperature", type=float, default=1.0, help="model card's recommended sampling (README.md:90)")
    ap.add_argument("--top-p", type=float, default=0.95)
    ap.add_argument("--effort-map", default="off=low,low=medium,high=high")
    ap.add_argument("--no-send-reasoning", dest="send_reasoning", action="store_false",
                    help="strip reasoning_content from earlier assistant turns when sending the history back")
    ap.add_argument("--tries", type=int, default=5)
    ap.add_argument("--timeout", type=float, default=600)
    a = ap.parse_args()
    if a.platform in ("cn", "plan") and a.cny_per_usd <= 0:
        raise SystemExit("--platform cn bills in CNY: pass --cny-per-usd <rate> so the USD budget can be enforced")
    parse_effort_map(a.effort_map)
    key = read_key(a.key_file)
    g = Generator(a, key)
    items = SETS[a.set].build_prompt_set(open(a.wiki, encoding="utf8").read() if a.wiki else None)
    split = {s: sum(1 for x in items if x["split"] == s) for s in ("cal", "panel")}
    print(f"set {a.set}: {len(items)} prompts, split {split}", flush=True)
    if a.set == "base" and a.wiki and split != BASE_SPLIT:
        print(f"warning: split {split} != sudoingx's {BASE_SPLIT}: --wiki is not exllamav3 v1.6.0's wiki.utf8",
              flush=True)
    done, g.spent_usd = load_done(a.out)
    todo = [it for it in items if it["id"] not in done]
    if a.limit:
        todo = todo[: a.limit]
    print(f"{len(done)} done, {len(todo)} to go, spent so far ${g.spent_usd:.4f} of ${a.budget_usd}", flush=True)
    t0, n_ok, n_err, n_skip = time.time(), 0, 0, 0
    with open(a.out, "a") as f, ThreadPoolExecutor(a.workers) as ex:
        futs = {ex.submit(g.run_item, it): it for it in todo}
        for fut in futs:
            it = futs[fut]
            try:
                r = fut.result()
            except Exception as e:  # noqa: BLE001
                n_err += 1
                print(f"[err] {it['id']} {it['domain']}: {str(e)[:200]}", flush=True)
                continue
            if r is None:
                n_skip += 1
                continue
            with g.lock:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
                f.flush()
            n_ok += 1
            if n_ok % 20 == 0:
                print(f"{n_ok} ok, {n_err} err, ${g.spent_usd:.4f}, {time.time() - t0:.0f}s", flush=True)
    if g.stop.is_set():
        print("[stop] fatal API error (auth or billing): fix the key or the account, then rerun to resume", flush=True)
    elif n_skip:
        print(f"[stop] budget ${a.budget_usd} reached: {n_skip} prompts not started", flush=True)
    print(f"DONE {n_ok} ok, {n_err} err, {n_skip} skipped; total spent ${g.spent_usd:.4f} "
          f"(incl. ${g.dropped_usd:.4f} on failed items), {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
