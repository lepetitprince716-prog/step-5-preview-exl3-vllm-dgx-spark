# SPDX-License-Identifier: Apache-2.0
# Derived from sudoingx dsv41q scripts/quantize/caldata/build.py and build_supplement.py (Apache-2.0): same layout,
# split rule, truncation rule and exllamav3 standard-text sampling; the renderer is Step 5's official chat template.
"""Token sets for the Step 5 passes, from generate_stepfun.py's conversations plus exllamav3's standard calibration
text (MIT, turboderp-org):

  cal.safetensors    tokens int32 [T], offsets int64 [n+1]            calibration sequences (variable length)
  cal_index.json     per sequence: source, id, domain, length (+ truncated_from when cut)
  panel.safetensors  tokens, offsets, score uint8 [T] (1 = assistant-generated position)   the KLD panel
  panel_index.json   per sequence: id, domain, length, scored, thinking, effort, api_effort, tools (+ truncated_from)
  sha256.txt         sha256 of cal.safetensors and panel.safetensors (also printed)

Rendering: the checkpoint's chat_template.jinja through transformers' apply_chat_template with the official tokenizer
(tokenize=True, so no extra special tokens: the template writes BOS itself), reasoning_effort = the effort the API was
called with (system line "Reasoning: <effort>"), tools as sent, preserve_thinking at the template default (true:
every assistant turn keeps <think>reasoning</think> before its answer), tool results in the template's tool_response
role. Tool-call arguments are parsed from their JSON string into a mapping before rendering (the template's string
branch needs a fromjson filter that transformers does not register; the mapping branch renders the same text).
The score mask covers what the model generated in every assistant turn: the template's {% generation %} span
(reasoning, </think>, answer, tool calls, <|im_end|> and the newline the template writes after it, chat_template.jinja
:57-82); the "<|im_start|>assistant\\n<think>" header is prompt. It is computed from token prefixes (turn_mask: the
conversation rendered up to each assistant turn as a generation prompt, and through that turn), which also checks
that history renders exactly like the generation form (it does with the default preserve_thinking=true).

Length: every sequence is cut to --max-len (default 4096) tokens, sudoingx's rule (truncate, never drop; the panel's
score mask is cut with it). The long-context prompts (wiki passages of 2,500-6,000 characters) mostly fit; any that do
not are truncated and listed with truncated_from. No random-token rows (sudoingx added 8): the data provenance rule
admits only API answers, sudoingx's prompts and exllamav3's standard text.

Split: each record's "split" from the prompt set (sudoingx: 299 cal / 201 panel in the base set; the instruct set is
all cal). Cal and panel are disjoint by prompt id (checked).

  python3 build.py /model /out --conversations base.jsonl instruct.jsonl --exllamav3 /path/to/exllamav3 \\
                   [--max-len 4096] [--std-rows 64] [--std-len 2048]
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import random

SEED = 20261008
STD_FILES = [("c4.utf8", 20), ("code.utf8", 20), ("multilingual.utf8", 10), ("technical.utf8", 10), ("wiki.utf8", 50),
             ("tiny.utf8", 5)]
STD_SUBDIR = os.path.join("exllamav3", "conversion", "standard_cal_data")
MAX_LEN = 4096


# ------------------------------------------------------------------------------------------------ pure helpers
def cut(ids: list[int], score: list[int] | None, max_len: int) -> tuple[list[int], list[int] | None, int | None]:
    """Truncate to max_len (sudoingx's rule). Returns (ids, score, original length if cut else None)."""
    if len(ids) <= max_len:
        return ids, score, None
    return ids[:max_len], (score[:max_len] if score is not None else None), len(ids)


def split_records(records: list[dict]) -> tuple[list[dict], list[dict]]:
    """(cal, panel) by each record's split, sorted by id. Raises on duplicate ids or an unknown split."""
    seen: set[str] = set()
    cal, panel = [], []
    for r in records:
        if r["id"] in seen:
            raise ValueError(f"duplicate prompt id {r['id']} (the same prompt in two files or twice in one)")
        seen.add(r["id"])
        if r["split"] == "cal":
            cal.append(r)
        elif r["split"] == "panel":
            panel.append(r)
        else:
            raise ValueError(f"{r['id']}: unknown split {r['split']!r}")
    if {r["id"] for r in cal} & {r["id"] for r in panel}:
        raise AssertionError("cal and panel share a prompt id")
    return sorted(cal, key=lambda r: r["id"]), sorted(panel, key=lambda r: r["id"])


def source_of(rec_id: str) -> str:
    """sudoingx's id prefixes: p = base set (prompts.py), i = instruct supplement (prompts_instruct.py)."""
    return {"p": "chat", "i": "instruct"}.get(rec_id[:1], "unknown")


def std_rows_per_file(total_rows: int) -> list[tuple[str, int]]:
    wsum = sum(w for _, w in STD_FILES)
    return [(fname, max(1, round(total_rows * w / wsum))) for fname, w in STD_FILES]


def std_windows(n_tokens: int, n_rows: int, length: int, rng: random.Random) -> list[tuple[int, int]]:
    """Seeded fixed-width windows (start, end) over a token stream, sudoingx's sampling."""
    out = []
    for _ in range(n_rows):
        s = rng.randrange(0, max(1, n_tokens - length))
        out.append((s, min(n_tokens, s + length)))
    return out


def pack(seqs: list[list[int]]) -> tuple[list[int], list[int]]:
    flat = [t for s in seqs for t in s]
    offs = [0]
    for s in seqs:
        offs.append(offs[-1] + len(s))
    return flat, offs


def turn_mask(full: list[int], turns: list[tuple[list[int], list[int]]], rec_id: str = "") -> list[int]:
    """Score mask from token prefixes. turns: per assistant turn (head, turn) = the ids of the conversation rendered up
    to that turn as a generation prompt (ends "<|im_start|>assistant\\n<think>") and rendered through that turn.
    Positions len(head) .. len(turn)-1 (reasoning, </think>, answer, tool calls, <|im_end|>, its newline) are scored.
    Raises unless head is a prefix of turn and turn a prefix of the full ids (history rendered like the generation)."""
    mask = [0] * len(full)
    end = 0
    for j, (head, turn) in enumerate(turns):
        if not (len(head) < len(turn) <= len(full) and turn[:len(head)] == head and full[:len(turn)] == turn
                and len(head) >= end):
            raise AssertionError(f"{rec_id}: assistant turn {j} is not rendered in history as it was generated "
                                 f"(token prefix mismatch)")
        mask[len(head):len(turn)] = [1] * (len(turn) - len(head))
        end = len(turn)
    return mask


def for_template(messages: list[dict]) -> list[dict]:
    """Messages as the template expects them: tool-call arguments as a mapping (see the module docstring)."""
    msgs = copy.deepcopy(messages)
    for m in msgs:
        for tc in m.get("tool_calls") or []:
            fn = tc.get("function", tc)
            if isinstance(fn.get("arguments"), str):
                try:
                    fn["arguments"] = json.loads(fn["arguments"] or "{}")
                except json.JSONDecodeError:
                    fn["arguments"] = {}
    return msgs


def std_dir_of(path: str) -> str:
    """An exllamav3 checkout (or its standard_cal_data directory itself) -> the standard_cal_data directory."""
    for d in (os.path.join(path, STD_SUBDIR), path):
        if os.path.exists(os.path.join(d, "c4.utf8")):
            return d
    raise SystemExit(f"no {STD_SUBDIR}/c4.utf8 under {path}")


def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for blk in iter(lambda: f.read(1 << 24), b""):
            h.update(blk)
    return h.hexdigest()


# ------------------------------------------------------------------------------------------------ rendering
class Renderer:
    def __init__(self, model_dir: str):
        # Not AutoTokenizer: with tokenizer_config's "LlamaTokenizerFast" class, transformers 5.x resolves to a
        # Python LlamaTokenizer that drops newline, space and CJK tokens (verified 2026-10-15 in the quant container;
        # the official inference/generate.py:378-387 avoids it the same way). The generic fast loader is id-identical
        # to tokenizer.json.
        from transformers import PreTrainedTokenizerFast
        self.tok = PreTrainedTokenizerFast.from_pretrained(model_dir)
        with open(os.path.join(model_dir, "chat_template.jinja"), encoding="utf8") as f:
            self.template = f.read()
        self.bos = self.tok.bos_token_id
        if self.bos is None:
            raise SystemExit("tokenizer has no BOS token")

    def render_text(self, rec: dict, generation_prompt: bool = False) -> str:
        return self.tok.apply_chat_template(for_template(rec["messages"]), tools=rec.get("tools") or None,
                                            chat_template=self.template, tokenize=False,
                                            add_generation_prompt=generation_prompt, **self._kw(rec))

    def _ids(self, msgs: list[dict], tools, kw: dict, generation_prompt: bool) -> list[int]:
        text = self.tok.apply_chat_template(msgs, tools=tools, chat_template=self.template, tokenize=False,
                                            add_generation_prompt=generation_prompt, **kw)
        return list(self.tok(text, add_special_tokens=False)["input_ids"])

    def encode(self, rec: dict) -> tuple[list[int], list[int]]:
        """(ids, score mask). The mask comes from token prefixes (turn_mask), not from transformers'
        return_assistant_tokens_mask: this tokenizer is ByteLevel with trim_offsets, so a newline token has an empty
        character span, char_to_token(<the "\\n" after <|im_end|>>) is None and transformers then marks everything
        from the first answer to the end of the sequence (seen in the quant container, transformers 5.17)."""
        msgs, tools, kw = for_template(rec["messages"]), rec.get("tools") or None, self._kw(rec)
        ids = self._ids(msgs, tools, kw, False)
        turns = [(self._ids(msgs[:j], tools, kw, True), self._ids(msgs[:j + 1], tools, kw, False))
                 for j, m in enumerate(msgs) if m["role"] == "assistant"]
        mask = turn_mask(ids, turns, rec["id"])
        if ids[:1] != [self.bos] or ids[1:2] == [self.bos]:
            raise AssertionError(f"{rec['id']}: expected exactly one leading BOS")
        if not any(mask):
            raise AssertionError(f"{rec['id']}: no assistant positions")
        return ids, mask

    @staticmethod
    def _kw(rec: dict) -> dict:
        eff = rec.get("api_effort")          # undefined in the template when absent: no "Reasoning:" line
        return {"reasoning_effort": eff} if eff else {}

    def tokens(self, text: str) -> list[int]:
        return self.tok.encode(text, add_special_tokens=False)


# ------------------------------------------------------------------------------------------------ main
def build(a: argparse.Namespace) -> None:
    from safetensors.torch import save_file
    import torch

    r = Renderer(a.model)
    records = []
    for path in a.conversations:
        with open(path, encoding="utf8") as f:
            records += [json.loads(line) for line in f if line.strip()]
    cal_recs, panel_recs = split_records(records)

    cal, cal_idx = [], []
    for rec in cal_recs:
        ids, _ = r.encode(rec)
        ids, _, full = cut(ids, None, a.max_len)
        cal.append(ids)
        cal_idx.append({"source": source_of(rec["id"]), "id": rec["id"], "domain": rec["domain"], "length": len(ids),
                        **({"truncated_from": full} if full else {})})
    panel, panel_idx = [], []
    for rec in panel_recs:
        ids, score = r.encode(rec)
        ids, score, full = cut(ids, score, a.max_len)
        panel.append((ids, score))
        panel_idx.append({"id": rec["id"], "domain": rec["domain"], "length": len(ids), "scored": sum(score),
                          "thinking": rec["thinking"], "effort": rec["effort"], "api_effort": rec.get("api_effort"),
                          "tools": bool(rec.get("tools")), **({"truncated_from": full} if full else {})})

    rng = random.Random(SEED)
    std_dir = std_dir_of(a.exllamav3)
    for fname, n_rows in std_rows_per_file(a.std_rows):
        with open(os.path.join(std_dir, fname), encoding="utf8") as f:
            ids = r.tokens(f.read())
        for s, e in std_windows(len(ids), n_rows, a.std_len, rng):
            cal.append(ids[s:e])
            cal_idx.append({"source": "exllamav3_std", "id": f"{fname}@{s}", "domain": fname.split(".")[0],
                            "length": e - s})

    os.makedirs(a.out, exist_ok=True)
    t, o = pack(cal)
    cal_path, panel_path = os.path.join(a.out, "cal.safetensors"), os.path.join(a.out, "panel.safetensors")
    save_file({"tokens": torch.tensor(t, dtype=torch.int32), "offsets": torch.tensor(o, dtype=torch.int64)}, cal_path)
    t, o = pack([p[0] for p in panel])
    sc = torch.tensor([x for p in panel for x in p[1]], dtype=torch.uint8)
    save_file({"tokens": torch.tensor(t, dtype=torch.int32), "offsets": torch.tensor(o, dtype=torch.int64),
               "score": sc}, panel_path)
    for name, obj in (("cal_index.json", cal_idx), ("panel_index.json", panel_idx)):
        with open(os.path.join(a.out, name), "w") as f:
            json.dump(obj, f, indent=0, ensure_ascii=False)
    by: dict[str, int] = {}
    for x in cal_idx:
        by[x["source"]] = by.get(x["source"], 0) + x["length"]
    n_cut = sum("truncated_from" in x for x in cal_idx + panel_idx)
    print(f"cal: {len(cal)} sequences ({len(cal_recs)} conversations), {sum(by.values()):,} tokens {by}")
    print(f"panel: {len(panel)} conversations, {sum(x['length'] for x in panel_idx):,} tokens, "
          f"{sum(x['scored'] for x in panel_idx):,} scored positions; {n_cut} sequences cut to {a.max_len}")
    lines = [f"{sha256_file(p)}  {os.path.basename(p)}" for p in (cal_path, panel_path)]
    with open(os.path.join(a.out, "sha256.txt"), "w") as f:
        f.write("\n".join(lines) + "\n")
    for line in lines:
        print("sha256", line)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("model", help="dir with chat_template.jinja, tokenizer.json, tokenizer_config.json (/model)")
    ap.add_argument("out")
    ap.add_argument("--conversations", nargs="+", required=True, help="generate_stepfun.py JSONL files")
    ap.add_argument("--exllamav3", required=True, help="exllamav3 checkout (v1.6.0) or its standard_cal_data dir")
    ap.add_argument("--max-len", type=int, default=MAX_LEN)
    ap.add_argument("--std-rows", type=int, default=64)
    ap.add_argument("--std-len", type=int, default=2048)
    a = ap.parse_args()
    if not 0 < a.std_len <= a.max_len <= MAX_LEN:
        raise SystemExit(f"need 0 < --std-len <= --max-len <= {MAX_LEN} (sequences longer than 4096 are not dense)")
    build(a)


if __name__ == "__main__":
    main()
