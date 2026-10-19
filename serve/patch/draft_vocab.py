"""Frequency-ranked draft vocabulary for the Step-5 MTP heads (FR-Spec style).

Counts token ids in the assistant turns (reasoning and content) of the calibration sets, ranks them by count, and
reports how much of the held-out turns and of extra text files the top-N ids cover. Writes the top-N ids (plus every
added/special token) as a JSON list that the plugin's draft head loads through STEP5_DRAFT_VOCAB.

Records with split == "cal" are counted; every other split (the KL panel) is held out.

    python3 draft_vocab.py --tokenizer /model/tokenizer.json --data /cal/base.jsonl /cal/instruct.jsonl \
        --extra /evals/humaneval-local/samples.raw.jsonl --sizes 16384,24576,32768,49152 --save 32768 --out /work
"""
from __future__ import annotations

import argparse
import collections
import json
import os
import sys

from tokenizers import Tokenizer


def assistant_texts(rec: dict) -> list[str]:
    out = []
    for m in rec.get("messages") or []:
        if m.get("role") != "assistant":
            continue
        for key in ("reasoning_content", "content"):
            v = m.get(key)
            if isinstance(v, str) and v:
                out.append(v)
        for call in m.get("tool_calls") or []:
            fn = call.get("function") or {}
            for key in ("name", "arguments"):
                v = fn.get(key)
                if isinstance(v, str) and v:
                    out.append(v)
    return out


def extra_texts(rec: dict) -> list[str]:
    return [v for k, v in rec.items() if k in ("solution", "response", "output", "content") and isinstance(v, str)]


def cjk_share(texts: list[str]) -> float:
    chars = sum(len(t) for t in texts)
    cjk = sum(1 for t in texts for ch in t if "　" <= ch <= "鿿" or "가" <= ch <= "힯"
              or "＀" <= ch <= "￯")
    return cjk / max(1, chars)


def count(tok: Tokenizer, texts: list[str], counter: collections.Counter) -> None:
    for enc in tok.encode_batch(texts, add_special_tokens=False):
        counter.update(enc.ids)


def coverage(ranked: list[int], counts: collections.Counter, n: int) -> float:
    total = sum(counts.values())
    top = set(ranked[:n])
    return sum(c for t, c in counts.items() if t in top) / max(1, total)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tokenizer", required=True)
    ap.add_argument("--data", nargs="+", required=True, help="calibration jsonl files (messages with splits)")
    ap.add_argument("--extra", nargs="*", default=[], help="jsonl files whose solution/response fields are held out")
    ap.add_argument("--sizes", default="16384,24576,32768,49152,65536")
    ap.add_argument("--save", default="", help="comma list of sizes to write as hot-N.json")
    ap.add_argument("--shards", type=int, default=2, help="tensor-parallel ranks the head's vocab is split across")
    ap.add_argument("--out", default=".")
    a = ap.parse_args()

    tok = Tokenizer.from_file(a.tokenizer)
    with open(a.tokenizer) as f:
        added = sorted({t["id"] for t in json.load(f).get("added_tokens", [])})
    vocab = tok.get_vocab_size(with_added_tokens=True)

    train, held, held_cjk = collections.Counter(), collections.Counter(), collections.Counter()
    splits, cjk_records = collections.Counter(), collections.Counter()
    for path in a.data:
        with open(path) as f:
            for line in f:
                rec = json.loads(line)
                split = rec.get("split", "?")
                splits[split] += 1
                texts = assistant_texts(rec)
                is_cjk = cjk_share(texts) >= 0.2
                cjk_records[split] += is_cjk
                count(tok, texts, train if split == "cal" else held)
                if split != "cal" and is_cjk:
                    count(tok, texts, held_cjk)
    extra = collections.Counter()
    for path in a.extra:
        with open(path) as f:
            texts = [t for line in f for t in extra_texts(json.loads(line))]
        count(tok, texts, extra)

    ranked = [t for t, _ in sorted(train.items(), key=lambda kv: (-kv[1], kv[0]))]
    print(f"vocab {vocab}, added tokens {len(added)}, records by split {dict(splits)}, "
          f"CJK records (>=20% CJK chars) by split {dict(cjk_records)}, held-out CJK tokens {sum(held_cjk.values())}")
    print(f"tokens: train {sum(train.values())} ({len(train)} distinct), held-out {sum(held.values())} "
          f"({len(held)} distinct), extra {sum(extra.values())} ({len(extra)} distinct)")
    per = -(-vocab // a.shards)
    for n in (int(s) for s in a.sizes.split(",")):
        ids = set(ranked[:n]) | set(added)
        split = [sum(1 for t in ids if r * per <= t < (r + 1) * per) for r in range(a.shards)]
        print(f"top {n:6d}: train {coverage(ranked, train, n):.5f}  held-out {coverage(ranked, held, n):.5f}  "
              f"held-out CJK {coverage(ranked, held_cjk, n):.5f}  extra {coverage(ranked, extra, n):.5f}  "
              f"ids per shard {split}")

    for n in (int(s) for s in a.save.split(",") if s):
        ids = sorted(set(ranked[:n]) | set(added))
        path = os.path.join(a.out, f"hot-{n}.json")
        with open(path, "w") as f:
            json.dump({"vocab_size": vocab, "ranked_from": a.data, "ids": ids}, f)
        print(f"wrote {path}: {len(ids)} ids")
    return 0


if __name__ == "__main__":
    sys.exit(main())
