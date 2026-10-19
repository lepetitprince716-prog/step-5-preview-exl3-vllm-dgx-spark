# Task benchmarks: local EXL3 2.4 bpw (body 5/6/6) vs official step-5-preview API, 2026-10-17

Both targets receive the same chat requests, greedy (temperature 0), max 16,384 generated tokens, thinking on (the
model has no thinking-off mode; thinking arrives as `reasoning`, only `content` is scored). Local = the TP2 serve
described in `serving.md` (`http://127.0.0.1:8888/v1`, 4 concurrent). API = `https://api.stepfun.com/step_plan/v1`,
model `step-5-preview`, 4 concurrent. Scripts: `evals/gen_evalplus.py`, `evals/run_evalplus.sh`, `evals/run_ifeval.sh`.
Scoring: EvalPlus 0.3.1 (HumanEval+ v0.1.10 164 tasks, MBPP+ v0.2.0 378 tasks), lm-evaluation-harness `ifeval`
(541 prompts) through `local-chat-completions` with the chat template applied.

| Benchmark | Local 2.4 bpw | Official API | Delta |
|---|---|---|---|
| HumanEval (base) | 95.7 | 97.0 | -1.3 |
| HumanEval+ | 93.9 | 94.5 | -0.6 |
| MBPP (base) | 95.8 | 95.8 | 0.0 |
| MBPP+ | 80.7 | 81.0 | -0.3 |
| IFEval prompt-level strict | 86.9 | 87.3 | -0.4 |
| IFEval instruction-level strict | 89.1 | 89.8 | -0.7 |

Notes:
- HumanEval local: 164 tasks, 0 generation errors, 4,193 s for the 95 tasks of the resumed run (the first run was
  interrupted by the 07:20 memory incident, see `serving.md`; `gen_evalplus.py` resumes from `samples.raw.jsonl`).
- One task of HumanEval (0.6 points) separates the two on the plus set; this is within the run-to-run noise of a
  164-task set (one task = 0.61 points) and MTP decoding is not bit-reproducible between runs.
- MBPP local: 378 tasks, 0 generation errors, 8,625 s for the 311 tasks of the resumed run (67 were kept from the
  run interrupted on 07:20). MBPP+ differs by one task (one task = 0.26 points).
- IFEval API: 541 prompts in 1 h 28 min; loose variants 88.5 (prompt) / 90.7 (instruction).
- IFEval local: 541 prompts in 8 h 25 min (4 concurrent streams, long thinking on many prompts); loose variants
  88.4 (prompt) / 90.1 (instruction). The prompt-level standard error is 1.4 points on both targets.
- Overall: the 2.4 bpw checkpoint sits 0.3-1.3 points under the official API on every benchmark, within one
  standard error on each.
- The serve was not restarted between runs; `spark status` stayed `api: 200` and the memory guard did not fire
  after the 07:20 incident.
- Raw outputs: head `~/evals/out/<dataset>-<target>/` (samples, `meta.jsonl` with token counts, `run.log`).
