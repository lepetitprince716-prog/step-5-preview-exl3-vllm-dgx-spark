#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# IFEval through lm-evaluation-harness against one OpenAI-compatible chat endpoint, detached (nohup), on the head.
#   bash run_ifeval.sh local      (served step5 on 127.0.0.1:8888)
#   bash run_ifeval.sh api        (official step-5-preview, Step Plan path, key ~/step5/stepfun.env)
# Output: ~/evals/out/ifeval-<target>/ (lm_eval results json + samples), log run.log ending with IFEVAL_EXIT=<code>.
# The dataset comes from hf-mirror.com (huggingface.co is unreachable from the Sparks); no HF token is sent.
# Environment: WORKERS (4), MAX_TOKENS (16384), LIMIT (unset = all), EVALS (~/evals), STEP5 (~/step5).
set -euo pipefail
TARGET=${1:?local | api}
EVALS=${EVALS:-$HOME/evals}; STEP5=${STEP5:-$HOME/step5}
VENV=$EVALS/venv
OUT=$EVALS/out/ifeval-$TARGET
mkdir -p "$OUT"
case "$TARGET" in
  local) BASE=http://127.0.0.1:8888/v1/chat/completions; MODEL=step5; export OPENAI_API_KEY=none ;;
  api)
    BASE=https://api.stepfun.com/step_plan/v1/chat/completions; MODEL=step-5-preview
    KEY=$(grep -E '^STEPFUN_API_KEY=' "$STEP5/stepfun.env" | head -n 1 | cut -d= -f2- | tr -d "'\"")
    [ -n "$KEY" ] || { echo "no STEPFUN_API_KEY in $STEP5/stepfun.env"; exit 1; }
    export OPENAI_API_KEY=$KEY ;;
  *) echo "target must be local or api"; exit 2 ;;
esac
WORKERS=${WORKERS:-4}; MAX_TOKENS=${MAX_TOKENS:-16384}
export HF_ENDPOINT=https://hf-mirror.com HF_HUB_DISABLE_IMPLICIT_TOKEN=1 HF_TOKEN= HF_DATASETS_OFFLINE=1 \
  HF_HUB_OFFLINE=1
CMD="$VENV/bin/lm_eval --model local-chat-completions \
  --model_args model=$MODEL,base_url=$BASE,num_concurrent=$WORKERS,max_retries=3,timeout=1800,tokenized_requests=False \
  --tasks ifeval --apply_chat_template --gen_kwargs max_gen_toks=$MAX_TOKENS,temperature=0 \
  --output_path $OUT --log_samples ${LIMIT:+--limit $LIMIT}"
nohup bash -c "$CMD; echo IFEVAL_EXIT=\$?" >> "$OUT/run.log" 2>&1 < /dev/null &
echo "started ifeval on $TARGET -> $OUT/run.log"
