#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Generate + sanitize + evaluate one EvalPlus dataset against one target, detached (nohup), on the head.
#   bash run_evalplus.sh local humaneval        (served step5 on 127.0.0.1:8888)
#   bash run_evalplus.sh api   mbpp             (official step-5-preview, Step Plan path, key ~/step5/stepfun.env)
# Output: ~/evals/out/<dataset>-<target>/{samples.raw.jsonl,meta.jsonl,run.log}, then evalplus.sanitize +
# evalplus.evaluate results in the same dir; the log ends with EVALPLUS_EXIT=<code>.
# Environment: WORKERS (4), MAX_TOKENS (16384), LIMIT (0 = all), EVALS (~/evals), STEP5 (~/step5).
set -euo pipefail
TARGET=${1:?local | api}; DATASET=${2:?humaneval | mbpp}
EVALS=${EVALS:-$HOME/evals}; STEP5=${STEP5:-$HOME/step5}
VENV=$EVALS/venv
OUT=$EVALS/out/$DATASET-$TARGET
mkdir -p "$OUT"
case "$TARGET" in
  local) BASE=http://127.0.0.1:8888/v1; MODEL=step5; KEYF="" ;;
  api)   BASE=https://api.stepfun.com/step_plan/v1; MODEL=step-5-preview; KEYF=$STEP5/stepfun.env ;;
  *) echo "target must be local or api"; exit 2 ;;
esac
WORKERS=${WORKERS:-4}; MAX_TOKENS=${MAX_TOKENS:-16384}; LIMIT=${LIMIT:-0}
export HF_HUB_DISABLE_IMPLICIT_TOKEN=1 HF_ENDPOINT=https://hf-mirror.com
# sanitize + evaluate run inside a memory-capped container (the venv is mounted at its own path, python 3.12 in the
# image): evaluate executes the generated code in parallel workers and took 4.5 GB on 2026-10-17, which drove the
# serving node's MemAvailable to 0 and made the memory guard kill the serve. Here the container, not the serve, dies.
EVAL_MEM=${EVAL_MEM:-4g}; EVAL_PAR=${EVAL_PAR:-4}; IMAGE=${IMAGE:-dsv41-quant:dev}
DOCK="docker run --rm --memory $EVAL_MEM --user $(id -u):$(id -g) -e HOME=/tmp -v $EVALS:$EVALS \
  -v $HOME/.cache/evalplus:/tmp/.cache/evalplus:ro --entrypoint $VENV/bin/python3 $IMAGE"
CMD="$VENV/bin/python3 $STEP5/evals/gen_evalplus.py --dataset $DATASET --base-url $BASE --model $MODEL --out $OUT \
  --workers $WORKERS --max-tokens $MAX_TOKENS --limit $LIMIT ${KEYF:+--api-key-file $KEYF} \
  && $DOCK -m evalplus.sanitize $OUT/samples.raw.jsonl \
  && $DOCK -m evalplus.evaluate --dataset $DATASET --samples $OUT/samples.raw-sanitized.jsonl --parallel $EVAL_PAR"
nohup bash -c "$CMD; echo EVALPLUS_EXIT=\$?" >> "$OUT/run.log" 2>&1 < /dev/null &
echo "started $DATASET on $TARGET -> $OUT/run.log"
