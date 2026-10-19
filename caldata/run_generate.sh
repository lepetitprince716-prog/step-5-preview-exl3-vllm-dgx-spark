#!/bin/bash
# Generate both calibration sets on the head node through the official StepFun API (China platform), one after the
# other, resumable (done ids are skipped). BUDGET is a safety fuse in USD, not a target.
#
#   nohup bash ~/step5/caldata/run_generate.sh > ~/step5/cal/generate-all.log 2>&1 &
set -uo pipefail
cd "$(dirname "$0")/.." || exit 1
BUDGET=${BUDGET:-30}
COMMON=(--platform "${PLATFORM:-plan}" --cny-per-usd "${CNY_PER_USD:-7.1}" --budget-usd "$BUDGET" --workers "${WORKERS:-5}"
        --max-tokens "${MAX_TOKENS:-3500}" --tries 4 --timeout 300)
mkdir -p cal
echo "STAGE base $(date -Is)"
python3 caldata/generate_stepfun.py cal/base.jsonl "${COMMON[@]}"
echo "STAGE instruct $(date -Is)"
python3 caldata/generate_stepfun.py cal/instruct.jsonl --set instruct "${COMMON[@]}"
echo "STAGE done $(date -Is)"
