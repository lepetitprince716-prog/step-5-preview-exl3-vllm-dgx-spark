#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# After the teacher run (step5-p6-ref) exits, evaluate the three body variants on the panel one after another (GPU).
#   tmux new-session -d -s p6chain 'bash ~/step5/run_p6_chain.sh > ~/step5/runs/p6/chain.log 2>&1'
# Prints CHAIN_DONE at the end; exits 1 on the first failed run.
set -euo pipefail
STEP5=${STEP5:-$HOME/step5}
cd "$STEP5"
docker wait step5-p6-ref >/dev/null 2>&1 || true
grep -q 'P6_EXIT=0' runs/p6/ref.stdout || { echo "teacher run failed (runs/p6/ref.stdout)"; exit 1; }
for spec in "b444:/dense" "b566:/dense/body_5-6-6" "b688:/dense/body_6-8-8"; do
  name=${spec%%:*}; body=${spec#*:}
  if grep -q 'P6_EXIT=0' "runs/p6/$name.stdout" 2>/dev/null; then
    echo "$name already done"; continue
  fi
  bash run_p6.sh quant "$name" --experts /p1 --body "$body"
  docker wait "step5-p6-$name" >/dev/null
  grep -q 'P6_EXIT=0' "runs/p6/$name.stdout" || { echo "$name failed (runs/p6/$name.stdout)"; exit 1; }
  date "+%H:%M $name done"
done
echo CHAIN_DONE
