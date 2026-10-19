#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Assemble the serving checkpoints for every body variant, sequentially, in one detached container (pure Python, no
# GPU). ~/models is mounted once so kept shards hardlink to the official snapshot instead of being copied.
#   bash run_p5_all.sh                      (variants: 4/4/4 -> step5-exl3, 5/6/6 -> step5-exl3-b566, 6/8/8 -> -b688)
#   VARIANTS="b688:body_6-8-8" bash run_p5_all.sh   (NAME_SUFFIX:dense_subdir pairs; "" suffix = p4's default dir)
# stdout: ~/step5/runs/p5/p5.<suffix or b444>.stdout, each ending with P5_EXIT=<code>.
set -euo pipefail
H=$HOME
STEP5=${STEP5:-$H/step5}
PLAN=${PLAN:-$STEP5/plans/plan.json}
IMAGE=${IMAGE:-dsv41-quant:dev}
VARIANTS=${VARIANTS:-":. b566:body_5-6-6 b688:body_6-8-8"}
P5EXTRA=${P5EXTRA:-}
mkdir -p "$STEP5/runs/p5"
CMD=""
for spec in $VARIANTS; do
  suffix=${spec%%:*}; sub=${spec#*:}
  out=/models/step5-exl3${suffix:+-$suffix}
  log=/out/p5.${suffix:-b444}.stdout
  CMD+="python3 -u p5_assemble.py /models/Step-5-Preview-FP8 /pass1_out /dense_out/$sub /plan/$(basename "$PLAN") $out $P5EXTRA >> $log 2>&1; echo P5_EXIT=\$? >> $log; "
done
docker rm -f step5-p5 >/dev/null 2>&1 || true
docker run -d --name step5-p5 --restart no --user "$(id -u):$(id -g)" -e HOME=/tmp \
  -v "$H/models:/models" -v "$STEP5:/q:ro" -v "$STEP5/runs/p1:/pass1_out:ro" -v "$STEP5/runs/p4:/dense_out:ro" \
  -v "$(dirname "$PLAN"):/plan:ro" -v "$STEP5/runs/p5:/out" -w /q --entrypoint bash "$IMAGE" -c "$CMD" >/dev/null
echo "step5-p5 started: $VARIANTS (stdout $STEP5/runs/p5/p5.*.stdout)"
