#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Dense pass (p4_dense.py) on both Sparks: each box streams every layer through pass 1's EXL3 experts and encodes the
# body linears of its strided half (--part 0/2 on the head, 1/2 on the worker). No exchange: the worker's output
# directory is rsynced to the head afterwards (RUNBOOK 5.1). Run on the head.
#   PLAN=~/step5/plans/plan.json ./run_p4_2box.sh
# Environment (all optional except PLAN): A / B (ssh aliases), MODEL, STEP5 (code dir), CAL / CALF, P1RUN (pass 1
# output dir), RUN (output dir), IMAGE, STEP5_CUDA_MEM_FRACTION, VARIANTS (default 5/6/6,6/8/8), P4_EXTRA (extra
# p4_dense.py arguments, container paths), DRY=1 (print, do not start).
set -euo pipefail
A=${A:-spark-head}; B=${B:-spark-worker}
sh_() {   # run on a box: locally when it is this host, else over ssh
  if [ "$1" = "$(hostname -s)" ] || [ "$1" = "$(hostname)" ]; then bash -c "$2"; else
    ssh -o BatchMode=yes -o ConnectTimeout=10 "$1" "$2"; fi
}
PLAN=${PLAN:?path of the plan json, same path on both Sparks}
H=$(sh_ "$A" 'echo $HOME'); HB=$(sh_ "$B" 'echo $HOME')
[ "$H" = "$HB" ] || { echo "home differs ($H vs $HB): run dirs must have the same path on both boxes"; exit 1; }
STEP5=${STEP5:-$H/step5}
MODEL=${MODEL:-$H/models/Step-5-Preview-FP8}
CAL=${CAL:-$STEP5/cal}; CALF=${CALF:-cal.safetensors}
P1RUN=${P1RUN:-$STEP5/runs/p1}
RUN=${RUN:-$STEP5/runs/p4}
IMAGE=${IMAGE:-dsv41-quant:dev}
MEMFRAC=${STEP5_CUDA_MEM_FRACTION:-0.70}
VARIANTS=${VARIANTS:-5/6/6,6/8/8}
P4_EXTRA=${P4_EXTRA:-}; DRY=${DRY:-0}
# same plan, token set, code and image on both boxes; pass 1 complete on both; GPUs idle
SUMS="sha256sum $PLAN $CAL/$CALF $STEP5/p4_dense.py $STEP5/p_torch.py $STEP5/p_util.py $STEP5/step5q/*.py 2>&1 \
  | cut -c1-16 | tr '\n' ' '"
ha=$(sh_ "$A" "$SUMS"); hb=$(sh_ "$B" "$SUMS")
[ "$ha" = "$hb" ] || { echo "plan / calibration / code differ between boxes: $ha vs $hb"; exit 1; }
ia=$(sh_ "$A" "docker image inspect -f {{.Id}} $IMAGE"); ib=$(sh_ "$B" "docker image inspect -f {{.Id}} $IMAGE")
[ "$ia" = "$ib" ] || { echo "$IMAGE differs between boxes: $ia vs $ib"; exit 1; }
P1LIST="cd $P1RUN && ls -l layer_*.part[01].safetensors | awk '{print \$5, \$9}' | sha256sum | cut -c1-16"
pa=$(sh_ "$A" "$P1LIST"); pb=$(sh_ "$B" "$P1LIST")
[ "$pa" = "$pb" ] || { echo "pass 1 file lists differ between boxes: $pa vs $pb"; exit 1; }
n1=$(sh_ "$A" "ls $P1RUN/layer_*.part[01].safetensors | wc -l")
[ "$n1" = 176 ] || { echo "pass 1 incomplete: $n1 part files, expected 176"; exit 1; }
for n in "$A" "$B"; do
  g=$(sh_ "$n" "nvidia-smi --query-compute-apps=pid --format=csv,noheader | wc -l")
  [ "$g" = 0 ] || { echo "$n has $g GPU processes: stop the serve first"; exit 1; }
done
echo "plan + calibration + code identical on both ($ha); image $ia; pass 1 complete ($pa); GPUs idle"
UID_GID=$(sh_ "$A" 'echo $(id -u):$(id -g)')
for spec in "$A:0" "$B:1"; do
  IFS=: read -r n part <<<"$spec"
  START="mkdir -p $RUN && bash $STEP5/guards.sh start && docker rm -f step5-p4 >/dev/null 2>&1; \
    docker run -d --name step5-p4 --gpus all --ipc=host --restart no --user $UID_GID -e HOME=/tmp \
    -e STEP5_CUDA_MEM_FRACTION=$MEMFRAC -v $MODEL:/model:ro -v $STEP5:/q:ro -v $CAL:/cal:ro \
    -v $(dirname "$PLAN"):/plan:ro -v $P1RUN:/pass1_out:ro -v $RUN:/out -w /q --entrypoint bash $IMAGE -c \
    'python3 -u /q/p4_dense.py /model /cal/$CALF /pass1_out /plan/$(basename "$PLAN") /out --part $part/2 \
       --variants $VARIANTS $P4_EXTRA >> /out/dense.p$part.stdout 2>&1; echo P4_EXIT=\$? >> /out/dense.p$part.stdout' \
    >/dev/null && echo '$n part $part/2 started (stdout $RUN/dense.p$part.stdout)'"
  if [ "$DRY" = 1 ]; then echo "== DRY $n:"; echo "$START"; continue; fi
  sh_ "$n" "$START"
done
