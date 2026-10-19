#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Pass 1 on both Sparks (ported from sudoingx dsv41_recipe scripts/quantize/run_pass1_2box.sh): each box encodes a
# strided half of every MoE layer's experts (--part 0/2 on the head, 1/2 on the worker) and exchange_sync.sh pushes the
# finished part files to the peer over the ConnectX link; both streams advance through all experts. Run on the head.
#   PLAN=~/step5/plans/plan.json ./run_pass1_2box.sh [first-last]          (default 3-90; a restart resumes from the
#                                                                          saved stream in each box's run dir)
# Partial re-encode (only experts whose K changed; the rest from an assembled checkpoint):
#   PLAN=.../plan_v2.json OLD_PLAN=.../plan_v1.json REUSE=~/models/step5-exl3-v1 RUN=~/step5/runs/p1_v2 ./run_pass1_2box.sh
# Environment (all optional except PLAN): A / B (ssh aliases, default spark-head / spark-worker), FAB_IF (ConnectX
# interface, default enp1s0f1np1 per this cluster's dsv41 profile; FAB_IF_A / FAB_IF_B per box), A_IP / B_IP (skip the
# lookup), MODEL, STEP5 (code dir, mounted at /q), CAL / CALF, RUN, IMAGE, STEP5_CUDA_MEM_FRACTION, P1_EXTRA (extra
# p3_pass1.py arguments, container paths), DRY=1 (print, do not start).
set -euo pipefail
A=${A:-spark-head}; B=${B:-spark-worker}
FAB_IF=${FAB_IF:-enp1s0f1np1}; FAB_IF_A=${FAB_IF_A:-$FAB_IF}; FAB_IF_B=${FAB_IF_B:-$FAB_IF}
sh_() {   # run on a box: locally when it is this host, else over ssh
  if [ "$1" = "$(hostname -s)" ] || [ "$1" = "$(hostname)" ]; then bash -c "$2"; else
    ssh -o BatchMode=yes -o ConnectTimeout=10 "$1" "$2"; fi
}
fab_ip() { sh_ "$1" "ip -4 -o addr show $2 | awk '{print \$4}' | cut -d/ -f1 | head -1" || true; }
A_IP=${A_IP:-$(fab_ip "$A" "$FAB_IF_A")}; B_IP=${B_IP:-$(fab_ip "$B" "$FAB_IF_B")}
[ -n "$A_IP" ] && [ -n "$B_IP" ] || { echo "no IPv4 on the fabric: $A/$FAB_IF_A '$A_IP', $B/$FAB_IF_B '$B_IP'"; exit 1; }
PLAN=${PLAN:?path of the plan json, same path on both Sparks}
LAYERS=${1:-3-90}
H=$(sh_ "$A" 'echo $HOME'); HB=$(sh_ "$B" 'echo $HOME')
[ "$H" = "$HB" ] || { echo "home differs ($H vs $HB): run dirs must have the same path on both boxes"; exit 1; }
STEP5=${STEP5:-$H/step5}
MODEL=${MODEL:-$H/models/Step-5-Preview-FP8}
CAL=${CAL:-$STEP5/cal}; CALF=${CALF:-cal.safetensors}
RUN=${RUN:-$STEP5/runs/p1}
IMAGE=${IMAGE:-dsv41-quant:dev}
MEMFRAC=${STEP5_CUDA_MEM_FRACTION:-0.70}
REUSE=${REUSE:-}; OLD_PLAN=${OLD_PLAN:-}; P1_EXTRA=${P1_EXTRA:-}; DRY=${DRY:-0}
if [ -n "$REUSE" ]; then
  [ -n "$OLD_PLAN" ] && [ "$(dirname "$OLD_PLAN")" = "$(dirname "$PLAN")" ] || { echo "REUSE needs OLD_PLAN next to PLAN"; exit 1; }
  RMOUNT="-v $REUSE:/reuse:ro"; RARGS="--reuse /reuse --old-plan /plan/$(basename "$OLD_PLAN")"
else
  RMOUNT=""; RARGS=""
fi
# same plan, token set, code and image on both boxes, GPUs idle
SUMS="sha256sum $PLAN $OLD_PLAN $CAL/$CALF $STEP5/p3_pass1.py $STEP5/p_torch.py $STEP5/p_util.py $STEP5/step5q/*.py \
  ${REUSE:+$REUSE/model.safetensors.index.json} 2>&1 | cut -c1-16 | tr '\n' ' '"
ha=$(sh_ "$A" "$SUMS"); hb=$(sh_ "$B" "$SUMS")
[ "$ha" = "$hb" ] || { echo "plan / calibration / code differ between boxes: $ha vs $hb"; exit 1; }
ia=$(sh_ "$A" "docker image inspect -f {{.Id}} $IMAGE"); ib=$(sh_ "$B" "docker image inspect -f {{.Id}} $IMAGE")
[ "$ia" = "$ib" ] || { echo "$IMAGE differs between boxes: $ia vs $ib"; exit 1; }
for n in "$A" "$B"; do
  g=$(sh_ "$n" "nvidia-smi --query-compute-apps=pid --format=csv,noheader | wc -l")
  [ "$g" = 0 ] || { echo "$n has $g GPU processes: stop the serve first"; exit 1; }
done
echo "plan + calibration + code identical on both ($ha); image $ia; GPUs idle"
for spec in "$A:0:$B_IP" "$B:1:$A_IP"; do
  IFS=: read -r n part peer <<<"$spec"
  SYNC="mkdir -p $RUN && bash $STEP5/guards.sh start && (tmux kill-session -t p1sync 2>/dev/null || true) && \
    tmux new-session -d -s p1sync 'bash $STEP5/exchange_sync.sh $part $peer $RUN >> $RUN/exchange_sync.p$part.log 2>&1'"
  START="docker rm -f step5-p1 >/dev/null 2>&1; docker run -d --name step5-p1 --gpus all --ipc=host --restart no \
    -e STEP5_CUDA_MEM_FRACTION=$MEMFRAC -v $MODEL:/model:ro -v $STEP5:/q:ro -v $CAL:/cal:ro \
    -v $(dirname "$PLAN"):/plan:ro -v $RUN:/out $RMOUNT -w /q --entrypoint bash $IMAGE -c \
    'python3 -u /q/p3_pass1.py /model /cal/$CALF /plan/$(basename "$PLAN") /out --layers $LAYERS \
       --part $part/2 $RARGS $P1_EXTRA >> /out/pass1.p$part.stdout 2>&1; echo P1_EXIT=\$? >> /out/pass1.p$part.stdout' \
    >/dev/null && echo '$n part $part/2 started (stdout $RUN/pass1.p$part.stdout)'"
  if [ "$DRY" = 1 ]; then echo "== DRY $n:"; echo "$SYNC"; echo "$START"; continue; fi
  sh_ "$n" "$SYNC"
  sh_ "$n" "$START"
done
