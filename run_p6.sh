#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# One p6_kld.py run in a detached container on this box (GPU idle required).
#   bash run_p6.sh ref ref                                   teacher -> runs/p6/ref/ref.safetensors
#   bash run_p6.sh quant b444 --experts /p1 --body /dense    runs/p6/b444/kld.json (ref.safetensors hardlinked in)
#   bash run_p6.sh quant b688 --experts /p1 --body /dense/body_6-8-8
#   bash run_p6.sh quant x --experts /p1                     experts only, body native
# Container mounts: /model official FP8, /cal, /p1 pass 1, /dense p4, /out runs/p6. stdout runs/p6/<name>.stdout,
# last line P6_EXIT=<code>. Container name step5-p6-<name>.
set -euo pipefail
MODE=${1:?mode: ref | quant | serving}; NAME=${2:?run name (output subdir)}; shift 2
H=$HOME
STEP5=${STEP5:-$H/step5}
IMAGE=${IMAGE:-dsv41-quant:dev}
MEMFRAC=${STEP5_CUDA_MEM_FRACTION:-0.70}
OUT=$STEP5/runs/p6/$NAME
mkdir -p "$OUT"
if [ "$MODE" != ref ] && [ ! -e "$OUT/ref.safetensors" ]; then
  [ -e "$STEP5/runs/p6/ref/ref.safetensors" ] || { echo "no teacher: run 'run_p6.sh ref ref' first"; exit 1; }
  ln "$STEP5/runs/p6/ref/ref.safetensors" "$OUT/ref.safetensors" 2>/dev/null \
    || cp "$STEP5/runs/p6/ref/ref.safetensors" "$OUT/ref.safetensors"
fi
g=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader | wc -l)
[ "$g" = 0 ] || { echo "$g GPU processes running: wait for them"; exit 1; }
docker rm -f "step5-p6-$NAME" >/dev/null 2>&1 || true
docker run -d --name "step5-p6-$NAME" --gpus all --ipc=host --restart no --user "$(id -u):$(id -g)" -e HOME=/tmp \
  -e STEP5_CUDA_MEM_FRACTION="$MEMFRAC" -v "$H/models/Step-5-Preview-FP8:/model:ro" -v "$STEP5:/q:ro" \
  -v "$STEP5/cal:/cal:ro" -v "$STEP5/runs/p1:/p1:ro" -v "$STEP5/runs/p4:/dense:ro" -v "$STEP5/runs/p6:/out" \
  -w /q --entrypoint bash "$IMAGE" -c \
  "python3 -u p6_kld.py /model /cal/panel.safetensors /cal/panel_index.json /out/$NAME --mode $MODE $* \
     >> /out/$NAME.stdout 2>&1; echo P6_EXIT=\$? >> /out/$NAME.stdout"  >/dev/null
echo "step5-p6-$NAME started (stdout $OUT.stdout)"
