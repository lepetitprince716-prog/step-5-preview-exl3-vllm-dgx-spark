#!/bin/bash
# Start a serving experiment with the cfgB settings (dual-HCA NCCL, 4096-token prefill chunks, skinny kernels for the
# router, g_proj and draft heads) plus any extra exp_launch.py start arguments given after the name.
# K=<n> sets the MTP draft length (default 2) and the matching CUDA graph sizes (up to 4 sequences x (n+1) tokens).
#   K=3 bash ~/step5/serve/exp_cfg.sh cfgK3 [--env K=V] [--mount SRC:DST] ...
set -euo pipefail
name=${1:?usage: exp_cfg.sh NAME [exp_launch start args]}
shift
P=$HOME/step5/serve/patch/step5_spark
D=/usr/local/lib/python3.12/dist-packages/step5_spark
k=${K:-2}
extra=()
if [ "$k" != 2 ]; then
    sizes="1,2,4,8"
    for ((n = 12; n <= 4 * (k + 1); n += 4)); do sizes="$sizes,$n"; done
    extra+=("--arg=--speculative-config={\"method\":\"mtp\",\"num_speculative_tokens\":$k}")
    extra+=("--arg=--compilation-config={\"cudagraph_mode\":\"FULL_DECODE_ONLY\",\"cudagraph_capture_sizes\":[$sizes]}")
fi
cd "$HOME/step5"
exec python3 serve/exp_launch.py start "$name" \
    --env NCCL_IB_HCA=rocep1s0f1,roceP2p1s0f1 --env ST_EXL3_ST_CHUNK_TOKENS=4096 --env STEP5_SKINNY=router,gproj,draft \
    --mount "$P/skinny.py:$D/skinny.py" --mount "$P/step5.py:$D/step5.py" --mount "$P/step5_mtp.py:$D/step5_mtp.py" \
    "${extra[@]}" "$@"
