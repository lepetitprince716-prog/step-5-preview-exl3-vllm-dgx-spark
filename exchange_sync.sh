#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Host-side mirror for a 2-box pass 1 (ported from sudoingx dsv41_recipe scripts/quantize/exchange_sync.sh; run on each
# Spark, in tmux, as the normal user):
#   exchange_sync.sh <my_part> <peer fabric IP> <run_dir>
# Pushes this box's encoded part files (layer_LL.part<my_part>.safetensors and .json) to the same path on the peer over
# the ConnectX link, once each (--ignore-existing; kept for assembly). Every pushed file is then evicted from the page
# cache on both boxes (rsync's read here, its write there): GB10 CUDA allocations cannot reclaim page cache (sudoingx
# FINDINGS 1-3; ~8 GB landing per layer during the peer's encode took a Spark down on Oct 8). Partial writes are never
# pushed: p3_pass1.py writes *.tmp and renames, and rsync renames into place on the peer.
set -u
PART=$1 PEER=$2 RUN=${3%/}
SSH="ssh -T -o Compression=no -c aes128-gcm@openssh.com"
evict() {   # $1 dir, then file names relative to it: flush + drop each file's pages here and on the peer
  local d=$1 f
  shift
  [ $# -gt 0 ] || return 0
  for f in "$@"; do sync -d "$d/$f" 2>/dev/null; dd if="$d/$f" iflag=nocache count=0 2>/dev/null; done
  $SSH "$PEER" "cd '$d' && for f in $*; do sync -d \"\$f\" 2>/dev/null; dd if=\"\$f\" iflag=nocache count=0 2>/dev/null; done"
}
while true; do
  sent=$(rsync -a --ignore-existing --out-format='%n' --include "layer_*.part$PART.safetensors" \
    --include "layer_*.part$PART.json" --exclude "*" -e "$SSH" "$RUN/" "$PEER:$RUN/" 2>/dev/null | grep -v '/$')
  [ -n "$sent" ] && evict "$RUN" $sent
  sleep 3
done
