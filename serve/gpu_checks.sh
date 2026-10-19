#!/bin/bash
# SPDX-License-Identifier: Apache-2.0
# Stage 2 GPU pre-flight checks (research/2026-10-15-step5-2spark-plan.md, 阶段 2), in order. Run on the HEAD Spark,
# one check per invocation; every check prints "CHECK n PASS|FAIL: <numbers>" and keeps a log in $RUNS/check<n>.log.
#
#   bash ~/step5/serve/gpu_checks.sh 1        both images: exllamav3 version, exl3_moe / exl3_moe_coop signatures,
#                                             EXL3_INT8_GEMV and the int8 GEMV max K, quantize.refit_scales present?
#   bash ~/step5/serve/gpu_checks.sh 2        synthetic checkpoint TP2 boot (launch2.sh): MemAvailable per node, KV pool,
#                                             decode tok/s (256 greedy tokens); body K variants (BODY_VARIANTS, same
#                                             experts); per-K-bucket decode cost (1 vs 3 buckets)
#   bash ~/step5/serve/gpu_checks.sh 3        the 23 ssmax_s tensors of the official checkpoint (CPU): min / max per layer
#   bash ~/step5/serve/gpu_checks.sh 4 [..]   lane A's tests/gpu_single_layer.py in the quant image (args passed on)
#   bash ~/step5/serve/gpu_checks.sh 5        EXL3 cross-version decode: encode with exllamav3 1.6.0 (dsv41-quant:dev),
#                                             reconstruct in the serving image (1.5.3), compare bit for bit (K 2-6, 8)
# Checks 1, 2, 4, 5 use the GPU and refuse to run unless the cluster is idle (spark stop: desired=none, nothing on
# :8888, no GPU process on either node). Check 2 takes about 1.5-2 hours (five boots); start it detached:
#   nohup setsid bash ~/step5/serve/gpu_checks.sh 2 > ~/step5/runs/gpu_checks/check2.out 2>&1 < /dev/null &
#
# Env: OFFICIAL (~/models/Step-5-Preview-FP8, same path on the worker) SYN (~/models/step5-synth, both nodes)
#      WORKER (spark-worker) IMG (serving image, pinned) QIMG (dsv41-quant:dev) RUNS (~/step5/runs/gpu_checks)
#      MIX_MAIN (2:0.55,3:0.35,4:0.10) MIX_ONE (2:1.0) MIX_THREE (2:0.98,3:0.01,4:0.01) MIN_AVAIL_GIB (4)
#      BODY_BASE (4/4/4) BODY_VARIANTS ("6/8/8 8/8/8")
#      KV_BYTES MAX_LEN NSPEC_MAIN (2) are passed to launch2.sh; FABRIC_ENV as in launch2.sh
set -euo pipefail

HERE=$(cd "$(dirname "$0")" && pwd)
Q=$(dirname "$HERE")
IMG=${IMG:-ghcr.io/0xsero/step-5-preview-spark@sha256:4e28f849a414a483b44be50a09198d76cbc859096b354796dc25ee2159e05a5e}
QIMG=${QIMG:-dsv41-quant:dev}
OFFICIAL=${OFFICIAL:-$HOME/models/Step-5-Preview-FP8}
SYN=${SYN:-$HOME/models/step5-synth}
WORKER=${WORKER:-spark-worker}
RUNS=${RUNS:-$Q/runs/gpu_checks}
MIX_MAIN=${MIX_MAIN:-2:0.55,3:0.35,4:0.10}
MIX_ONE=${MIX_ONE:-2:1.0}                    # one K bucket per layer
MIX_THREE=${MIX_THREE:-2:0.98,3:0.01,4:0.01} # three buckets, within 1.5% of MIX_ONE's bytes
MIN_AVAIL_GIB=${MIN_AVAIL_GIB:-4}
BODY_BASE=${BODY_BASE:-4/4/4}                # attention/shared/mlp K of the main boot
BODY_VARIANTS=${BODY_VARIANTS:-6/8/8 8/8/8}  # extra boots, same expert mix: decode tok/s and MemAvailable each
NSPEC_MAIN=${NSPEC_MAIN:-2}
PORT=8888
MOE_LAYERS=88
UID_GID="$(id -u):$(id -g)"
mkdir -p "$RUNS"

wssh() { timeout "${T:-60}" ssh -o BatchMode=yes -o ConnectTimeout=10 "$WORKER" "$@"; }
log() { echo "$(date +%H:%M:%S) $*"; }
avail_gib() { awk '/MemAvailable/{printf "%.1f", $2/1048576}' /proc/meminfo; }
w_avail_gib() { T=20 wssh "awk '/MemAvailable/{printf \"%.1f\", \$2/1048576}' /proc/meminfo"; }

idle_or_die() {
  local n=$1 d busy wbusy
  d=$(cat "$HOME/.cache/spark-ctl/desired" 2>/dev/null || echo none)
  [ "$d" = none ] || { echo "CHECK $n FAIL: spark watcher keeps '$d' serving; run 'spark stop' first"; exit 1; }
  if curl -s -m 3 -o /dev/null "http://127.0.0.1:$PORT/v1/models"; then
    echo "CHECK $n FAIL: something answers on 127.0.0.1:$PORT; run 'spark stop' first"; exit 1
  fi
  busy=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | grep -c . || true)
  wbusy=$(T=20 wssh "nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | grep -c . || true")
  [ "${busy:-0}" = 0 ] && [ "${wbusy:-0}" = 0 ] \
    || { echo "CHECK $n FAIL: GPU processes running (head $busy, worker $wbusy)"; exit 1; }
}

# ---------------------------------------------------------------------------------------------------------- check 1
check1() {
  idle_or_die 1
  # the encoder image first (info: exllamav3 1.6.0, refit_scales), then the serving image (the CHECK 1 line)
  docker run --rm --gpus all --user "$UID_GID" --entrypoint python3 -e HOME=/tmp -v "$Q:/q:ro" -w /q "$QIMG" \
    serve/gpu_probe.py sigs --label quant || echo "quant image: sigs failed (info only)"
  docker run --rm --gpus all --entrypoint python3 -v "$Q:/q:ro" -w /q "$IMG" serve/gpu_probe.py sigs --label serve
}

# ---------------------------------------------------------------------------------------------------------- check 2
sync_worker_code() {  # synth_ckpt.py is pure Python: it runs on each host with its own python3
  tar -C "$Q" -cf - serve/synth_ckpt.py p5_assemble.py p_util.py step5q/__init__.py step5q/plan.py step5q/names.py \
    | T=60 wssh "mkdir -p ~/step5 && tar -C ~/step5 -xf -"
}

synth_both() {  # $1 k-mix, $2 extra synth args; head and worker in parallel
  local args="--meta $OFFICIAL --from-official $OFFICIAL --k-mix $1 ${2:-}" rc=0
  log "synth $1 ${2:-} on both nodes -> $SYN"
  # shellcheck disable=SC2086
  python3 "$Q/serve/synth_ckpt.py" "$SYN" $args > "$RUNS/synth-head.log" 2>&1 &
  local pid=$!
  T=7200 wssh "python3 ~/step5/serve/synth_ckpt.py $SYN $args" > "$RUNS/synth-worker.log" 2>&1 || rc=1
  wait "$pid" || rc=1
  tail -n 2 "$RUNS/synth-head.log"; tail -n 2 "$RUNS/synth-worker.log"
  [ "$rc" = 0 ] || { echo "CHECK 2 FAIL: synth_ckpt failed (see $RUNS/synth-*.log)"; exit 1; }
}

boot() {  # $1 NSPEC, $2 "soft" = return 1 instead of failing the check; leaves the server running
  log "boot NSPEC=$1"
  if ! WORKERS=$WORKER MODEL_DIR=$SYN NSPEC=$1 bash "$HERE/launch2.sh" start > "$RUNS/boot.log" 2>&1; then
    tail -n 30 "$RUNS/boot.log"
    WORKERS=$WORKER bash "$HERE/launch2.sh" stop >/dev/null 2>&1 || true
    [ "${2:-}" = soft ] && return 1
    echo "CHECK 2 FAIL: TP2 boot failed (NSPEC=$1, $RUNS/boot.log)"; exit 1
  fi
  tail -n 1 "$RUNS/boot.log"
}

BODIES=""
body_variant() {  # $1 body K a/s/m: rewrite exl3/body on both nodes, boot with MTP k=NSPEC_MAIN, append to BODIES
  local a w t r
  synth_both "$MIX_MAIN" "--body-only --body-k $1"
  if boot "$NSPEC_MAIN" soft; then
    sleep 30
    a=$(avail_gib); w=$(w_avail_gib); t=$(decode_tps || echo 0)
    stop_server
    r="body $1: MemAvailable head $a GiB worker $w GiB, decode $t tok/s"
  else
    r="body $1: boot FAILED ($RUNS/boot.log)"
  fi
  log "$r"
  BODIES="$BODIES$r; "
}

decode_tps() {  # prints decode tok/s (median of 3 x 256 greedy tokens)
  python3 "$HERE/gpu_probe.py" decode "http://127.0.0.1:$PORT" step5 --tokens 256 --reps 3 ${API_KEY_FILE:+--key-file "$API_KEY_FILE"} \
    | tee -a "$RUNS/decode.jsonl" | python3 -c 'import json,sys;print(json.load(sys.stdin)["decode_tok_s"])'
}

stop_server() { WORKERS=$WORKER bash "$HERE/launch2.sh" stop >/dev/null 2>&1 || true; sleep 20; }

check2() {
  idle_or_die 2
  [ -d "$OFFICIAL" ] || { echo "CHECK 2 FAIL: no official checkpoint at $OFFICIAL"; exit 1; }
  sync_worker_code
  trap stop_server EXIT                 # never leave a rank holding memory after a failed step
  synth_both "$MIX_MAIN" "--body-k $BODY_BASE"
  local a0 w0 a1 w1 kv tps_main tps1 tps3 per_bucket
  a0=$(avail_gib); w0=$(w_avail_gib)
  boot "$NSPEC_MAIN"
  sleep 30
  a1=$(avail_gib); w1=$(w_avail_gib)
  kv=$(docker logs st5-r0 2>&1 | python3 "$HERE/gpu_probe.py" kvlog | tr '\n' ';' || true)
  tps_main=$(decode_tps)
  log "main mix $MIX_MAIN body $BODY_BASE: MemAvailable head $a0 -> $a1 GiB, worker $w0 -> $w1 GiB; $kv" \
    "decode $tps_main tok/s (MTP k=$NSPEC_MAIN with random drafts: acceptance ~0)"
  stop_server

  local v                              # body-K variants, same experts (frontier review 2b)
  for v in $BODY_VARIANTS; do body_variant "$v"; done
  synth_both "$MIX_MAIN" "--body-only --body-k $BODY_BASE"

  synth_both "$MIX_ONE" --experts-only
  boot 0; tps1=$(decode_tps); stop_server
  synth_both "$MIX_THREE" --experts-only
  boot 0; tps3=$(decode_tps); stop_server
  per_bucket=$(python3 -c "t1,t3=$tps1,$tps3;d=(1e3/t3-1e3/t1)/2;print(f'{d:.3f} ms/token per extra bucket ({d*1e3/$MOE_LAYERS:.1f} us per layer)')")
  log "1 bucket ($MIX_ONE) $tps1 tok/s, 3 buckets ($MIX_THREE) $tps3 tok/s: $per_bucket"
  local ok=PASS
  python3 -c "import sys;sys.exit(0 if min($a1,$w1)>=$MIN_AVAIL_GIB and min($tps_main,$tps1,$tps3)>0 else 1)" || ok=FAIL
  echo "CHECK 2 $ok: main mix $MIX_MAIN MemAvailable head $a1 GiB worker $w1 GiB (floor $MIN_AVAIL_GIB), KV [$kv]," \
    "decode $tps_main tok/s (MTP k=$NSPEC_MAIN); body variants (MTP k=$NSPEC_MAIN): ${BODIES:-none}" \
    "no-MTP decode 1 bucket $tps1 / 3 buckets $tps3 tok/s; $per_bucket"
}

# ---------------------------------------------------------------------------------------------------------- check 3
check3() {
  docker run --rm --user "$UID_GID" --entrypoint python3 -v "$OFFICIAL:/model:ro" -v "$Q:/q:ro" -w /q "$QIMG" \
    serve/gpu_probe.py ssmax /model
}

# ---------------------------------------------------------------------------------------------------------- check 4
check4() {
  idle_or_die 4
  [ -f "$Q/tests/gpu_single_layer.py" ] || { echo "CHECK 4 FAIL: tests/gpu_single_layer.py (lane A) is missing"; exit 1; }
  if docker run --rm --gpus all --user "$UID_GID" --entrypoint python3 -e STEP5_CUDA_MEM_FRACTION -e HOME=/tmp \
      -v "$OFFICIAL:/model:ro" -v "$Q:/q" -w /q "$QIMG" tests/gpu_single_layer.py /model "$@"; then
    echo "CHECK 4 PASS: tests/gpu_single_layer.py exited 0"
  else
    echo "CHECK 4 FAIL: tests/gpu_single_layer.py reported failures (lines not OK above)"; exit 1
  fi
}

# ---------------------------------------------------------------------------------------------------------- check 5
check5() {
  idle_or_die 5
  local out="$RUNS/xver"
  mkdir -p "$out"
  docker run --rm --gpus all --user "$UID_GID" --entrypoint python3 -e HOME=/tmp -v "$Q:/q:ro" -v "$out:/out" -w /q \
    "$QIMG" serve/gpu_probe.py xver-encode /out
  docker run --rm --gpus all --entrypoint python3 -v "$Q:/q:ro" -v "$out:/out:ro" -w /q "$IMG" \
    serve/gpu_probe.py xver-decode /out
}

# ---------------------------------------------------------------------------------------------------------- check 6
check6() {  # tokenizer ids through the serving image's loader (vLLM / AutoTokenizer) vs tokenizer.json; CPU only
  docker image inspect "$IMG" >/dev/null 2>&1 || { echo "CHECK 6 FAIL: serving image not present; pull it in the maintenance window"; exit 1; }
  local fixed="$RUNS/tok-fixed"
  rm -rf "$fixed"; mkdir -p "$fixed"
  (cd "$Q" && python3 -c "import p5_assemble; p5_assemble.copy_tokenizer('$OFFICIAL', '$fixed')")
  echo "official tokenizer_config (LlamaTokenizerFast): a mismatch here is the known transformers 5.x trap"
  docker run --rm --entrypoint python3 -v "$OFFICIAL:/model:ro" -v "$Q:/q:ro" -w /q "$IMG" serve/gpu_probe.py tokcheck /model || true
  echo "assembled-style tokenizer_config (PreTrainedTokenizerFast): must match"
  if docker run --rm --entrypoint python3 -v "$fixed:/model:ro" -v "$Q:/q:ro" -w /q "$IMG" serve/gpu_probe.py tokcheck /model; then
    echo "CHECK 6 PASS: serving-image tokenizer ids equal tokenizer.json with tokenizer_class PreTrainedTokenizerFast"
  else
    echo "CHECK 6 FAIL: ids differ even with the rewritten tokenizer_class; serving would mangle prompts"; exit 1
  fi
}

n=${1:-}
[ -n "$n" ] && shift
case "$n" in
  1|2|3|4|5|6) "check$n" "$@" 2>&1 | tee "$RUNS/check$n.log"; exit "${PIPESTATUS[0]}" ;;
  *) echo "usage: $0 <1|2|3|4|5|6> [args for check 4]" >&2; exit 2 ;;
esac
