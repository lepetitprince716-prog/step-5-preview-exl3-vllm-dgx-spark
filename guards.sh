#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Memory guards for long GPU jobs on a Spark (ported from sudoingx dsv41_recipe scripts/quantize/guards.sh), in tmux
# (they die with a reboot, so start them again after one):
#   dropcache  drop the page cache when free memory is under 20 GiB AND at least 4 GiB of it is reclaimable (Buffers +
#              Cached - Shmem; GB10 CUDA allocations cannot reclaim it), logged in $LOGS/dropcache.log. Page cache
#              only (echo 1). Needs passwordless `sudo -n tee /proc/sys/vm/drop_caches`.
#   memwatch   used / available GiB every 10 s into $LOGS/memwatch_<name>.log
#   servewatch kill the serve container ($SERVE_CONTAINER) when MemAvailable < 1.5 GiB (a dead serve costs a restart,
#              a wedged GB10 a power cycle), logged in $LOGS/servewatch.log
#   bash guards.sh start [name]   before encodes: dropcache + memwatch (name defaults to the hostname)
#   bash guards.sh serve [name]   for a serve: stop dropcache, start memwatch + servewatch
#   bash guards.sh stop           kill all three
# Environment: STEP5_LOGS (default ~/step5/logs), SERVE_CONTAINER (docker name filter, default ^st5-r[0-9]+$).
set -u
L=${STEP5_LOGS:-$HOME/step5/logs}
SERVE_CONTAINER=${SERVE_CONTAINER:-^st5-r[0-9]+\$}   # serve/launch2.sh names its containers st5-r0 / st5-r1
me=$(readlink -f "$0")
case "${1:-start}" in
  start)
    mkdir -p "$L"
    tmux has-session -t dropcache 2>/dev/null || tmux new-session -d -s dropcache "bash $me loop-dropcache"
    tmux has-session -t memwatch 2>/dev/null || tmux new-session -d -s memwatch "bash $me loop-memwatch ${2:-$(hostname)}"
    tmux ls 2>/dev/null | grep -E '^(dropcache|memwatch):' ;;
  serve)
    mkdir -p "$L"
    tmux kill-session -t dropcache 2>/dev/null
    tmux has-session -t memwatch 2>/dev/null || tmux new-session -d -s memwatch "bash $me loop-memwatch ${2:-$(hostname)}"
    tmux has-session -t servewatch 2>/dev/null || tmux new-session -d -s servewatch "bash $me loop-servewatch"
    tmux ls 2>/dev/null | grep -E '^(dropcache|memwatch|servewatch):' ;;
  stop)
    for s in dropcache memwatch servewatch; do tmux kill-session -t $s 2>/dev/null; done; true ;;
  loop-dropcache)
    while true; do
      read -r f r < <(awk '/^MemFree:/{m=$2} /^Buffers:/{b=$2} /^Cached:/{c=$2} /^Shmem:/{s=$2}
                            END{printf "%d %d\n", m / 1048576, (b + c - s) / 1048576}' /proc/meminfo)
      if [ "$f" -lt 20 ] && [ "$r" -ge 4 ]; then
        sync; echo 1 | sudo -n tee /proc/sys/vm/drop_caches >/dev/null
        echo "$(date +%T) dropped page cache (free was $f GiB, $r GiB reclaimable)" >> "$L/dropcache.log"
      fi
      sleep 5
    done ;;
  loop-servewatch)
    while true; do
      a=$(awk '/^MemAvailable:/{printf "%d", $2 / 1024}' /proc/meminfo)     # MiB
      if [ "$a" -lt 1536 ]; then
        for c in $(docker ps -q --filter "name=$SERVE_CONTAINER"); do
          docker rm -f "$c" >/dev/null 2>&1
          echo "$(date +%T) MemAvailable $a MiB: killed $c ($SERVE_CONTAINER)" >> "$L/servewatch.log"
        done
      fi
      sleep 2
    done ;;
  loop-memwatch)
    while true; do
      echo "$(date +%H:%M:%S) $(free -g | awk '/Mem:/{print "used", $3, "avail", $7}')" >> "$L/memwatch_$2.log"
      sleep 10
    done ;;
  *) echo "usage: $0 start|serve|stop [name]"; exit 2 ;;
esac
