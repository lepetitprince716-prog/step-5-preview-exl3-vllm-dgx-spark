#!/bin/bash
# Fill ~/step5/fabric.env on the head from the live fabric addresses of both nodes. Prints no address: only the line
# count of the written file. Run on the head as the normal user:
#   bash ~/step5/serve/make_fabric_env.sh
set -euo pipefail
IF=${IF:-enp1s0f1np1}
WORKER=${WORKER:-spark-worker}
OUT=${OUT:-$HOME/step5/fabric.env}
HERE=$(cd "$(dirname "$0")" && pwd)
head_ip=$(ip -4 -o addr show "$IF" | awk '{print $4}' | cut -d/ -f1 | head -1)
worker_ip=$(ssh -o BatchMode=yes -o ConnectTimeout=10 "$WORKER" "ip -4 -o addr show $IF" | awk '{print $4}' | cut -d/ -f1 | head -1)
[ -n "$head_ip" ] && [ -n "$worker_ip" ] || { echo "no IPv4 on $IF on one of the nodes; set IF=<ifname>" >&2; exit 1; }
umask 077
sed -e "s/^HEAD_IP=.*/HEAD_IP=$head_ip/" -e "s/^WORKER_IPS=.*/WORKER_IPS=$worker_ip/" \
    -e "s/^HEAD_IFNAME=.*/HEAD_IFNAME=$IF/" -e "s/^WORKER_IFNAMES=.*/WORKER_IFNAMES=$IF/" \
    "$HERE/fabric.env.example" > "$OUT"
chmod 600 "$OUT"
echo "written $OUT ($(grep -c . "$OUT") lines, both addresses filled)"
