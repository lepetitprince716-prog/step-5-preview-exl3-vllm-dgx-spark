#!/bin/bash
# Sample this node's GPU (SM clock, power, temperature, utilization) every 500 ms and the CPU busy share for N seconds;
# print one line of medians. Run on both nodes while a benchmark runs:
#   bash ~/step5/serve/gpu_sample.sh 60
set -euo pipefail
N=${1:-30}
OUT=$(mktemp)
trap 'rm -f "$OUT"' EXIT
read -r _ u0 n0 s0 i0 w0 q0 sq0 st0 _ < /proc/stat
timeout "$N" nvidia-smi --query-gpu=clocks.sm,power.draw,temperature.gpu,utilization.gpu \
    --format=csv,noheader,nounits -lms 500 > "$OUT" || true
read -r _ u1 n1 s1 i1 w1 q1 sq1 st1 _ < /proc/stat
busy=$(( (u1 - u0) + (n1 - n0) + (s1 - s0) + (q1 - q0) + (sq1 - sq0) + (st1 - st0) ))
idle=$(( (i1 - i0) + (w1 - w0) ))
python3 - "$OUT" "$busy" "$idle" "$(hostname)" <<'EOF'
import statistics, sys
rows = []
for line in open(sys.argv[1]):
    parts = [p.strip() for p in line.split(",")]
    try:
        rows.append([float(p) for p in parts])
    except ValueError:
        continue
busy, idle = int(sys.argv[2]), int(sys.argv[3])
cpu = 100.0 * busy / max(1, busy + idle)
if not rows:
    print(sys.argv[4], "no GPU samples; cpu busy %.1f%%" % cpu)
    sys.exit(0)
cols = list(zip(*rows))
med = [statistics.median(c) for c in cols]
print(f"{sys.argv[4]}: {len(rows)} samples, SM clock median {med[0]:.0f} MHz (min {min(cols[0]):.0f}), power {med[1]:.1f} W, "
      f"temp {med[2]:.0f} C, GPU util {med[3]:.0f}%, CPU busy {cpu:.1f}%")
EOF
