#!/bin/bash
# Run a command inside the Step-5 serving image on this node's GPU, with ~/step5/serve/patch at /patch (read-only)
# and ~/step5/runs/patch at /work, ~/models at /models and ~/step5 at /step5 (read-only). Refuses while a serving or
# experiment container is up.
#   bash ~/step5/serve/patch/run_in_image.sh python3 /patch/tests/skinny_check.py /patch
set -euo pipefail
if docker ps --format '{{.Names}}' | grep -Eq '^(st5-r0|st5-r1|step5)'; then
    echo "a Step-5 container is running on this node; stop it first" >&2
    exit 1
fi
IMAGE=$(python3 -c 'import json,os;print(json.load(open(os.path.expanduser("~/step5/runs/exp/inspect-r0.json")))["Config"]["Image"])')
mkdir -p "$HOME/step5/runs/patch"
exec docker run --rm --gpus all --ipc host --network none \
    -v "$HOME/step5/serve/patch:/patch:ro" -v "$HOME/step5/runs/patch:/work" -w /work \
    -v "$HOME/models:/models:ro" -v "$HOME/step5:/step5:ro" \
    -e TRITON_CACHE_DIR=/work/triton-cache -e PYTHONDONTWRITEBYTECODE=1 \
    --entrypoint "" "$IMAGE" "$@"
