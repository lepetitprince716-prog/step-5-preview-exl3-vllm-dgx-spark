#!/bin/bash
# Build step5_spark.tf_exl3._C inside the serving image as the calling user; the module lands beside the sources.
# Needs no GPU. Prints the tail of the build log, the module and the architectures its cubins were built for.
set -euo pipefail
IMG=${IMG:-ghcr.io/0xsero/step-5-preview-spark@sha256:4e28f849a414a483b44be50a09198d76cbc859096b354796dc25ee2159e05a5e}
ARCH=${ARCH:-12.1}
SRC=$(cd "$(dirname "$0")" && pwd)
rm -f "$SRC"/_C*.so
docker run --rm --user "$(id -u):$(id -g)" -e HOME=/tmp -e TORCH_CUDA_ARCH_LIST="$ARCH" --memory 32g \
  --entrypoint bash -v "$SRC:/src" -w /src "$IMG" -c '
    set -o pipefail
    start=$(date +%s)
    python3 setup.py build_ext --inplace --build-temp /tmp/tf_exl3_build --build-lib /tmp/tf_exl3_lib > /tmp/build.log 2>&1 || { tail -n 60 /tmp/build.log; exit 1; }
    tail -n 5 /tmp/build.log
    echo "build took $(( $(date +%s) - start )) s"
    ls -la /src/_C*.so
    cuobjdump --list-elf /src/_C*.so | sed "s/.*\.\(sm_[0-9a-z]*\)\..*/\1/" | sort | uniq -c'
