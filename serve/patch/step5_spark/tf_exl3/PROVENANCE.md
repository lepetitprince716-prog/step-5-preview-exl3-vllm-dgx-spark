# Provenance: step5_spark/tf_exl3

| Field | Value |
|---|---|
| Source | https://github.com/bertholomus/TensorFold, branch `deepseek-v41-tp2`, commit `8b1fe210d5e0d68e0cdb3b14755c135465c9ffa8` (2026-10-12) |
| Upstream of that fork | https://github.com/ashhart/TensorFold 0.6.6, commit `cb2ebf0` |
| License | Apache-2.0 (`LICENSE`); TensorFold code written before 0.6.0 keeps its MIT notice (`LICENSE-MIT`). Attribution: `NOTICE` |
| Taken | `src/tensorfold/cuda/exl3/`: `experts.py`, `experts.cpp`, `experts.cu`, `experts_grouped.cuh`, `experts_cb0.cu`, `experts_cb1.cu`, `experts_cb2.cu`; the tree's `LICENSE` and `LICENSES/MIT.txt` |
| Used by | `exl3_quant.Exl3MoEMethod.apply` with `ST_EXL3_TF=1`: the routed experts of layers with fewer than 64 tokens (`experts.routed`, its fused decode path) |

## Changes

| File | Change |
|---|---|
| `experts.cu` | `gateup_epilogue_kernel` takes `act_mode` 2: SwiGLU in fp32 with the gate clamped after SiLU, `min(silu(g), limit) * clamp(u, -limit, limit)` (Step-5's activation, as vLLM and ExLlamaV3 compute it; upstream's modes clamp the gate before SiLU) |
| `experts_grouped.cuh` | the fused decode epilogue takes the same `act_mode` 2 |
| `experts.py` | `ACT_F32_POST = 2`; `_ext()` imports the prebuilt `step5_spark.tf_exl3._C` instead of building with `tensorfold.cuda.build.load` |
| `experts_bind.cpp` | upstream `experts.cpp`, renamed only: setuptools names objects by stem, so `experts.cpp` and `experts.cu` would both build `experts.o` |
| added | `__init__.py`, `setup.py`, `build.sh`, `NOTICE`, `PROVENANCE.md`, `LICENSE-MIT` |

Unchanged: `experts_cb0.cu`, `experts_cb1.cu`, `experts_cb2.cu`, `LICENSE`.

Not slimmed: the prompt-chunk kernels (`grouped_rows`, `grouped_mma*`, `grouped_down3`) and the six-launch decode paths
are compiled but unused, since step5_spark keeps `st_moe_ext` for 64 tokens or more and takes the fused decode path
below; they stay so these files remain upstream's plus the changes listed above.

## Build

`bash build.sh` on a Spark builds `_C.cpython-312-aarch64-linux-gnu.so` beside the sources inside the serving image:
sm_121 only, about 4 minutes, no GPU needed. The serving containers mount this directory at
`/usr/local/lib/python3.12/dist-packages/step5_spark/tf_exl3`.

## Checks

`tests/tf_moe_check.py` (2026-10-18; layers 3, 30, 47 and 90, expert-TP rank 0's slice, 352 experts, top-8):

- the lane decode is bit-identical to the image's ExLlamaV3 `reconstruct` for every bit width present (K2 to K5, mul1);
- routed output against float64: rms error 3.4e-4 to 4.3e-4, against 9.1e-4 to 9.6e-4 for the exllamav3 `exl3_moe`
  buckets (with the SwiGLU limit clamping up to 62% of activations at the larger input scale);
- bf16 against fp16 inputs, rows alone against rows inside a window, CUDA graph replay against the eager call:
  bit-identical;
- a layer's routed experts on random routings, CUDA graph of 50 calls: 105-132 us at 1 token, 291-374 us at 3 and
  1031-1333 us at 12, against 277-314, 553-635 and 1697-1791 us for the `exl3_moe` buckets; 206-229 GB/s at 1 to 63
  tokens.

End to end (2026-10-18, `notes/serving.md`): with this kernel the two Sparks decode 41.1 tok/s single stream against
28.2 with the `exl3_moe` buckets (cfgT against cfgE), and the KL against the teacher on 29 panel sequences (65,498
positions) is 0.01964 against 0.01945, a difference of 1.4 standard errors (sequences as clusters); top-1 0.9569
against 0.9570.

## Updating

Copy the seven files from a newer commit, re-apply the changes above (each changed file notes them at its top), run
`build.sh`, then `tests/tf_moe_check.py`.
