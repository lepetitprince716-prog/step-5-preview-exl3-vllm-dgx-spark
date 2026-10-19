# Step-5-Preview EXL3 on two DGX Sparks

Code behind [kefallrain/Step-5-Preview-EXL3-2.4bpw-2xSpark](https://huggingface.co/kefallrain/Step-5-Preview-EXL3-2.4bpw-2xSpark),
a 2.4 bpw EXL3 quantization of [stepfun-ai/Step-5-Preview](https://huggingface.co/stepfun-ai/Step-5-Preview) (604B
total, 27B active) that serves on two NVIDIA DGX Spark (GB10, 128 GB each) with vLLM tensor parallel 2:

- the quantization pipeline that produced the checkpoint (calibration set, statistics, per-expert bit allocation,
  sequential EXL3 encoding, assembly, KL evaluation);
- patches to the [0xSero step5_spark](https://huggingface.co/0xSero/Step-5-Preview-Spark) vLLM plugin that make
  decode 65 % faster, including a vendored copy of TensorFold's grouped EXL3 routed-experts kernel;
- the benchmark, smoke-test and KL-capture scripts behind the numbers below, and the measurement notes.

The model card on Hugging Face describes the checkpoint itself (files, recipe, fidelity, task benchmarks).

## Decode speed

Single stream, 384 generated tokens, MTP k=2, `serve/decode_bench.py` (kit_bench prompt set b); 2 and 4 concurrent
streams on the same set; prefill from kit_bench. All runs on the same two Sparks and checkpoint.

| Configuration | set b tok/s | 2 / 4 streams, tok/s total | prefill 8K / 32K tok/s |
|---|---|---|---|
| plugin as shipped | 24.9 | 34.0 / 50.7 | 989 / 907 |
| + dual-HCA NCCL, 4096-token prefill chunks, skinny GEMV for router, g_proj and draft heads | 27.0 | 35.0 / 51.5 | 1046 / 1019 |
| + int8 draft heads and MTP blocks, fused routing prep (`STEP5_SKINNY=router,gproj,draft8,mtp8`, `ST_EXL3_PREP=1`) | 28.2 | 37.5 / 54.0 | 1076 / 1019 |
| + TensorFold grouped EXL3 experts below 64 tokens (`ST_EXL3_TF=1`) | **41.1** | **53.2 / 73.8** | 1073 / 1013 |

The last row is the production configuration since 2026-10-19 (re-measured after the switch: 42.1 tok/s). A decode
step takes about 60 ms against 87 ms before the kernel change; the MTP acceptance length is unchanged (2.47 of 3).

## Output fidelity of the serving engine

The kernel change was checked against the teacher (the official FP8 model run through the reference implementation)
on every 6th sequence of the held-out panel: 29 sequences, 65,498 scored positions, captured from the serving engine
itself without MTP or prefix caching, in 48-token prefill chunks so that every MoE call takes the decode-path kernel.

| Routed-experts kernel | KL mean | KL p99 | top-1 | NLL (teacher 0.4907) |
|---|---|---|---|---|
| exllamav3 `exl3_moe` (as shipped) | 0.01945 | 0.2321 | 95.70 % | 0.5064 |
| TensorFold grouped (`ST_EXL3_TF=1`) | 0.01964 | 0.2316 | 95.69 % | 0.5061 |

The difference is +0.00019 (+1.0 %) with a standard error of 0.00014 (sequences as clusters): no measurable change.
Per layer, the TensorFold path is closer to a float64 reference than the shipped one (rms error 3.4-4.3e-4 against
9.1-9.6e-4, `serve/patch/tests/tf_moe_check.py`).

## Using the serving patches

The serving image is `ghcr.io/0xsero/step-5-preview-spark@sha256:4e28f849a414a483b44be50a09198d76cbc859096b354796dc25ee2159e05a5e`.
The patches replace files of its `step5_spark` package by read-only bind mounts, so the image stays unchanged.

1. Build the TensorFold extension once on a Spark: `bash serve/patch/step5_spark/tf_exl3/build.sh` runs the build
   inside the serving image (sm_121, about 4 minutes, no GPU needed) and leaves `_C.cpython-312-aarch64-linux-gnu.so`
   beside the sources. Copy the `tf_exl3` directory with the module to the same path on the second Spark.
   `serve/patch/tests/tf_moe_check.py` checks the kernel against the shipped path on real layers.
2. Add to both ranks' `docker run`:

   ```text
   -e NCCL_IB_HCA=<both RoCE HCAs, comma-separated>
   -e ST_EXL3_ST_CHUNK_TOKENS=4096
   -e STEP5_SKINNY=router,gproj,draft8,mtp8
   -e ST_EXL3_PREP=1
   -e ST_EXL3_TF=1
   -v <repo>/serve/patch/step5_spark/<name>:/usr/local/lib/python3.12/dist-packages/step5_spark/<name>:ro
   ```

   with one `-v` for each of `skinny.py`, `step5.py`, `step5_mtp.py`, `exl3_quant.py`, `moe_prep.py` and `tf_exl3`.
   The patched files must exist at the same path on both nodes.

The two-node launcher itself (`serve/launch2.sh`) ships with the checkpoint on Hugging Face; it does not include the
lines above. `serve/exp_launch.py` starts A/B experiments as clones of running production containers with explicit
edits, and `serve/kld_capture.py` with `serve/patch/kld_eval.py` measures the serving engine's KL as in the table above.

## Reproducing the quantization

`RUNBOOK.md` (in Chinese) lists every step in order with its command and completion check. In short:

| Step | Script | What it does |
|---|---|---|
| 0 | `caldata/` | calibration conversations (published sudoingx prompts, answers from the official API) and token sets |
| 1 | `p0_stats.py`, `p0b_merge_stats.py` | per-layer activation statistics and expert Hessians from the FP8 teacher |
| 2 | `p1_sensitivity.py`, `p1b_relerr.py` | per-layer output-noise sensitivity and measured per-K expert error |
| 3 | `p2_allocate.py` | per-expert bit allocation at 2.4 bpw (`p2_uniform.py`: uniform control) |
| 4 | `p3_pass1.py`, `run_pass1_2box.sh` | sequential EXL3 encoding of the experts, half of each layer per machine |
| 5 | `p4_dense.py`, `run_p4_2box.sh` | EXL3 encoding of the attention, shared-expert and dense projections |
| 6 | `p5_assemble.py` | checkpoint assembly (`serve/verify.py` checks it) |
| 7 | `p6_kld.py` | KL, top-1 and NLL against the teacher on the held-out panel |

The calibration data, the panel and the API key are not in this repository. `tests/` holds the CPU tests
(`pytest -m 'not gpu'`) and the single-layer GPU checks; `notes/` holds the measurement logs (`serving.md` for the
speed and KL work above, `panel.md` for fidelity, `evals.md` for task benchmarks, `gpu_checks.md` for the pre-flight
checks). Host names in scripts and notes are the generic `spark-head` and `spark-worker`.

## License

Apache License 2.0 (`LICENSE`); third-party code keeps its own SPDX headers and notices (`NOTICE`,
`serve/patch/step5_spark/tf_exl3/`). The model weights are not here; they are distributed with the checkpoint under
the StepFun Community License.
