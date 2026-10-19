# Serving, 2026-10-17 (head spark-head rank 0, worker spark-worker rank 1)

Checkpoint `~/models/step5-exl3-b566` on both nodes (worker copy hardlinked to `step5-exl3` for the expert files).
`serve/launch2.sh start` with the defaults: TP2, expert TP, body exl3, MTP k=2, RoCE all-reduce on, KV 8e9 B per rank,
max model len 131072, max 4 seqs, batch 4096. Log `runs/serve/launch-b566.log`.

| Item | Result |
|---|---|
| first boot | ready after 769 s (compile cache cleared) |
| MemAvailable after boot | head 5.2 GiB, worker 6.5 GiB (floor 4) |
| smoke (`serve/smoke.py --long 20000`) | chat OK; tool call parsed (`get_weather`, `{"city": "Paris"}`); reasoning split from content; 25,526-token prompt prefilled in 30.8 s |
| decode, kit_bench set b, 384 tokens, 2 reps (`serve/decode_bench.py`) | median 25.0 tok/s (code 25.4, prose 21.7, structured 25.0), TTFT 0.3 s; `runs/serve/decode-b566-mtp2.json` |
| MTP k=2 | mean acceptance length 2.59-2.66 of 3 (vLLM SpecDecoding metrics), generation throughput 24.3-24.7 tok/s single stream |
| prefill, kit_bench (`runs/serve/prefill-b566.json`) | 8192: 1003 tok/s (TTFT 8.2 s); 32768: 899 tok/s (36.4 s); 126976: 662 tok/s (191.8 s) |

Notes:
- kit_bench's decode command sees no samples on this model: the image's vLLM streams the thinking as `delta.reasoning`
  and kit_bench only times `delta.content` chunks. `serve/decode_bench.py` wraps it (same prompts, same tok/s formula).
- Two greedy runs of the same prompt differ in text (kit_bench `identical_texts: false`): MTP verification and batching
  are not bit-reproducible run to run.
- 262K context is out at this memory: KV 8e9 holds about 178K tokens per rank; 262144 needs about 11.7e9 and the head
  has 5.2 GiB left. Per-request cap stays 131072.
- Incident 2026-10-17 07:20: `evalplus.evaluate` (parallel test execution, 4.5 GB RSS) on the head drove MemAvailable
  to 0 and launch2's memguard killed st5-r0. The spark watcher did not restart it: the memguard started by the previous
  `spark restart` had inherited spark-ctl's job lock (fd 9) and held it for two hours, so the watcher's `flock -n`
  always failed. Fixes: `evals/run_evalplus.sh` runs sanitize + evaluate inside `docker run --memory 4g`;
  `launch2.sh` starts the memguard with `9>&-`. Rule: nothing memory-hungry runs unconfined on a serving node.
- Untested: MTP k=3 (the model has 3 MTP layers; k=2 saturates at 2.00/2 verification depth), `p6_kld.py --mode serving`
  against the plugin's capture (RUNBOOK 6), 2+ concurrent streams, `spark` idle/wake cycle.

## Decode speed work, 2026-10-18 (experiment instances via `serve/exp_launch.py`, launcher `serve/exp_cfg.sh`)

Production `step5` profile stopped from 12:02; port 8888 served the experiment instances. `serve/cmp_runs.py`:

| run | set b tok/s (acceptance) | set a | 2 / 4 streams | prefill 8K / 32K |
|---|---|---|---|---|
| base (production settings) | 24.94 (2.473) | - | 34.0 / 50.7 | 989 / 907 |
| cfgB: dual-HCA NCCL, 4096-token prefill chunks, skinny router / g_proj / draft heads | 26.99 (2.501) | 25.52 | 35.0 / 51.5 | 1046 / 1019 |
| cfgE: cfgB + int8 draft heads and MTP blocks (`STEP5_SKINNY=router,gproj,draft8,mtp8`) + `ST_EXL3_PREP=1` | 28.24 (2.467) | 27.97 | 37.5 / 54.0 | 1076 / 1019 |
| cfgT: cfgE + `ST_EXL3_TF=1` (TensorFold's grouped EXL3 experts below 64 tokens) | 41.13 (2.472) | 37.82 | 53.2 / 73.8 | 1073 / 1013 |

- cfgT's step is about 60 ms (acceptance / tok/s) against cfgE's 87.4 ms; prefill is unchanged (chunks of 64 tokens or
  more keep st_moe_ext). Ready after 625 s; MemAvailable head 5.4 GiB, worker 7.1 GiB.
- The kernel (`patch/step5_spark/tf_exl3/`, provenance and changes in its `PROVENANCE.md`): one set of three launches
  covers every bit width of a layer, the SwiGLU and the routing-weighted combine run inside, 206-229 GB/s from DRAM. On
  four real layers (`patch/tests/tf_moe_check.py`) it is 2.1-2.8x faster than the exl3_moe buckets at 1 token and
  1.5-2.0x at 3 tokens, and its error against float64 is about a third of theirs (rms 3.4e-4 to 4.3e-4 against
  9.1e-4 to 9.6e-4).
- cfgT's launch: `exp_cfg.sh cfgT --env STEP5_SKINNY=router,gproj,draft8,mtp8 --env ST_EXL3_PREP=1 --env ST_EXL3_TF=1`
  with `--mount` of `patch/step5_spark/{exl3_quant.py,moe_prep.py,tf_exl3}` onto the image's `step5_spark` package.
- KL against the teacher (`serve/kld_capture.py` on the head, `patch/kld_eval.py` in the image): every 6th panel
  sequence (29 sequences, every domain, 65,498 scored positions), captured from two instances without MTP or prefix
  caching in 48-token prefill chunks, so every MoE call takes the decode-path kernel. A = the exl3_moe buckets (cfgE's
  kernel at up to 48 rows; its rows alone, inside 12 and inside 48 differ by about 1e-7 relative in
  `tf_moe_check.py`, so A stands for cfgE's decode), B = TF.

  | run | KL mean | KL p99 | top-1 | NLL (teacher 0.4907 here) |
  |---|---|---|---|---|
  | A exl3_moe | 0.01945 | 0.2321 | 0.9570 | 0.5064 |
  | B TF | 0.01964 | 0.2316 | 0.9569 | 0.5061 |

  B - A = +0.00019 (+1.0 % of A), standard error 0.00014 with the sequences as clusters (1.4 SE): no measurable
  difference; B is lower in 13 of 29 sequences; the two argmaxes agree at 98.65 % of positions. The layer-streamed
  reference's b566 KL (0.02047, `notes/panel.md`) covers the whole panel, so it is not the same set of positions.
  Results: head `~/step5/runs/patch/kld_pair/kld_pair.json`.
- Capture layout, checked by `kld_capture.py check` on every sequence: each engine step writes a one-row sampling call
  and then the step's prompt-logprob rows. The sampling row agrees with the block's last row within fp16 tolerance, not
  bit for bit (a one-row lm_head call rounds differently). Both ranks write captures (the logits are all-gathered);
  the head's copy is used.
- Greedy texts: cfgT twice 6/6 identical; cfgE against cfgT 1/6 (they split at near-tie wording). cfgE's own
  run-to-run repeatability was not measured, so the 6/6 says nothing about the old kernel.
- Capture data on disk, about 106 GB: head `~/step5/runs/patch/kcap{A,B}` (raw, 18 GB each) and
  `kcap{A,B}_packed` (17 GB each, root-owned), worker `~/.step5-sparks/cache/kcap{A,B}` (the worker's duplicate raw
  captures, 18 GB each).
- In production since 2026-10-19 02:37 (`spark` profile `step5`, watcher back on). `serve/patch/apply_cfgt_launch2.py`
  added cfgT's settings to `serve/launch2.sh` for both ranks: the env `NCCL_IB_HCA=rocep1s0f1,roceP2p1s0f1` (was a
  per-rank argument), `ST_EXL3_ST_CHUNK_TOKENS=4096`, `STEP5_SKINNY=router,gproj,draft8,mtp8`, `ST_EXL3_PREP=1`,
  `ST_EXL3_TF=1`, and read-only mounts of `~/step5/serve/patch/step5_spark/{skinny.py,step5.py,step5_mtp.py,
  exl3_quant.py,moe_prep.py,tf_exl3}` over `/usr/local/lib/python3.12/dist-packages/step5_spark/`. Head copy md5
  ddd8f9ff -> 21e37fff (the Mac copy is identical), backup `launch2.sh.bak-before-cfgt`.
  - cfgT2 (exp_launch) ran 10 h before the switch with no traceback, CUDA error or out-of-memory line in either
    rank's log.
  - `spark switch step5 --fg` (`runs/serve/switch-cfgt.log`): result ok after 612 s (start 589 s); MemAvailable head
    6.7 GiB, worker 8.0 GiB. `docker inspect` on both ranks: the five keys set, the six mounts read-only, every
    source an existing file or directory.
  - `serve/smoke.py --long 20000`: chat, tool call (`get_weather`, Paris) and reasoning OK. Its long request stops at
    32 tokens inside the reasoning; with 1024 tokens the 25,526- and 80,526-token prompts get the right answer
    (2499, 7499; the second in 97.8 s).
  - decode set b: median 42.13 tok/s, acceptance 2.468 (`runs/serve/decode-prod-cfgt.json`).
  - launch2.sh's fabric check still prints the per-rank HCA it validates (`HCAs rocep1s0f1: ok`); the containers run
    with the fixed two-HCA `NCCL_IB_HCA` above.
- `launch2.sh` does not copy the patch files to the worker: after changing a file under `serve/patch/step5_spark/`,
  copy it to the same path on the worker before a restart (`exp_launch.py start` does this and checks md5).
