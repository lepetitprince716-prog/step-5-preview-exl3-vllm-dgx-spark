# Panel fidelity (p6_kld.py), 2026-10-17, worker spark-worker

Teacher: official FP8 experts dequantized to BF16 through the same layer-streamed reference (`runs/p6/ref`, 397,006
scored positions, mean NLL 0.4684, 1871 s). Quantized runs: pass 1 experts (`runs/p1`, plan.json: 2.4 bpw routed
experts) + one p4 body K set each, 2330 s per run. Results in `runs/p6/<name>/kld.json` on the worker; one-line summary
`p6_summary.py`.

| body (attn/shared/mlp) | KL mean | KL p99 | top-1 | NLL (teacher 0.4684) | delta KL vs 4/4/4 |
|---|---|---|---|---|---|
| 4/4/4 | 0.02249 | 0.270 | 0.9543 | 0.4866 | - |
| 5/6/6 | 0.02047 | 0.252 | 0.9574 | 0.4852 | -9.0% |
| 6/8/8 | 0.02012 | 0.249 | 0.9577 | 0.4849 | -10.5% |

Per domain (KL / top-1), 5/6/6: code 0.016 / 0.962 (291,575 positions, dominates the mean), agentic 0.021 / 0.954,
code_debug 0.022 / 0.952, writing 0.020 / 0.967, math 0.011 / 0.967, knowledge 0.035 / 0.927, chat 0.041 / 0.932,
multilingual 0.043 (ru) to 0.120 (de) with 282-5,128 positions each (zh 0.059 / 0.930). The domain ordering is the
same for all three bodies.

Decision: body 5/6/6 (`~/models/step5-exl3-b566`). It takes 9.0 of the 10.5 % KL reduction that 6/8/8 reaches; the
last 1.7 % would cost the second +15 % of body decode bytes and about 0.55 GiB more per node (GPU check 2: 6/8/8 costs
about 1.1 GiB per node over 4/4/4).

Not measured here: the serving engine's own output (`--mode serving` with the plugin's capture, RUNBOOK 6) and the
uniform-K control plan (`plans/plan-uniform.json`, needs its own pass 1).
