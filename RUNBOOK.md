# Step 5 双 Spark 量化：执行手册

按顺序机械执行，每步一条命令，看到写明的完成标志再进下一步。所有命令在头节点 `spark-head` 上跑。路径：官方 FP8 `~/models/Step-5-Preview-FP8`，代码 `~/step5`，校准集 `~/step5/cal`，计划 `~/step5/plans`，运行目录 `~/step5/runs`，产物 `~/models/step5-exl3`。两台的这些路径必须相同。

容器约定（量化镜像 `dsv41-quant:dev`，单机 pass）：

```
RUNQ='docker run --rm --gpus all --user $(id -u):$(id -g) --entrypoint python3 -e HOME=/tmp -e STEP5_CUDA_MEM_FRACTION=0.70 -v $HOME/models/Step-5-Preview-FP8:/model:ro -v $HOME/step5:/q -v $HOME/step5/cal:/cal:ro -v $HOME/step5/plans:/plan -w /q'
```

## 0. 下载与校准数据（GPU 照常服务，可并行）

| 步骤 | 命令 | 完成标志 |
|---|---|---|
| 0.1 FP8 权重 | 已在跑：`MBPS=40 PARALLEL=4 nohup bash ~/step5/download.sh > ~/step5/download.log 2>&1 &` | `grep STEP5-DOWNLOAD-DONE ~/step5/download.log` |
| 0.2 校准对话 | 已在跑：`nohup bash ~/step5/caldata/run_generate.sh > ~/step5/cal/generate-all.log 2>&1 &`（base 后接 instruct，保险丝 30 USD） | 日志出现 `STAGE done`；`DONE ... err` 行里 err 为 0 |
| 0.3 token 集 | `docker run --rm --entrypoint python3 --memory 16g -v $HOME/models/Step-5-Preview-FP8:/model:ro -v $HOME/step5:/q -w /q dsv41-quant:dev caldata/build.py /model cal --conversations cal/base.jsonl cal/instruct.jsonl --exllamav3 /usr/local/lib/python3.12/dist-packages/exllamav3/conversion/standard_cal_data`（容器里装的就是 v1.6.0 的标准文本，wiki.utf8 2,093,275 字节） | 打印两个 sha256；`cal/cal.safetensors`、`cal/panel.safetensors`、两个 index 存在 |
| 0.4 fabric.env | 用户在头节点填 `~/step5/fabric.env`（照 `serve/fabric.env.example`：HEAD_IP、HEAD_IFNAME、HEAD_HCAS、WORKER_IPS、WORKER_IFNAMES、WORKER_HCAS；不进仓库） | 文件存在，权限 600 |

## 1. 维护窗口开始（先告诉用户）

| 步骤 | 命令 | 完成标志 |
|---|---|---|
| 1.1 停 v41tf | `~/spark-ctl/spark stop` | `spark status` 显示 `desired: none`，两台 MemAvailable > 110 GiB |
| 1.2 权重到副节点 | `rsync -a --info=progress2 ~/models/Step-5-Preview-FP8/ spark-worker:models/Step-5-Preview-FP8/`，完成后两台 `python3 ~/spark-ctl/pagecache.py evict ~/models/Step-5-Preview-FP8` | 副节点 `du -sh` 与头节点一致；`python3 ~/step5/dl_limited.py --help` 不需要，校验用 `cd ~/models/Step-5-Preview-FP8 && sha256sum -c <(awk '{print $1"  "$3}' ~/step5/Step-5-Preview-FP8.sha256.manifest)` 在副节点跑一次 |
| 1.3 代码到副节点 | `rsync -a --exclude runs --exclude cal/*.jsonl ~/step5/ spark-worker:step5/` | 两台 `sha256sum ~/step5/p3_pass1.py` 相同 |
| 1.4 服务镜像 | 两台 `docker pull ghcr.io/0xsero/step-5-preview-spark@sha256:4e28f849a414a483b44be50a09198d76cbc859096b354796dc25ee2159e05a5e` | `docker image inspect` 成功 |

## 2. GPU 预检（`serve/gpu_checks.sh`，一次一项，全部 PASS 才进 3）

| 项 | 命令 | 看什么 |
|---|---|---|
| 1 | `bash ~/step5/serve/gpu_checks.sh 1` | 服务镜像 exllamav3 版本（预期 1.5.3）、`exl3_moe` 签名 |
| 6 | `bash ~/step5/serve/gpu_checks.sh 6` | 改类名后分词 MATCH |
| 3 | `bash ~/step5/serve/gpu_checks.sh 3` | 23 个 `ssmax_s` 的值（预期常数） |
| 4 | `bash ~/step5/serve/gpu_checks.sh 4` | 单层对照：注意力、FP8 反量化、Triton 内核、一个专家的 K3 编码误差 |
| 5 | `bash ~/step5/serve/gpu_checks.sh 5` | 1.6.0 编码、1.5.3 重建逐位一致 |
| 2 | `nohup setsid bash ~/step5/serve/gpu_checks.sh 2 > ~/step5/runs/gpu_checks/check2.out 2>&1 < /dev/null &`（约 1 小时，三次启动） | 每台 MemAvailable、KV 池、解码 tok/s、桶数代价；据此把 `--bpw` 定在 2.4 或更低 |

## 3. 统计、灵敏度、分配（头节点一台）

两台都先 `bash ~/step5/guards.sh start`（每 5 s 清页缓存 + 内存看护）。

| 步骤 | 命令 | 完成标志 |
|---|---|---|
| 3.1 p0 | `eval $RUNQ -v $HOME/step5/runs/p0:/out dsv41-quant:dev p0_stats.py /model /cal/cal.safetensors /out/stats --index /cal/cal_index.json --stream-dir /out/stats/stream --keep-stream-layers 3,30,60,89`（约 1.5–2.5 h；四层快照各约 7 GB） | `runs/p0/stats/` 每层文件齐全（88 层）；`stream/` 有四层快照 |
| 3.2 p0b | `eval $RUNQ -v $HOME/step5/runs/p0:/out dsv41-quant:dev p0b_merge_stats.py /out/stats_w /out/stats`（默认域权重：panel 域 1、instruction 1.5、标准文本 0.5） | `runs/p0/stats_w/` |
| 3.3 p1 | 不依赖 p0，与 3.1 同时在副节点跑（副节点需先有 `dsv41-quant:dev`：`docker save dsv41-quant:dev \| ssh spark-worker docker load`）：`ssh spark-worker` 后 `eval $RUNQ -v $HOME/step5/runs/p1:/out dsv41-quant:dev p1_sensitivity.py /model /cal/cal.safetensors /out/alpha.json --seqs 16 --len 512 --sigma 0.1 --work /out/sens_work`（约 2–3 h），完成后 `rsync -a spark-worker:step5/runs/p1/alpha.json ~/step5/runs/p1/` | `alpha.json` 有 3–90 层，形状前重后轻 |
| 3.4 p1b | `eval $RUNQ -v $HOME/step5/runs/p0:/stats:ro -v $HOME/step5/runs/p1:/out dsv41-quant:dev p1b_relerr.py /model /cal/cal.safetensors /stats/stats/stream /out/rel_err.json --layers 3,30,60,89 --experts 64 --ks 2,3,4,5 --alpha /out/alpha.json --swap-kl-layer 30`（约 1–2 h） | `rel_err.json` 有四层段；swap 层的 KL 与 alpha 预测同量级 |
| 3.5 p2 | `eval $RUNQ -v $HOME/step5/runs/p0:/stats -v $HOME/step5/runs/p1:/p1:ro dsv41-quant:dev p2_allocate.py /stats/stats_w --alpha /p1/alpha.json --bpw 2.4 --decode-bpw 2.6 --rel-err /p1/rel_err.json --super-share 0.20 --super-k 5 --super-max --layer-guard 5,4 --floor-layers 88-90:3 --max-kinds 3 --attn-k 4 --shared-k 4 --mlp-k 4 -o /plan/plan.json`；再 `--score /plan/plan.json`；另出一份等字节均匀对照 plan（前 20 层 K3 其余 K2）`--score` 存档 | 每台字节数 ≤ 预检 2 给出的上限；每层 K 种类 ≤ 3；88–90 层 ≥ K3；`enforce_kinds` 前后预测代价差 < 1% |
| 3.6 计划到副节点 | `rsync -a ~/step5/plans/ spark-worker:step5/plans/` | 两台 sha256 相同 |

## 4. pass 1 两台编码（约 15 小时起）

`PLAN=$HOME/step5/plans/plan.json P1_EXTRA="--keep-stream-at 87" bash ~/step5/run_pass1_2box.sh`（估 17–25 h）。看 `~/step5/runs/p1/pass1.p0.*.stdout`（头）和副节点的 `pass1.p1.*.stdout`；每层一行。中断后同一条命令续跑。完成标志：两台 `runs/p1/` 下 `layer_03..layer_90` 的 part 文件齐全，`P1_EXIT=0`。2.4 还是 2.5 bpw 在阶段 3 之后、这一步之前定（body 6/8/8 相当于专家 0.05 bpw 的内存）。

## 5. 稠密、装配、面板

| 步骤 | 命令 | 完成标志 |
|---|---|---|
| 5.1 p4 | 头节点一条：`PLAN=$HOME/step5/plans/plan.json bash ~/step5/run_p4_2box.sh`（预检两台计划/代码/镜像/pass 1 清单一致，再以 1000:1000 起容器 `step5-p4`：头 `--part 0/2`，副 `--part 1/2`，`--variants 5/6/6,6/8/8`；stdout `runs/p4/dense.p{0,1}.stdout`，结尾 `P4_EXIT=`）；完成后 `rsync -a spark-worker:step5/runs/p4/ ~/step5/runs/p4/` | 两台 `P4_EXIT=0`；头上 `body_00..body_91` 齐全，`body_5-6-6/`、`body_6-8-8/` 同样齐全 |
| 5.2 p5 | `eval $RUNQ -v $HOME/step5/runs/p1:/pass1_out:ro -v $HOME/step5/runs/p4:/dense_out:ro -v $HOME/models:/models dsv41-quant:dev p5_assemble.py /model /pass1_out /dense_out /plan/plan.json /models/step5-exl3`；两个 body 变体各装一份（`/dense_out/body_5-6-6` → `/models/step5-exl3-b566`，`body_6-8-8` → `-b688`） | `verify.py --body-format exl3` 返回 0；`tokenizer_config.json` 的类是 `PreTrainedTokenizerFast`；`config.json` 无 `sparse_config` |
| 5.3 p6 | `--mode ref` 一次（teacher），`--mode quant --experts /pass1_out --body <三个 dense 目录各一次>` | 面板 KLD、top-1、NLL 按领域打印；判据 ΔKL 对 Δ解码字节（5/6/6 约 +15%，6/8/8 约 +30%）；记进 HANDOFF |
| 5.4 产物到副节点 | `rsync -a ~/models/step5-exl3/ spark-worker:models/step5-exl3/` | sha256 清单一致 |

## 6. 服务

`MODEL_DIR=$HOME/models/step5-exl3-b566 bash ~/step5/serve/launch2.sh start`（读 `~/step5/fabric.env`；起不来先 `ROCE=0`）。冒烟：`python3 serve/smoke.py --long 20000`（贪心、工具调用、思考输出、长提示各一条）；测速 `python3 serve/decode_bench.py --kit ~/dsv41-tensorfold/engine-tp2/tools/dsv41/kit_bench.py --set b --tokens 384 --reps 2`（kit_bench 自己的 decode 子命令数不到 `delta.reasoning` 块）；预填充 `kit_bench.py ... prefill --lengths 8192,32768,126976`。通过后 `spark-ctl` 的 `step5` profile（已加，2026-10-17）`spark switch step5`。262K 在当前内存下不可行（头节点剩 5 GiB，KV 要再加 3.7 GB），单请求上限 131072。结果：`notes/serving.md`。

## 回退

任何一步失败：停下，记录到 `notes/`，不要跳步。服务不成则 `~/spark-ctl/spark wake` 拉回 v41tf。
