#!/usr/bin/env bash
# 新论文主线：目标Global + Local D2，默认configs/paper.yaml，不是旧B3。
# 统一参数放在子命令前：--config PATH、--set KEY=VALUE。
# score --dataset NAME [--manifest CSV] [--limit N] [--output DIR] [--dry-run]
# replay --window-scores CSV [--output DIR] [--dry-run]：只重算已有raw。
# evaluate --run-dir DIR --pairs CSV [--dry-run]：严格固定配对，不重新抽样。
# predict --video PATH --dataset NAME [--indices-json JSON] [--output DIR] [--dry-run]
# bootstrap --run-dir 候选目录 --baseline-run 基线目录 --pairs CSV [--output DIR] [--dry-run]
# components --run-dir 完整主线分数 --pairs CSV [--output DIR]：五项固定组件移除。
# bootstrap读取components结果时，显式指定--variant与--baseline-variant。
# fit --dataset NAME [--stop-after gaussian|cdf] [--output DIR] [--resume] [--dry-run]
# --set fit.feature_source=video fit：从原视频提取Uniform K3/video256；默认cache复用已验证小资产。
# --set 'runtime.devices=[cuda:0,cuda:1]' score/fit：固定视频分片，每卡一进程，不拆单视频batch。
# export --run-dir 已完成fit目录 [--output DIR] [--dry-run]：不替换默认参考包。
# 示例：bash scripts/run_main.sh --set runtime.device=cuda:1 predict --video datasets/example.mp4 --dataset genvideo --dry-run
# score/replay/fit恢复必须同输入/配置/源码；bootstrap只比较已有冻结分数。
set -euo pipefail
cd "$(dirname "$0")/.."
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1
conda run --no-capture-output -n stall python scripts/run.py "$@"
