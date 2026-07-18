# 复现入口

这里记录 Alpha-STALLED 清理后的公开复现命令。命令假设数据集索引、全局
score CSV、patch score CSV 和校准文件都位于 `configs/alpha_stalled.yaml`
记录的路径。

## 1. 原版 STALL 全局分数

```bash
conda run --no-capture-output -n stall python src/eval.py \
  --csv cache/indexes/<dataset>.csv \
  --emb-cache cache/embeddings/<dataset> \
  --params precomputed/stall_params_vatex_dino_v3.npz \
  --duration 2 \
  --compact \
  --output-csv results/paper_scores/<dataset>_global.csv
```

## 2. Patch 同网格二阶时序分数

```bash
conda run --no-capture-output -n stall python src/eval_patch_fast.py \
  --csv cache/indexes/<dataset>.csv \
  --patch-emb-cache cache/patch_embeddings/<dataset> \
  --patch-params <configs/alpha_stalled.yaml 中的数据集 patch 参数> \
  --patch-temp-mode same_grid_second_order \
  --patch-spat-weight 0.10 \
  --patch-temp-weight 0.90 \
  --output-csv results/paper_scores/<dataset>_patch_second_order.csv
```

## 3. Alpha-STALLED 融合和指标

```bash
conda run --no-capture-output -n stall python tools/eval_alpha_stalled.py \
  --global-csv results/paper_scores/<dataset>_global.csv \
  --patch-csv results/paper_scores/<dataset>_patch_second_order.csv \
  --alpha 0.60 \
  --output-csv results/paper_scores/<dataset>_alpha_stalled.csv \
  --metrics-csv results/paper_tables/<dataset>_alpha_stalled_metrics.csv
```

## 4. Alpha sweep

```bash
conda run --no-capture-output -n stall python tools/fuse_scores.py \
  --dataset <dataset> \
  --tag same_grid_second_order \
  --global-csv results/paper_scores/<dataset>_global.csv \
  --patch-csv results/paper_scores/<dataset>_patch_second_order.csv \
  --patch-score-col patch_final_score \
  --alphas 0:1:0.05 \
  --output-dir results/paper_sweeps/<dataset>_alpha
```

## 5. 已有 score CSV 的指标计算

组件消融和局部时序消融已经有逐视频 score CSV 时，使用统一指标入口：

```bash
conda run --no-capture-output -n stall python tools/eval_score_csv.py \
  --csv results/paper_scores/<dataset>_patch_second_order.csv \
  --score-col patch_final_score \
  --output-csv results/paper_tables/<dataset>_patch_only_metrics.csv
```

完整消融命令见 `scripts/ablations/README.md`。

如需从已纳入 release 的 `results/paper_scores/` CSV 一次性重建所有论文表格
和 sweep 汇总，运行：

```bash
bash scripts/reproduce/rebuild_paper_assets.sh
```
