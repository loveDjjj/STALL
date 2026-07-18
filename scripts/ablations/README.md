# 消融复现说明

所有消融指标都应通过 `src/metrics.py` 重新计算，不再使用早期一次性探索脚本。
统一入口为：

```bash
conda run --no-capture-output -n stall python tools/eval_score_csv.py \
  --csv <score-file.csv> \
  --score-col <score-column> \
  --output-csv results/paper_tables/<name>_metrics.csv
```

除非实验显式声明相反方向，所有分数均解释为“越高越接近真实视频”。

## 1. Global/Patch 组件消融

全局分支：

```bash
conda run --no-capture-output -n stall python tools/eval_score_csv.py \
  --csv results/paper_scores/comgenvid_global.csv \
  --score-col final_score \
  --output-csv results/paper_tables/comgenvid_global_only_metrics.csv
```

Patch 分支：

```bash
conda run --no-capture-output -n stall python tools/eval_score_csv.py \
  --csv results/paper_scores/comgenvid_patch_second_order.csv \
  --score-col patch_final_score \
  --output-csv results/paper_tables/comgenvid_patch_only_metrics.csv
```

完整 Alpha-STALLED：

```bash
conda run --no-capture-output -n stall python tools/eval_score_csv.py \
  --csv results/paper_scores/comgenvid_alpha_stalled.csv \
  --score-col final_score \
  --output-csv results/paper_tables/comgenvid_alpha_stalled_metrics.csv
```

`videofeedback` 和 `genvideo` 使用同样模式替换数据集名前缀。

## 2. ComGenVid 局部时序定义消融

这些 score CSV 已作为 release 资产保存在 `results/paper_scores/`。重新计算
指标：

```bash
conda run --no-capture-output -n stall python tools/eval_score_csv.py \
  --csv results/paper_scores/comgenvid_patch_spatial.csv \
  --score-col patch_final_score \
  --output-csv results/paper_tables/comgenvid_patch_spatial_metrics.csv

conda run --no-capture-output -n stall python tools/eval_score_csv.py \
  --csv results/paper_scores/comgenvid_patch_lag1.csv \
  --score-col patch_final_score \
  --output-csv results/paper_tables/comgenvid_patch_lag1_metrics.csv

conda run --no-capture-output -n stall python tools/eval_score_csv.py \
  --csv results/paper_scores/comgenvid_patch_multilag.csv \
  --score-col patch_final_score \
  --output-csv results/paper_tables/comgenvid_patch_multilag_metrics.csv

conda run --no-capture-output -n stall python tools/eval_score_csv.py \
  --csv results/paper_scores/comgenvid_patch_motionhard.csv \
  --score-col patch_final_score \
  --output-csv results/paper_tables/comgenvid_patch_motionhard_metrics.csv

conda run --no-capture-output -n stall python tools/eval_score_csv.py \
  --csv results/paper_scores/comgenvid_patch_motionsoft.csv \
  --score-col patch_final_score \
  --output-csv results/paper_tables/comgenvid_patch_motionsoft_metrics.csv

conda run --no-capture-output -n stall python tools/eval_score_csv.py \
  --csv results/paper_scores/comgenvid_patch_second_order_ablation.csv \
  --score-col patch_final_score \
  --output-csv results/paper_tables/comgenvid_patch_second_order_metrics.csv
```

## 3. 固定 alpha sweep

`tools/fuse_scores.py` 会调用统一指标实现：

```bash
conda run --no-capture-output -n stall python tools/fuse_scores.py \
  --dataset comgenvid \
  --tag same_grid_second_order \
  --global-csv results/paper_scores/comgenvid_global.csv \
  --patch-csv results/paper_scores/comgenvid_patch_second_order.csv \
  --patch-score-col patch_final_score \
  --alphas 0:1:0.05 \
  --output-dir results/paper_sweeps/comgenvid_alpha
```

`videofeedback` 和 `genvideo` 使用各自的 `results/paper_scores/*` 输入重复运行。

## 4. 泄漏边界

Persistence、sample fallback、split-minus selector、source-aware gate 和
test-batch rank fusion 目前只作为诊断上限。除非重写为不依赖测试标签、生成器
身份和测试批次 rank 的推理路径，否则不能混入主消融表。
