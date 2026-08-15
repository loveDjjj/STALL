# 消融复现说明

当前正式消融必须先在 `reports/u0_experiment_registry.csv` 和
`docs/RESULT_STATUS.md` 中确认协议角色。U0 的 K1/K3、Global/Local、D1/D2、
calibration 和 bootstrap 分析使用对应的 `tools/analyze_u0_*.py` 或报告中记录的
专用分析器；不能把 legacy K1 CSV 与 locked K3 分数直接拼接。

下面的 `eval_score_csv.py` 入口只适用于已经具有相同数据身份、label方向和协议
定义的单个 score CSV。它负责指标计算，不负责证明两个 CSV 是公平对照：

```bash
conda run --no-capture-output -n stall python tools/eval_score_csv.py \
  --csv <score-file.csv> \
  --score-col <score-column> \
  --output-csv results/paper_tables/<name>_metrics.csv
```

除非实验显式声明相反方向，所有分数均解释为“越高越接近真实视频”。
AP 还必须显式注明 `AP_real` 或 `AP_fake`，结果汇总必须注明 `Macro-3` 或
`All-N`。

## 当前 U0 消融入口

| 实验 | 权威分析/报告 |
|---|---|
| Global/Local、K1/K3、alpha/beta | `tools/analyze_u0_core_ablation.py`、`reports/u0_core_ablation.md` |
| Local D1/D2 | `tools/analyze_second_order_independent_calibration.py`、`reports/second_order_and_independent_calibration.md` |
| calibration size/seed | `tools/analyze_u0_calibration_sensitivity.py`、`reports/u0_calibration_size_and_seed.md` |
| cross-dataset calibration | `tools/analyze_u0_cross_dataset_calibration.py`、`reports/u0_cross_dataset_calibration.md` |
| 23-source K1/K3 factorial | `tools/analyze_duration_aware_k1_factorial.py`、`reports/original_k1_full23_factorial.md` |

以下各节是 legacy K1 `paper_scores` 的复算说明。

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
