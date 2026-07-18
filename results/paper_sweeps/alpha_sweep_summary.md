# Alpha-STALLED 固定 alpha sweep 汇总

由 `tools/fuse_scores.py` 和 `src/metrics.py` pairwise balanced 指标生成。

| 数据集 | Best AUC alpha | Best AUC / AP | Best AP alpha | Best AP AUC / AP | 冻结 alpha=0.60 AUC / AP | Summary CSV |
|---|---:|---:|---:|---:|---:|---|
| comgenvid | 0.20 | 0.9296 / 0.9327 | 0.20 | 0.9296 / 0.9327 | 0.9198 / 0.9211 | `results/paper_sweeps/comgenvid_alpha/comgenvid_same_grid_second_order_summary.csv` |
| videofeedback | 0.80 | 0.8687 / 0.8784 | 0.75 | 0.8681 / 0.8786 | 0.8628 / 0.8750 | `results/paper_sweeps/videofeedback_alpha/videofeedback_same_grid_second_order_summary.csv` |
| genvideo | 0.65 | 0.8380 / 0.8281 | 0.60 | 0.8374 / 0.8283 | 0.8374 / 0.8283 | `results/paper_sweeps/genvideo_alpha/genvideo_same_grid_second_order_summary.csv` |

best-alpha 列是对已有 score 文件的诊断性 oracle sweep。除非另行定义验证协议，论文主线 release baseline 仍使用 `configs/alpha_stalled.yaml` 中记录的冻结 alpha。
