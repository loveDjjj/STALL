# Alpha-STALLED 主实验汇总

由 refactor 分支上的以下命令生成：

```text
tools/eval_alpha_stalled.py
alpha = 0.60
metrics = src/metrics.py pairwise balanced AUC/AP
```

| 数据集 | 融合 score CSV | Metrics CSV | 平均 AUC | 平均 AP |
|---|---|---|---:|---:|
| ComGenVid | `results/paper_scores/comgenvid_alpha_stalled.csv` | `results/paper_tables/comgenvid_alpha_stalled_metrics.csv` | 0.9198 | 0.9211 |
| VideoFeedback | `results/paper_scores/videofeedback_alpha_stalled.csv` | `results/paper_tables/videofeedback_alpha_stalled_metrics.csv` | 0.8628 | 0.8750 |
| GenVideo | `results/paper_scores/genvideo_alpha_stalled.csv` | `results/paper_tables/genvideo_alpha_stalled_metrics.csv` | 0.8374 | 0.8283 |

这些数值属于 pre-U0 score-CSV release，不是当前 locked U0。
历史审计见 `research_archive/docs/pre_u0_release/reproducibility_audit.md`。
