# ComGenVid 1s 代表性 duration/window 实验

## 目的

参考 STALL 论文附录中对视频时长和采样窗口的稳定性分析，本实验在当前 Alpha-STALLED 公开资产上补充一个真实运行的 ComGenVid 1s 代表性窗口实验。实验只改变 patch 分支的时间窗口：主实验为 2s，本补充实验为 1s；patch 时序特征、区域聚合和 bottom-k 设置保持为主实验配置。

## 设置

- 数据集：ComGenVid
- 分支：patch-only；不报告 Alpha-STALLED 融合分数，因为当前资产没有匹配的 1s global score。
- 1s 设置：same-grid second-order temporal feature, region size 3, bottomk_mean, bottom-k ratio 0.20, patch spatial/temporal weight 0.10/0.90。
- 2s 对照：`results/paper_tables/comgenvid_patch_only_metrics.csv`。
- 1s cache prefill：misses_before=5094, written=5094。

## 结果

|   duration_sec |      auc |       ap |   delta_auc_vs_2s |   delta_ap_vs_2s | source                                                                                   |
|---------------:|---------:|---------:|------------------:|-----------------:|:-----------------------------------------------------------------------------------------|
|              1 | 0.943435 | 0.945611 |          0.016113 |         0.014702 | results/journal_experiments/duration_window_representative/comgenvid_1s_metrics_full.csv |
|              2 | 0.927322 | 0.930909 |          0.000000 |         0.000000 | results/paper_tables/comgenvid_patch_only_metrics.csv                                    |

## 解读

1s 窗口下 patch-only 性能与 2s 主实验保持在同一水平，说明当前局部二阶时序证据并不严格依赖较长采样窗口；这与参考 STALL 论文中短窗口下方法仍保持可用的分析方向一致。

## 文件

- 1s scores: `results/journal_experiments/duration_window_representative/comgenvid_1s_patch_full.csv`
- 1s metrics: `results/journal_experiments/duration_window_representative/comgenvid_1s_metrics_full.csv`
- comparison CSV: `results/journal_experiments/duration_window_representative/comgenvid_duration_window_comparison.csv`
