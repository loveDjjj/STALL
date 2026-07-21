# Rich evidence upgrade

本轮优化按 algorithmic paper 的实验链条补强图表：主张必须绑定比较、消融或边界证据。

## 新增主文表格

- `tables/per_generator_component_matrix.tex`：20 个生成器级 global/patch/Alpha AUC/AP 和 delta。
- `tables/per_generator_bootstrap_delta.tex`：20 个生成器级 paired bootstrap AUC 差值区间。
- `tables/patch_temporal_comprehensive.tex`：ComGenVid patch spatial、D=1、multi-lag、motion-hard/soft、D=2/3/4 完整局部证据对照。
- `tables/hyperparameter_sensitivity_matrix.tex`：region、aggregation、bottom-k 多数据集敏感性大表。
- `tables/runtime_window_boundary.tex`：时间窗口、运行时间和存储成本边界。
- `tables/keyframe_case_summary.tex`：关键帧解释案例的分支分数、Patch-Global 差值和关键帧时间。

## 新增主文图

- `figures/results/global_patch_likelihood_joint_panel.*`：仿 STALL Fig. 1 的全局--局部联合分数散点图，但纵轴使用本文新增 patch second-order 分支。
- `figures/results/component_per_generator_matrix.*`：生成器级 AUC 热图和分支 delta 条形图。
- `figures/results/patch_temporal_design_landscape.*`：局部证据变体排序和相对 D=2 损失。
- `figures/results/hyperparameter_sensitivity_grid.*`：region、aggregation、bottom-k 和 frozen hyperparameter gap 的四面板图。

## 边界

- D=1 已由 `comgenvid_patch_lag1_metrics.csv` 纳入，不是新缺口。
- 全局不同阶数目前不是已有 release 的 CSV 级实验；已记录到 `notes/global_derivative_order_gap_analysis.md`，不在主文伪造结果。
