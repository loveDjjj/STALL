# 图表库存

## 已放入论文目录

| 文件 | 来源 | 建议位置 | 作用 |
|---|---|---|---|
| `figures/method/alpha_stalled_pipeline.pdf` | `IEEE_Conference_Template/Alpha-STALLED 完整方法流程图.pdf` | 主文方法图 | 展示全局--局部流程 |
| `figures/results/macro_average_bootstrap_alpha_delta.pdf/.svg/.png` | `results/paper_figures/` | 主文结果 | 支撑宏平均提升稳定性 |
| `figures/results/per_generator_auc_delta_heatmap.pdf/.svg/.png` | `results/paper_figures/` | 主文或补充 | 展示逐生成器增益和负迁移 |
| `figures/results/global_patch_likelihood_joint_panel.pdf/.svg/.png` | `results/paper_scores/*_alpha_stalled.csv` | 主文消融 | 类似原 STALL Fig. 1，但展示 global score 与本文 patch second-order score 的联合分布 |
| `figures/results/component_per_generator_matrix.pdf/.svg/.png` | `results/paper_sensitivity/component_per_generator_delta.csv` | 主文消融 | 展示 20 个生成器上 global/patch/Alpha 的 AUC 矩阵和分支 delta |
| `figures/results/comgenvid_temporal_derivative_order.pdf/.svg/.png` | `results/journal_experiments/temporal_derivative_order/` | 主文消融 | 支撑二阶优于三阶/四阶 |
| `figures/results/patch_temporal_design_landscape.pdf/.svg/.png` | `results/paper_tables/` 与 `results/journal_experiments/temporal_derivative_order/` | 主文消融 | 把 spatial、D=1、multi-lag、motion、D=2/3/4 放在同一图中 |
| `figures/results/patch_likelihood_assumption_audit.pdf/.svg/.png` | `results/journal_experiments/patch_likelihood_assumption_audit/` | 主文机制诊断 | 验证 patch token、region-pooled token 和二阶 patch 差分的白化 covariance、正态性、方向余弦与位置共享残差 |
| `figures/results/cross_dataset_frozen_hyperparams.pdf/.svg/.png` | `results/paper_figures/` | 主文或补充 | 展示冻结超参数迁移 gap |
| `figures/results/hyperparameter_sensitivity_grid.pdf/.svg/.png` | `results/journal_experiments/*sensitivity/` 与 frozen hyperparams | 主文消融 | 四面板展示 region、aggregation、bottom-k 和 frozen oracle gap |
| `figures/results/real_calib_size_curves.pdf/.svg/.png` | `results/journal_experiments/real_calib_size_cross_dataset_summary/real_calib_size_cross_dataset_curve.csv` | 主文消融或补充 | 展示真实校准数量对 patch-only AUC 的影响，以及 best-alpha 融合相对 global-only 的 AUC 增益 |
| `figures/results/alpha_gain_summary.pdf/.svg/.png` | `results/journal_experiments/real_calib_size_cross_dataset_summary/real_calib_size_cross_dataset_best_rows.csv` | 主文消融或补充 | 对比冻结 alpha 与 best/oracle alpha 相对 global-only 的 AUC 增益，突出最优融合权重的数据集相关性 |
| `figures/results/keyframe_patch_anomaly_cases_main.pdf/.png` | 从完整 6 案例图裁剪 | 主文 | 展示两个代表性失败/边界案例，避免主文图过密 |
| `figures/results/keyframe_patch_anomaly_cases.pdf/.svg/.png` | `results/journal_experiments/keyframe_case_explanations/` | 备用/补充 | 完整 6 案例源图 |
| `figures/supplementary/keyframe_patch_anomaly_cases_full.pdf/.svg/.png` | `results/journal_experiments/keyframe_case_explanations/` | 补充 | 完整 6 案例图，适合补充材料 |
| `figures/supplementary/failure_case_audit_priority.pdf/.svg/.png` | `results/paper_figures/` | 补充 | 展示 P0/P1/P2 failure audit 分布 |

## 候选补充图

| 文件 | 来源 | 建议位置 | 作用 |
|---|---|---|---|
| AIGVDBench aggregation distribution | `results/research_summary/` 中保留结论，原始一次性图未纳入 release | 需要时重建 | 支撑 aggregation mismatch 解释 |

## 后续可能需要重画

- LaTeX 主文优先引用 PDF，SVG 作为可编辑源文件，PNG 作为 Overleaf 或投稿系统兼容备份。
- 若目标期刊要求单栏图清晰度，建议把 `per_generator_auc_delta_heatmap` 和 `cross_dataset_frozen_hyperparams` 重画为更高纵横比的双栏图。
- `keyframe_patch_anomaly_cases_main` 已裁剪为两个案例，主文可读性更好；完整图保留为补充资产。
- 方法图当前来自 PDF，Overleaf 可直接引用；若后续需要统一字体，建议重新导出矢量 PDF。

## 关键表格资产

| 文件 | 来源 | 建议位置 | 作用 |
|---|---|---|---|
| `tables/validation_fusion_loso.tex` | 冻结表格；结论级数据见 `results/research_summary/` | 主文消融或补充 | 对比 alpha-only LOSO、three-branch LOSO 与 global-only |
