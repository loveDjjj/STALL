# 图表强化完成审计

日期：2026-07-23

本审计按用户提出的图表强化目标逐项核对当前论文状态。核心原则是：每个机制主张都需要对应表格、图像或明确的边界说明；没有实验支撑的内容不伪造数值。

## 目标到证据映射

| 用户目标 | 当前证据 | 完成判断 |
|---|---|---|
| V 节图表太少，需要增加大表格和多种绘图形式 | `sections/05_ablation_analysis.tex` 现在包含 12 个表格输入和 8 个图/图表组合：`temporal_order`、`patch_temporal_comprehensive`、`patch_likelihood_assumption_audit`、`local_d1_d2_ablation`、`u0_region_aggregation_sensitivity`、`hyperparameter_sensitivity_matrix`、`per_generator_component_matrix`、`per_generator_bootstrap_delta`、`u0_failed_directions`、`u0_oas_candidate`、`u0_injection`、`u0_compute_cost`；图包括 `global_patch_likelihood_joint_panel`、`component_per_generator_matrix`、`real_calib_size_curves`、`alpha_gain_summary`、`patch_temporal_design_landscape`、`patch_likelihood_assumption_audit`、`hyperparameter_sensitivity_grid`、`keyframe_patch_anomaly_cases_main` | 已完成 |
| D=1 缺失，temporal derivative 对比不全面 | `tables/patch_temporal_comprehensive.tex` 已纳入 D=1 lag-1、D=1 multi-lag、D=2、D=3、D=4，以及 spatial-only、motion-hard/soft；`tables/temporal_order.tex` 也补入 D=1 | 已完成 |
| 类似原文 Figure 1 的 spatio-temporal likelihood 分析，但突出本文创新 | `figures/results/global_patch_likelihood_joint_panel.pdf/.png/.svg` 使用 global STALL 分数为横轴、本文新增 patch second-order 分数为纵轴，三数据集 real/generated 散点展示全局--局部联合结构 | 已完成 |
| Fig. 4 temporal order 原来数据单薄 | `tables/patch_temporal_comprehensive.tex` + `figures/results/patch_temporal_design_landscape.pdf` 将 spatial、一阶、多滞后、扰动、高阶和二阶放在同一协议下比较 | 已完成 |
| Fig. 5 frozen hyperparameter 原来形式单一 | `tables/hyperparameter_sensitivity_matrix.tex` + `figures/results/hyperparameter_sensitivity_grid.pdf` 统一展示 region、aggregation、bottom-k 和 leave-one-dataset-out oracle gap | 已完成 |
| 真实校准数量和 alpha 微调需要进入论文证据链 | `tables/real_calib_alpha_summary.tex` + `tables/validation_fusion_loso.tex` + `figures/results/real_calib_size_curves.pdf` + `figures/results/alpha_gain_summary.pdf` 已接入 V 节，展示真实校准数量曲线、固定 alpha、best/oracle alpha、alpha-only LOSO 和 three-branch LOSO 的部署边界 | 已完成 |
| Fig. 6 关键帧解释案例只有图，数据不足 | `tables/keyframe_case_summary.tex` 补充 6 个关键帧案例的 global、patch、Alpha 分数、Patch-Global 差值和关键帧时间 | 已完成 |
| 参考多数据权威表格形式 | 新增 `per_generator_component_matrix.tex` 和 `per_generator_bootstrap_delta.tex`，按 20 个生成器展开 AUC/AP、delta 和 bootstrap CI；`hyperparameter_sensitivity_matrix.tex` 按多数据集、多设置组织 | 已完成 |
| 全局分支不同阶数可以考虑和分析是否需要 | `notes/global_derivative_order_gap_analysis.md` 和 V 节文字说明当前 release 没有 global order CLI；global D sweep 需要重定义全局时序特征、重建校准参数并三数据集重跑，当前不报告不存在的数值 | 已完成为“分析边界”，未伪造实验 |
| 缺数据或实验可以慢慢补，不断优化 | 已记录 P2/P3 后续项：global order sweep、外部 baseline、校准源/大小、backbone、扰动实验等；当前主文只使用已存在且可追溯的 CSV/图像证据 | 当前阶段完成，后续可扩展 |

## 当前 V 节图表资产

### 主文表格

- `tables/component_ablation.tex`
- `tables/per_generator_component_matrix.tex`
- `tables/per_generator_bootstrap_delta.tex`
- `tables/real_calib_alpha_summary.tex`
- `tables/validation_fusion_loso.tex`
- `tables/patch_temporal_comprehensive.tex`
- `tables/patch_likelihood_assumption_audit.tex`
- `tables/hyperparameter_sensitivity_matrix.tex`
- `tables/runtime_window_boundary.tex`
- `tables/keyframe_case_summary.tex`

### 主文图像

- `figures/results/global_patch_likelihood_joint_panel.pdf`
- `figures/results/component_per_generator_matrix.pdf`
- `figures/results/real_calib_size_curves.pdf`
- `figures/results/alpha_gain_summary.pdf`
- `figures/results/patch_temporal_design_landscape.pdf`
- `figures/results/patch_likelihood_assumption_audit.pdf`
- `figures/results/hyperparameter_sensitivity_grid.pdf`
- `figures/results/keyframe_patch_anomaly_cases_main.pdf`

## 质量边界

- 未在本机编译 LaTeX；本机没有 LaTeX 环境，Overleaf 建议使用 XeLaTeX。
- 已做静态检查：`\\input`、`\\includegraphics`、`\\label`/`\\ref` 均可解析，无缺失文件或重复标签。
- 全局 temporal derivative order 不作为已完成实验；这是新增实验方向，不是现有 CSV 汇总。
- 外部 baseline、backbone、扰动、校准集大小等仍是高成本后续实验，适合审稿明确要求时补充。
