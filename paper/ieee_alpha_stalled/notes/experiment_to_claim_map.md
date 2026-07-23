# 实验到论文论点映射

| 论文论点 | 已有证据 | 可写强度 | 注意边界 |
|---|---|---|---|
| Alpha-STALLED 在三个 benchmark 上相对 global-only 有稳定提升 | `results/paper_tables/ablation_summary.md`; `results/journal_experiments/macro_average_bootstrap/macro_average_paired_bootstrap_delta.md` | 可写为“在当前 release 协议下三数据集宏平均提升为正” | 不写成相对所有 SOTA baseline 均领先 |
| 局部二阶时序是 patch 分支主要贡献 | `paper/ieee_alpha_stalled/tables/patch_temporal_comprehensive.tex`; `figures/results/patch_temporal_design_landscape.pdf`; `results/journal_experiments/temporal_derivative_order/comgenvid_temporal_derivative_order.md` | 可写为“ComGenVid 代表性消融支持 D=2 优于 spatial、D=1、多滞后、扰动和 D=3/D=4” | D=3/D=4 只在 ComGenVid 实跑，不能写成三数据集定理 |
| Patch 高斯似然假设在新增局部分支上是否合理 | `results/journal_experiments/patch_likelihood_assumption_audit/patch_likelihood_assumption_summary.csv`; `paper/ieee_alpha_stalled/tables/patch_likelihood_assumption_audit.tex`; `figures/results/patch_likelihood_assumption_audit.pdf` | 可写为“二阶 patch 差分在白化 covariance、方向余弦、AD/DP 和位置共享残差上最接近原 STALL 的高斯似然假设” | raw patch/region-pooled token 的坐标级正态性较弱；该实验是统计诊断，不是严格证明 |
| Patch 分支与全局分支互补 | `paper/ieee_alpha_stalled/tables/per_generator_component_matrix.tex`; `figures/results/global_patch_likelihood_joint_panel.pdf`; `figures/results/component_per_generator_matrix.pdf` | 可写为“互补性随数据集和生成器变化” | ComGenVid patch-only 高于 fusion，应避免绝对化“融合总是最好” |
| Region/aggregation 是局部异常尺度边界 | `paper/ieee_alpha_stalled/tables/hyperparameter_sensitivity_matrix.tex`; `figures/results/hyperparameter_sensitivity_grid.pdf`; `region_sensitivity_summary.md`; `aggregation_sensitivity_summary.md`; `cross_dataset_frozen_hyperparams.md` | 可写为“局部空间支持域受数据集影响” | 不把某个 region 写成普适最优 |
| Patch 分支依赖足够且有代表性的真实视频校准 | `results/research_summary/`; `paper/ieee_alpha_stalled/figures/results/real_calib_size_curves.pdf`; `paper/ieee_alpha_stalled/notes/real_calib_alpha_experiment_draft.md` | 可写为“25/50 real 通常不足，100/200/300 后多数数据集出现可用 patch 信号；收益不严格单调” | 同时强调来源代表性、质量噪声、aggregation 和 alpha 选择协议 |
| 固定 alpha 不是普适最优，可调 alpha 是部署适配方向 | `results/research_summary/`; `paper/ieee_alpha_stalled/tables/real_calib_alpha_summary.tex`; `figures/results/alpha_gain_summary.pdf` | 最优 alpha 明显随数据集变化；frozen alpha 是默认，validation-selected alpha 是扩展策略 | best/oracle alpha 不能当作部署协议 |
| AIGVDBench 外部失败可通过 aggregation 调整缓解 | `results/research_summary/experiment_metrics.csv` | region3 mean + fixed alpha=0.60 在三源协议高于 global-only | CogVideoX1.5 仍有 patch 负迁移，不能写成全面优于 global |
| 空间 patch 与时序 patch 分开融合的历史诊断 | `results/research_summary/`; `paper/ieee_alpha_stalled/tables/validation_fusion_loso.tex` | 外部集合有少量 AP 线索，原始集合相对 alpha-only 几乎无增益 | 不再作为默认第三分支；后续研究 local D2 residual |
| 全局不同导数阶数暂不作为主文结果 | `paper/ieee_alpha_stalled/notes/global_derivative_order_gap_analysis.md`; `src/create_params.py`; `src/eval.py`; `src/patch_matching.py` | 可写为“当前导数阶数消融限定于新增 patch 分支” | 不能报告 global D=2/D=3/D=4 数字；这需要重建全局校准和重跑 |
| 方法不使用生成样本训练 | 方法定义与 release 脚本 | 可写 | 需要明确 patch 校准真实来源，避免和严格外部校准混淆 |
| D3 对照需要重跑才能公平比较 | `d3_protocol_audit.md` | 可写为 protocol audit | 不能写 D3 数值对比表，除非后续同协议实跑 |
| 关键帧能解释方法边界 | `keyframe_case_explanations.md`; `paper/ieee_alpha_stalled/tables/keyframe_case_summary.tex`; 主文关键帧图 | 可写为“事后解释，并报告分支分数与关键帧时间” | 不能声称自动图证明具体语义原因，人工复核前只能描述分数关系 |
