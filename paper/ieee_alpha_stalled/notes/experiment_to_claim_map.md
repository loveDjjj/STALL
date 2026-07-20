# 实验到论文论点映射

| 论文论点 | 已有证据 | 可写强度 | 注意边界 |
|---|---|---|---|
| Alpha-STALLED 在三个 benchmark 上相对 global-only 有稳定提升 | `results/paper_tables/ablation_summary.md`; `results/journal_experiments/macro_average_bootstrap/macro_average_paired_bootstrap_delta.md` | 可写为“在当前 release 协议下三数据集宏平均提升为正” | 不写成相对所有 SOTA baseline 均领先 |
| 局部二阶时序是 patch 分支主要贡献 | `results/paper_tables/ablation_summary.md`; `results/journal_experiments/temporal_derivative_order/comgenvid_temporal_derivative_order.md` | 可写为“ComGenVid 代表性消融支持” | D=3/D=4 只在 ComGenVid 实跑，不能写成三数据集定理 |
| Patch 分支与全局分支互补 | 三数据集 global-only / patch-only / fusion 消融 | 可写为“互补性随数据集变化” | ComGenVid patch-only 高于 fusion，应避免绝对化“融合总是最好” |
| Region/aggregation 是局部异常尺度边界 | `region_sensitivity_summary.md`; `aggregation_sensitivity_summary.md`; `cross_dataset_frozen_hyperparams.md` | 可写为“局部空间支持域受数据集影响” | 不把某个 region 写成普适最优 |
| 方法不使用生成样本训练 | 方法定义与 release 脚本 | 可写 | 需要明确 patch 校准真实来源，避免和严格外部校准混淆 |
| D3 对照需要重跑才能公平比较 | `d3_protocol_audit.md` | 可写为 protocol audit | 不能写 D3 数值对比表，除非后续同协议实跑 |
| 关键帧能解释方法边界 | `keyframe_case_explanations.md` 与图 | 可写为“事后解释” | 不能声称自动图证明具体语义原因，人工复核前只能描述分数关系 |
