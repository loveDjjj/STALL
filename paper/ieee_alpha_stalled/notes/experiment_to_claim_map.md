# U0 实验到论文论点映射

日期：2026-07-25

| 论文论点 | 权威证据 | 可写强度 | 禁止外推 |
|---|---|---|---|
| U0 优于 Original STALL | `reports/u0_core_ablation.md`；AP +0.0295，CI [+0.0241,+0.0358] | 三开发 benchmark 的受控主比较 | 不写成优于所有检测器 |
| 全 23 生成器上优于原窗口 STALL | `reports/original_k1_full23_factorial.md`；全覆盖 All-23 AP_fake/AP_real +0.0383/+0.0288 | 同视频协议扩展审计 | 不与论文 AP 方向或不同样本行直接相减 |
| Local 是主要增益、K3 是次要独立增益 | 同上；平衡 cohort AP_fake +0.0377 与 +0.0085，CI 均为正 | 原窗口因子分解 | 不写 K3 逐生成器单调改善 |
| K3 覆盖优于同核心 K1 | `reports/u0_core_ablation.md`；AP +0.0087，CI [+0.0057,+0.0120] | 主要因果对照 | 不用历史 clean K1 归因 K |
| Global/Local 互补 | `reports/u0_core_ablation.md`；双向加入 AP CI 全正 | 两分支均有独立贡献 | 不写逐数据集/生成器总是改善 |
| D2 是 Local 主要有效项 | K1 PatchD2 AP 0.8214；完整 Local 0.8144 | 统一三数据集核心消融支持 | PatchSpatial 不得写成贡献来源 |
| lag/higher derivative 被排除 | `reports/local_residual_multiscale_audit.md` | 代表性历史机制对照 | 不写成三数据集统一定理 |
| K3 是保留的复杂度 | `reports/multi_window_joint_typicality.md`；K5/all/lower-tail 未过门槛 | K3 有正 CI 且成本 2.30x | 不称 K3 对所有数据集最优 |
| residual 不改善共同运动误判 | `reports/local_residual_multiscale_final.md`；AP -0.0088，CI 全负 | 明确负结果 | 不声称共同运动已被建模解决 |
| Joint 不优于固定融合 | `reports/multi_window_joint_typicality.md` | J1/J2/J3 均低于 0.6/0.4 | 不声称线性权重是普适最优参数 |
| 需要约 200 条目标域 real | `reports/u0_calibration_size_and_seed.md`、`reports/u0_cross_dataset_calibration.md` | 当前协议部署条件 | 不写 target-data-free |
| 新域生成器确认 | `reports/u0_locked_external_validation.md` | 对 ModelScope/Pika 的锁定确认 | 不外推到所有新生成器或无域校准 |
| 轻扰动变化有限、严重扰动退化 | `reports/u0_robustness.md`、`bootstrap_deltas.csv` | 报告效应量和区间 | 区间跨 0 不等于等效/鲁棒证明 |
| Local 是分布性证据 | `reports/u0_localization_and_injection.md` | 自然数据排序互补 | 不写通用语义定位、单调异常响应或因果解释 |
| release 可复现 | `reports/alpha_stalled_u0_final_package.md`、release validator | 数值和协议可复现 | 不等同于第三方跨硬件完整复现 |

## 主要统计层级

- 主要终点：Macro-3 real-positive AP。
- 主要比较：U0 vs Original STALL；U0 K3 vs Unified K1。
- 次要终点：AUC、fake-positive AP、单数据集、逐生成器。
- 覆盖率终点：全 45,185 fake 的固定 50/50 先验 AP；All-23 为 23 个生成器等权。
- 探索性分析：组件敏感性、失败候选、conflict subset、注入、扰动和分组诊断。
- 所有探索结果报告效应量与边界，但不作未经校正的全族显著性声明。

## 当前不可写的标题级结论

- State-of-the-art generated-video detection。
- 无需目标域真实视频即可泛化。
- Local map 能定位生成伪影的语义位置。
- 三窗口对每个数据集、生成器和时长组都提高性能。
- alpha=0.6 或 beta=0.1 是普适最优权重。
