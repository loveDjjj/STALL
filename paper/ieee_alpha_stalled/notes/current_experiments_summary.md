# 当前实验总览

本文件把 `results/` 中已经完成的论文实验和审计资产映射到当前稿件。主文不必逐项展开所有数字，但投稿前应保证每个核心结论都能追溯到这里列出的证据文件。

## 主线结果

| 实验 | 证据文件 | 关键结果 | 主文用途 |
|---|---|---|---|
| Alpha-STALLED 三数据集主结果 | `results/paper_tables/alpha_stalled_main_summary.md` | ComGenVid 0.9198/0.9211；VideoFeedback 0.8628/0.8750；GenVideo 0.8374/0.8283 | 主表 |
| Global/Patch/Fusion 消融 | `results/paper_tables/ablation_summary.md` | 三数据集均有 global-only、patch-only、fusion 对照 | 证明全局与局部证据互补 |
| ComGenVid 局部时序定义消融 | `results/paper_tables/ablation_summary.md` | 同网格二阶 0.9273/0.9309，高于 spatial/lag-1/multi-lag/motion variants | 支撑局部二阶设计 |
| 生成器宏平均 paired bootstrap | `results/journal_experiments/macro_average_bootstrap/macro_average_paired_bootstrap_delta.md` | Alpha-STALLED 相对 global-only 的三数据集 $\Delta$AUC 95% CI 均大于 0 | 支撑稳定性 |

## 机制和敏感性实验

| 实验 | 证据文件 | 关键结果 | 写作边界 |
|---|---|---|---|
| Temporal derivative order | `results/journal_experiments/temporal_derivative_order/comgenvid_temporal_derivative_order.md` | D=2 优于 D=3/D=4 | 代表性机制对照，只在 ComGenVid |
| Region sensitivity | `results/journal_experiments/region_sensitivity/region_sensitivity_summary.md` | ComGenVid region=3 更好；GenVideo region=2；VideoFeedback region=1 | 局部空间支持域存在数据集边界 |
| Aggregation sensitivity | `results/journal_experiments/aggregation_sensitivity/aggregation_sensitivity_summary.md` | GenVideo/VideoFeedback mean 优于 bottom-k | aggregation 不是普适优劣 |
| Bottom-k sensitivity | `results/journal_experiments/bottomk_sensitivity/bottomk_sensitivity_summary.md` | ComGenVid bottom-k 比例增大仍保持收益 | 用作 ComGenVid 局部异常分布分析 |
| Cross-dataset frozen hyperparameters | `results/journal_experiments/cross_dataset_frozen_hyperparams/cross_dataset_frozen_hyperparams.md` | beta/aggregation gap 小，region gap 大 | 区分 oracle sweep 与冻结泛化 |

## 协议、覆盖和成本

| 实验/审计 | 证据文件 | 当前结论 | 写作边界 |
|---|---|---|---|
| D3 protocol audit | `results/journal_experiments/d3_protocol_audit/d3_protocol_audit.md` | 已审计类别平衡、FPS、encoder、AP 方向 | 没有重跑 D3，不能写主表对照 |
| Patch coverage gaps | `results/paper_tables/patch_coverage_gaps.md` | VideoFeedback Hotshot-XL、GenVideo HotShot-XL/MoonValley 不在当前 patch release baseline | 主表脚注或补充材料说明 |
| Duration/window representative run | `results/journal_experiments/duration_window_representative/comgenvid_duration_window_representative.md` | ComGenVid 1s patch-only 0.9434/0.9456 | 不报告 1s fusion |
| Runtime/storage audit | `results/journal_experiments/runtime_storage_audit/runtime_storage_audit.md` | patch cache 合计约 713.22 GiB，global cache 约 8.23 GiB | cache 体积不等于提交体积 |
| Video-stage runtime benchmark | `results/journal_experiments/video_stage_runtime_benchmark/video_stage_runtime_benchmark.md` | 小样本三阶段合计 20.083 s | 只代表阶段成本，不外推全量耗时 |

## 解释性和失败模式

| 实验 | 证据文件 | 当前结论 | 写作边界 |
|---|---|---|---|
| Failure case audit | `results/journal_experiments/failure_case_audit/failure_case_audit.md` | VideoFeedback 存在稳定负迁移生成器；真实视频低分案例可作为边界 | 不观看原视频前不写具体语义原因 |
| Keyframe case explanations | `results/journal_experiments/keyframe_case_explanations/keyframe_case_explanations.md` | 已有 6 个关键帧解释案例 | 事后解释，不参与训练或推理 |
| Patch anomaly visualization | `results/journal_experiments/case_visualizations/patch_anomaly_case_summary.md` | 已有 patch anomaly map 资产 | 主文空间有限时放补充 |

## 尚未完成但可执行的高成本实验

- 同协议外部 baseline：D3-DINOv3、D3-X-CLIP16、AEROBLADE、RIGID、ZED、T2VE、AIGVDet。
- Calibration source/size：不同真实校准源或校准集大小下重建 patch/global 参数。
- Backbone ablation：MobileNet、ResNet、ViCLIP、VideoMAE 等需要重新提取特征。
- Image/temporal perturbation：JPEG、blur、crop、noise、reverse、shuffle、flash 等需要重新处理视频或帧并重建 cache。
- Patch 二阶正态性检验：可从 cache 抽样做 AD/DP 检验，但目前只作为低优先级理论补充。
