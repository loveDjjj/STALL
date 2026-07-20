# 参考文献实验体系对齐审计

本审计以 `2603.15026v2` 的主文与附录实验为参照，检查当前 Alpha-STALLED 结果资产的覆盖程度。
它只检查已有文件，不重新提取特征、不运行模型。

## 总览

| priority | status | count |
|---|---|---:|
| P1 | partial_csv_runtime_complete_end_to_end_pending | 1 |
| P2 | complete_for_our_method | 1 |
| P2 | complete_without_raw_keyframes | 1 |
| P2 | partial | 1 |
| P2 | partial_but_sufficient | 1 |
| P2 | partial_dirty_file_needs_review | 1 |
| P3 | missing_but_cost_audited | 1 |
| P3 | missing_optional_theory | 1 |
| P3 | missing_requires_feature_extraction | 1 |
| P3 | missing_requires_video_or_cache_rebuild | 1 |
| P3 | missing_requires_video_or_frame_reprocessing | 1 |
| done | complete | 1 |
| done | complete_extra | 1 |
| done | complete_for_patch_branch | 1 |
| done | representative_run_complete | 1 |

## 逐项对照

| reference section | reference experiment | our counterpart | status | priority | evidence | next action |
|---|---|---|---|---|---|---|
| Table 1 / Sec. 5.2 | 三 benchmark 主结果与检测器对比 | 三数据集 global-only / patch-only / Alpha-STALLED 主表 | complete_for_our_method | P2 | results/paper_tables/alpha_stalled_main_summary.md; results/paper_tables/ablation_summary.md | 若投稿需要完整 SOTA baseline 表，再单独建立 baseline 环境重跑；否则不建议优先做。 |
| Fig. 6c / D.6.6 | calibration source ablation | 真实视频百分位校准假设与 cross-dataset frozen hyperparameter | partial | P2 | results/journal_experiments/cross_dataset_frozen_hyperparams/cross_dataset_frozen_hyperparams.md | 需要重新构造真实校准集并生成 patch params；计算成本高，建议只在审稿要求时做一个代表数据集。 |
| Fig. 7a / D.6.5 | calibration set size ablation | 当前只审计 cache/storage；未按校准集大小重建参数 | missing_but_cost_audited | P3 | results/journal_experiments/runtime_storage_audit/runtime_storage_audit.md | 需要从真实集抽样、多次重建 patch params；优先级低于 duration/window 代表实跑。 |
| Table 2 / D.6.4 | backbone encoder ablation | 当前 Alpha-STALLED 固定 DINOv3；无 MobileNet/ResNet/ViCLIP/VideoMAE 对照 | missing_requires_feature_extraction | P3 | results/paper_tables/alpha_stalled_main_summary.md | 不建议当前补；若要做，先选 ComGenVid 小规模 smoke，再决定是否铺开。 |
| Fig. 14 / D.2.1 | spatial-only / temporal-only / combined component ablation | global-only / patch-only / Alpha-STALLED；ComGenVid patch spatial 与多种 temporal 定义 | complete | done | results/paper_tables/ablation_summary.md; results/paper_scores/comgenvid_patch_spatial.csv; results/paper_scores/comgenvid_patch_second_order.csv | 写论文时强调 global STALL 与 patch 二阶时序的互补，而不是沿用原文 spatial/temporal 命名。 |
| Fig. 15 / D.2.2 | frame-level aggregation min/mean/max ablation | patch region size、mean vs bottom-k aggregation、bottom-k ratio | complete_for_patch_branch | done | results/journal_experiments/region_sensitivity/region_sensitivity_summary.md; results/journal_experiments/aggregation_sensitivity/aggregation_sensitivity_summary.md; results/journal_experiments/bottomk_sensitivity/bottomk_sensitivity_summary.md | 不建议补原文 min/max；当前 region/aggregation 结果更贴合我们的创新点。 |
| Fig. 13 / D.1 | temporal derivative order D=1/2/3/4 | ComGenVid lag-1、multi-lag、motion-hard/soft、同网格二阶时序消融 | partial_but_sufficient | P2 | results/paper_scores/comgenvid_patch_lag1.csv; results/paper_scores/comgenvid_patch_multilag.csv; results/paper_scores/comgenvid_patch_second_order_ablation.csv | 不优先补；如需更像原文，可只在 ComGenVid 增加 D=3/4 patch temporal 对照。 |
| Fig. 8 / D.6.1-D.6.3 | step size / FPS / video length ablation | duration/window feasibility audit + ComGenVid 1s representative run | representative_run_complete | done | results/journal_experiments/duration_window_feasibility/duration_window_feasibility.md; results/journal_experiments/duration_window_representative/comgenvid_duration_window_representative.md | 当前足以作为短窗口边界证据；除非审稿要求，不建议直接铺开 3 数据集 × 4 时长。 |
| Fig. 7b / D.5 | image perturbation robustness | 当前无 JPEG/blur/crop/noise 扰动结果 | missing_requires_video_or_frame_reprocessing | P3 | no_local_evidence_required_paths | 不建议当前补；若审稿要求，先做 GenVideo 小子集和 global-only/Alpha-STALLED 对照。 |
| D.4 | temporal perturbation robustness | 当前无 reverse/shuffle/flash 扰动实验 | missing_requires_video_or_cache_rebuild | P3 | no_local_evidence_required_paths | 不建议当前补；可在未来用 400 个真实视频做小规模机制验证。 |
| B | normality / Gaussian assumption tests | 当前没有针对 patch 二阶差分的 AD/DP 正态性检验 | missing_optional_theory | P3 | no_local_evidence_required_paths | 可作为低优先级理论补充；优先先写明继承 STALL 全局校准假设，patch 分支为经验增强。 |
| A.4 | D3 baseline protocol audit | 当前已有 release 复现/结果说明，但未系统审计 D3 protocol | partial_dirty_file_needs_review | P2 | results/stall_repro_comparison.md | 提交前单独审查 `results/stall_repro_comparison.md` diff；不要混入当前补充实验提交。 |
| E | inference time / memory analysis | runtime/storage audit + CSV-stage runtime benchmark | partial_csv_runtime_complete_end_to_end_pending | P1 | results/journal_experiments/runtime_storage_audit/runtime_storage_audit.md; results/journal_experiments/runtime_benchmark/csv_stage_runtime_benchmark.md | 若论文需要完整 runtime 表，固定 GPU/batch/cache 状态重跑视频级阶段；当前 CSV-stage benchmark 可直接报告为轻量后处理成本。 |
| D.7 / Fig. 19-20 | qualitative examples | patch anomaly case visualization + failure case audit | complete_without_raw_keyframes | P2 | results/journal_experiments/case_visualizations/patch_anomaly_cases.svg; results/journal_experiments/failure_case_audit/failure_case_audit.md | 版面允许时补关键帧；当前统计和 patch map 已足够支撑 failure-mode 选择。 |
| Statistical reporting | 显著性/稳定性分析 | 逐生成器 paired bootstrap + 宏平均 paired bootstrap | complete_extra | done | results/paper_sensitivity/paired_bootstrap_delta_summary.csv; results/journal_experiments/macro_average_bootstrap/macro_average_paired_bootstrap_delta.md | 写作时区分宏平均总体提升与 VideoFeedback 内部负迁移。 |

## 当前建议

- `done` 项已经足够支撑当前手稿，不建议重复。
- `P1` 项是若继续投入最值得做的内容：当前主要剩视频级端到端 runtime benchmark。
- `P2` 项只在版面、审稿或对比需求明确时做：外部 baseline、关键帧可视化、D3 protocol 审查。
- `P3` 项需要重新提特征或重建大量 cache，当前不建议作为下一步默认任务。

## 尚未完成但可执行的缺口

| priority | reference experiment | gap | concrete next action |
|---|---|---|---|
| P1 | inference time / memory analysis | 已有 storage footprint、历史日志片段和可复现 CSV-stage 计时；仍缺原始视频解码、DINOv3 embedding、patch prefill、全量 patch eval 的固定环境端到端 benchmark。 | 若论文需要完整 runtime 表，固定 GPU/batch/cache 状态重跑视频级阶段；当前 CSV-stage benchmark 可直接报告为轻量后处理成本。 |
| P2 | D3 baseline protocol audit | 该文件当前有未提交修改，且 D3 protocol audit 不是 Alpha-STALLED 主创新所必需。 | 提交前单独审查 `results/stall_repro_comparison.md` diff；不要混入当前补充实验提交。 |
| P2 | qualitative examples | 已有 patch anomaly map；若要更直观，需要再叠加原视频关键帧。 | 版面允许时补关键帧；当前统计和 patch map 已足够支撑 failure-mode 选择。 |
| P2 | temporal derivative order D=1/2/3/4 | 未完全复刻 D=1/2/3/4 有限差分；但已证明我们的同网格二阶 patch 证据优于多种局部时序替代。 | 不优先补；如需更像原文，可只在 ComGenVid 增加 D=3/4 patch temporal 对照。 |
| P2 | calibration source ablation | 还没有用 VATEX/MSR-VTT/Panda70M/DiDeMo 等不同真实校准源重建 Alpha-STALLED patch params。 | 需要重新构造真实校准集并生成 patch params；计算成本高，建议只在审稿要求时做一个代表数据集。 |
| P2 | 三 benchmark 主结果与检测器对比 | 尚未重跑 AEROBLADE/RIGID/ZED/D3/T2VE/AIGVDet 等外部 baseline；当前论文可重点和 STALL global-only / patch branch 比较。 | 若投稿需要完整 SOTA baseline 表，再单独建立 baseline 环境重跑；否则不建议优先做。 |
| P3 | normality / Gaussian assumption tests | 可从 patch embedding cache 抽样做正态性检验，但 cache 体积大，且理论上我们主要继承原文 DINOv3 likelihood 假设。 | 可作为低优先级理论补充；优先先写明继承 STALL 全局校准假设，patch 分支为经验增强。 |
| P3 | temporal perturbation robustness | 需要重排/插帧并重新提特征；patch cache 不能直接表达扰动后视频。 | 不建议当前补；可在未来用 400 个真实视频做小规模机制验证。 |
| P3 | calibration set size ablation | 缺少 1k/5k/10k/... real calibration size 下的 patch/global 校准重建和评测。 | 需要从真实集抽样、多次重建 patch params；优先级低于 duration/window 代表实跑。 |
| P3 | image perturbation robustness | 需要对视频帧施加扰动并重新提 DINOv3/patch 特征；不属于现有 CSV 可推导。 | 不建议当前补；若审稿要求，先做 GenVideo 小子集和 global-only/Alpha-STALLED 对照。 |
| P3 | backbone encoder ablation | 需要为其他 backbone 重新提 global/patch embedding 与校准参数；不是 CSV 级分析。 | 不建议当前补；若要做，先选 ComGenVid 小规模 smoke，再决定是否铺开。 |
