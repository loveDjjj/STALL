# `results/` 数据说明

本目录只保留 Alpha-STALLED release 需要的轻量论文资产。历史调参、fallback、
persistence、source/rank selector、debug run 等结果已经删除或归档，不属于当前
主线。

## 顶层文件

| 路径 | 说明 | 是否提交 |
|---|---|---|
| `README.md` | 本说明文件 | 是 |
| `alpha_stalled_project_manuscript_zh.md` | 中文项目手稿和方法/实验审计 | 是 |
| `alpha_stalled_full_pipeline_flow_zh.svg` | Alpha-STALLED 完整方法流程图 SVG 源文件 | 是 |
| `stall_repro_comparison.md` | 原版 STALL 复现对照说明；该文件历史上已修改，提交前需单独审查 | 视 diff 决定 |

## `paper_scores/`

逐视频分数 CSV。该目录是主实验和消融指标的直接输入。

| 文件模式 | 含义 |
|---|---|
| `<dataset>_global.csv` | 原版 STALL 全局分支逐视频分数 |
| `<dataset>_patch_second_order.csv` | patch 同网格二阶时序分支逐视频分数 |
| `<dataset>_alpha_stalled.csv` | global + patch 融合后的 Alpha-STALLED 逐视频分数 |
| `comgenvid_patch_spatial.csv` | ComGenVid 局部空间分支消融 |
| `comgenvid_patch_lag1.csv` | ComGenVid 同网格 lag-1 消融 |
| `comgenvid_patch_multilag.csv` | ComGenVid multi-lag 消融 |
| `comgenvid_patch_motionhard.csv` | ComGenVid motion-hard 消融 |
| `comgenvid_patch_motionsoft.csv` | ComGenVid motion-soft 消融 |
| `comgenvid_patch_second_order_ablation.csv` | ComGenVid 同网格二阶消融输入 |

核心列：

| 列名 | 说明 |
|---|---|
| `subset` | `real` 表示真实视频；`annotated` 表示生成视频 |
| `source_model` | 真实来源或生成器名称 |
| `filename` | 视频文件名 |
| `final_score` | 分数越高越接近真实视频 |
| `patch_final_score` | patch 分支分数，越高越接近真实视频 |
| `global_score`, `patch_score`, `alpha` | 融合文件中的全局分数、patch 分数和融合权重 |

## `paper_tables/`

由 `paper_scores/` 计算出的指标和汇总。

| 文件 | 说明 |
|---|---|
| `alpha_stalled_main_summary.md` | 三个数据集主实验平均 AUC/AP 汇总 |
| `ablation_summary.md` | 组件消融和 ComGenVid 局部时序定义消融汇总 |
| `<dataset>_alpha_stalled_metrics.csv` | Alpha-STALLED 逐生成器指标 |
| `<dataset>_global_only_metrics.csv` | global-only 逐生成器指标 |
| `<dataset>_patch_only_metrics.csv` | patch-only 逐生成器指标 |
| `comgenvid_patch_*_metrics.csv` | ComGenVid 各局部消融指标 |
| `patch_coverage_gaps.md` / `.csv` | 短视频来源未纳入当前 baseline 的原因 |

指标口径统一为 `src/metrics.py` 的 pairwise balanced AUC/AP。

## `paper_sweeps/`

固定 alpha sweep 结果，由 `tools/fuse_scores.py` 生成。

| 路径 | 说明 |
|---|---|
| `alpha_sweep_summary.md` | 三个数据集 best-alpha 和冻结 alpha=0.60 对照 |
| `<dataset>_alpha/*_summary.csv` | 某数据集不同 alpha 的平均指标 |
| `<dataset>_alpha/*_per_model.csv` | 某数据集不同 alpha 的逐生成器指标 |
| `<dataset>_alpha/*_summary.md` | 某数据集 alpha sweep 可读汇总 |

best-alpha 仅作为诊断性 oracle sweep。论文主线 baseline 仍使用
`configs/alpha_stalled.yaml` 中记录的冻结 alpha。

## `paper_sensitivity/`

基于现有逐视频分数生成的期刊版补充分析，不重新提取特征。

| 文件 | 说明 |
|---|---|
| `journal_experiment_gap_analysis.md` | 期刊版实验缺口、已补充分析和后续实跑优先级 |
| `journal_experiment_runbook.md` | 后续 region、aggregation、case visualization 等实跑实验命令模板 |
| `beta_sensitivity_summary.csv` | patch 内部空间/时序权重 beta 的平均 AUC/AP 曲线 |
| `beta_sensitivity_per_model.csv` | beta sweep 的逐生成器指标 |
| `component_per_generator_delta.csv` | global、patch、Alpha-STALLED 逐生成器 AUC/AP 差值 |
| `alpha_score_distribution_summary.csv` | Alpha-STALLED 分数分布统计 |
| `bootstrap_ci_summary.csv` | 逐生成器 AUC/AP bootstrap 置信区间 |
| `paired_bootstrap_delta_summary.csv` | 同一次重采样下的方法差异 ΔAUC/ΔAP 置信区间 |
| `failure_case_candidates.csv` | 后续人工审计和 patch anomaly map 可视化的候选样本 |
| `failure_case_candidates_summary.md` | 候选样本的可读摘要和案例图选择建议 |

## `paper_figures/`

由现有 CSV 派生的期刊版补充图表，默认同时保存 SVG 和 PNG：

| 文件 | 说明 |
|---|---|
| `alpha_sensitivity_curves.*` | 三数据集 alpha 敏感性曲线 |
| `beta_sensitivity_patch_only.*` | patch-only beta 敏感性曲线 |
| `beta_sensitivity_fused_alpha0p60.*` | 固定 alpha=0.60 下 beta 敏感性曲线 |
| `per_generator_auc_delta_heatmap.*` | 逐生成器 AUC 增益/退化 heatmap |
| `alpha_score_distribution_panel.*` | 真实/生成 Alpha-STALLED 分数分布 |
| `bootstrap_alpha_minus_global_auc_ci.*` | Alpha-STALLED 相对 global-only 的 AUC 增益及 paired bootstrap 区间 |
| `macro_average_bootstrap_alpha_delta.*` | 生成器宏平均 paired bootstrap 下 Alpha-STALLED 相对 global-only 的 ΔAUC 区间 |
| `failure_case_audit_priority.*` | failure / boundary 候选样本的 P0/P1/P2 优先级分布 |
| `comgenvid_bottomk_sensitivity.*` | ComGenVid bottom-k 聚合比例敏感性曲线 |
| `comgenvid_region_sensitivity.*` | ComGenVid patch region size 敏感性曲线 |
| `genvideo_region_sensitivity.*` | GenVideo patch region size 敏感性曲线 |
| `videofeedback_region_sensitivity.*` | VideoFeedback patch region size 敏感性曲线 |
| `genvideo_aggregation_sensitivity.*` | GenVideo mean 与 bottom-k aggregation 敏感性曲线 |
| `videofeedback_aggregation_sensitivity.*` | VideoFeedback mean 与 bottom-k aggregation 敏感性曲线 |
| `cross_dataset_frozen_hyperparams.*` | leave-one-dataset-out frozen hyperparameter 与目标 oracle 的 AUC 差距 |

## `journal_experiments/`

需要全量 patch cache 的期刊补充实跑实验。当前只保留轻量 CSV/Markdown 结果，
不提交 patch cache 或临时校准参数。

| 路径 | 说明 |
|---|---|
| `bottomk_sensitivity/` | bottom-k 超参数敏感性实跑结果 |
| `bottomk_sensitivity/bottomk_sensitivity_summary.md` | 已完成 bottom-k 点的可读汇总 |
| `region_sensitivity/` | patch region size 敏感性实跑结果 |
| `region_sensitivity/region_sensitivity_summary.md` | 已完成 region 点的可读汇总 |
| `aggregation_sensitivity/` | mean 与 bottom-k aggregation 敏感性实跑结果 |
| `aggregation_sensitivity/aggregation_sensitivity_summary.md` | 已完成 aggregation 对照点的可读汇总 |
| `case_visualizations/` | VideoFeedback 代表案例的 patch anomaly map 和案例清单 |
| `case_visualizations/patch_anomaly_cases.*` | 代表案例分数条、空间 anomaly map 和时序 anomaly 曲线 |
| `case_visualizations/selected_patch_cases.csv` | 案例图使用的固定样本清单 |
| `case_visualizations/patch_anomaly_case_summary.md` | 案例选择依据和图中 anomaly 计算口径 |
| `keyframe_case_explanations/` | VideoFeedback failure/boundary 案例的原视频关键帧、patch anomaly map 和解释表 |
| `keyframe_case_explanations/keyframe_patch_anomaly_cases.*` | 关键帧 + 分数条 + patch anomaly map + 时序 anomaly 曲线复合图 |
| `keyframe_case_explanations/keyframe_case_explanations.md` | 关键帧选择规则、案例解释和使用边界 |
| `d3_protocol_audit/` | D3 baseline protocol 审计，不包含外部 D3 重跑分数 |
| `d3_protocol_audit/d3_protocol_audit.md` | D3 采样、类别平衡、FPS 处理、encoder 和指标方向审计 |
| `d3_protocol_audit/d3_protocol_dataset_summary.csv` | 三个 index 的 FPS、duration 和 2s window 覆盖摘要 |
| `temporal_derivative_order/` | ComGenVid D=3/D=4 patch temporal derivative 代表性对照 |
| `temporal_derivative_order/comgenvid_temporal_derivative_order.md` | D=2/D=3/D=4 平均 AUC/AP 对照和结论 |
| `temporal_derivative_order/comgenvid_temporal_derivative_order.*` | temporal derivative order 对照图 |
| `duration_window_feasibility/` | 1s/2s/3s/4s duration/window 敏感性实跑前的 index/cache 覆盖审计 |
| `duration_window_feasibility/duration_window_feasibility.md` | duration/window 是否可直接 patch eval 的可读结论 |
| `duration_window_feasibility/duration_window_summary.csv` | 数据集级 duration/window index 与 compact patch cache 覆盖汇总 |
| `duration_window_representative/` | ComGenVid 1s 代表性 duration/window patch-only 实跑结果 |
| `duration_window_representative/comgenvid_duration_window_representative.md` | 1s 与 2s 主实验 patch-only AUC/AP 对照和解读 |
| `duration_window_representative/comgenvid_duration_window_comparison.csv` | 1s vs 2s duration/window 对照表 |
| `duration_window_representative/comgenvid_1s_patch_full.csv` | ComGenVid 1s patch-only 逐视频分数 |
| `duration_window_representative/comgenvid_1s_metrics_full.csv` | ComGenVid 1s patch-only 逐生成器指标 |
| `cross_dataset_frozen_hyperparams/` | 基于已有 sweep 的跨数据集冻结超参数 transfer / leave-one-dataset-out 分析 |
| `cross_dataset_frozen_hyperparams/cross_dataset_frozen_hyperparams.md` | alpha、beta、region、aggregation 冻结配置与目标 oracle gap 的可读结论 |
| `cross_dataset_frozen_hyperparams/*_leave_one_out.csv` | 每类超参数的 leave-one-dataset-out 选择结果 |
| `cross_dataset_frozen_hyperparams/*_transfer_matrix.csv` | 每类超参数的 source-to-target transfer matrix |
| `runtime_storage_audit/` | 当前 cache、结果资产体积、已有日志和固定 benchmark 的 runtime/storage 审计 |
| `runtime_storage_audit/runtime_storage_audit.md` | storage footprint、可追溯 runtime 片段、CSV-stage 和 video-stage benchmark 汇总 |
| `runtime_storage_audit/storage_audit.csv` | cache、precomputed、results 等目录的文件数和体积 |
| `runtime_storage_audit/runtime_log_audit.csv` | 已有补充实验日志中可解析的 tqdm elapsed time |
| `runtime_benchmark/` | 不重提特征、不重建 cache 的 CSV-stage runtime benchmark |
| `runtime_benchmark/csv_stage_runtime_benchmark.md` | 融合、metrics、alpha sweep 和审计脚本的固定命令计时报告 |
| `runtime_benchmark/csv_stage_runtime_benchmark.csv` | 每个 timed command 的命令、return code、elapsed seconds 和输出尾部 |
| `runtime_benchmark/runtime_benchmark_environment.md` | CPU/GPU/conda/package 环境记录 |
| `video_stage_runtime_benchmark/` | 使用原始视频和干净临时 cache 的小样本 video-stage runtime benchmark |
| `video_stage_runtime_benchmark/video_stage_runtime_benchmark.md` | global compact embedding、patch prefill 和 patch cached scoring 的代表性阶段成本 |
| `video_stage_runtime_benchmark/video_stage_runtime_benchmark.csv` | 每个 video-stage timed command 的命令、return code、elapsed seconds 和 cache 体积 |
| `video_stage_runtime_benchmark/*_scores.csv` | runtime 小样本输出分数，仅用于验证命令成功和输出规模 |
| `macro_average_bootstrap/` | 与论文 Average 行一致的生成器宏平均 paired bootstrap 结果 |
| `macro_average_bootstrap/macro_average_paired_bootstrap_delta.md` | Alpha-STALLED / patch / global 方法差值的宏平均 ΔAUC/ΔAP 置信区间 |
| `macro_average_bootstrap/macro_average_paired_bootstrap_delta.csv` | 宏平均 paired bootstrap 数值表 |
| `failure_case_audit/` | 合并候选样本、逐生成器 CI、宏平均 CI 和 index 元数据的失败/边界案例审计 |
| `failure_case_audit/failure_case_audit.md` | P0/P1/P2 案例优先级、推荐用途和论文写作建议 |
| `failure_case_audit/failure_case_audit.csv` | 300 个候选样本的完整审计表，含 video_path、duration、分数冲突和生成器 CI |
| `failure_case_audit/failure_case_audit_summary.csv` | 按数据集、风险类型和优先级汇总的案例数量 |
| `reference_alignment_audit/` | 对照 `2603.15026v2` 主文和附录实验体系的覆盖矩阵 |
| `reference_alignment_audit/reference_experiment_alignment.md` | 哪些参考文献实验已覆盖、哪些是 P1/P2/P3 缺口及下一步建议 |
| `reference_alignment_audit/reference_experiment_alignment.csv` | reference section、实验项、当前证据路径、缺口和优先级的机器可读表 |

## `alpha_stalled_manuscript_tables/`

论文表格的 SVG/PNG 渲染资产：

- `zero_shot_detection.*`
- `component_ablation.*`
- `temporal_ablation.*`

## 当前 baseline 数值

| 数据集 | 平均 AUC | 平均 AP | 覆盖范围 |
|---|---:|---:|---|
| ComGenVid | 0.9198 | 0.9211 | 2 个生成来源 |
| VideoFeedback | 0.8628 | 0.8750 | 10 个 2 秒生成来源 |
| GenVideo | 0.8374 | 0.8283 | 8 个 2 秒生成来源 |

短视频来源覆盖缺口见 `paper_tables/patch_coverage_gaps.md`。

## 重建命令

从当前 `paper_scores/` 重建融合分数、metrics 和 markdown 汇总：

```bash
bash scripts/reproduce/rebuild_paper_assets.sh
```

验证 release 资产：

```bash
conda run --no-capture-output -n stall python tools/verify_alpha_stalled_release.py
```
