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
| `comgenvid_bottomk_sensitivity.*` | ComGenVid bottom-k 聚合比例敏感性曲线 |
| `comgenvid_region_sensitivity.*` | ComGenVid patch region size 敏感性曲线 |
| `genvideo_region_sensitivity.*` | GenVideo patch region size 敏感性曲线 |
| `videofeedback_region_sensitivity.*` | VideoFeedback patch region size 敏感性曲线 |
| `genvideo_aggregation_sensitivity.*` | GenVideo mean 与 bottom-k aggregation 敏感性曲线 |
| `videofeedback_aggregation_sensitivity.*` | VideoFeedback aggregation 基线点；bottom-k 对照待补 |

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
