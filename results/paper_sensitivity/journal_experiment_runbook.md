# 期刊版补充实验 Runbook

本文档基于当前已有基础消融，列出仍值得补充的实验、原因、命令模板和预期产物。原则是先做能回应审稿质疑的实验，而不是重复已有 global/patch/temporal 基础消融。

## 已完成且不建议重复的部分

- 三数据集 global-only、patch-only、Alpha-STALLED 组件消融。
- ComGenVid 上 patch 空间、lag-1、multi-lag、motion-hard、motion-soft、同网格二阶时序定义消融。
- 三数据集 global/patch 融合权重 alpha sweep。
- 基于现有 CSV 的 beta sweep、逐生成器 delta heatmap、分数分布、bootstrap CI、paired ΔAUC/ΔAP CI、失败样本候选。
- 生成器宏平均 paired bootstrap：
  已与论文 Average 行口径对齐，三数据集 Alpha-STALLED 相对 global-only 的宏平均 ΔAUC 95% CI 均大于 0。
- ComGenVid / region=3 / bottom-k ratio = 0.10/0.15/0.20/0.25/0.30/0.35/0.40/0.50
  的全量 patch eval 和敏感性曲线。
- ComGenVid / same-grid second-order / mean aggregation 的 region=1/2/3
  全量 patch eval 和敏感性曲线，当前 region=3 最优，支持 ComGenVid 主线使用更大局部空间支持域。
- GenVideo / same-grid second-order / mean aggregation 的 region=1/2/3
  全量 patch eval 和敏感性曲线，当前 region=2 最优，支持主线 region 选择。
- VideoFeedback / same-grid second-order / mean aggregation 的 region=1/2/3
  全量 patch eval 和敏感性曲线，当前 region=1 最优，说明该数据集对更小局部空间支持域更敏感。
- GenVideo / region=2 / same-grid second-order 的 aggregation 对照：
  bottom-k ratio = 0.20/0.50 均低于 mean baseline，支持 GenVideo 主线使用 mean aggregation。
- VideoFeedback / region=1 / same-grid second-order 的 aggregation 对照：
  bottom-k ratio = 0.20/0.50 均低于 mean baseline，支持 VideoFeedback 主线使用 mean aggregation。
- VideoFeedback 代表性 patch anomaly case visualization：
  已从 failure candidates 中选取 6 个案例，并基于已有 patch cache 生成空间 anomaly map 和时序 anomaly 曲线。
- Failure / boundary case audit：
  已将 300 个 failure candidates 与逐生成器 CI、宏平均 CI 和 index 元数据合并，输出 P0/P1/P2 审计优先级。
- Cross-dataset frozen hyperparameter：
  已基于现有 alpha/beta/region/aggregation 敏感性结果完成 transfer matrix 和 leave-one-dataset-out 分析，
  用于区分目标数据集 oracle sweep 与跨数据集冻结配置。
- Runtime / storage 代价审计：
  已读取当前 cache、precomputed、results 目录体积，解析已有补充实验日志中的 tqdm elapsed time，
  并完成不重提特征的 CSV-stage runtime benchmark；
  完整端到端 runtime 表仍需固定环境单独 benchmark。
- Reference alignment audit：
  已对照 `2603.15026v2` 主文和附录实验体系，生成当前 Alpha-STALLED 结果覆盖矩阵和 P1/P2/P3 缺口优先级。
- Duration/window representative run：
  已完成 ComGenVid 1s patch-only 代表性实跑，并与 2s 主实验 patch-only 指标对照。

## P0：patch region size 敏感性

为什么要做：当前主线配置对不同数据集使用不同 patch region size。期刊审稿可能质疑是否存在数据集特异调参。ComGenVid、GenVideo 与 VideoFeedback 均已完成 region=1/2/3 mean aggregation。结果分别支持 ComGenVid 使用 region=3、GenVideo 使用 region=2、VideoFeedback 使用 region=1，说明 region size 对应局部时序证据的空间尺度，而不是一个固定装饰性参数。

建议网格：

```text
patch_region_size ∈ {1, 2, 3}
patch_temp_mode = same_grid_second_order
aggregation = mean 或当前数据集主线 aggregation
beta = 0.10 作为 frozen baseline；另用现有 beta sweep 做补充解释
```

命令模板，以 `<dataset>` 替换 `comgenvid/videofeedback/genvideo`：

```bash
conda run --no-capture-output -n stall python src/create_patch_params.py \
  --csv cache/indexes/<dataset>.csv \
  --patch-emb-cache cache/patch_embeddings/<dataset> \
  --output precomputed/patch_params_<dataset>_region<R>_second_order_journal.npz \
  --patch-temp-mode same_grid_second_order \
  --patch-region-size <R> \
  --aggregation mean \
  --real-only

conda run --no-capture-output -n stall python src/eval_patch_fast.py \
  --csv cache/indexes/<dataset>.csv \
  --patch-emb-cache cache/patch_embeddings/<dataset> \
  --patch-params precomputed/patch_params_<dataset>_region<R>_second_order_journal.npz \
  --patch-temp-mode same_grid_second_order \
  --patch-region-size <R> \
  --aggregation mean \
  --patch-spat-weight 0.10 \
  --patch-temp-weight 0.90 \
  --output-csv results/journal_experiments/region_sensitivity/<dataset>_region<R>_patch.csv

conda run --no-capture-output -n stall python tools/eval_score_csv.py \
  --csv results/journal_experiments/region_sensitivity/<dataset>_region<R>_patch.csv \
  --score-col patch_final_score \
  --output-csv results/journal_experiments/region_sensitivity/<dataset>_region<R>_metrics.csv
```

预期产物：

```text
results/journal_experiments/region_sensitivity/*_patch.csv
results/journal_experiments/region_sensitivity/*_metrics.csv
results/journal_experiments/region_sensitivity/region_sensitivity_summary.md
```

## P0：aggregation / bottom-k 敏感性

为什么要做：ComGenVid 上 bottom-k 有明显作用，且已完成加密网格；
GenVideo 和 VideoFeedback 已完成 bottom-k 对照且均支持 mean。
ComGenVid 的 region=3 mean 已接近 bottom-k 主线，但 bottom-k=0.50 仍略高，
说明 region size 与 aggregation 都应作为独立边界分析报告。

建议网格：

```text
aggregation ∈ {mean, bottomk_mean}
bottomk_ratio ∈ {0.10, 0.20, 0.30, 0.50}  # 只对 bottomk_mean 生效
patch_region_size ∈ {当前主线 region}
patch_temp_mode = same_grid_second_order
```

命令模板：

```bash
conda run --no-capture-output -n stall python src/create_patch_params.py \
  --csv cache/indexes/<dataset>.csv \
  --patch-emb-cache cache/patch_embeddings/<dataset> \
  --output precomputed/patch_params_<dataset>_<AGG>_<BK>_second_order_journal.npz \
  --patch-temp-mode same_grid_second_order \
  --patch-region-size <MAIN_REGION> \
  --aggregation <AGG> \
  --bottomk-ratio <BK> \
  --real-only

conda run --no-capture-output -n stall python src/eval_patch_fast.py \
  --csv cache/indexes/<dataset>.csv \
  --patch-emb-cache cache/patch_embeddings/<dataset> \
  --patch-params precomputed/patch_params_<dataset>_<AGG>_<BK>_second_order_journal.npz \
  --patch-temp-mode same_grid_second_order \
  --patch-region-size <MAIN_REGION> \
  --aggregation <AGG> \
  --bottomk-ratio <BK> \
  --patch-spat-weight 0.10 \
  --patch-temp-weight 0.90 \
  --output-csv results/journal_experiments/aggregation_sensitivity/<dataset>_<AGG>_<BK>_patch.csv
```

预期产物：

```text
results/journal_experiments/aggregation_sensitivity/*_patch.csv
results/journal_experiments/aggregation_sensitivity/*_metrics.csv
results/journal_experiments/aggregation_sensitivity/aggregation_sensitivity_summary.md
```

## P0：可解释案例图

为什么要做：已有指标说明 patch 分支总体有帮助，但 VideoFeedback 存在负迁移。期刊稿需要说明边界，而不是只展示平均数。当前已完成 patch-cache 级 anomaly map；若需要更强直观性，可在同一图中再叠加原视频关键帧。

输入候选：

```text
results/paper_sensitivity/failure_case_candidates.csv
results/paper_sensitivity/failure_case_candidates_summary.md
```

建议优先选择：

- VideoFeedback / Text2Video-Zero：稳定负迁移。
- VideoFeedback / VideoCrafter2：稳定负迁移。
- 一个 GenVideo / Sora 正迁移样本：展示 patch 分支显著补强 global 的情况。
- 一个真实视频低分样本：展示误伤边界。

预期图：

```text
results/journal_experiments/case_visualizations/patch_anomaly_cases.svg
results/journal_experiments/case_visualizations/patch_anomaly_cases.png
results/journal_experiments/case_visualizations/selected_patch_cases.csv
results/journal_experiments/case_visualizations/patch_anomaly_case_summary.md
```

图中建议包含：原视频关键帧、patch anomaly heatmap、global/patch/Alpha-STALLED 三个分数，以及一句失败/成功机制说明。

## P1：duration/window 敏感性

为什么要做：VideoFeedback 和 GenVideo 有短视频来源，当前主线报告 2 秒覆盖，短视频来源作为 coverage gap 单独说明。若投期刊，最好补充 1s/2s/3s/4s 的边界。

当前状态：已完成 `results/journal_experiments/duration_window_feasibility/` 审计，并进一步完成 ComGenVid 1s 代表性实跑。1s patch-only 平均 AUC/AP 为 0.9434 / 0.9456，2s 主实验 patch-only 平均 AUC/AP 为 0.9273 / 0.9309。该结果说明当前局部二阶时序证据在 ComGenVid 上不严格依赖 2s 窗口，可作为短窗口边界证据。VideoFeedback 的 1s cache 只覆盖 Hotshot-XL 生成视频、缺真实 1s cache；3s/4s 基本需要重新 prefill patch cache。

已完成产物：

```text
results/journal_experiments/duration_window_representative/comgenvid_1s_patch_full.csv
results/journal_experiments/duration_window_representative/comgenvid_1s_metrics_full.csv
results/journal_experiments/duration_window_representative/comgenvid_duration_window_comparison.csv
results/journal_experiments/duration_window_representative/comgenvid_duration_window_representative.md
scripts/run_comgenvid_duration1_representative.sh
tools/summarize_duration_window_representative.py
```

建议网格：

```text
duration ∈ {1, 2, 3, 4}
对每个 duration 独立使用对应 index/window 和 patch params
```

风险：该实验可能需要重建 index 和 cache，耗时高于 CSV 级分析。

建议执行前置步骤：

```bash
conda run --no-capture-output -n stall python tools/analyze_duration_window_feasibility.py
```

若继续扩展，不建议直接铺开三数据集 1s/3s/4s 全网格；应先明确审稿或论文叙事是否需要，因为当前 ComGenVid 1s 结果已经能支撑短窗口可用性边界。

## P1：cross-dataset frozen hyperparameter

为什么要做：现有 alpha/beta sweep 是诊断性 oracle。期刊稿应明确 frozen hyperparameter 是否能跨数据集泛化。

当前状态：已完成 `results/journal_experiments/cross_dataset_frozen_hyperparams/` 分析。
该分析不重新提取特征、不重新评测视频，只读取已有敏感性 CSV。

已采用协议：

- source-to-target transfer matrix：用 source dataset 上平均 AUC 最优的超参数，套到 target dataset；
- leave-one-dataset-out：用另外两个数据集平均 AUC 最优的超参数，冻结到 held-out target；
- 报告 frozen AUC/AP、target oracle AUC/AP，以及 ΔAUC/ΔAP。

重建命令：

```bash
conda run --no-capture-output -n stall python tools/analyze_cross_dataset_frozen_hyperparams.py
```

关键结论：

- beta frozen 平均 |ΔAUC| = 0.0020，aggregation frozen 平均 |ΔAUC| = 0.0004，说明这两类超参数跨数据集损失较小。
- alpha frozen 平均 |ΔAUC| = 0.0119，ComGenVid 作为 held-out target 时损失最大（-0.0224）。
- region frozen 平均 |ΔAUC| = 0.0198，VideoFeedback 作为 held-out target 时损失最大（-0.0288）。
- 论文中应把 region size 写作局部运动尺度/空间支持域的边界参数；beta 和主 region 下的 aggregation 可以写作相对稳健的辅助选择。

预期产物：

```text
results/journal_experiments/cross_dataset_frozen_hyperparams/*_transfer_matrix.csv
results/journal_experiments/cross_dataset_frozen_hyperparams/*_leave_one_out.csv
results/journal_experiments/cross_dataset_frozen_hyperparams/cross_dataset_frozen_hyperparams.md
results/paper_figures/cross_dataset_frozen_hyperparams.svg
results/paper_figures/cross_dataset_frozen_hyperparams.png
```

## P1：runtime 和存储开销

为什么要做：方法论文需要说明代价。建议报告：

- global-only scoring 时间；
- patch cache prefill 时间；
- patch-only scoring 时间；
- fusion 和 CSV 级分析时间；
- embedding cache 与 patch cache 大小。

当前状态：已完成 `results/journal_experiments/runtime_storage_audit/` 审计和
`results/journal_experiments/runtime_benchmark/` CSV-stage benchmark。
当前 compact patch cache 合计约 682.55 GiB，global embedding cache 合计约 8.23 GiB；
已有日志只能支持补充实验片段耗时，不足以作为完整端到端 runtime 表。
CSV-stage benchmark 在 RTX 5090 ×2 环境中完成 18 个固定命令，总耗时 22.645 秒，
其中三数据集主融合+metrics 合计 3.640 秒，三数据集 alpha sweep 合计 4.883 秒。

重建命令：

```bash
conda run --no-capture-output -n stall python tools/audit_runtime_storage_costs.py
```

CSV-stage benchmark 重建命令：

```bash
conda run --no-capture-output -n stall python tools/benchmark_csv_stage_runtime.py
```

若要生成正式 runtime 表，建议固定如下条件后单独重跑：

```text
GPU 型号和数量
batch size
是否已有 global embedding cache
是否已有 compact patch cache
dataset / duration / region / aggregation 配置
计时范围：从命令启动到输出 CSV 完成
```

当前可直接引用的结论是 storage footprint、已有补充实验日志片段，以及 CSV-stage 后处理成本；
不要把这些写成完整 pipeline runtime。

## P2：生成器宏平均 paired bootstrap

为什么要做：逐生成器 paired bootstrap 能解释哪些生成器稳定提升或负迁移，但论文主表的 Average 行还需要一个总体不确定性口径。

当前状态：已完成 `results/journal_experiments/macro_average_bootstrap/` 分析。
每个 bootstrap 轮次先对每个生成器分别 balanced resampling 并计算 AUC/AP，再对生成器取宏平均。

重建命令：

```bash
conda run --no-capture-output -n stall python tools/bootstrap_macro_average_delta.py
```

关键结论：

- ComGenVid: Alpha-STALLED 相对 global-only 宏平均 ΔAUC = +0.0669，95% CI [+0.0618, +0.0714]。
- VideoFeedback: Alpha-STALLED 相对 global-only 宏平均 ΔAUC = +0.0153，95% CI [+0.0127, +0.0179]。
- GenVideo: Alpha-STALLED 相对 global-only 宏平均 ΔAUC = +0.0344，95% CI [+0.0266, +0.0433]。

该结果可以支持“总体平均指标稳定提升”的表述；逐生成器 paired bootstrap 仍用于说明 VideoFeedback 的局部负迁移边界。

## P2：失败/边界样本审计

为什么要做：已有案例图只展示少量样本。期刊稿还需要说明这些案例如何选择，避免被认为是 cherry-picking。

当前状态：已完成 `results/journal_experiments/failure_case_audit/` 审计。
该分析合并 `failure_case_candidates.csv`、逐生成器 paired bootstrap、宏平均 paired bootstrap 和 index 元数据；
不观看视频、不重新提取特征、不参与调参。

重建命令：

```bash
conda run --no-capture-output -n stall python tools/audit_failure_cases.py
```

关键结论：

- 300 个候选样本全部匹配到 index 元数据，包括 `video_path`、duration、fps 和 frame count。
- P0 案例共 86 个，其中 VideoFeedback 占 51 个，主要来自 Text2Video-Zero / VideoCrafter2 稳定负迁移、patch-global conflict 和 Panda70M 真实误伤。
- 该审计可作为案例选择依据；若要解释“低运动、压缩伪影、语义域偏移”等具体视觉原因，还需要人工观看关键帧。

## P2：参考文献实验体系对齐审计

为什么要做：参考文献 `2603.15026v2` 覆盖了大量主文和附录实验，包括 calibration、backbone、扰动、FPS/duration、效率和 qualitative examples。Alpha-STALLED 不应机械照搬所有实验，而应先判断哪些已经由现有证据覆盖，哪些真正需要补。

当前状态：已完成 `results/journal_experiments/reference_alignment_audit/` 审计。
该分析只检查已有文件，不重新提取特征、不运行模型。

重建命令：

```bash
conda run --no-capture-output -n stall python tools/audit_reference_experiment_alignment.py
```

关键结论：

- `done`：组件消融、patch aggregation/region 边界、统计稳定性已经足够，不建议重复。
- `P1`：若继续投入，最值得做的是代表数据集 duration/window 实跑，或固定环境视频级端到端 runtime benchmark。
- `P2`：外部 baseline/D3 protocol、原视频关键帧 qualitative examples、calibration source 只在投稿或审稿明确要求时做。
- `P3`：backbone、image/temporal perturbation、calibration size/source 大多需要重新提特征或重建大量 cache，当前不建议默认执行。

## 推荐执行顺序

1. 根据版面决定是否投入 patch cache prefill 来补 duration/window 实跑；当前已完成可行性审计。
2. Cross-dataset frozen hyperparameter 已完成，可直接写入泛化/边界分析。
3. Macro-average paired bootstrap 已完成，可用于总体显著性/稳定性表述。
4. Reference alignment audit 已完成；CSV-stage runtime benchmark 也已完成。若继续补，只建议做三类需要额外资源或人工判断的工作：视频级端到端 runtime benchmark、代表数据集 duration cache prefill、失败样本关键帧人工语义解释。Backbone、扰动鲁棒性、calibration size/source 暂列 P3，不作为默认下一步。
