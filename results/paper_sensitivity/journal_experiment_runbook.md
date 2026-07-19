# 期刊版补充实验 Runbook

本文档基于当前已有基础消融，列出仍值得补充的实验、原因、命令模板和预期产物。原则是先做能回应审稿质疑的实验，而不是重复已有 global/patch/temporal 基础消融。

## 已完成且不建议重复的部分

- 三数据集 global-only、patch-only、Alpha-STALLED 组件消融。
- ComGenVid 上 patch 空间、lag-1、multi-lag、motion-hard、motion-soft、同网格二阶时序定义消融。
- 三数据集 global/patch 融合权重 alpha sweep。
- 基于现有 CSV 的 beta sweep、逐生成器 delta heatmap、分数分布、bootstrap CI、paired ΔAUC/ΔAP CI、失败样本候选。
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

当前状态：已完成 `results/journal_experiments/duration_window_feasibility/` 审计。三数据集只有 2s compact patch cache 可直接完整评测；VideoFeedback 的 1s cache 只覆盖 Hotshot-XL 生成视频，缺真实 1s cache，不能做真实视频校准；3s/4s 基本需要重新 prefill patch cache。

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

若决定实跑，优先选择一个代表数据集补 1s/2s 对照，不建议直接铺开三数据集 1s/3s/4s 全网格。

## P1：cross-dataset frozen hyperparameter

为什么要做：现有 alpha/beta sweep 是诊断性 oracle。期刊稿应明确 frozen hyperparameter 是否能跨数据集泛化。

建议协议：

- 用 ComGenVid 选 alpha/beta/region，在 VideoFeedback 和 GenVideo 上冻结评测；
- 用 VideoFeedback 选 alpha/beta/region，在 ComGenVid 和 GenVideo 上冻结评测；
- 报告 oracle 与 frozen transfer 的差距。

## P1：runtime 和存储开销

为什么要做：方法论文需要说明代价。建议报告：

- global-only scoring 时间；
- patch cache prefill 时间；
- patch-only scoring 时间；
- fusion 和 CSV 级分析时间；
- embedding cache 与 patch cache 大小。

## 推荐执行顺序

1. 根据版面决定是否投入 patch cache prefill 来补 duration/window 实跑；当前已完成可行性审计。
2. 若需要更强泛化论证，再补 cross-dataset frozen hyperparameter。
3. 最后补 runtime 和存储开销，作为方法代价说明。
