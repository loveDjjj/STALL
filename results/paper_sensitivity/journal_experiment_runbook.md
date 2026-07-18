# 期刊版补充实验 Runbook

本文档基于当前已有基础消融，列出仍值得补充的实验、原因、命令模板和预期产物。原则是先做能回应审稿质疑的实验，而不是重复已有 global/patch/temporal 基础消融。

## 已完成且不建议重复的部分

- 三数据集 global-only、patch-only、Alpha-STALLED 组件消融。
- ComGenVid 上 patch 空间、lag-1、multi-lag、motion-hard、motion-soft、同网格二阶时序定义消融。
- 三数据集 global/patch 融合权重 alpha sweep。
- 基于现有 CSV 的 beta sweep、逐生成器 delta heatmap、分数分布、bootstrap CI、paired ΔAUC/ΔAP CI、失败样本候选。
- ComGenVid / region=3 / bottom-k ratio = 0.10/0.15/0.20/0.25/0.30/0.35/0.40/0.50
  的全量 patch eval 和敏感性曲线。

## P0：patch region size 敏感性

为什么要做：当前主线配置对不同数据集使用不同 patch region size。期刊审稿可能质疑是否存在数据集特异调参。需要至少证明 region=1/2/3 的趋势，而不是只报告最终配置。

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
VideoFeedback/GenVideo 当前主线使用 mean，仍需要证明 aggregation 选择不是偶然。

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

为什么要做：已有指标说明 patch 分支总体有帮助，但 VideoFeedback 存在负迁移。期刊稿需要说明边界，而不是只展示平均数。

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
```

图中建议包含：原视频关键帧、patch anomaly heatmap、global/patch/Alpha-STALLED 三个分数，以及一句失败/成功机制说明。

## P1：duration/window 敏感性

为什么要做：VideoFeedback 和 GenVideo 有短视频来源，当前主线报告 2 秒覆盖，短视频来源作为 coverage gap 单独说明。若投期刊，最好补充 1s/2s/3s/4s 的边界。

建议网格：

```text
duration ∈ {1, 2, 3, 4}
对每个 duration 独立使用对应 index/window 和 patch params
```

风险：该实验可能需要重建 index 和 cache，耗时高于 CSV 级分析。

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

1. 先跑 region size 敏感性。
2. 再跑 aggregation/bottom-k 敏感性。
3. 同时从 failure candidates 选 4-6 个案例，做 patch anomaly map。
4. 最后根据版面决定是否补 duration 和 runtime。
