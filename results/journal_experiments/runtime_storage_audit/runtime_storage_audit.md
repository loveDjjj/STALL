# Runtime / storage 代价审计

本审计读取本地文件系统、已有日志和已生成的固定 benchmark 结果。
目录体积是当前状态的直接证据；运行时间分为历史补充实验日志、CSV-stage benchmark 和小样本 clean-cache video-stage benchmark。

## 存储规模

| item | path | files | size MiB | size GiB |
|---|---|---:|---:|---:|
| index_csv | `cache/indexes` | 6 | 25.59 | 0.025 |
| global_embedding_cache_comgenvid | `cache/embeddings/comgenvid` | 5098 | 327.60 | 0.320 |
| global_embedding_cache_videofeedback | `cache/embeddings/videofeedback` | 37661 | 2660.13 | 2.598 |
| global_embedding_cache_genvideo | `cache/embeddings/genvideo` | 18188 | 5434.78 | 5.307 |
| compact_patch_cache_comgenvid | `cache/patch_embeddings/comgenvid` | 10198 | 94187.66 | 91.980 |
| compact_patch_cache_videofeedback | `cache/patch_embeddings/videofeedback` | 37661 | 443759.39 | 433.359 |
| compact_patch_cache_genvideo | `cache/patch_embeddings/genvideo` | 15623 | 192388.48 | 187.879 |
| precomputed_calibration_params | `precomputed` | 75 | 603.26 | 0.589 |
| paper_scores | `results/paper_scores` | 15 | 21.45 | 0.021 |
| paper_tables | `results/paper_tables` | 19 | 0.01 | 0.000 |
| paper_sensitivity | `results/paper_sensitivity` | 10 | 0.24 | 0.000 |
| paper_sweeps | `results/paper_sweeps` | 10 | 0.05 | 0.000 |
| journal_experiments | `results/journal_experiments` | 92 | 53.34 | 0.052 |
| paper_figures | `results/paper_figures` | 30 | 2.00 | 0.002 |

## 汇总判断

- compact patch cache 当前合计 713.22 GiB，global embedding cache 当前合计 8.23 GiB。
- `results/` 下当前论文资产合计 77.10 MiB；提交范围仍应以 `.gitignore` 与 `git status` 为准，优先提交 CSV/Markdown/SVG/必要 PNG。
- cache、precomputed 参数和新增运行日志不应提交。
- duration/window sweep 若扩展到 1s/3s/4s，主要新增成本会落在 compact patch cache prefill，而不是 CSV 级融合分析。
- video-stage runtime benchmark 已补充原始视频解码、DINOv3 embedding、patch prefill 和 patch cache scoring 的代表性小样本 clean-cache 计时。

## 已有日志中可解析的运行片段

| log | parsed tqdm elapsed | last line |
|---|---:|---|
| `results/journal_experiments/aggregation_sensitivity/genvideo_aggregation_run.log` | 30:05 | 2026年 07月 19日 星期日 23:26:14 CST |
| `results/journal_experiments/aggregation_sensitivity/videofeedback_aggregation_run.log` | 57:06 | 2026年 07月 20日 星期一 01:26:48 CST |
| `results/journal_experiments/region_sensitivity/comgenvid_region_mean_run.log` | 01:25 | 2026年 07月 19日 星期日 22:12:16 CST |
| `results/journal_experiments/region_sensitivity/videofeedback_region_mean_run.log` | 57:31 | ===== 2026-07-19 00:48:58 VideoFeedback region mean sensitivity all done ===== |

## 固定 video-stage benchmark

结果文件：`journal_experiments/video_stage_runtime_benchmark/video_stage_runtime_benchmark.md`。

| stage | elapsed sec | cache files | cache MiB | score rows |
|---|---:|---:|---:|---:|
| `global_compact_embedding_and_scoring` | 8.359 | 6 | 0.386 | 6 |
| `patch_compact_cache_prefill` | 7.928 | 6 | 73.888 | 1 |
| `patch_cached_scoring` | 3.796 | 6 | 73.888 | 6 |

## 尚未覆盖的 runtime 边界

- 尚未清空三数据集全部 cache 后重跑全量端到端总耗时；这会产生大量重复计算和 I/O，不建议作为默认补充任务。
- 当前可以报告 storage footprint、CSV-stage 后处理成本、代表性 video-stage clean-cache 阶段成本，以及已有全量补充实验日志片段。

论文中应明确区分小样本阶段成本、已有全量实验片段耗时和全数据集总耗时，避免把小样本 clean-cache benchmark 线性外推为完整 benchmark。
