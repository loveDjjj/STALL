# Runtime / storage 代价审计

本审计只读取本地文件系统和已有日志，不重新运行 global scoring、patch prefill 或 patch eval。
因此，目录体积是当前状态的直接证据；运行时间只报告日志中可解析的补充实验片段，不能替代完整端到端 benchmark。

## 存储规模

| item | path | files | size MiB | size GiB |
|---|---|---:|---:|---:|
| index_csv | `cache/indexes` | 6 | 25.59 | 0.025 |
| global_embedding_cache_comgenvid | `cache/embeddings/comgenvid` | 5098 | 327.60 | 0.320 |
| global_embedding_cache_videofeedback | `cache/embeddings/videofeedback` | 37661 | 2660.13 | 2.598 |
| global_embedding_cache_genvideo | `cache/embeddings/genvideo` | 18188 | 5434.78 | 5.307 |
| compact_patch_cache_comgenvid | `cache/patch_embeddings/comgenvid` | 5098 | 62779.95 | 61.309 |
| compact_patch_cache_videofeedback | `cache/patch_embeddings/videofeedback` | 37661 | 443759.39 | 433.359 |
| compact_patch_cache_genvideo | `cache/patch_embeddings/genvideo` | 15623 | 192388.48 | 187.879 |
| precomputed_calibration_params | `precomputed` | 74 | 595.23 | 0.581 |
| paper_scores | `results/paper_scores` | 15 | 21.45 | 0.021 |
| paper_tables | `results/paper_tables` | 19 | 0.01 | 0.000 |
| paper_sensitivity | `results/paper_sensitivity` | 10 | 0.23 | 0.000 |
| paper_sweeps | `results/paper_sweeps` | 10 | 0.05 | 0.000 |
| journal_experiments | `results/journal_experiments` | 71 | 52.23 | 0.051 |
| paper_figures | `results/paper_figures` | 26 | 1.83 | 0.002 |

## 汇总判断

- compact patch cache 当前合计 682.55 GiB，global embedding cache 当前合计 8.23 GiB。
- `results/` 下当前论文资产合计 75.81 MiB；提交范围仍应以 `.gitignore` 与 `git status` 为准，优先提交 CSV/Markdown/SVG/必要 PNG。
- cache、precomputed 参数和新增运行日志不应提交。
- duration/window sweep 若扩展到 1s/3s/4s，主要新增成本会落在 compact patch cache prefill，而不是 CSV 级融合分析。

## 已有日志中可解析的运行片段

| log | parsed tqdm elapsed | last line |
|---|---:|---|
| `results/journal_experiments/aggregation_sensitivity/genvideo_aggregation_run.log` | 30:05 | 2026年 07月 19日 星期日 23:26:14 CST |
| `results/journal_experiments/aggregation_sensitivity/videofeedback_aggregation_run.log` | 57:06 | 2026年 07月 20日 星期一 01:26:48 CST |
| `results/journal_experiments/region_sensitivity/comgenvid_region_mean_run.log` | 01:25 | 2026年 07月 19日 星期日 22:12:16 CST |
| `results/journal_experiments/region_sensitivity/videofeedback_region_mean_run.log` | 57:31 | ===== 2026-07-19 00:48:58 VideoFeedback region mean sensitivity all done ===== |

## 仍需实测的 runtime 项

- global-only scoring 的端到端时间；
- patch cache prefill 的端到端时间和 GPU 配置；
- patch-only eval 在三数据集主配置上的端到端时间；
- fusion / metrics / sensitivity CSV 级分析时间。

论文中目前可以稳妥报告 storage footprint 和已有补充实验片段耗时；完整 runtime 表应在固定 GPU、固定 batch size、清空/固定 cache 状态后单独重跑一次。
