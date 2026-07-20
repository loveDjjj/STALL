# Video-stage runtime benchmark

本 benchmark 使用原始视频和干净临时 cache，计时 Alpha-STALLED 中无法由 CSV 直接复现的视频级阶段。默认设置为 ComGenVid 2s，每个 `(subset, source_model)` 取 2 个视频，因此共 6 个视频；该设置用于端到端阶段成本审计，不作为性能指标。

环境记录见 `video_stage_runtime_environment.md`。

## Timed stages

| stage | elapsed sec | return code | cache files | cache MiB | score rows |
|---|---:|---:|---:|---:|---:|
| `global_compact_embedding_and_scoring` | 8.359 | 0 | 6 | 0.386 | 6 |
| `patch_compact_cache_prefill` | 7.928 | 0 | 6 | 73.888 | 1 |
| `patch_cached_scoring` | 3.796 | 0 | 6 | 73.888 | 6 |

## Summary

- successful stages: 3 / 3
- total successful elapsed time: 20.083 sec
- `global_compact_embedding_and_scoring`: 1.393 sec per score row
- `patch_compact_cache_prefill`: 1.321 sec per cache file
- `patch_cached_scoring`: 0.633 sec per score row

## Interpretation

该结果补足 CSV-stage benchmark 不能覆盖的 raw-video 阶段：视频解码、DINOv3 embedding 写 cache、patch cache prefill 以及从 patch cache 读取后的快速评分。由于这是小样本 clean-cache benchmark，论文中应把它表述为代表性阶段成本，不要外推为全数据集总耗时。全数据集总耗时仍应结合已有日志和 cache/storage 审计说明。

## Output files

- stage table: `results/journal_experiments/video_stage_runtime_benchmark/video_stage_runtime_benchmark.csv`
- environment: `results/journal_experiments/video_stage_runtime_benchmark/video_stage_runtime_environment.md`
- score CSVs: `results/journal_experiments/video_stage_runtime_benchmark/comgenvid_2s_debug2_global_scores.csv`, `results/journal_experiments/video_stage_runtime_benchmark/comgenvid_2s_debug2_patch_scores.csv`
- patch prefill summary: `results/journal_experiments/video_stage_runtime_benchmark/comgenvid_2s_debug2_patch_prefill.csv`
