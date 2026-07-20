# CSV-stage runtime benchmark

本 benchmark 只覆盖不需要原始视频、不重提 DINOv3 特征、不重建 patch cache 的阶段。
因此它不能替代完整端到端 runtime 表；它用于量化论文资产重建、融合和 CSV 级分析的实际成本。

环境记录见 `runtime_benchmark_environment.md`。

## Timed commands

| stage | elapsed sec | return code |
|---|---:|---:|
| `fuse_main_comgenvid` | 1.157 | 0 |
| `metrics_comgenvid_global` | 1.080 | 0 |
| `metrics_comgenvid_patch_second_order` | 1.098 | 0 |
| `metrics_comgenvid_alpha_stalled` | 1.089 | 0 |
| `alpha_sweep_comgenvid` | 1.253 | 0 |
| `fuse_main_videofeedback` | 1.324 | 0 |
| `metrics_videofeedback_global` | 1.129 | 0 |
| `metrics_videofeedback_patch_second_order` | 1.171 | 0 |
| `metrics_videofeedback_alpha_stalled` | 1.145 | 0 |
| `alpha_sweep_videofeedback` | 1.984 | 0 |
| `fuse_main_genvideo` | 1.198 | 0 |
| `metrics_genvideo_global` | 1.114 | 0 |
| `metrics_genvideo_patch_second_order` | 1.110 | 0 |
| `metrics_genvideo_alpha_stalled` | 1.115 | 0 |
| `alpha_sweep_genvideo` | 1.608 | 0 |
| `run_analyze_duration_window_feasibility` | 1.050 | 0 |
| `run_audit_runtime_storage_costs` | 2.145 | 0 |
| `run_audit_reference_experiment_alignment` | 0.810 | 0 |

## Summary

- successful commands: 18 / 18
- total successful elapsed time: 22.579 sec
- main fusion + metrics total: 3.678 sec
- score metrics total: 10.050 sec
- alpha sweep total: 4.845 sec
- audit scripts total: 4.006 sec

## Video-stage benchmark status

| item | status | where to look |
|---|---|---|
| global embedding extraction | covered by representative clean-cache video-stage benchmark | `results/journal_experiments/video_stage_runtime_benchmark/video_stage_runtime_benchmark.md` |
| compact patch cache prefill | covered by representative clean-cache video-stage benchmark | `results/journal_experiments/video_stage_runtime_benchmark/video_stage_runtime_benchmark.md` |
| patch cached scoring | covered by representative clean-cache video-stage benchmark | `results/journal_experiments/video_stage_runtime_benchmark/video_stage_runtime_benchmark.md` |
| full-dataset clean-cache total runtime | intentionally not run by this CSV-stage script | only run if reviewers require a full end-to-end wall-clock table |
