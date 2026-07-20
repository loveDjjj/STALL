# CSV-stage runtime benchmark

本 benchmark 只覆盖不需要原始视频、不重提 DINOv3 特征、不重建 patch cache 的阶段。
因此它不能替代完整端到端 runtime 表；它用于量化论文资产重建、融合和 CSV 级分析的实际成本。

环境记录见 `runtime_benchmark_environment.md`。

## Timed commands

| stage | elapsed sec | return code |
|---|---:|---:|
| `fuse_main_comgenvid` | 1.144 | 0 |
| `metrics_comgenvid_global` | 1.118 | 0 |
| `metrics_comgenvid_patch_second_order` | 1.099 | 0 |
| `metrics_comgenvid_alpha_stalled` | 1.111 | 0 |
| `alpha_sweep_comgenvid` | 1.269 | 0 |
| `fuse_main_videofeedback` | 1.295 | 0 |
| `metrics_videofeedback_global` | 1.129 | 0 |
| `metrics_videofeedback_patch_second_order` | 1.176 | 0 |
| `metrics_videofeedback_alpha_stalled` | 1.137 | 0 |
| `alpha_sweep_videofeedback` | 1.998 | 0 |
| `fuse_main_genvideo` | 1.202 | 0 |
| `metrics_genvideo_global` | 1.138 | 0 |
| `metrics_genvideo_patch_second_order` | 1.143 | 0 |
| `metrics_genvideo_alpha_stalled` | 1.129 | 0 |
| `alpha_sweep_genvideo` | 1.615 | 0 |
| `run_analyze_duration_window_feasibility` | 1.071 | 0 |
| `run_audit_runtime_storage_costs` | 2.074 | 0 |
| `run_audit_reference_experiment_alignment` | 0.797 | 0 |

## Summary

- successful commands: 18 / 18
- total successful elapsed time: 22.645 sec
- main fusion + metrics total: 3.640 sec
- score metrics total: 10.180 sec
- alpha sweep total: 4.883 sec
- audit scripts total: 3.942 sec

## Pending end-to-end benchmark items

| item | why not included here | required next step |
|---|---|---|
| global embedding extraction | requires raw video decoding and DINOv3 inference | run fixed-dataset video scoring benchmark with cache state recorded |
| compact patch cache prefill | dominant storage/runtime cost; can run for many hours and hundreds of GiB | choose one representative dataset/duration and time prefill from a clean cache |
| full patch eval | existing logs are available, but not a clean fixed benchmark | rerun selected dataset/region/aggregation with fixed GPU and batch size |
| raw-video keyframe visualization | requires video decoding and visual selection | run only if manuscript needs raw-frame qualitative panels |
