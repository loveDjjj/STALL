# Experiment launchers

本目录只保存可恢复的实验命令编排，不实现采样、评分、校准或指标公式。共享数学
和 I/O 必须位于 `src/alpha_stalled/`，单次分析入口位于 `tools/`。

现有脚本属于历史或专项入口；它们创建的目录和命名不应被当作新实验模板：

| 脚本 | 角色 |
|---|---|
| `run_multi_window_shard.sh` | 历史 multi-window shard 编排 |
| `run_local_residual_shard.sh` | 已拒绝 residual 方向的 shard 编排 |
| `fit_unified_region_params.sh` | 历史 region 参数拟合 |
| `run_journal_patch_sensitivity.sh` | 历史 patch sensitivity 批处理 |

新实验统一写入 `results/runs/<experiment_id>/`，并由捕获包装器启动：

```bash
conda run --no-capture-output -n stall \
  python tools/capture_experiment_run.py \
  --experiment-id <experiment_id> \
  --protocol-id <protocol_id> -- \
  python tools/<experiment>.py \
  --output-dir results/runs/<experiment_id>
```

包装器不经 shell 执行 wrapped argv，在启动命令前原子写入 `run_capture.json`，并
拒绝复用已有实验目录。非零退出和 `KeyboardInterrupt` 也会记录完成时间与退出码。
`--require-clean` 可用于要求启动时 Git worktree 完全干净；默认允许脏树但完整记录
`git status --porcelain`。

命令退出后可独立验证 capture：

```bash
conda run --no-capture-output -n stall \
  python tools/verify_run_capture.py \
  --capture results/runs/<experiment_id>/run_capture.json
```

`--allow-running` 只用于诊断仍标记为 running 的记录，不是运行完成证据。

运行完成后，以
`configs/run_manifests/captured_experiment.yaml.template` 为字段模板创建该实验的
YAML spec，再生成 run-local manifest：

```bash
conda run --no-capture-output -n stall \
  python tools/build_run_manifest.py \
  --spec configs/run_manifests/<experiment_id>.yaml \
  --output results/runs/<experiment_id>/run_manifest.json
```

失败且没有完整指标/输出的运行只保留 `run_capture.json`，日志写入
`logs/<experiment_id>/`；不得填写虚假 metrics 来强行生成 completed manifest。
结论级实验通过准入后，再登记到
`reports/u0_experiment_registry.csv` 并将其 manifest 提升到
`results/research_summary/run_manifests/`。
