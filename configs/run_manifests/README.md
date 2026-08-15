# Run manifest specifications

本目录保存可审查的 YAML spec；生成的带哈希 JSON 保存到
`results/runs/<experiment_id>/run_manifest.json`。通过准入并登记为结论级证据后，
manifest 才提升到 `results/research_summary/run_manifests/`。spec 定义事实，builder
负责读取实际文件并写入 `bytes` 和 `sha256`，因此 YAML 中禁止手工填写这两个字段。

## Provenance modes

- `captured`：新运行的默认模式。必须记录带时区的开始/结束时间、exact invocation，
  且 `provenance_gaps` 为空。
- `reconstructed`：只用于历史运行。必须使用 canonical reproduction commands，
  并明确列出无法从现有证据恢复的时间、命令或环境信息。

不得为了让历史实验看起来完整而伪造 exact invocation 或开始时间。

## Capture first

新运行不能在完成后手填 `captured` provenance。必须先用包装器启动；它会在命令
执行前保存 Git commit/dirty status、环境、exact argv 和带时区开始时间，在退出后
保存完成时间与 exit code：

```bash
conda run --no-capture-output -n stall \
  python tools/capture_experiment_run.py \
  --experiment-id <experiment_id> \
  --protocol-id <protocol_id> -- \
  python tools/<experiment>.py \
  --output-dir results/runs/<experiment_id>
```

输出目录固定为 `results/runs/<experiment_id>/` 且必须尚不存在。spec 中只声明
`provenance.capture_path`：

```yaml
provenance:
  capture_path: results/runs/<experiment_id>/run_capture.json
```

builder 会验证 capture 已完成、身份/协议一致、退出码与 outcome 一致，并自动把
capture 作为 `artifacts.inputs` 中的 `run_capture` 哈希；禁止手工重复登记。字段模板
见 `captured_experiment.yaml.template`。

独立验证 capture（默认要求已经完成）：

```bash
conda run --no-capture-output -n stall \
  python tools/verify_run_capture.py \
  --capture results/runs/<experiment_id>/run_capture.json
```

## Artifact groups

- `inputs`：配置、模型、参数、split 和数据 manifest；
- `intermediates`：raw shard、checkpoint 汇总或后续分析直接读取的缓存；
- `outputs`：逐视频分数、指标、验证和发布元数据。

每个 `role` 和 `path` 在一个 run 内必须唯一。路径必须相对仓库根目录，不能通过
`..` 或符号链接指向仓库外部。

## Commands

运行完成后生成 run-local JSON（默认拒绝覆盖已有文件）：

```bash
conda run --no-capture-output -n stall \
  python tools/build_run_manifest.py \
  --spec configs/run_manifests/<experiment_id>.yaml \
  --output results/runs/<experiment_id>/run_manifest.json
```

确认 JSON 可由 spec 和当前 artifact 确定性重建：

```bash
conda run --no-capture-output -n stall \
  python tools/build_run_manifest.py \
  --spec configs/run_manifests/<experiment_id>.yaml \
  --output results/runs/<experiment_id>/run_manifest.json \
  --check
```

完整验证会重新计算所有 artifact 哈希。`--skip-hashes` 只适合快速结构诊断，不能
作为发布证据：

```bash
conda run --no-capture-output -n stall \
  python tools/verify_run_manifest.py \
  --manifest results/runs/<experiment_id>/run_manifest.json
```

通用 `verify_run_manifest.py` 还要求实验已经进入 registry，因此未准入的 run-local
manifest 先用 builder 的 `--check` 验证；不能为了通过 registry 检查提前把诊断实验
登记为论文证据。
