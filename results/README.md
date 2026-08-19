# 实验结果

每个执行实例使用独立目录：

```text
results/runs/<run-name>/
├── resolved_config.yaml
├── run_manifest.json
├── video_scores.csv
├── window_scores.csv
├── dataset_metrics.csv
└── generator_metrics.csv
```

`resolved_config.yaml` 是命令行覆盖后的真实配置，`run_manifest.json` 记录 Git
commit、配置哈希、方法、采样和校准协议；直接缓存运行还会记录缓存契约哈希、各数据集
校准数量与短视频排除数量。`bootstrap_metrics.csv` 在存在可比较分支时输出最终分数相对
Global-only/Local-only 的视频级 AUC 置信区间。不要把候选结果直接放进 `release/`。
