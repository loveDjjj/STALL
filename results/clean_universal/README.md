# Clean Universal U0 轻量结果

本目录保存 `reports/clean_universal_cross_layer_final.md` 对应的机器可读结果。
正式方法为预声明的 U0：统一 `region=1`、temporal `mean`、DINO layer 23、
`beta=0.1`、K=3 MW2、`alpha=0.6`，Macro-3 AUC/AP 为
`0.872499/0.872243`。

主要文件：

- `universal_*`：U0-U5 与 HistoricalTuned 的数据集、生成器和 bootstrap 结果。
- `cross_layer_*`：C0-C2 指标、分组、bootstrap 与准入结论。
- `calibration_audit.csv`：每数据集 200 条独立真实校准、0 fake、0 评测重叠审计。
- `config_registry.csv`：U0 主方法、U1-U5 sensitivity 和 HistoricalTuned 的角色定义。
- `numerical_audit/`：VideoFeedback rank-1023 whitening 的 batch-shape/float64 审计。

逐窗口、逐视频分数、whitening 参数、raw shard 和 layer17 feature cache 体积较大，
只保留在本地，不进入版本控制。
