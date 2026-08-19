# 冻结发布

`release/` 只保存已经确认、不可随日常实验改写的版本身份。每个正式发布必须包含
冻结配置、数据 manifest、参数资产、输入哈希和参考指标。

由严格缓存主链生成的发布还会一并复制 `window_scores.csv` 与
`bootstrap_metrics.csv`（若该 run 生成了它们），以保留窗口级证据和统计比较；导入
外部基线 CSV 的运行则只要求基础逐视频与指标产物。

日常运行、消融和重复实验必须写入 `results/runs/`。当方法结构、采样和校准协议
全部确定后，才从某个完整 run 创建新的 `release/alpha_stall_vN/`。
