# 锁定 U0 预计算资产

本目录保存复现锁定 U0 主实验所必需、且不能从通用 K=3 特征缓存无损推导的资产。它们属于方法协议的一部分，不能用目标数据集的 evaluation 数据重新拟合替换。

- `local/<dataset>_region1_mean.npz`：三个开发集的 Local Spatial / D2 高斯参数，以及独立 K=1 calibration 窗口分数 CDF 参考。
- `cache_overrides/annotated/WildScrape/D325.pt`：该视频的锁定 U0 第二个窗口包含帧 64；通用 K=3 缓存的帧并集未包含该帧，因此保留该条完整 DINOv3 特征覆盖。读取时仅对这一条缓存键优先使用该文件。

`D325.pt` 的 SHA256 为 `2ecdedaef2af2bc4747fa379d8d1f3f83cbba7e78cc27a059f7c4f52ca47d248`。通用缓存仍保持原样；覆盖资产的存在不会修改 packed shard 或其索引。
