# 预计算校准文件

`precomputed/` 存放小型校准资产，用于把 DINOv3 特征统计转换成真实视频
似然百分位。这些文件只由真实校准视频生成；白化参数拟合阶段不使用生成视频。

## 主线 release 文件

| 文件 | 作用 |
|---|---|
| `stall_params_vatex_dino_v3.npz` | 原版 STALL 在 VATEX 真实视频上的全局校准 |
| `patch_params_comgenvid_real_second_order_region3_bottomk0p20_v2.npz` | ComGenVid 主线 patch 校准 |
| `patch_params_videofeedback_real_same_grid_second_order_region1_mean_v2.npz` | VideoFeedback 主线 patch 校准 |
| `patch_params_genvideo_real_same_grid_second_order_region2_mean_v2.npz` | GenVideo 已完成来源的主线 patch 校准 |

## 消融文件

其他 `patch_params_*.npz` 属于 Section 6 消融或参数 sweep 的校准资产，包括：

- ComGenVid 的 spatial、lag-1、multi-lag、motion-hard、motion-soft，以及
  second-order region/bottom-k sweep；
- VideoFeedback 和 GenVideo 的 lag-1/second-order region 与聚合方式 sweep。

这些文件默认不进入 release commit。只有当对应 score CSV 可以通过公开命令
重新生成时，才需要进一步整理。`debug_` 前缀、非 `v2` 重复版本和未使用探索
sweep 可以在记录路径映射后移动到 `research_archive/` 或继续由 `.gitignore`
排除。

当前保留/排除决策见 `docs/restructure/asset_manifest.md`。
