# 预计算校准文件

`precomputed/` 存放小型校准资产，用于把 DINOv3 特征统计转换成真实视频
似然百分位。这些文件只由真实校准视频生成；白化参数拟合阶段不使用生成视频。

## 历史与兼容文件

当前 locked U0 的 Local 参数不在本目录，而在 `release/u0/params/`；其文件路径和
SHA-256 由 `configs/alpha_stalled_u0_locked.yaml` 与
`release/u0/config_and_checkpoint_hashes.json` 锁定。下列文件服务于原版 STALL 或
pre-U0 dataset-specific score-CSV 流程，不能据此声明当前 U0 配置。

所有已提交参数文件的机器身份、NPZ数组契约和生命周期统一登记在
`configs/parameter_assets.yaml`，可读快照见 `reports/parameter_asset_inventory.md`。
使用 `tools/verify_parameter_assets.py` 验证；README中的文件名不能替代该总账。

| 文件 | 作用 |
|---|---|
| `stall_params_vatex_dino_v3.npz` | 原版 STALL 在 VATEX 真实视频上的全局校准 |
| `patch_params_comgenvid_real_second_order_region3_bottomk0p20_v2.npz` | ComGenVid 历史 region3/bottom-k patch 校准 |
| `patch_params_videofeedback_real_same_grid_second_order_region1_mean_v2.npz` | VideoFeedback 历史 region1/mean patch 校准 |
| `patch_params_genvideo_real_same_grid_second_order_region2_mean_v2.npz` | GenVideo 历史 region2/mean patch 校准 |

## 消融文件

其他 `patch_params_*.npz` 属于 Section 6 消融或参数 sweep 的校准资产，包括：

- ComGenVid 的 spatial、lag-1、multi-lag、motion-hard、motion-soft，以及
  second-order region/bottom-k sweep；
- ComGenVid 的 `same_grid_third_order` / `same_grid_fourth_order`
  temporal derivative 对照参数。对应结果已汇总到
  `results/journal_experiments/temporal_derivative_order/`，`.npz` 文件默认被
  `.gitignore` 排除，可按该目录 markdown 中的命令重建；
- VideoFeedback 和 GenVideo 的 lag-1/second-order region 与聚合方式 sweep。

这些文件默认不进入 release commit。只有当对应 score CSV 可以通过公开命令
重新生成时，才需要进一步整理。`debug_` 前缀、非 `v2` 重复版本和未使用探索
sweep 可以在记录路径映射后移动到 `research_archive/` 或继续由 `.gitignore`
排除。

该阶段的保留/排除决策见
`research_archive/docs/pre_u0_release/asset_manifest.md`；当前资产治理见
`docs/PROJECT_ORGANIZATION.md`。
