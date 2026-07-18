# Alpha-STALLED 资产清单

本清单区分论文主线资产、消融资产和本地/归档资产。路径是当前 release
判断的依据。

## 论文主线校准文件

| 范围 | 路径 | 作用 | 是否进入 release |
|---|---|---|---|
| Global STALL | `precomputed/stall_params_vatex_dino_v3.npz` | 原版 STALL 的 VATEX 真实视频白化和百分位校准 | 是 |
| ComGenVid patch | `precomputed/patch_params_comgenvid_real_second_order_region3_bottomk0p20_v2.npz` | ComGenVid 同网格二阶 patch 主线校准 | 是 |
| VideoFeedback patch | `precomputed/patch_params_videofeedback_real_same_grid_second_order_region1_mean_v2.npz` | VideoFeedback 同网格二阶 patch 主线校准 | 是 |
| GenVideo patch | `precomputed/patch_params_genvideo_real_same_grid_second_order_region2_mean_v2.npz` | GenVideo 已完成来源的同网格二阶 patch 主线校准 | 是 |

## 论文主线分数文件

release 只暴露 `results/paper_scores/` 和 `results/paper_tables/` 下的自包含
副本，不依赖已删除的历史结果目录。

| 数据集 | Global 输入 | Patch 输入 | 融合分数 | 指标表 |
|---|---|---|---|---|
| ComGenVid | `results/paper_scores/comgenvid_global.csv` | `results/paper_scores/comgenvid_patch_second_order.csv` | `results/paper_scores/comgenvid_alpha_stalled.csv` | `results/paper_tables/comgenvid_alpha_stalled_metrics.csv` |
| VideoFeedback | `results/paper_scores/videofeedback_global.csv` | `results/paper_scores/videofeedback_patch_second_order.csv` | `results/paper_scores/videofeedback_alpha_stalled.csv` | `results/paper_tables/videofeedback_alpha_stalled_metrics.csv` |
| GenVideo | `results/paper_scores/genvideo_global.csv` | `results/paper_scores/genvideo_patch_second_order.csv` | `results/paper_scores/genvideo_alpha_stalled.csv` | `results/paper_tables/genvideo_alpha_stalled_metrics.csv` |

## 必需消融资产

| 消融 | 数据集范围 | Score CSV | 指标列 |
|---|---|---|---|
| 组件：global only | ComGenVid, VideoFeedback, GenVideo | `results/paper_scores/*_global.csv` | `final_score` |
| 组件：patch only | ComGenVid, VideoFeedback, GenVideo | `results/paper_scores/*_patch_second_order.csv` | `patch_final_score` |
| 组件：global + patch | ComGenVid, VideoFeedback, GenVideo | `results/paper_scores/*_alpha_stalled.csv` | `final_score` |
| 局部空间 | ComGenVid | `results/paper_scores/comgenvid_patch_spatial.csv` | `patch_final_score` |
| 同网格 lag-1 | ComGenVid | `results/paper_scores/comgenvid_patch_lag1.csv` | `patch_final_score` |
| Multi-lag | ComGenVid | `results/paper_scores/comgenvid_patch_multilag.csv` | `patch_final_score` |
| Motion-hard | ComGenVid | `results/paper_scores/comgenvid_patch_motionhard.csv` | `patch_final_score` |
| Motion-soft | ComGenVid | `results/paper_scores/comgenvid_patch_motionsoft.csv` | `patch_final_score` |
| 同网格二阶 | ComGenVid | `results/paper_scores/comgenvid_patch_second_order_ablation.csv` | `patch_final_score` |
| 跨数据集 patch 分数 | VideoFeedback, GenVideo | `results/paper_scores/*_patch_second_order.csv` | `patch_final_score` |
| 固定 alpha sweep | ComGenVid, VideoFeedback, GenVideo | `results/paper_sweeps/` | `final_score` |
| 短视频 patch 覆盖缺口 | VideoFeedback, GenVideo | `results/paper_tables/patch_coverage_gaps.md` | 不适用 |

## 非 release 资产

以下内容只应保留在本地或归档目录中，默认不进入 release：

- `datasets/` 下的原始视频数据；
- DINOv3 权重和本地 clone；
- `cache/embeddings/` 与 `cache/patch_embeddings/`；
- debug shard、日志和一次性运行输出；
- 依赖测试标签、生成器来源或测试批次 rank 的 persistence/fallback/selector 输出。
