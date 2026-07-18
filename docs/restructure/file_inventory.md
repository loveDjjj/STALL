# 文件清单与处理原则

本清单对应当前 `refactor/alpha-stalled-reproducible-release` 分支状态。

## 已跟踪基线

仓库原始基线仍主要是官方 STALL：

- 根文件：`.gitignore`、`README.md`、`environment.yml`
- 原版核心：`src/stall.py`、`src/eval.py`、`src/metrics.py`、
  `src/create_params.py`、`src/dataset_utils.py`、`src/video_index.py`、
  `src/whitening_transform.py`
- 校准：`precomputed/stall_params_vatex_dino_v3.npz`
- demo 数据和 notebook
- 已跟踪结果说明：`results/stall_repro_comparison.md`

Alpha-STALLED 扩展主要由本次 release refactor 新增。

## 主线新增文件

| 路径 | 处理 |
|---|---|
| `configs/alpha_stalled.yaml` | 保留；冻结方法和评测协议 |
| `src/stall_patch.py` | 保留；DINOv3 global + patch token 提取 |
| `src/dataset_utils_patch.py` | 保留；patch cache 路径和加载辅助 |
| `src/patch_math.py` | 保留；patch 校准和测试共用的 numpy 数学函数 |
| `src/create_patch_params.py` | 保留；patch 白化/校准拟合 |
| `src/eval_patch_fast.py` | 保留；从 cache 计算同网格 patch 分数 |
| `src/patch_matching.py` | 保留；公开说明聚焦同网格模式，motion matching 属于消融/归档 |
| `tools/prefill_patch_cache.py` | 保留；cache 构建器 |
| `tools/eval_alpha_stalled.py` | 保留；主融合入口 |
| `tools/fuse_scores.py` | 保留；alpha sweep，使用统一指标 |
| `tools/eval_score_csv.py` | 保留；组件和局部消融指标入口 |
| `tools/verify_alpha_stalled_release.py` | 保留；release 资产 verifier |
| `tests/test_alpha_stalled_core.py` | 保留；融合、指标、分数方向和 patch math 回归测试 |
| `docs/restructure/*.md` | 保留；cleanup 和 release 审计文档 |
| `scripts/reproduce/README.md` | 保留；公开复现命令 |
| `scripts/reproduce/rebuild_paper_assets.sh` | 保留；重建 paper assets |
| `scripts/ablations/README.md` | 保留；公开消融命令 |

## 论文复现结果

| 路径 | 作用 |
|---|---|
| `results/alpha_stalled_project_manuscript_zh.md` | 当前中文手稿/审计源 |
| `results/alpha_stalled_full_pipeline_flow_zh.svg` | 方法图源文件 |
| `results/alpha_stalled_manuscript_tables/` | 手稿表格图 |
| `results/paper_scores/` | 自包含 global、patch 和 fused score CSV |
| `results/paper_tables/` | 主实验和消融 metrics CSV |
| `results/paper_tables/patch_coverage_gaps.md` | 短视频 patch 来源排除说明 |
| `results/paper_sweeps/` | 使用统一指标的 alpha sweep 汇总 |

`results/` 已被物理精简为 release 资产。历史探索结果不再保留在该目录。

## 校准文件

| 路径 | 作用 |
|---|---|
| `precomputed/stall_params_vatex_dino_v3.npz` | 原版 STALL VATEX 校准 |
| `precomputed/patch_params_comgenvid_real_second_order_region3_bottomk0p20_v2.npz` | ComGenVid 主线 patch 参数 |
| `precomputed/patch_params_videofeedback_real_same_grid_second_order_region1_mean_v2.npz` | VideoFeedback 主线 patch 参数 |
| `precomputed/patch_params_genvideo_real_same_grid_second_order_region2_mean_v2.npz` | GenVideo 主线 patch 参数 |

其他 `precomputed/patch_params_*` 文件属于消融或 sweep 资产，默认由
`.gitignore` 保持本地，不进入 release commit。

## 归档类别

| 类别 | 文件模式 |
|---|---|
| motion matching | `matching_*`, `patch_motion_*`, `patch_trajectory_*`, motion-hard/soft 脚本 |
| morphology/tail | `morphology_*`, `patch_tail_*`, `hard_tail_*`, `likelihood_tail_*` |
| frequency/neighbor | `patch_frequency_*`, `patch_neighbor_*`, temporal-frequency 脚本 |
| patchfield | `patchfield_*`, `apply_universal_patchfield_addon.py` |
| adaptive/reliability | `adaptive_fusion.py`, `reliability_fusion.py`, reliability 脚本 |
| fallback/routing | `apply_sample_fallback_rule.py`, `sample_*fallback*`, split/source routing 工具 |
| fresh/hotshot 脚手架 | `build_*runbook*`, `verify_*runbook*`, `fresh_*`, `hotshot_*` |
| table/report 生成 | 只保留最终 table renderer；一次性 report exporter 归档 |
| 早期文档 | `docs/patch_stall_implementation_plan.md` 已移到 `research_archive/docs/` |

## 本地资产

以下路径是运行时资产，不应进入常规开源提交：

- `datasets/`
- `cache/embeddings/`
- `cache/patch_embeddings/`
- `cache/patch_embedding_shards_debug/`
- `logs/`
- `debug_outputs/`
- `dinov3/` 下的本地 clone 和权重

## 当前状态

主线文件、`results/paper_scores/`、`results/paper_tables/` 和
`results/paper_sweeps/` 已就位。剩余决策主要是：是否公开更多消融校准
`.npz`，或保持本地并按需重新生成。
