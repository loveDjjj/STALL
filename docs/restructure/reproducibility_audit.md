# 可复现性审计

分支：`refactor/alpha-stalled-reproducible-release`

当前可从已有 score CSV 复现的无泄漏 Alpha-STALLED 路径为：

```text
final_score = alpha * global_score + (1 - alpha) * patch_score
alpha = 0.60
metrics = src/metrics.py pairwise balanced AUC/AP
```

## Smoke-test 命令

ComGenVid：

```bash
conda run --no-capture-output -n stall python tools/eval_alpha_stalled.py \
  --global-csv results/paper_scores/comgenvid_global.csv \
  --patch-csv results/paper_scores/comgenvid_patch_second_order.csv \
  --alpha 0.60 \
  --output-csv /tmp/alpha_stalled_smoke/comgenvid_alpha.csv \
  --metrics-csv /tmp/alpha_stalled_smoke/comgenvid_metrics.csv
```

VideoFeedback：

```bash
conda run --no-capture-output -n stall python tools/eval_alpha_stalled.py \
  --global-csv results/paper_scores/videofeedback_global.csv \
  --patch-csv results/paper_scores/videofeedback_patch_second_order.csv \
  --alpha 0.60 \
  --output-csv /tmp/alpha_stalled_smoke/videofeedback_alpha.csv \
  --metrics-csv /tmp/alpha_stalled_smoke/videofeedback_metrics.csv
```

GenVideo：

```bash
conda run --no-capture-output -n stall python tools/eval_alpha_stalled.py \
  --global-csv results/paper_scores/genvideo_global.csv \
  --patch-csv results/paper_scores/genvideo_patch_second_order.csv \
  --alpha 0.60 \
  --output-csv /tmp/alpha_stalled_smoke/genvideo_alpha.csv \
  --metrics-csv /tmp/alpha_stalled_smoke/genvideo_metrics.csv
```

## 当前验证值

| 数据集 | Global CSV | Patch CSV | 平均 AUC | 平均 AP |
|---|---|---|---:|---:|
| ComGenVid | `results/paper_scores/comgenvid_global.csv` | `results/paper_scores/comgenvid_patch_second_order.csv` | 0.9198 | 0.9211 |
| VideoFeedback | `results/paper_scores/videofeedback_global.csv` | `results/paper_scores/videofeedback_patch_second_order.csv` | 0.8628 | 0.8750 |
| GenVideo | `results/paper_scores/genvideo_global.csv` | `results/paper_scores/genvideo_patch_second_order.csv` | 0.8374 | 0.8283 |

## 说明

- 上述数值由 `src/metrics.py` 的统一 pairwise balanced 指标路径生成。
- `tools/verify_alpha_stalled_release.py` 检查 release score CSV、metric CSV、
  alpha-sweep 汇总、配置文档和主校准文件是否存在，并确认 alpha=0.60 的
  sweep 行与主指标一致。
- `docs/restructure/release_staging_manifest.md` 记录应提交的 release 文件和
  应保持本地的运行时资产。
- `results/alpha_stalled_project_manuscript_zh.md`、
  `results/paper_tables/alpha_stalled_main_summary.md` 和
  `results/paper_tables/ablation_summary.md` 使用这些 release-baseline 数值。
  早期 raw-space、rank-space 或完整范围输出只作为历史审计，不作为主线结果。
- `tools/fuse_scores.py` 已统一调用 `src/metrics.py`；release alpha sweep 位于
  `results/paper_sweeps/`。

## 尚未覆盖的验证缺口

- 尚未从 patch cache 对三个数据集完整重跑 patch scoring；当前 release 验证
  以已有 score CSV 融合为主。
- 是否继续使用预冻结 alpha，或用独立验证协议选择 alpha，需要单独决策。
  当前 `results/paper_sweeps/` 的 best-alpha 行是对已有 score CSV 的诊断性
  oracle sweep。
- 自动化测试仍是轻量核心测试。`tests/test_alpha_stalled_core.py` 覆盖融合、
  CSV 一对一合并拒绝、统一指标入口、分数方向、同网格二阶特征、patch region
  pooling、bottom-k 聚合和经验百分位校准。更重的集成测试还包括：从磁盘
  patch cache 打分、DINOv3 特征提取和样例视频端到端 CLI。
- 短视频来源覆盖尚未补全：VideoFeedback Hotshot-XL 以及 GenVideo
  HotShot/MoonValley 的 patch 分数不属于当前 release baseline。状态见
  `results/paper_tables/patch_coverage_gaps.md`。
