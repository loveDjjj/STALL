# Release 提交清单

> 历史文档：本清单描述locked U0之前的dataset-specific score-CSV release。
> 当前发布身份与验证入口分别为`u0_locked_v1`和`tools/verify_u0_locked_release.py`。
> 本文件保留原始路径文字，作为当时提交边界的历史记录。

分支：`refactor/alpha-stalled-reproducible-release`

本清单记录 Alpha-STALLED 可复现 release commit 应提交的文件，以及应继续保持
本地或归档的文件。原则是保守提交：不在本清单或 `asset_manifest.md` 中的文件，
不要直接加入 release commit。

## 提交前检查

```bash
conda run --no-capture-output -n stall python tools/verify_alpha_stalled_release.py
conda run --no-capture-output -n stall python -m unittest tests/test_alpha_stalled_core.py
conda run --no-capture-output -n stall python -m py_compile \
  src/patch_math.py src/patch_matching.py src/create_patch_params.py \
  tools/eval_score_csv.py tools/eval_alpha_stalled.py tools/fuse_scores.py \
  tools/verify_alpha_stalled_release.py src/metrics.py \
  tests/test_alpha_stalled_core.py
```

## 应提交到 release 的文件

### 已修改跟踪文件

```text
.gitignore
README.md
```

`results/stall_repro_comparison.md` 在本次 refactor 前已经处于 dirty 状态。
如果要提交，需要单独审查 diff；不要自动并入 Alpha-STALLED cleanup commit。

### 新增配置和文档

```text
FILE_STRUCTURE.md
configs/alpha_stalled.yaml
docs/restructure/asset_manifest.md
docs/restructure/file_inventory.md
docs/restructure/release_scope.md
docs/restructure/reproducibility_audit.md
docs/restructure/release_staging_manifest.md
precomputed/README.md
research_archive/README.md
research_archive/docs/patch_stall_implementation_plan.md
scripts/reproduce/README.md
scripts/reproduce/rebuild_paper_assets.sh
scripts/ablations/README.md
```

### 主线代码

```text
src/stall_patch.py
src/dataset_utils_patch.py
src/patch_math.py
src/patch_matching.py
src/create_patch_params.py
src/eval_patch_fast.py
tools/prefill_patch_cache.py
tools/eval_alpha_stalled.py
tools/eval_score_csv.py
tools/fuse_scores.py
tools/verify_alpha_stalled_release.py
tools/inspect_dinov3_tokens.py
tools/render_manuscript_tables.js
tools/summarize_metrics_average_rows.py
tests/test_alpha_stalled_core.py
```

### 主线校准文件

```text
precomputed/patch_params_comgenvid_real_second_order_region3_bottomk0p20_v2.npz
precomputed/patch_params_videofeedback_real_same_grid_second_order_region1_mean_v2.npz
precomputed/patch_params_genvideo_real_same_grid_second_order_region2_mean_v2.npz
```

原版全局 STALL 校准文件已被跟踪：

```text
precomputed/stall_params_vatex_dino_v3.npz
```

### 论文资产和 release 结果

```text
results/README.md
results/alpha_stalled_project_manuscript_zh.md
results/alpha_stalled_full_pipeline_flow_zh.svg
results/alpha_stalled_manuscript_tables/
results/paper_scores/
results/paper_tables/
results/paper_sweeps/
```

### 研究归档代码

```text
research_archive/src/
research_archive/scripts/
research_archive/tools/
```

这些文件用于实验考古，不属于默认 Alpha-STALLED 检测器。提升规则见
`research_archive/README.md`。

## 不应提交

本地运行时资产：

```text
datasets/
cache/embeddings/
cache/patch_embeddings/
cache/patch_embedding_shards_debug/
cache/indexes/
debug_outputs/
logs/
dinov3/
```

默认不提交 patch 校准 sweep：

```text
precomputed/patch_params_*.npz
precomputed/debug_patch_params_*.npz
```

`.gitignore` 只放行三个主线 patch 校准文件，其余保持本地。

历史结果目录已从 `results/` 物理删除；如果之后重新产生类似目录，除非某个
论文表或审计明确需要，否则不要提交。

## 建议 staging 命令

审查 `results/stall_repro_comparison.md` 后，可用以下命令 staged release 文件：

```bash
git add \
  .gitignore README.md FILE_STRUCTURE.md \
  configs/alpha_stalled.yaml \
  docs/restructure \
  precomputed/README.md \
  precomputed/patch_params_comgenvid_real_second_order_region3_bottomk0p20_v2.npz \
  precomputed/patch_params_videofeedback_real_same_grid_second_order_region1_mean_v2.npz \
  precomputed/patch_params_genvideo_real_same_grid_second_order_region2_mean_v2.npz \
  research_archive \
  results/README.md \
  results/alpha_stalled_project_manuscript_zh.md \
  results/alpha_stalled_full_pipeline_flow_zh.svg \
  results/alpha_stalled_manuscript_tables \
  results/paper_scores \
  results/paper_tables \
  results/paper_sweeps \
  scripts/reproduce/README.md \
  scripts/reproduce/rebuild_paper_assets.sh \
  scripts/ablations/README.md \
  src/stall_patch.py src/dataset_utils_patch.py src/patch_math.py \
  src/patch_matching.py src/create_patch_params.py src/eval_patch_fast.py \
  tools/prefill_patch_cache.py tools/eval_alpha_stalled.py \
  tools/eval_score_csv.py tools/fuse_scores.py \
  tools/verify_alpha_stalled_release.py tools/inspect_dinov3_tokens.py \
  tools/render_manuscript_tables.js tools/summarize_metrics_average_rows.py \
  tests/test_alpha_stalled_core.py
```
