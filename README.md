# Alpha-STALLED / STALL

本仓库基于 STALL 官方实现整理而来，用于复现 Alpha-STALLED：在原版全局
STALL 的基础上，引入 DINOv3 patch token 的同网格二阶时序似然，并用冻结
的全局--局部分数融合完成训练自由生成视频检测。

原论文：

```text
Training-free Detection of Generated Videos via Spatial-Temporal Likelihoods
CVPR 2026
arXiv: https://arxiv.org/abs/2603.15026
```

当前 release 主线为：

```text
原版 STALL 全局分支
+ DINOv3 patch 同网格二阶时序分支
+ 固定协议的 global/local 标量融合
```

分数方向统一为：分数越高，视频越接近真实视频。主方法推理时不使用测试批次
rank、真实/生成标签、生成器身份或来源路由。

## 当前仓库结构

| 路径 | 作用 |
|---|---|
| `FILE_STRUCTURE.md` | 全局文件架构、功能代码职责和提交边界说明 |
| `configs/alpha_stalled.yaml` | Alpha-STALLED 冻结配置和结果路径 |
| `src/stall.py`, `src/eval.py` | 原版 STALL 全局分支 |
| `src/stall_patch.py` | DINOv3 全局 token 与 patch token 提取 |
| `src/create_patch_params.py` | patch 分支白化和真实视频百分位校准 |
| `src/eval_patch_fast.py` | 基于 patch cache 的同网格二阶时序打分 |
| `tools/eval_alpha_stalled.py` | 全局分数与局部分数融合并计算论文指标 |
| `tools/eval_score_csv.py` | 对已有 score CSV 计算统一 AUC/AP 指标 |
| `tools/fuse_scores.py` | 固定 alpha sweep |
| `tools/verify_alpha_stalled_release.py` | release 资产一致性检查 |
| `scripts/reproduce/rebuild_paper_assets.sh` | 从 `results/paper_scores/` 重建论文表格 |
| `results/README.md` | `results/` 下逐视频分数、指标表、sweep 和图表资产说明 |
| `results/paper_scores/` | release 自包含逐视频分数 |
| `results/paper_tables/` | 主实验、消融和覆盖缺口表 |
| `results/paper_sweeps/` | alpha sweep 结果 |
| `results/paper_sensitivity/` | 期刊版敏感性、bootstrap CI 和逐生成器差值分析 |
| `results/paper_figures/` | 期刊版补充分析图 |
| `results/journal_experiments/` | 需要 patch cache 的期刊补充实跑实验轻量结果 |
| `research_archive/` | 非主线探索代码和历史脚本 |

`results/` 已被精简，只保留论文/release 必需资产。历史调参、fallback、
source/rank selector、persistence 等诊断结果不属于当前主线。各类结果文件的
列含义和使用边界见 `results/README.md`。

## 环境安装

建议使用仓库自带环境名 `stall`：

```bash
conda env create -f environment.yml
conda activate stall
```

安装 PyTorch 时按本机 CUDA 版本选择命令。例如 CUDA 12.4：

```bash
conda run --no-capture-output -n stall python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
```

如果只使用 `--hf-dataset OmerXYZ/comgenvid` 的预计算 embedding 路径，可以不
安装 DINOv3。若要从本地视频提特征，需要准备 DINOv3：

```bash
git clone https://github.com/facebookresearch/dinov3 dinov3
mkdir -p dinov3/weights
```

下载 `ViT-L/16 distilled` 权重
`dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth`，放到：

```text
dinov3/weights/dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth
```

如需使用其他位置：

```bash
export DINO_V3_REPO_DIR=/path/to/dinov3
export DINO_V3_WEIGHTS=/path/to/dinov3/weights/dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth
```

## 快速复现 release 结果

如果 `results/paper_scores/` 中的全局和 patch 分数已存在，直接重建融合
分数、指标表和 alpha sweep：

```bash
bash scripts/reproduce/rebuild_paper_assets.sh
```

单独融合一个数据集：

```bash
conda run --no-capture-output -n stall python tools/eval_alpha_stalled.py \
  --global-csv results/paper_scores/comgenvid_global.csv \
  --patch-csv results/paper_scores/comgenvid_patch_second_order.csv \
  --patch-score-col patch_final_score \
  --alpha 0.60 \
  --output-csv results/paper_scores/comgenvid_alpha_stalled.csv \
  --metrics-csv results/paper_tables/comgenvid_alpha_stalled_metrics.csv
```

对已有分数 CSV 计算统一 pairwise balanced 指标：

```bash
conda run --no-capture-output -n stall python tools/eval_score_csv.py \
  --csv results/paper_scores/comgenvid_patch_second_order.csv \
  --score-col patch_final_score \
  --output-csv results/paper_tables/comgenvid_patch_only_metrics.csv
```

运行一致性检查和轻量测试：

```bash
conda run --no-capture-output -n stall python tools/verify_alpha_stalled_release.py
conda run --no-capture-output -n stall python -m unittest tests/test_alpha_stalled_core.py
```

当前已验证的 release baseline：

| 数据集 | 平均 AUC | 平均 AP |
|---|---:|---:|
| ComGenVid | 0.9198 | 0.9211 |
| VideoFeedback | 0.8628 | 0.8750 |
| GenVideo | 0.8374 | 0.8283 |

VideoFeedback 当前覆盖 10 个 2 秒来源；GenVideo 当前覆盖 8 个 2 秒来源。
短视频来源的 patch 覆盖缺口见 `results/paper_tables/patch_coverage_gaps.md`。

## 从本地视频重新生成分数

第一步，建立索引 CSV：

```bash
mkdir -p cache/indexes
conda run --no-capture-output -n stall python src/video_index.py \
  --real-dir datasets/<dataset>/real/ \
  --fake-dir datasets/<dataset>/fake/ \
  --output cache/indexes/<dataset>.csv
```

第二步，计算原版 STALL 全局分数：

```bash
mkdir -p cache/embeddings/<dataset> results/paper_scores
conda run --no-capture-output -n stall python src/eval.py \
  --csv cache/indexes/<dataset>.csv \
  --emb-cache cache/embeddings/<dataset>/ \
  --output-csv results/paper_scores/<dataset>_global.csv \
  --workers 8 --video-batch 8
```

第三步，预填充 patch embedding cache 并计算 patch 分数。具体参数以
`configs/alpha_stalled.yaml` 为准：

```bash
conda run --no-capture-output -n stall python tools/prefill_patch_cache.py \
  --csv cache/indexes/<dataset>.csv \
  --patch-emb-cache cache/patch_embeddings/<dataset>

conda run --no-capture-output -n stall python src/eval_patch_fast.py \
  --csv cache/indexes/<dataset>.csv \
  --patch-emb-cache cache/patch_embeddings/<dataset> \
  --patch-params precomputed/<dataset-specific-patch-params>.npz \
  --patch-temp-mode same_grid_second_order \
  --patch-spat-weight 0.10 \
  --patch-temp-weight 0.90 \
  --output-csv results/paper_scores/<dataset>_patch_second_order.csv
```

最后运行 `tools/eval_alpha_stalled.py` 融合全局和局部分数。

## 原版 STALL 入口

不使用 patch 分支时，可以直接运行原版 STALL。

HuggingFace 预计算 embedding（ComGenVid）：

```bash
conda run --no-capture-output -n stall python src/eval.py --hf-dataset OmerXYZ/comgenvid
```

本地视频目录：

```bash
conda run --no-capture-output -n stall python src/eval.py \
  --real-dir path/to/real/ \
  --fake-dir path/to/fake/
```

大数据集建议使用“索引 CSV + embedding cache”模式，以便复用缓存。

## 校准文件

原版全局 STALL 使用：

```text
precomputed/stall_params_vatex_dino_v3.npz
```

Alpha-STALLED patch 分支使用三个主线校准文件：

```text
precomputed/patch_params_comgenvid_real_second_order_region3_bottomk0p20_v2.npz
precomputed/patch_params_videofeedback_real_same_grid_second_order_region1_mean_v2.npz
precomputed/patch_params_genvideo_real_same_grid_second_order_region2_mean_v2.npz
```

这些参数只由真实视频拟合，不使用生成视频。

如需用自定义真实视频重新拟合原版 STALL 参数：

```bash
conda run --no-capture-output -n stall python src/create_params.py \
  --real-dir /path/to/real/videos/ \
  --output precomputed/my_params.npz
```

## 主要文档

| 文档 | 说明 |
|---|---|
| `FILE_STRUCTURE.md` | 仓库目录、功能代码、结果资产和提交边界总览 |
| `results/README.md` | `results/` 数据文件、指标表和图表资产说明 |
| `results/alpha_stalled_project_manuscript_zh.md` | 中文项目手稿和审计说明 |
| `docs/restructure/release_scope.md` | release 范围 |
| `docs/restructure/asset_manifest.md` | 资产保留/排除清单 |
| `docs/restructure/reproducibility_audit.md` | 可复现性审计 |
| `docs/restructure/release_staging_manifest.md` | 提交清单 |
| `scripts/reproduce/README.md` | 复现入口 |
| `scripts/ablations/README.md` | 消融复现命令 |

## 引用

如果使用原版 STALL，请引用：

```bibtex
@inproceedings{hayun2026trainingfreedetectiongeneratedvideos,
  title     = {Training-free Detection of Generated Videos via Spatial-Temporal Likelihoods},
  author    = {{Ben Hayun}, Omer and Betser, Roy and Levi, Meir Yossef and Kassel, Levi and Gilboa, Guy},
  booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  year      = {2026},
  eprint    = {2603.15026},
  archivePrefix = {arXiv},
  primaryClass  = {cs.CV},
  url       = {https://arxiv.org/abs/2603.15026},
}
```
