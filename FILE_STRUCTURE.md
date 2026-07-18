# 文件架构说明

本文档说明当前 Alpha-STALLED/STALL release 分支中各类文件的作用、是否应提交，
以及 `results/` 中数据资产的含义。

## 顶层目录

| 路径 | 作用 | 是否提交 |
|---|---|---|
| `README.md` | 面向用户的中文快速说明、安装和复现入口 | 是 |
| `FILE_STRUCTURE.md` | 当前文件架构、功能代码和提交边界说明 | 是 |
| `configs/alpha_stalled.yaml` | Alpha-STALLED 冻结配置、分数路径和实验协议 | 是 |
| `src/` | 原版 STALL 和 Alpha-STALLED 的核心功能代码 | 是 |
| `tools/` | 复现、融合、指标、cache 和 release 检查工具 | 是 |
| `scripts/reproduce/` | 主实验复现入口 | 是 |
| `scripts/ablations/` | 消融实验复现说明 | 是 |
| `precomputed/` | 小型校准参数 | 只提交主线校准文件 |
| `results/` | 自包含论文分数、指标和图表资产 | 只提交 release 子集 |
| `docs/restructure/` | 重构审计、资产清单和提交清单 | 是 |
| `research_archive/` | 历史探索代码归档，不属于默认检测器 | 可提交作实验考古 |
| `datasets/` | 原始视频数据 | 不提交 |
| `cache/` | embedding cache、patch cache、索引 CSV | 不提交 |
| `dinov3/` | DINOv3 本地 clone 和权重 | 不提交 |
| `logs/`, `debug_outputs/` | 运行日志和 debug 输出 | 不提交 |

## 功能代码文件

### 原版 STALL 主线

| 文件 | 作用 |
|---|---|
| `src/stall.py` | DINOv3 加载、视频帧转 embedding、原版空间/时序似然和 STALL 推理 |
| `src/eval.py` | 原版 STALL 评测 CLI，支持 HuggingFace embedding、本地目录和索引 CSV |
| `src/create_params.py` | 从真实视频或预计算 embedding 拟合原版 STALL 校准参数 |
| `src/dataset_utils.py` | 原版 STALL 的数据加载、CSV 加载和 embedding cache 流程 |
| `src/video_index.py` | 扫描视频目录，生成带固定窗口帧索引的 enriched CSV |
| `src/metrics.py` | 统一 pairwise balanced AUC/AP 指标实现 |
| `src/whitening_transform.py` | 白化变换实现 |
| `src/organize_videofeedback.py` | VideoFeedback 数据目录整理工具 |

### Alpha-STALLED patch 主线

| 文件 | 作用 |
|---|---|
| `src/stall_patch.py` | 在原版 STALL 基础上提取 DINOv3 global token 和 patch token |
| `src/dataset_utils_patch.py` | patch embedding cache 的路径、预填充和加载工具 |
| `src/create_patch_params.py` | 从真实 patch cache 拟合 patch 白化参数和真实百分位校准 |
| `src/eval_patch_fast.py` | 不加载 DINOv3，直接从 patch cache 快速计算 patch-only 分数 |
| `src/patch_matching.py` | patch region pooling、同网格时序差分和历史 motion matching 辅助函数 |
| `src/patch_math.py` | `bottomk_mean`、经验百分位等轻量 numpy 函数 |

### 工具入口

| 文件 | 作用 |
|---|---|
| `tools/eval_alpha_stalled.py` | 融合 global score 和 patch score，输出 Alpha-STALLED 分数和指标 |
| `tools/eval_score_csv.py` | 对已有 score CSV 计算统一论文指标 |
| `tools/fuse_scores.py` | 扫描 alpha，生成 fixed-alpha sweep |
| `tools/verify_alpha_stalled_release.py` | 检查 release 资产是否存在且内部一致 |
| `tools/prefill_patch_cache.py` | 为 patch 分支预填充 patch embedding cache |
| `tools/inspect_dinov3_tokens.py` | 检查 DINOv3 global/patch token 输出形状 |
| `tools/render_manuscript_tables.js` | 渲染手稿表格图 |
| `tools/summarize_metrics_average_rows.py` | 汇总 metrics CSV 中的 Average 行 |

## `results/` 数据资产

`results/` 只保留自包含 release 资产，详细说明见 `results/README.md`。

- `results/paper_scores/`：逐视频 score CSV，可复算主实验和消融指标。
- `results/paper_tables/`：由 score CSV 计算出的 metrics、summary 和覆盖缺口。
- `results/paper_sweeps/`：alpha sweep 输出。
- `results/paper_sensitivity/`：期刊版补充敏感性、bootstrap CI 和逐生成器差值分析。
- `results/paper_figures/`：由现有 CSV 派生的补充分析图，SVG 为主，PNG 为预览。
- `results/journal_experiments/`：需要 patch cache 的期刊补充实跑实验轻量结果。
- `results/alpha_stalled_project_manuscript_zh.md`：中文手稿和项目审计。
- `results/alpha_stalled_full_pipeline_flow_zh.svg` 与
  `results/alpha_stalled_manuscript_tables/`：论文图表资产。

历史探索结果目录已经从 `results/` 删除，不再作为 release 数据提交。

## `.gitignore` 提交边界

当前 `.gitignore` 的策略是：

1. 默认忽略大体量或可再生成资产：`datasets/`、`cache/`、`dinov3/`、日志和
   debug 输出。
2. 默认忽略整个 `results/`，再只放行 release 需要的小型文件和目录。
3. 默认忽略 `precomputed/patch_params_*.npz` sweep 文件，再只放行三个主线
   patch 校准参数。

应提交：

- 源码、工具、测试、配置、中文说明文档；
- `results/paper_scores/`、`results/paper_tables/`、`results/paper_sweeps/`；
- 三个主线 patch 校准 `.npz` 和原版全局校准文件；
- `research_archive/` 中已经归档的探索代码，如果希望保留实验考古。

不应提交：

- 原始视频数据；
- DINOv3 repo 和权重；
- embedding/patch cache；
- 运行日志、debug 输出、临时结果；
- 未纳入 release 的 patch 校准 sweep 和历史结果目录。

## 验证命令

```bash
conda run --no-capture-output -n stall python tools/verify_alpha_stalled_release.py
conda run --no-capture-output -n stall python -m unittest tests/test_alpha_stalled_core.py
```
