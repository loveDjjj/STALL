# Alpha STALL 实验主干

本仓库只维护我们的 `Alpha STALL` 方法。原文 `STALL` 基线应从官方
代码单独运行；其结果作为外部基线导入论文汇总，绝不在这里维护兼容复现代码。
所有日常实验均从一份基础配置启动，实验差异由 shell 脚本传入的显式参数表达，
不再按某个主实验或历史版本复制代码。

## 快速开始

所有命令使用 Conda 环境 `stall`：

```bash
bash scripts/run_alpha_stall.sh --dry-run
bash scripts/run_ablation.sh local_d1_refit --dry-run
```

`--dry-run` 只验证最终配置并打印执行计划，不会创建结果目录、读取缓存或启动计算。
正式运行一开始就会在 `results/runs/<run-name>/` 写入 `resolved_config.yaml`、
`run_manifest.json`、`progress.json`、`command.txt` 和 `logs/run.log`；终端输出会同步到日志。

正常主实验会从严格 DINOv3 特征缓存读取完整 8 FPS 下采样序列，确定性选择 K=1/K=3
窗口，以 calibration real 拟合参数与两级 CDF，再生成逐窗口分数、逐视频分数、pooled
数据集指标、论文配对宏平均指标、生成器指标和 bootstrap 对比：

```bash
bash scripts/run_cache_rebuild.sh
bash scripts/run_alpha_stall.sh
```

缓存未建立或目标 GPU 空闲显存不足 12GiB 时，主实验会明确停止，不会复用 legacy
缓存或抢占其他 GPU 任务。少于 16 帧的短视频按基础配置一致排除，并被记录在 run
manifest。`--scores-csv` 只用于导入原文 STALL 官方代码等外部方法的
既有逐视频分数：

```bash
bash scripts/run_alpha_stall.sh \
  --scores-csv path/to/video_scores.csv
```

逐视频分数必须具有 `video_id`、`dataset`、`subset`、`source_model` 和
`final_score` 字段。主方法的全部流程都在同一 runner 中执行，不再新建按实验命名的
算法工具脚本。

## 目录职责

| 路径 | 职责 |
|---|---|
| `configs/benchmark.yaml` | 唯一基础配置；每个字段都有中文说明 |
| `scripts/` | Conda 启动脚本和唯一 Python CLI，不放算法实现 |
| `src/` | Alpha STALL 源码；根目录放执行与算法主链，子目录按数据、分支和评测组织 |
| `data/manifests/` | 开发与外部数据集的校准/评测身份清单，不存放视频 |
| `cache/patch_embeddings_k3_2s_8fps/` | 严格、可验证且 K=1/K=2/K=3 共用的 Global+patch 特征缓存 |
| `datasets/` | 原始视频数据；仅保留当前开发和外部评测所需子集 |
| `results/runs/` | 每次运行的可追溯结果 |
| `release/` | 已确认实验的冻结版本 |
| `logs/` | 监督与概略修复记录；原始运行日志仍在 `results/runs/*/logs/` |
| `analysis/` | 已完成运行的结果解读与跨实验结论 |

## 运行命名

运行名必须包含实验意图，例如 `alpha_stall`、`alpha_stall_local_d1`、
`alpha_stall_global_k1`。结果目录不可默认覆盖；需要明确传入 `--overwrite`。

`release/` 与 `results/` 不重复：前者是已经确认且冻结的版本身份，后者是每次
候选、消融、重复和汇总运行的完整过程记录。
