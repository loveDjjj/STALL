# 运行入口

所有运行从 `configs/benchmark.yaml` 开始。Shell 脚本只声明实验意图和显式
override；算法、校准、评分、指标和产物写入都在 `src/`。

| 脚本 | 用途 |
|---|---|
| `run_experiment.py` | 唯一 Python runner；合并 `--set` 覆盖并写入运行产物 |
| `build_manifest.py` | 从原始视频目录生成包含采样帧索引的 manifest |
| `rebuild_patch_cache.py` | 审计并分批、可续跑地构建严格 DINOv3 Global+patch 缓存 |
| `run_cache_rebuild.sh` | 使用 `stall` 环境启动当前全部 manifest 的 K=3 均匀窗口缓存重建 |
| `wait_for_cache_gpu.sh` | 等待 GPU 0 空闲达到阈值后安全启动缓存重建 |
| `run_alpha_stall.sh` | Alpha STALL 默认方法 |
| `run_ablation.sh` | 结构与时间覆盖消融 |
| `run_calibration.sh` | 校准种子与样本量实验 |
| `run_external.sh` | 冻结方法的外部评测 |
| `freeze_release.sh` | 将一个已完成 run 冻结为正式发布版本 |

脚本默认通过 `conda run -n stall` 执行。先用 `--dry-run` 检查最终配置；不传参数时
runner 从严格缓存执行完整主方法，导入已有外部方法分数时传入 `--scores-csv <path>`。
每次运行都会在 `results/runs/<run-name>/` 保存
`resolved_config.yaml` 和 `run_manifest.json`。

缓存重建默认写入 `cache/patch_embeddings_k3_2s_8fps/`，不会复用旧的
`cache/patch_embeddings/`。执行前可使用：

```bash
bash scripts/run_cache_rebuild.sh --audit-only
```

确认显存空闲后再实际启动。缺少 `2_sec_idxs` 的短视频会写入
`cache/patch_embeddings_k3_2s_8fps/audit/short_or_ineligible_videos.csv`；它们不会被
伪装成缓存失败，也不会写入不能参与 2 秒评测的特征。缓存只保存 K=3 的首、中、末
均匀窗口帧并去重，因此 K=1/K=2/K=3 均可复用同一严格缓存；超过 K=3 的配置会被
runner 明确拒绝。

原文 STALL 不在本仓库运行。请使用官方仓库完成基线推理，再把其标准化逐视频分数
作为外部结果交给论文汇总流程。
