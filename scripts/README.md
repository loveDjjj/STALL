# 运行入口

所有运行从 `configs/benchmark.yaml` 开始。Shell 脚本只声明实验意图和显式
override；算法、校准、评分、指标和产物写入都在 `src/`。

| 脚本 | 用途 |
|---|---|
| `run_experiment.py` | 唯一 Python runner；合并 `--set` 覆盖并写入运行产物 |
| `build_paper_tables.py` | 从已有逐视频分数增量生成论文配对宏平均表，不重跑 GPU |
| `build_manifest.py` | 从原始视频目录生成包含采样帧索引的 manifest |
| `rebuild_patch_cache.py` | 审计并分批、可续跑地构建严格 DINOv3 Global+patch 缓存 |
| `run_cache_rebuild.sh` | 使用 `stall` 环境启动当前全部 manifest 的 K=3 均匀窗口缓存重建 |
| `run_cache_pack.sh` | 将已验证的单视频 K=3 缓存迁移为 32 视频顺序 shard，并边校验边删除旧文件 |
| `wait_for_cache_gpu.sh` | 等待 GPU 0 空闲达到阈值后安全启动缓存重建 |
| `run_alpha_stall.sh` | Alpha STALL 默认方法 |
| `run_correspondence.sh` | Stage 1 C0-C3 same-grid/hard/soft/confidence 受控实验 |
| `run_trajectory.sh` | Stage 2 T1-T5 单次 cache 扫描轨迹几何矩阵；T0/Global 复用 C0 |
| `run_conditional.sh` | Stage 3 conditioned D2/geometry 单次 cache 扫描矩阵 |
| `run_ablation.sh` | 单个结构与时间覆盖消融；名称区分 locked 与 refit 协议 |
| `run_development_matrix.sh` | 顺序执行最终无 Spatial 方法的开发集核心消融矩阵 |
| `run_refit_control_matrix.sh` | 复核旧 Spatial 方法的 D1/D2、K1/K3 重拟合结果，仅作历史对照 |
| `run_spatial_control_matrix.sh` | 补齐 Local Spatial 独立能力、局部融合和最终模型必要性 factorial |
| `run_final_no_spatial_matrix.sh` | 为最终 Global+D2-only 方法补齐 Full D1 与 K=1 严格对照 |
| `build_ablation_refit_comparisons.py` | 从已完成 run 增量构建 D1/D2、K1/K3、Spatial 的配对差值和 AUC bootstrap 数据 |
| `run_calibration.sh` | 校准种子与样本量实验 |
| `run_external.sh` | 冻结方法的外部评测 |
| `run_external_control_matrix.sh` | GenVidBench 最终 K3、受控 K1 与 Global-only 外部验证 |
| `build_external_genvidbench_comparisons.py` | 对齐外部 run，生成 K3/K1 与 Final/Global 的 AUC/AP bootstrap CI |
| `freeze_release.sh` | 将一个已完成 run 冻结为正式发布版本 |

脚本默认通过 `conda run -n stall` 执行。先用 `--dry-run` 检查最终配置；它不会创建
结果目录。正式运行会在 `results/runs/<run-name>/` 立即写入 `resolved_config.yaml`、
`run_manifest.json`（状态为 `running`）、`progress.json`、`command.txt` 和
`logs/run.log`，并在结束时写入标准 CSV。终端输出与错误输出也会同时写入 `run.log`。
导入已有外部方法分数时传入 `--scores-csv <path>`。

所有 `run_*.sh` 都透传 `--dry-run`、`--overwrite` 和 `--set KEY=VALUE`。常用资源覆盖：

```bash
# 指定单卡。
bash scripts/run_alpha_stall.sh --set runtime.device=cuda:1

# 双卡并行 evaluation 评分；校准拟合仍固定在第一张卡。
bash scripts/run_alpha_stall.sh --set 'runtime.devices=[cuda:0,cuda:1]'

# 调整缓存预取吞吐，不改变采样、方法或逐窗口浮点公式。
bash scripts/run_alpha_stall.sh --set runtime.score_batch_size=16 --set runtime.cache_io_workers=4
```

Stage 1 只复用现有 Patch token 缓存并改变 Local correspondence：

```bash
bash scripts/run_correspondence.sh --dry-run
bash scripts/run_correspondence.sh --variant c1
bash scripts/run_correspondence.sh --variant c3 --set runtime.score_batch_size=8
```

正式 C0-C3 固定 `radius=1`、`temperature=0.07` 和 `spatial_penalty=0.05`。
`--radius 2` 只供 C0-C3 决策后的预注册敏感性检查，不得根据单个数据集 fake
结果选择。

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

两张 GPU 同时重建同一根缓存时，两个进程必须分别设置
`--shard-index 0 --shard-count 2` 与 `--shard-index 1 --shard-count 2`。分片按稳定缓存键
划分，因此可安全续跑且不会发生并发覆盖。

重建入口会优先使用随机帧定位；极少数 H.264 容器在目标帧边界随机定位失败时，会用
顺序解码取得相同的 manifest 帧索引，并在 `audit/sequential_decode_fallbacks_shard*.csv`
记录该降级事件。若顺序解码帧数比 manifest 声明少，会以实际帧数重算该行的降采样与
窗口索引，并在 `audit/manifest_frame_count_repairs_shard*.csv` 记录修正前后帧数；不能
静默跳过可恢复视频。

严格缓存构建完成后，可执行 `bash scripts/run_cache_pack.sh` 将多个单视频 `.pt`
迁移为按数据集、split 排列的 32 视频 shard。迁移不是有损压缩，主要减少机械盘随机
读取和文件数量；每个 shard 写入、读回校验、提交 index 后才删除对应旧文件。迁移期间
不得同时运行实验或缓存重建；先用 `--dry-run` 和 `--limit-shards 1` 验证。

原文 STALL 不在本仓库运行。请使用官方仓库完成基线推理，再把其标准化逐视频分数
作为外部结果交给论文汇总流程。

`dataset_metrics.csv` 是全量唯一视频 pooled 指标，适合检查运行和真实类别比例；它的
AP 会受类别不平衡影响，不能直接写入论文主表。论文使用每个生成器与等量真实视频配对
后再做生成器宏平均的 `pairwise_metrics.csv`。已完成 run 可直接补写该表：

```bash
conda run -n stall python scripts/build_paper_tables.py --run-name alpha_stall
```
