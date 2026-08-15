# 脚本入口与生命周期

`scripts/` 只负责可恢复的命令编排，不实现采样、特征、评分、校准或指标公式。
正式算法必须位于 `src/alpha_stalled/`，单步 CLI 位于 `tools/`。脚本文件存在不代表
其输出属于当前主结果；协议和证据资格以配置、实验注册表及结果状态账本为准。

所有脚本均通过 `bash scripts/<name>.sh ...` 启动，不依赖 executable bit。新实验应
优先由 `tools/capture_experiment_run.py` 包装，输出写入唯一的
`results/runs/<experiment_id>/`，日志写入 `logs/<experiment_id>/`。

## Locked U0 复现

| 脚本 | 生命周期 | 作用 |
|---|---|---|
| `run_u0_locked_shard.sh` | `formal_release` | 三数据集 K3 raw-window scoring 和完整 shard finalization |
| `run_u0_locked_calibration_shard.sh` | `formal_release` | 锁定 600-real K1 calibration-reference scoring |
| `run_u0_locked_k1_cache_shard.sh` | `paper_evidence` | 统一 K1 direct control 的 cache scoring |

前两个脚本由 `configs/run_manifests/alpha_stalled_u0_locked.yaml` 记录。它们允许 scorer
在完整 checkpoint 后非零退出，再由 finalizer 严格核验输出；日志中的单次崩溃不能
单独证明失败或成功，最终以 shard、hash 和 release verifier 为准。

## 论文证据与扩展

| 脚本 | 生命周期 | 作用 |
|---|---|---|
| `run_second_order_ablation.sh` | `paper_evidence` | Local D1/D2 与 independent-real calibration 受控实验 |
| `run_u0_external_genvidbench_shard.sh` | `paper_evidence` | 锁定后 GenVidBench 外部确认 |
| `run_u0_injection_shard.sh` | `paper_evidence` | 合成注入响应与定位边界 |
| `run_u0_robustness_shard.sh` | `paper_evidence` | calibration/evaluation 两 split 鲁棒性评分 |
| `run_u0_robustness_split_shard.sh` | `paper_evidence` | 单 split 鲁棒性评分的调度入口 |
| `run_u0_final_validation_shard.sh` | `compatibility` | 顺序组合 injection、external、robustness；不定义独立协议或结果 |
| `run_duration_aware_23source.sh` | `coverage_extension` | duration-aware 23-source 参数、评分和可选分析 |
| `run_duration_aware_calibration_size_optimization.sh` | `coverage_extension` | real-only calibration-size select/confirm 两阶段流程 |
| `run_full_coverage_original_k1.sh` | `coverage_extension` | full-coverage Original STALL K1 双 GPU 初始运行 |
| `run_remaining_original_k1_parallel.sh` | `coverage_extension` | K1 未完成 cache/video shard 的恢复运行 |

`run_full_coverage_original_k1.sh` 与 `run_remaining_original_k1_parallel.sh` 的运行日志
默认写入 `logs/full_coverage_paper_protocol/`，机器可读分数仍写入对应 results 目录。

## 历史兼容入口

| 脚本 | 生命周期 | 作用 |
|---|---|---|
| `run_comgenvid_duration1_representative.sh` | `historical_frozen` | pre-U0 ComGenVid region3/bottom20 的 1 秒 patch-only 代表实验 |

该历史脚本使用 `configs/alpha_stalled.yaml` 的 dataset-specific 配置，不能作为当前
region1/mean U0 或 duration-aware 23-source 的复现入口。

## 子目录

| 路径 | 边界 |
|---|---|
| `experiments/README.md` | 已拒绝或历史专项 launcher；新实验不得照搬其目录命名 |
| `reproduce/README.md` | 当前 locked 验证入口及 pre-U0 paper-score 兼容重建 |
| `ablations/README.md` | 当前消融证据和 legacy score-CSV 指标复算说明 |

## 维护约束

1. 每个顶层 shell 脚本必须在本索引中出现。
2. 子目录脚本必须在该目录 README 中出现。
3. shell 脚本必须使用 Bash shebang、`set -euo pipefail` 并通过 `bash -n`。
4. 新脚本不得把 `.log` 写入 `results/`，不得通过目录名暗示准入成功。
5. 改变 split、窗口、分支、校准或指标时必须创建新 protocol ID，不能静默复用 U0。
