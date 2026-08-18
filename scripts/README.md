# 脚本入口与生命周期

`scripts/` 只负责可恢复的命令编排；算法和指标实现位于 `src/alpha_stalled/`，
单步 CLI 位于 `tools/`。当前脚本只覆盖锁定 U0、核心 D1/D2 对照和外部确认。

## 当前入口

| 脚本 | 作用 |
|---|---|
| `run_u0_locked_shard.sh` | 三数据集 K3 raw-window scoring |
| `run_u0_locked_calibration_shard.sh` | 600-real K1 calibration-reference scoring |
| `run_u0_locked_k1_cache_shard.sh` | 统一 K1 direct control 的 cache scoring |
| `run_second_order_ablation.sh` | Local D1/D2 受控对照 |
| `run_u0_external_genvidbench_shard.sh` | 锁定后的 GenVidBench 外部确认 |

新实验必须使用唯一 protocol ID 和独立输出目录。大体量 raw、日志和缓存只保留在
服务器；Git 中只登记 manifest、hash 和最终汇总。

## 约束

1. shell 脚本必须使用 Bash shebang、`set -euo pipefail` 并通过 `bash -n`。
2. 不得把日志或逐窗口结果写入 Git 追踪的 `results/`。
3. 改变 split、窗口、分支、校准或指标时必须创建新 protocol ID。
4. 已删除的探索 launcher 只在 `docs/EXPLORATION_LOG.md` 中保留结论。
