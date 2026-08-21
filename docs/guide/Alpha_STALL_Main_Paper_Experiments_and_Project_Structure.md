# Alpha STALL 实验与项目结构

## 核心原则

代码主干按职责组织，不按某个主实验组织。本仓库只支持我们的 Alpha STALL；
原文 STALL 由官方代码独立运行，并以外部基线结果参与论文比较。

唯一基础配置是 `configs/benchmark.yaml`。所有消融通过 `scripts/run_*.sh` 的
`--set` 参数完成，脚本注释必须说明本次实验目的和改变的字段。

## 论文实验映射

| 问题 | 启动方式 | 仅改变的配置 |
|---|---|---|
| Global-only / Local / 完整方法消融 | `run_ablation.sh` | Local 分支、空间证据和时序阶数；名称注明 `locked` 或 `refit` |
| 开发集核心矩阵 | `run_development_matrix.sh` | 顺序执行全部待跑核心消融，不重复完整 K=3 主实验 |
| Local 与时间覆盖归因 | `run_ablation.sh` | Local 是否启用、`sampling.num_windows`；K=1 无锁定帧协议时必须标为 `refit` |
| 校准稳定性与样本量 | `run_calibration.sh` | 校准 seed 与真实视频数量 |
| 外部泛化 | `run_external.sh` | 外部数据集身份及其真实校准数据 |

每次运行输出到 `results/runs/<run-name>/`。论文表格从这些标准化 CSV 汇总，
而不是从按实验命名的源码或历史工具中读取。

## 执行顺序

1. 先确认完整 Alpha STALL 方法在三个开发数据集都能跑通。
2. 执行 Global-only、Local 结构和 D1/D2 消融，确定最终 Local 定义。D2 消融复用锁定参数；D1 必须标为重新拟合。
3. 执行 K=1/K=3 因子实验。完整 K=3 复用锁定主实验；当前 K=1 仅作为重新拟合参考，不能声称是严格锁定的单因素对照。
4. 执行真实校准 split 与数量稳定性实验。
5. 冻结最终方法版本，再执行外部泛化；官方 STALL 基线在其官方仓库单独完成。
