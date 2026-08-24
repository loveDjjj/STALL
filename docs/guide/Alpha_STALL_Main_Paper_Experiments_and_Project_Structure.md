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
| 重拟合控制矩阵 | `run_refit_control_matrix.sh` | `local_d2_refit` 与 `full_d2_k3_refit`，为 D1/D2、K1/K3 提供同协议对照 |
| Spatial 控制矩阵 | `run_spatial_control_matrix.sh` | Spatial-only、Local Spatial+D2、完整模型无 Spatial 和同协议 Global-only |
| 最终无 Spatial 矩阵 | `run_final_no_spatial_matrix.sh` | 为 Global+D2-only 正式方法补齐 Full D1 与 K=1 对照 |
| Local 与时间覆盖归因 | `run_ablation.sh` | Local 是否启用、`sampling.num_windows`；K=1 无锁定帧协议时必须标为 `refit` |
| 校准稳定性与样本量 | `run_calibration.sh` | 校准 seed 与真实视频数量 |
| 外部泛化 | `run_external.sh` | 外部数据集身份及其真实校准数据 |
| 外部控制矩阵 | `run_external_control_matrix.sh` | GenVidBench 最终 K3、同方法 K1 和 Global-only K3 |

每次运行输出到 `results/runs/<run-name>/`。论文表格从这些标准化 CSV 汇总，
而不是从按实验命名的源码或历史工具中读取。

完成全部 refit 时序、窗口与 Spatial 控制实验后，运行
`python scripts/build_ablation_refit_comparisons.py --overwrite`。它不重跑
特征和评分，只对齐既有 `video_scores.csv`，将差值与 AUC bootstrap 写入
`results/runs/ablation_refit_comparisons/`。

## 执行顺序

1. 先确认完整 Alpha STALL 方法在三个开发数据集都能跑通。
2. 执行 Global-only、Local 结构和 D1/D2 消融，确定最终 Local 定义。锁定 D2 结果用于历史主协议；`local_d2_refit` 与 `local_d1_refit` 才构成当前代码的受控比较。
3. 执行 K=1/K=3 因子实验。完整 K=3 锁定主实验用于论文主表；`full_d2_k3_refit` 与 `full_k1_refit` 才构成当前代码的受控窗口数比较。
4. Spatial factorial 已确认 Spatial 稳定降低 Macro 指标；正式方法固定为 Global+D2-only。
5. 执行 `run_final_no_spatial_matrix.sh`，重新建立无 Spatial 方法的 D1/D2 与 K1/K3 对照。
6. 执行真实校准 split 与数量稳定性实验。
7. 冻结最终方法版本，再执行 `run_external_control_matrix.sh`；官方 STALL 基线在其官方仓库单独完成。
