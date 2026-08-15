# U0 External GenVidBench

本目录对应实验 `u0_external_genvidbench` 和协议
`u0_external_genvidbench_v1`。它冻结 U0 在外部 GenVidBench 上的确认性评测资产，
不参与 strict-20 主方法选择，也不替代 `release/u0/`。

| 资产 | 作用 |
|---|---|
| `calibration_manifest.json` | 外部评测使用的真实校准成员 |
| `evaluation_manifest.json` | 外部评测成员 |
| `frame_indices.json` | 确定性窗口帧索引 |
| `params/region1_mean.npz` | 仅由真实校准拟合的 Local 参数 |
| `local_params_metadata.json` | Local参数身份与拟合元数据 |
| `data_audit.json` | 数据数量、交集和可用性审计 |

结果、限制和相对 STALL 的配对比较见
`reports/u0_locked_external_validation.md`；机器可读实验身份见
`reports/u0_experiment_registry.csv`。新增外部数据集必须新建独立 release 目录，
不得向本目录追加来源不明的临时资产。
