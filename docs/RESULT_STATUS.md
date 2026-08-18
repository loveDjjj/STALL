# Alpha-STALLED 结果状态账本

本文档只管理当前仍保留的结论边界。已删除的探索方向、失败原因和替代决定见
`docs/EXPLORATION_LOG.md`。

## 状态定义

| 状态 | 含义 |
|---|---|
| `main_method_locked` | 当前正式方法，可进入主表 |
| `main_baseline` | 正式受控基线 |
| `valid_direct_control` | 只改变声明因素的因果对照 |
| `valid_ablation` | 协议有效的组件/机制消融 |
| `external_confirmation` | 锁定后的外部确认，不回调主方法 |
| `diagnostic_only` | 机制、敏感性或部署边界，不作主性能声明 |
| `invalid_leakage` | 校准或选择使用了评测身份，禁止用于性能声明 |

## 当前正式链条

| 结果 | Macro AUC/AP_real | 状态 | 允许结论 |
|---|---:|---|---|
| Original STALL，固定 strict-20 交集 | 0.8388/0.8428 | `main_baseline` | 当前固定交集上的 Global 基线 |
| Unified K1 region1/mean | 0.8632/0.8636 | `valid_direct_control` | 与 K3 比较窗口覆盖贡献 |
| Locked Alpha-STALLED U0 K3 | 0.8741/0.8723 | `main_method_locked` | 当前论文正式方法 |

U0 相对 Original STALL 的 Macro AUC/AP_real 增益为 `+0.0353/+0.0295`；
相对统一 K1 的增益为 `+0.0109/+0.0087`。完整逐视频分数和窗口身份只保留在
`release/u0/`，轻量汇总位于 `results/research_summary/`。

## 核心机制与外部确认

| 实验 | 结果 | 状态 | 结论边界 |
|---|---:|---|---|
| K3 Global-only | 0.8485/0.8486 | `valid_ablation` | Local 的增量对照 |
| K3 Local-only | 0.8357/0.8292 | `valid_ablation` | Global 的增量对照 |
| Local D1-only | 0.8237/0.8171 | `valid_ablation` | 一阶局部时序对照 |
| Local D2-only | 0.8323/0.8281 | `valid_ablation` | 二阶局部时序对照 |
| Full model with Local D1 | 0.8721/0.8695 | `valid_direct_control` | 完整一阶版本 |
| GenVidBench locked external | 0.8410/0.8495 | `external_confirmation` | 新域真实校准下的外部确认 |

D1/D2、校准稳定性和外部确认的协议说明保留在 `reports/` 对应报告中；其
原始逐窗口输出不属于 Git 资产。

## 不能复用的结论

- 含评测真实视频进入校准的结果属于泄漏，不能用于模型选择或性能声明。
- dataset-specific region/aggregation、pre-U0 U0、K5/all-window、bottom-k、
  residual、multiscale、D3/D4、OAS、routing 和 injection 等方向不属于当前主文。
- robustness、duration-aware 23-source 和全覆盖协议是部署/覆盖诊断，不替代
  strict-20 U0。
- 未在完全相同协议下重跑的外部检测器不得写成全面 SOTA。

所有新增结论必须同步更新 `reports/u0_experiment_registry.csv`、
`results/research_summary/experiment_metrics.csv` 和对应报告，并运行：

```bash
conda run --no-capture-output -n stall \
  python tools/verify_experiment_registry.py
```
