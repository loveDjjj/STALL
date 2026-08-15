# Alpha-STALLED 结果状态账本

本文档只管理“某个数字能承担什么证据角色”。具体逐数据集和逐生成器结果以
对应机器可读 CSV、release 和报告为准。数值高低不能替代协议有效性判断。

## 状态定义

| 状态 | 含义 |
|---|---|
| `main_method_locked` | 当前正式方法，可进入主表 |
| `main_baseline` | 正式受控基线 |
| `valid_direct_control` | 与主方法只有声明因素不同的因果对照 |
| `valid_ablation` | 协议有效的组件/机制消融 |
| `external_confirmation` | 锁定后外部确认，不允许回调主方法 |
| `coverage_extension` | 不同覆盖协议的扩展结果，不替换主表 |
| `diagnostic_only` | 用于机制、敏感性或部署边界，不作主性能声明 |
| `rejected` | 协议有效但未过准入门槛 |
| `superseded` | 被更统一或更稳定的实现替代 |
| `invalid_leakage` | 存在样本校准泄漏，禁止用于性能声明 |

## 当前正式链条

| 结果 | Macro AUC/AP_real | 状态 | 允许结论 |
|---|---:|---|---|
| Original STALL，固定 strict-20 交集 | 0.8388/0.8428 | `main_baseline` | 当前固定交集上的正式 Global 基线 |
| Unified K1 region1/mean | 0.8632/0.8636 | `valid_direct_control` | 与 K3 比较可归因窗口覆盖 |
| Locked Alpha-STALLED U0 K3 | 0.8741/0.8723 | `main_method_locked` | 当前论文正式方法 |

U0 相对 Original STALL 的 Macro AUC/AP_real 增益为 `+0.0353/+0.0295`；
AP 95% paired-bootstrap CI 为 `[+0.0241,+0.0358]`。U0 相对 Unified K1 的
增益为 `+0.0109/+0.0087`；AP CI 为 `[+0.0057,+0.0120]`。

## 有效消融与确认

| 实验 | 结果 | 状态 | 结论边界 |
|---|---:|---|---|
| K3 Global-only | 0.8485/0.8486 | `valid_ablation` | Local 加入后 AP +0.0237 |
| K3 Local-only | 0.8357/0.8292 | `valid_ablation` | Global 加入后 AP +0.0431 |
| Local D1-only | 0.8237/0.8171 | `valid_ablation` | D1/D2 唯一变量对照 |
| Local D2-only | 0.8323/0.8281 | `valid_ablation` | 相对 D1 AP +0.0111 |
| Full model with Local D1 | 0.8721/0.8695 | `valid_direct_control` | 完整一阶版本 |
| Full model with Local D2 | 0.8741/0.8723 | `main_method_locked` | 相对 Full D1 AP +0.0028 |
| Independent calibration，3-seed mean | 0.8718/0.8684 | `diagnostic_only` | 校准 bank 稳定性，不是泄漏修复 |
| GenVidBench locked external | 0.8410/0.8495 | `external_confirmation` | 新域真实校准下的新生成器确认 |

完整 D1/D2 和独立校准结果见
`reports/second_order_and_independent_calibration.md`。

## 覆盖扩展

`duration_aware_23source` 恢复三个无有效 2 秒跨度的生成器，保存全部 45,185
条生成视频分数。其 full-coverage K3 Alpha：

| 汇总 | AUC/AP_fake/AP_real | 状态 |
|---|---:|---|
| Macro-3 | 0.8733/0.8629/0.8748 | `coverage_extension` |
| All-23 | 0.8619/0.8500/0.8618 | `coverage_extension` |

该协议含 1 秒/8 帧短视频流、不同校准规模和固定先验 AP，不能替换 strict-20
主结果。它用于回答生成器覆盖与平衡子集是否掩盖增益。

## 历史与被替代结果

| 结果 | Macro AUC/AP_real | 状态 | 原因 |
|---|---:|---|---|
| Historical clean K1 | 0.8570/0.8600 | `diagnostic_only` | 无身份泄漏，但配置曾由目标 fake 指标开发 |
| Historical dataset-specific K3 | 0.8694/0.8697 | `diagnostic_only` | region/aggregation 按目标 fake 选择 |
| Pre-release temporal-unified U0 | 0.8725/0.8722 | `superseded` | PatchSpatial 仍继承历史 dataset-specific 聚合 |
| Historical leakage-affected | 0.8737/0.8750 | `invalid_leakage` | 评测真实视频进入 patch calibration |
| 早期 ComGenVid 0.9198/0.9211 | dataset-specific | `invalid_leakage` | 属于同一历史泄漏链，不是 release baseline |

`invalid_leakage` 结果只能出现在协议审计中，不能进入摘要、主表、最佳结果或
参数选择论证。

## 已拒绝方向

| 方向 | Macro AP | 状态/原因 |
|---|---:|---|
| K=5 mean | 0.8672 | `rejected`，成本更高且低于 K3 |
| all non-overlap | 0.8695 | `rejected`，无 AP 增益且成本与时长耦合 |
| K3 Local bottom-2 | 0.8684 | `rejected`，lower-tail 放大噪声 |
| K3 Local hybrid | 0.8691 | `rejected` |
| Joint Typicality J2 | 0.8437 | `rejected` |
| Joint Typicality J3 | 0.8602 | `rejected` |
| spatial-mean residual D2 | 0.8609 | `rejected`，delta -0.0088，CI 全负 |
| fine+coarse | 0.8687 | `rejected`，未超过 fine-only |
| late+final | 0.8698 | `rejected`，增益约 +0.0002 且 CI 跨 0 |
| cross-layer min | 0.8695 | `rejected` |
| motion gate | 0.8692 | `rejected` |
| OAS covariance | 0.872342 | `rejected`，AP +0.000043，CI 跨 0 |

D3/D4 patch derivatives、multi-lag、hard/soft matching、Global raw D3、三分支和
gate/clip/cap 也未超过 same-grid D2 主线。除非建立新的预注册假设，不重复运行。

## 部署与机制诊断

- 目标域 200-real 校准的 AP 为 0.8723；单一源 off-domain 平均为 0.8489。
- 25/50/100/200-real 五 seed Macro AP 为 0.8237/0.8487/0.8628/0.8680。
- Local 对校准域和样本量的敏感性明显高于 Global。
- CRF35、重复帧、低帧率等严重扰动导致明确性能下降。
- 合成注入不支持把 Local D2 解释成通用语义伪影定位器。
- 当前未完成 D3、AEROBLADE、RIGID、ZED、T2VE、AIGVDet 在完全相同协议下的
  重跑，因此不得声称全面 SOTA。

## 更新规则

新增或修改结论级结果时必须同步：

1. `reports/u0_experiment_registry.csv` 的机器可读状态；
2. `results/research_summary/experiment_metrics.csv` 的指标行；
3. 对应实验报告中的协议、命令和准入结论；
4. 本状态账本；
5. 由固定结果生成脚本产生的论文表格。

登记后必须运行 `python tools/verify_experiment_registry.py`，确认 ID、协议、状态、
父子关系、证据路径和 locked U0 数字仍一致。`paper_status` 只描述展示角色，不能
替代本文件定义的 `status`。

禁止只修改论文数字而不保留逐视频分数、协议身份和指标生成来源。
