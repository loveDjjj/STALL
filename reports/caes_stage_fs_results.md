# CAES Stage FS：时间窗口选择实验结果

> 完成日期：2026-09-04  
> 数据范围：ComGenVid、VideoFeedback、GenVideo 固定 21,421 条视频  
> 基线：same-grid Local D2-only + official STALL Global + uniform K=3  
> 唯一受控变量：窗口位置

## 1. 结论

**CAES 第一轮 adaptive temporal selection 未通过预注册 Go gate，不能作为当前论文主创新。**

最优候选是无需真实参考的 `feature_change` Top-K。它把 Macro-3 AUC/AP 从
`0.8741/0.8729` 提高到 `0.8774/0.8769`，增益为 `+0.0033/+0.0040`，配对
bootstrap 区间均为正。但该增益：

- 低于预注册的 `+0.005` 门槛；
- 只来自 GenVideo，ComGenVid 与 VideoFeedback 均下降；
- 低 FPR 改善也只在 GenVideo 上一致；
- 属于已有 motion/change heuristic，而不是 CAES 的 real-only forensic signal。

更关键的是，CAES 核心候选 `real_anomaly`、`real_anomaly+NMS` 和
`stratified real_anomaly` 全部低于 uniform。因此当前证据不支持“1 FPS Global
真实异常可以在固定预算下找到更有效的 forensic windows”。

## 2. 严格协议与正确性

- FS0 直接调用历史 `uniform_windows`，完全复现正式基线。
- FS1-FS5 对 calibration real 和 evaluation 执行相同 selector。
- 每个 selector 使用自己的 matched window/video real CDF。
- Global、Local D2、DINO、K、窗口长度、融合权重和 calibration IDs 固定。
- 全量 adaptive dense 评分处理 21,421 条 evaluation 视频。
- 按需 FS0 与 630 GB strict cache 在 27 条跨数据集/来源样本、65 个窗口上：
  Global/Patch feature 与三个 raw score 的最大绝对差均为 `0.0`。

初版汇总曾让各 selector 分别构造真实视频配对。VideoFeedback 中行顺序不同造成
每个生成器 300 条 real 中有 18 条身份不一致。正式结果已改为：先按 `video_id`
对齐所有 selector，只由 FS0 构造一次 real/fake 配对，所有候选共享同一身份。
旧表保存为 `*_legacy_unmatched.csv`，不得用于论文。

## 3. 主结果

| Selector | ComGenVid | VideoFeedback | GenVideo | Macro-3 | Macro delta |
|---|---:|---:|---:|---:|---:|
| FS0 Uniform | **0.9015/0.9110** | **0.8621/0.8687** | 0.8586/0.8391 | 0.8741/0.8729 | - |
| FS1 Random | 0.8944/0.9036 | 0.8590/0.8657 | 0.8471/0.8282 | 0.8669/0.8658 | -0.0072/-0.0071 |
| FS2 Feature-change | 0.8986/0.9072 | 0.8600/0.8663 | **0.8735/0.8572** | **0.8774/0.8769** | **+0.0033/+0.0040** |
| FS3 Real-anomaly | 0.8966/0.9034 | 0.8596/0.8658 | 0.8423/0.8288 | 0.8662/0.8660 | -0.0079/-0.0069 |
| FS4 Real-anomaly + NMS | 0.8962/0.9037 | 0.8561/0.8614 | 0.8388/0.8240 | 0.8637/0.8630 | -0.0104/-0.0099 |
| FS5 Stratified anomaly | 0.8991/0.9080 | 0.8617/0.8681 | 0.8507/0.8296 | 0.8705/0.8686 | -0.0035/-0.0044 |

每格为生成器平衡的 AUC/real-positive AP。

## 4. Feature-change 的统计与域差异

| Scope | AUC delta [95% CI] | AP delta [95% CI] |
|---|---:|---:|
| ComGenVid | -0.0029 [-0.0055, +0.0001] | -0.0039 [-0.0068, -0.0009] |
| VideoFeedback | -0.0021 [-0.0025, -0.0016] | -0.0023 [-0.0029, -0.0018] |
| GenVideo | +0.0149 [+0.0113, +0.0192] | +0.0181 [+0.0139, +0.0222] |
| Macro-3 | +0.0033 [+0.0017, +0.0050] | +0.0040 [+0.0022, +0.0054] |

GenVideo 的八个生成器 AUC/AP 均同向提升，说明该域确实存在 motion/change peak
更富含 Local D2 证据的现象。但它没有跨数据集泛化，不能据此选择为统一主方法。

## 5. 分支机制

| Selector | Global Macro-3 | Local Macro-3 | Final Macro-3 |
|---|---:|---:|---:|
| Uniform | 0.8523/0.8532 | 0.8270/0.8229 | 0.8741/0.8729 |
| Feature-change | 0.8514/0.8515 | **0.8327/0.8312** | **0.8774/0.8769** |

Feature-change 相对 uniform：

- Global AUC/AP：`-0.0009/-0.0017`；
- Local AUC/AP：`+0.0057/+0.0083`；
- Final AUC/AP：`+0.0033/+0.0040`。

因此其有限收益来自 Local D2，而不是选择器直接优化了 Global STALL。另一方面，
1 FPS real-anomaly selector 同时损害 Global 与 Local，说明 coarse Global typicality
与 dense Local D2 forensic evidence 并不对齐。

## 6. 低 FPR

Feature-change 相对 uniform：

| Dataset | Fake TPR@0.1% FPR | Fake TPR@1% FPR | FPR@95% Fake TPR |
|---|---:|---:|---:|
| ComGenVid | +0.0529 | -0.0153 | +0.0200（更差） |
| VideoFeedback | -0.0030 | -0.0127 | +0.0100（更差） |
| GenVideo | +0.0115 | +0.0043 | -0.0306（更好） |

低 FPR 没有跨域一致改善，不能触发 borderline 继续条件。

## 7. 覆盖机制

三个 K≈3 数据集/子集上，uniform 的平均中心跨度最大。以 ComGenVid 为例：

| Selector | 平均中心跨度（秒） | 平均唯一 dense 帧 |
|---|---:|---:|
| Uniform | 5.66 | 44.92 |
| Feature-change | 1.90 | 27.78 |
| Real-anomaly | 1.00 | 24.01 |
| Real-anomaly + NMS | 2.00 | 31.96 |
| Stratified anomaly | 3.83 | 41.85 |

Top-K anomaly 明显聚集在局部时间段。NMS增加了间距，但不能恢复 uniform 的全局
覆盖；stratified 最接近 uniform，却仍然退化。这与 KFS-Bench 等工作的观察一致：
只提高局部 relevance 不能替代稳定 coverage。

0.5 秒候选网格让 185/21,421 条视频的普通 adaptive selector 比 FS0 少一个合法
窗口；按 selector 单独剔除这些视频后，Feature-change 的 Macro 增益仍为
`+0.00333/+0.00401`，结论不变。NMS主动减少窗口的数量更多，这是其设计行为，
也是其计算预算与效果均不占优的原因之一。

## 8. 计算成本

| Dataset | Calibration unique frames/time | Evaluation unique frames/time |
|---|---:|---:|
| ComGenVid | 12,189 / 192 s | 233,032 / 3,430 s |
| VideoFeedback | 4,324 / 33 s | 67,696 / 526 s |
| GenVideo | 17,565 / 125 s | 844,926 / 6,615 s |

全部 dense 提取与评分约 2.98 小时，最终统计约 5 分钟；score-only 结果远小于
复制一份 Patch cache。FS1-FS5共享每视频窗口并集，只做一次 DINO提取。

## 9. 决策

1. **Stage FS 判定为 No-Go。** 不继续 wavelet、DPP、bandit 或复杂 change-point。
2. 不把 Feature-change 写成最终方法；保留为重要的域依赖正对照和机制诊断。
3. Stage FS 的原始预注册决策是不继续cross-fitting；后续按研究要求仍完成了三种
   selector的5-fold控制。结果见`reports/caes_tail_crossfit_results.md`：Uniform与
   Feature-change严格不变，Real-anomaly提高约`+0.0024/+0.0023`，但仍低于Uniform。
4. 正式方法继续使用 uniform K=3。
5. 按预注册计划转入 FS0 上的 Tail/CVaR aggregation，然后检验 real-only fusion。

权威产物：

- `results/runs/caes_stage_fs/`
- `results/analysis/caes_stage_fs/`
- `results/caes/fs0_equivalence/`
