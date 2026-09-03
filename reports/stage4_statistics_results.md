# Stage 4：高维 D2 统计模型实验结果

> 完成日期：2026-09-03  
> 实现提交：`005f4d3`  
> 数据范围：三个开发数据集固定 21,421 条可用视频

## 1. 受控设计

| ID | Covariance | 参数选择 |
|---|---|---|
| S0 | empirical | 当前正式 D2 基线 |
| S1 | Ledoit-Wolf | calibration real closed-form shrinkage |
| S2 | OAS | calibration real closed-form shrinkage |

三种方法共享 same-grid normalized vector D2、最多 300,000 个 real-only 拟合行、K=3、Global、CDF 和融合。S1/S2 通过一次 cache 扫描共同评分。

## 2. 最终结果

| Variant | Macro-3 AUC | Macro-3 AP | Delta vs S0 |
|---|---:|---:|---:|
| S0 empirical | 0.8740587 | 0.8729366 | - |
| S1 Ledoit-Wolf | 0.8740482 | 0.8729792 | -0.0000106/+0.0000426 |
| S2 OAS | 0.8740481 | 0.8729792 | -0.0000106/+0.0000425 |

S1 与 S2 的 AUC/AP 差异低于 `6e-8`，在当前协议下可视为数值等价。

## 3. Shrinkage 系数

| Dataset | Ledoit-Wolf | OAS |
|---|---:|---:|
| ComGenVid | 0.0039297 | 0.0039410 |
| VideoFeedback | 0.0052984 | 0.0053113 |
| GenVideo | 0.0056409 | 0.0056330 |

系数仅约 `0.4%-0.6%`，说明 300,000 个 patch-D2 样本下经验 covariance 已较稳定。

## 4. 成对 bootstrap

按论文生成器平衡配对规则执行 1,000 次视频级成对重采样：

| Comparison | Macro AUC delta [95% CI] | Macro AP delta [95% CI] |
|---|---:|---:|
| S1 - S0 | -0.0000113 [-0.0000642,+0.0000662] | +0.0000400 [-0.0000339,+0.0001571] |
| S2 - S0 | -0.0000125 [-0.0000644,+0.0000733] | +0.0000379 [-0.0000351,+0.0001511] |

区间跨零且幅度远小于主线门槛，n=200 时 shrinkage 不构成性能改进。

## 5. 计算与存储

S1/S2 共同结果目录约 65 MB，未增加 feature cache。两候选各 21,421 条视频、无重复或非有限值；复用的 Global 分数逐视频完全一致。评分期峰值 PyTorch allocated 显存约 3.58 GiB。

## 6. 决策

**n=200 的 covariance shrinkage 不进入主方法。**

H4 尚有一个窄问题值得保留：在 25/50/100 个 calibration real 时，Ledoit-Wolf/OAS 是否能改善 empirical 的小样本稳定性。该问题放入 CAES 计划的 few-real 阶段，不阻塞时间选择主线。Student-t/kNN 依赖的低维 geometry 已在 Stage 2 大幅失败，因此继续剪枝。
