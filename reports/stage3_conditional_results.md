# Stage 3：条件真实动力学实验结果

> 完成日期：2026-09-03  
> 实现提交：`2a24c30`  
> 条件状态：same-grid 当前一阶速度范数  
> 条件边界：每数据集 calibration real 的 1/3、2/3 分位

## 1. 受控设计

| ID | Dynamics | Likelihood |
|---|---|---|
| D0 | normalized vector D2 | 无条件 Gaussian，当前正式基线 |
| D1 | normalized vector D2 | `p(D2 | current-speed bin)` |
| D2 | 4-D geometry | 无条件 Gaussian，Stage 2 T5 |
| D3 | 4-D geometry | `p(geometry | current-speed bin)` |

每个条件候选将 calibration real 的全部位置按 speed 分为 slow/medium/fast 三个等频区间。三个 bin 合计仍最多 reservoir 300,000 行，每 bin 上限 100,000，避免条件模型获得比 D0 更大的拟合预算。fake 仅用于最终评价。

## 2. 真实速度边界

| Dataset | q33 speed | q67 speed |
|---|---:|---:|
| ComGenVid | 1.8600 | 3.7854 |
| VideoFeedback | 1.9565 | 3.7770 |
| GenVideo | 1.5002 | 3.4796 |

三个数据集的边界不同，说明这是 target-real-conditioned 模型，不是 universal motion partition。边界由每次 run 保存的 NPZ 直接读取，未使用 evaluation 或 fake。

## 3. 最终融合结果

| Variant | ComGenVid AUC/AP | VideoFeedback AUC/AP | GenVideo AUC/AP | Macro-3 AUC/AP |
|---|---:|---:|---:|---:|
| D0 unconditioned D2 | **0.9015/0.9110** | 0.8621/0.8687 | 0.8586/0.8391 | **0.8741/0.8729** |
| D1 conditioned D2 | 0.8887/0.9013 | 0.8598/**0.8737** | **0.8597/0.8446** | 0.8694/0.8732 |
| D2 unconditioned geometry | 0.8313/0.8383 | **0.8074/0.8304** | 0.5866/0.6323 | 0.7418/0.7670 |
| D3 conditioned geometry | **0.8421/0.8477** | 0.7623/0.7910 | **0.6611/0.6940** | **0.7552/0.7776** |

D1 相对 D0 的 Macro AUC/AP 为 `-0.00466/+0.00025`，AUC 只在 GenVideo 提升，AP 在 GenVideo 与 VideoFeedback 提升；没有指标同时满足 Macro `+0.005` 和至少 2/3 数据集提升。

D3 相对 D2 的 Macro AUC/AP 为 `+0.01342/+0.01059`，且 ComGenVid、GenVideo 同向，因此通过“conditional vs unconditional geometry”门槛。但 D3 相对真正可用的 D0 仍低 `-0.11888/-0.09537`，不构成方法候选。

## 4. Local-only 结果

| Variant | Local Macro AUC | Local Macro AP |
|---|---:|---:|
| D0 unconditioned D2 | **0.8270** | **0.8229** |
| D1 conditioned D2 | 0.7715 | 0.7833 |
| D2 unconditioned geometry | 0.4411 | **0.5026** |
| D3 conditioned geometry | 0.4632 | 0.5011 |

D1 的 Local-only AUC/AP 明显下降，进一步说明最终 AP 的微小变化来自与固定 Global 的融合交互，不是条件 Local 本身更强。

## 5. 成对 bootstrap

按论文生成器平衡配对规则执行 1,000 次成对重采样：

| Comparison | Macro AUC delta [95% CI] | Macro AP delta [95% CI] |
|---|---:|---:|
| D1 - D0 | -0.00461 [-0.00615, -0.00301] | +0.00019 [-0.00188, +0.00219] |
| D3 - D2 | +0.01350 [+0.00858, +0.01858] | +0.01044 [+0.00562, +0.01549] |
| D3 - D0 | -0.11879 [-0.12549, -0.11271] | -0.09488 [-0.10193, -0.08861] |

条件化对 geometry 的改善具有统计支持，但无法弥补低维 descriptor 丢失的 D2 特征方向信息。对 vector D2，条件化使 AUC 显著下降且 AP 区间跨零。

## 6. 部署指标

D1 在 GenVideo pooled AP 和 0.1% FPR recall 上略有改善，但在 ComGenVid、VideoFeedback 的低 FPR 指标没有一致收益。D3 的低 FPR 表现同样缺乏跨数据集一致性。不能以单个域/指标选择 conditional candidate。

## 7. 计算与显存

两候选共同评分：

- ComGenVid：977.9 s；
- VideoFeedback：365.7 s；
- GenVideo：3,054.3 s；
- 合计约 73.3 分钟；结果目录约 79 MB。

初版条件实现曾在连续三个 1024-D bin 的 eigendecomposition/GEMM 后累积 CUDA allocator cache。正式 run 增加逐 bin 临时张量删除与 `empty_cache`，并使用 batch=16；run manifest 记录的评分期 PyTorch allocated 峰值为 2.63 GiB，GPU 1 最终稳定完成。三数据集、两候选的边界、逐 bin `mean/W`、窗口参考、calibration IDs 和 SHA256 均已保存。

## 8. 决策

**H3 获得机制性部分支持，但作为主方法被否定。**

条件化能够改善本来很弱的低维 geometry，说明 `p(dynamics|motion state)` 在统计上并非无效；然而它没有改善当前最强的 vector D2，且引入三倍高维 covariance、target-specific speed boundaries 和更高计算复杂度。因此：

1. 不把 conditional likelihood 加入正式方法。
2. 不 sweep bin 数、不尝试 neural conditional density。
3. 不以 D3-vs-D2 的正增益包装主创新，因为两者绝对性能均远低于 D0。
4. Stage 4 保留 vector D2 的 Ledoit-Wolf/OAS 对照，检验高维统计稳定性。
5. Student-t/kNN 原计划依赖低维 geometry；该表示在 n=200 已大幅失败，因此按 gate 剪枝，不再消耗全量 cache 运行。
