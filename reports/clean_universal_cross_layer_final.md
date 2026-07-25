# Alpha-STALLED Clean Universal 与跨层验证最终报告

日期：2026-07-24

> **发布审计勘误（2026-07-24）：** `0.8725/0.8722` 的分析器将 PatchD2
> 统一为 region1+mean，但 PatchSpatial 仍继承历史 K=3 窗口分数，因此
> ComGenVid PatchSpatial 使用 bottom-20，另外两个数据集使用 mean。以下数值
> 仍是无样本校准泄漏的 temporal-unified 实验结果，但不再称为全分支统一发布版。
> `configs/alpha_stalled_u0_locked.yaml` 会以 region1+mean 重算两个 Local 组件，
> clean-from-empty 的稳定结果为 Macro-3 `0.8741/0.8723`，已取代本报告的
> pre-release 方法状态；下方表格保留为历史消融证据。

## 1. 最终决策

本实验当时将正式主方法改为预声明的 **U0**：三个数据集统一使用 `region=1`、temporal
`mean`、DINO layer 23、`beta=0.1`、K=3 MW2 和 `alpha=0.6`。U0 在固定
21,421 个评测视频上达到 Macro-3 AUC/AP **`0.8725/0.8722`**。

旧 K=3 MW2 `0.8694/0.8697` 没有 sample-calibration leakage，但 region 和
aggregation 是根据各目标数据集 fake AUC/AP 选择的。因此它只能标记为
**Historical dataset-specific tuned baseline**，不能继续作为严格统一主方法。

U0 相对历史 tuned baseline 提升 `+0.00312 AUC / +0.00257 AP`；Macro AP 的 1,000 次
paired bootstrap 95% CI 为 **`[+0.00111,+0.00381]`**。跨层 C1 和运动门控
C2 均未通过准入，最终不增加跨层分支。

## 2. 四类结果必须分开

| 类别 | Macro-3 AUC/AP | 样本校准泄漏 | fake用于拟合/校准 | dataset-specific fake选参 | 用途 |
|---|---:|---:|---:|---:|---|
| 历史 leakage-affected B8 | 0.8737/0.8750 | 是 | 历史混合 | 是 | 无效历史审计值 |
| Historical dataset-specific tuned | 0.8694/0.8697 | 否 | 否 | **是** | 历史补充结果，不是主结果 |
| Pre-release temporal-unified U0 | **0.8725/0.8722** | 否 | 否 | **PatchSpatial 是** | 审计结果，不是锁定发布版 |
| C1/C2 跨层候选 | 0.8707/0.8695；0.8695/0.8692 | 否 | 否 | PatchSpatial 是 | 失败消融 |

这里的两个问题不同：

1. **sample-calibration leakage**：测试真实视频进入 whitening/CDF。历史 B8
   存在该问题，数值不能作为有效主结果。
2. **fake-label hyperparameter selection**：校准样本虽然独立，但使用测试 fake
   AUC/AP 为不同数据集选择 region/aggregation。`0.8694/0.8697` 属于这一类。

U0 的 whitening、window CDF、video CDF、运动阈值和跨层二次 CDF全部只使用
每数据集固定 200 条独立真实校准视频；校准 fake 数为 0，校准/评测视频交集为 0。
但 alpha、beta、K 和整体架构是在查看过这三个开发数据集的生成视频结果后冻结的；
因此这三个数据集同时承担开发与评测角色，不能作为 untouched confirmation sets。

## 3. Clean Universal U0

窗口级公式冻结为：

```text
G_k = 0.5 GlobalSpatial_k + 0.5 GlobalT1_k
T_k = CDF_real200(mean GaussianLikelihood(D2(region1(layer23))))
L_k = 0.1 PatchSpatial_k + 0.9 T_k
```

视频级仍为：

```text
G_raw = mean_k(G_k)
L_raw = mean_k(L_k)
G, L = effective-K matched real-video CDFs
S = 0.6 G + 0.4 L
```

所有数据集使用完全相同的 region、aggregation、layer 和权重。U0-U5 在实验前
已经固定；U1-U5 只作敏感性分析，不允许根据下表重新选择主配置。

| 配置 | Region | Temporal aggregation | ComGenVid | VideoFeedback | GenVideo | Macro-3 |
|---|---:|---|---:|---:|---:|---:|
| **U0** | **1** | **mean** | **0.8986/0.9092** | **0.8623/0.8684** | **0.8566/0.8391** | **0.8725/0.8722** |
| U1 | 2 | mean | 0.8951/0.9067 | 0.8521/0.8562 | 0.8601/0.8410 | 0.8691/0.8680 |
| U2 | 3 | mean | 0.8897/0.9026 | 0.8481/0.8495 | 0.8578/0.8364 | 0.8652/0.8628 |
| U3 | 1 | bottom-20% | 0.8874/0.9012 | 0.8615/0.8674 | 0.8415/0.8259 | 0.8635/0.8648 |
| U4 | 2 | bottom-20% | 0.8891/0.9021 | 0.8518/0.8564 | 0.8413/0.8224 | 0.8607/0.8603 |
| U5 | 3 | bottom-20% | 0.8857/0.8996 | 0.8497/0.8504 | 0.8363/0.8157 | 0.8573/0.8552 |
| HistoricalTuned | 3/1/2 | bottom20/mean/mean | 0.8857/0.8996 | 0.8623/0.8684 | 0.8601/0.8410 | 0.8694/0.8697 |

U0 相对历史 tuned baseline：

| 范围 | AUC 变化 | AP 变化 |
|---|---:|---:|
| ComGenVid | +0.01290 | +0.00964 |
| VideoFeedback | 0 | 0 |
| GenVideo | -0.00353 | -0.00193 |
| **Macro-3** | **+0.00312** | **+0.00257** |

U0 在 16/20 个生成器上不下降。主要 AP 改善来自 WildScrape `+0.0178`、VEO3
`+0.0120`、ComGenVid Sora `+0.0073`、Crafter `+0.0049`、Gen2 `+0.0032`
和 ModelScope `+0.0024`。下降集中在 GenVideo Lavie `-0.0191`、Show-1
`-0.0161`、Sora `-0.0075` 和 MorphStudio `-0.0009`。VideoFeedback 的十个
生成器逐分数与 HistoricalTuned 相同，因为其历史配置本来就是 region1+mean。

U0 在测试后恰好也是六个统一敏感性配置中 Macro 最好者，但这不是选择依据；
U0 的主方法身份在运行 U1-U5 前已经固定。

需要注意，旧 unified-multiscale 报告中的 “MS1 region1” 只统一了 region，仍沿用
各数据集原 aggregation。ComGenVid 的 MS1 因而实际是 `region1+bottom20`，与本次
U3 对应（旧/新 AP 均约 `0.9012`），不是 `region1+mean` U0。本次才首次把 region
和 aggregation 同时统一，并对完整 K3 窗口直接重算；因此不能用旧 MS1
`0.8687/0.8695` 代替 U0 `0.8725/0.8722`。

## 4. 最后一次跨层验证

跨层实验严格建立在 U0 上：

```text
C0 = calibrated layer23
C1_raw = min(calibrated layer17, calibrated layer23)
C1 = CDF_real_windows(C1_raw)
C2 = C1 if real-only motion CDF < 0.5 else layer23
```

layer17 和 layer23 分别使用独立 200-real whitening、Gaussian likelihood 和
window CDF；C1 的第二层 CDF同样只使用校准真实窗口。C2 的运动中位数分别为
ComGenVid `5.3404`、VideoFeedback `5.1554`、GenVideo `4.8285`，没有使用
fake 搜索阈值。

| 配置 | ComGenVid | VideoFeedback | GenVideo | Macro-3 |
|---|---:|---:|---:|---:|
| C0/U0 | 0.8986/0.9092 | 0.8623/0.8684 | 0.8566/0.8391 | **0.8725/0.8722** |
| C1 | 0.8984/0.9068 | 0.8632/0.8686 | 0.8504/0.8331 | 0.8707/0.8695 |
| C2 | 0.8930/0.9035 | 0.8647/0.8705 | 0.8506/0.8337 | 0.8695/0.8692 |

| 准入项 | C1 | C2 | 要求 |
|---|---:|---:|---:|
| Macro AP 变化 | -0.00271 | -0.00301 | >= +0.003 |
| 最差数据集 AP 变化 | -0.00597 | -0.00571 | >= -0.003 |
| 生成器不下降 | 4/20 | 10/20 | >= 12/20 |
| 高运动 fake AP 变化 | -0.01153 | +0.01170 | >= -0.005 |
| Macro AP 95% CI | [-0.00342,-0.00195] | [-0.00440,-0.00176] | 稳定正向 |
| 最终准入 | 否 | 否 | 全部满足 |

C1 仅在低运动 fake 上微增 `+0.00047 AP`，中/高运动分别下降 `-0.00223` 和
`-0.01153`。C2 修复高运动到 `+0.01170`，却让低/中运动下降 `-0.00740` 和
`-0.00521`，因此总指标仍显著下降。C1/C2 均被拒绝。

## 5. VideoFeedback rank-1023 数值稳定性

U0 layer23-region1 和 layer17-region1 的 whitening 都保留 1023/1024 维，完整
协方差因丢弃一维而奇异。保留子空间统计为：

| Operator | eigenvalue min/max | retained condition number |
|---|---:|---:|
| layer23-region1 | 6.13e-7 / 1.09e-2 | 1.78e4 |
| layer17-region1 | 1.65e-6 / 4.28e-2 | 2.60e4 |

同一 7,076 个窗口的 batch4/batch8 对照：

| Operator | raw 最大误差 | percentile 改变窗口 | 评测改变视频 | 最终分数最大误差 | 排名位置改变视频 | AUC/AP 变化 |
|---|---:|---:|---:|---:|---:|---:|
| layer23 | 84.38 | 12 | 3 | 0.0676 | 396 | +0.000073/+0.000116 |
| layer17 | 96.40 | 12 | 3 | 0.0648 | 1,026 | +0.000072/+0.000128 |

排名改变数量较大主要来自离散 CDF 分数的大量 ties，并不代表同样多的视频 raw
score 发生大变化；最终分数实际改变为 14 和 64 个视频。

对三个受影响的评测视频重新提取 token 并用 float64 计算后，六个 layer/window
percentile **均在 `3e-8` 内与 batch4 一致**。float64 raw 与 batch4 的最大误差为 `0.00140`，
而与 batch8 的最大误差为 `96.40`。因此 batch8 变化是病态 whitening 下的
float32 GEMM batch-shape 数值问题，不是视频内容或方法增益。正式结果固定
`video_batch_size=4`；batch8 结果不得用于方法比较。

## 6. 完整性与成本

- 评测视频：ComGenVid 4,298、VideoFeedback 3,500、GenVideo 13,623，共 21,421。
- 包含校准的窗口总数：58,496；所有键唯一，frame index 与冻结 K=3 文件逐行一致。
- 每数据集 200 条 calibration real；calibration fake 为 0；与 evaluation overlap 为 0。
- DINO 一次前向同时提取 layer17/23，并计算六个 region/aggregation raw score。
- 双 RTX 5090 顺序运行三个数据集约 62 分钟；本轮本地输出约 4.5 GB，主要是
  layer17 real calibration feature cache。轻量结果与报告进入版本控制，大缓存不提交。

## 7. 最终方法

最终方法仍然只有 **Global + Local 两个分支**。Local 使用统一 final-layer
region1 mean D2；不增加 layer17、min 跨层融合或运动 gate。正式数值为：

```text
ComGenVid      0.8986 / 0.9092
VideoFeedback  0.8623 / 0.8684
GenVideo       0.8566 / 0.8391
Macro-3        0.8725 / 0.8722
```

这比历史 dataset-specific tuned baseline 更严格、更简单，并且 Macro AUC/AP 都更高。
