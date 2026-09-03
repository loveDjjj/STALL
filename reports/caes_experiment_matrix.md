# CAES 实验矩阵与预注册决策

> 所有主参数来源仅允许：已有方法合同、结构性规则、real-only rule或明确prior。三个development benchmark用于Go/No-Go，最终confirmation必须使用未参与选择的数据。

## 1. 固定合同

| 项目 | 固定值 |
|---|---|
| Backbone | frozen DINOv3 ViT-L/16 final normalized tokens |
| Dense input | 8 FPS、16 unique frames、2秒连续窗口 |
| Budget | K=3，短视频使用existing effective-K |
| Global | official STALL VATEX，0.5 Spatial + 0.5 T1 |
| Local | same-grid normalized vector D2，mean likelihood |
| Local Spatial | disabled |
| Calibration | 每数据集200互斥real，seed17 |
| Fusion | 0.60 Global + 0.40 Local |
| Pairwise | seed42，real-positive AP |
| 唯一变化 | window positions |

## 2. Coarse与候选固定值

| 参数 | 值 | 来源 |
|---|---:|---|
| Coarse FPS | 1 | AKS/KFS常用低成本扫描；且与8FPS grid整除对齐 |
| Candidate stride | 0.5秒/4个8FPS位置 | 2秒窗口的1/4，兼顾位置分辨率与候选数 |
| Window | 2秒/16帧 | 当前方法合同 |
| K | 3 | 当前方法合同 |
| Random seed | 17 + video hash | 与calibration seed分离且确定性 |
| FS3 score | mean real anomaly | 对2秒区间的稳定证据，不用fake二选一 |
| FS4 NMS IoU | 0.5 | 标准结构性去重阈值，不做grid search |
| FS5 strata | 3 | 与K一一对应，无coverage权重 |

## 3. Stage FS

| ID | Selector | 目标问题 | Matched real calibration |
|---|---|---|---|
| FS0 | exact uniform K=3 | baseline reproduction | 是，复用C0 |
| FS1 | deterministic random K=3 | uniform是否仅因稳定coverage获益 | 是，独立null |
| FS2 | mean feature-change Top-K | 无real model的motion baseline | 是，独立null |
| FS3 | mean real-anomaly Top-K | forensic relevance是否有效 | 是，独立null |
| FS4 | FS3 + temporal NMS | 去除局部重复是否有效 | 是，独立null |
| FS5 | 3 strata内各选最大real anomaly | relevance+coverage能否稳定兼得 | 是，独立null |

每个候选同时保存feature-change mean/max与real-anomaly mean/max，但主selector只使用表中预注册列。

## 4. 必报产物

每个FS保存：

- WindowManifest（全部candidate与selected）；
- selector config/reference/content hashes；
- calibration IDs；
- raw window scores与calibrated video scores；
- dataset/per-generator/pairwise Macro AUC/AP；
- Fake Recall@Real FPR 0.1%/1%；
- FPR@95% Fake TPR；
- paired bootstrap AUC/AP CI；
- coarse/dense runtime；
- decoded/processed unique frames；
- effective-K与选择失败计数；
- selected center span、pairwise distance和stratum coverage。

## 5. Go/No-Go

### Go

最优合法adaptive selector相对FS0：

- Macro AUC或AP至少`+0.005`；
- 同一指标至少2/3 datasets提升；
- paired bootstrap方向稳定。

### Strong Go

Macro增益`+0.008`至`+0.010`以上，且低FPR不退化。

### Borderline

`+0.002`至`+0.005`。只有0.1%/1% FPR或预注册长视频子集明显改善时继续。

### No-Go

增益`<+0.002`、只改善一个dataset或matched real FPR明显恶化。立即停止wavelet/DPP/bandit/change-point，转Tail。

## 6. Cross-fitting

首轮standard matched calibration；若FS Go，补5-fold OOF real null并比较：

| ID | Calibration selector reference | Calibration video scores | Test reference |
|---|---|---|---|
| CF0 | 全部200real | in-sample matched selection | 全部200real |
| CF1 | 每fold其余160real | held-out 40real selection/score | 全部200real |

CF1重点报告real score shift和实际0.1%/1% FPR，不期待AUC必然提升。

## 7. Shortcut audit

对FS0与最优FS比较selected windows：

| Diagnostic | 定义/近似 |
|---|---|
| Motion magnitude | coarse `||g[t+1]-g[t]||` |
| Feature change | selector原始change signal |
| Shot boundary distance | coarse cosine-change peak最近距离 |
| Duplicate ratio | dense相邻Global/像素近零差分比例 |
| Blur proxy | Laplacian variance，window mean/min |
| Brightness change |相邻帧灰度均值绝对差 |
| Black-frame ratio |亮度低于固定real-only阈值比例 |
| Text overlay proxy | 第一轮不引入OCR；只在人工/后续诊断标记 |

所有阈值从calibration real分布或结构定义取得，不按fake调整。

## 8. Stage Tail（仅FS稳定后）

| ID | Local aggregation | 固定比例来源 |
|---|---|---|
| TA0 | current mean | baseline |
| TA1 | worst 5% mean | 预注册稀疏极端对照 |
| TA2 | CVaR 10% | primary tail候选 |
| TA3 | CVaR 20% |较宽tail敏感性 |

只改变Local D2 likelihood field聚合；calibration real也执行相同tail。若FS No-Go，Tail仍在FS0上运行。

## 9. Stage Fusion

固定最优合法selector/aggregation：

| ID | Fusion |
|---|---|
| F0 | 0.6 Global + 0.4 Local |
| F1 | 0.5/0.5 |
| F2 | equal mean of calibrated evidence |
| F3 | real-only p-value combination |
| F4 | Cauchy combination |

性能等价时优先无alpha方案；不得用三个benchmark fake重新调连续权重。

## 10. Cross-domain calibration

完整`calibration domain x evaluation domain`：VF/GV/CGV各自到三域，并增加三域混合Universal Real Bank。报告AUC/AP、低FPR、real/fake score shift、per-generator degradation和实际FPR偏移。

## 11. Few-real

在同一cache scan比较：

```text
n_real = 25,50,100,200
covariance = empirical, Ledoit-Wolf, OAS
```

每个n使用固定nested calibration IDs，至少5个预注册seed。目标是sample efficiency与variance，不是选择n=200最优模型。

## 12. 新数据集

只有FS/Tail候选冻结后才选择untouched confirmation。优先小型ViF-Bench或CoCoVideo公开evaluation subset；不得在confirmation上回调selector参数。

## 13. 时间与空间估计

| 阶段 | 预计时间 | 新增空间 |
|---|---:|---:|
| 模块/tests | 0.5-1天 | <50MB |
| 1 FPS coarse cache（dev） | 1-3小时 | 0.5-1.0GB |
| FS smoke | 1-2小时 | <200MB |
| FS1-FS5 dense extraction/score | 8-16小时（GPU1，selector窗口并集single-pass） | scores 0.2-1GB；含field 1-3GB |
| bootstrap/report | 1-2小时CPU | <100MB |

总新增空间保守控制在5GB以内，远低于69GB余量。若selected窗口并集接近所有候选而导致dense帧数异常增加，运行器必须在提取前输出预算并拒绝超过预设上限。
