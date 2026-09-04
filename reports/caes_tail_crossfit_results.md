# CAES 三选择器 Tail 与 5-fold Cross-fitting 结果

> 完成日期：2026-09-04  
> Selector：Uniform、Feature-change、Real-anomaly  
> 数据：ComGenVid、VideoFeedback、GenVideo固定21,421条视频  
> 检测器：official STALL Global + same-grid Local D2，固定0.60/0.40融合

## 1. 实验完成范围

三种selector均完成用户要求的三项后续验证：

1. 固定时长分层与paired bootstrap；
2. Mean、worst 5%、CVaR 10%、CVaR 20% Local D2位置证据聚合；
3. standard与5-fold OOF selector calibration比较。

Tail缓存只保存每个窗口`[T-2,14,14]` float32 Local D2 likelihood field，
不保存Patch token。全量共1,379个可恢复shard、约1.3 GB。每种selector与
calibration mode分别用calibration real位置似然建立经验CDF，将位置似然转换为
`A=1-F_real(likelihood)`，再对top anomaly 5/10/20%取均值。输出取负号，使分数
方向继续保持“越高越真实”。

## 2. 正确性审计

- 三种standard Mean直接复用冻结的Stage FS窗口/视频分数。
- 21,421条视频上，三种Mean的Global、Local、Final最大绝对差均为`0.0`。
- Uniform和Feature-change不依赖selector real reference，其standard/crossfit5
  在四种聚合上的逐视频最大差均为`0.0`。
- Real-anomaly每折160条real拟合selector reference、40条held-out执行选择；
  每条calibration real恰好held out一次，train/held-out ID与哈希均保存。
- 所有run产物和1,379个field shard通过内容哈希验证。

## 3. Tail主结果

### 3.1 Standard calibration

| Selector | Mean | Worst 5% | CVaR 10% | CVaR 20% |
|---|---:|---:|---:|---:|
| Uniform | **0.8741/0.8729** | 0.8600/0.8595 | 0.8612/0.8610 | 0.8631/0.8630 |
| Feature-change | **0.8774/0.8769** | 0.8608/0.8599 | 0.8626/0.8621 | 0.8649/0.8650 |
| Real-anomaly | **0.8662/0.8660** | 0.8523/0.8528 | 0.8538/0.8545 | 0.8555/0.8566 |

每格为共享FS0视频身份的Macro-3 AUC/real-positive AP。三个selector均呈现
`Mean > CVaR20 > CVaR10 > Worst5`，Tail越尖锐，性能下降越明显。

### 3.2 CVaR 10%相对Mean

| Selector | AUC delta [95% CI] | AP delta [95% CI] |
|---|---:|---:|
| Uniform | -0.0129 [-0.0142,-0.0115] | -0.0120 [-0.0134,-0.0104] |
| Feature-change | -0.0147 [-0.0162,-0.0132] | -0.0147 [-0.0163,-0.0131] |
| Real-anomaly | -0.0124 [-0.0139,-0.0108] | -0.0114 [-0.0129,-0.0098] |

所有区间完全低于零，Tail失败不是抽样噪声。结果说明当前Local D2的有效信号更像
跨位置的分布性偏移，而不是少量极端patch；只取最异常位置会放大真实视频中的镜头
边界、遮挡、快速运动、压缩噪声和局部低似然，造成real false positive。

## 4. Cross-fitting结果

### 4.1 三selector控制

| Selector | Crossfit行为 | 结果 |
|---|---|---:|
| Uniform | 不依赖selector reference，OOF清单等于standard | 四聚合逐视频差0 |
| Feature-change | 不依赖selector reference，OOF清单等于standard | 四聚合逐视频差0 |
| Real-anomaly | 每折重新拟合1 FPS real reference并对held-out real选窗 | 分数发生预期变化 |

这两个严格为零的控制证明Real-anomaly变化来自OOF selector reference，而不是
评测代码、视频顺序或重复校准。

### 4.2 Real-anomaly Mean

| Dataset | Standard | Crossfit5 | Delta AUC/AP |
|---|---:|---:|---:|
| ComGenVid | 0.8966/0.9034 | 0.8972/0.9039 | +0.0006/+0.0005 |
| VideoFeedback | 0.8596/0.8658 | 0.8622/0.8686 | +0.0026/+0.0028 |
| GenVideo | 0.8423/0.8288 | 0.8463/0.8324 | +0.0040/+0.0035 |
| Macro-3 | 0.8662/0.8660 | **0.8686/0.8683** | **+0.0024/+0.0023** |

Macro paired bootstrap：

- AUC `+0.00240`，95% CI `[+0.00196,+0.00284]`；
- AP `+0.00230`，95% CI `[+0.00175,+0.00283]`。

Cross-fitting显著修正了in-sample selector reference偏差，尤其在GenVideo上；但
crossfit Real-anomaly仍比Uniform低约`-0.0055/-0.0046`，比Feature-change更低，
因此它提高了协议严谨性，却没有把Real-anomaly变成最佳方法。

## 5. 与视频长度的交互

固定时长分析见`reports/caes_duration_results.md`。补充检查Tail/crossfit后：

- ≥16秒有限子集上，Feature-change Mean仍最好：`0.8156/0.7559`；
- Feature-change CVaR10降至`0.7949/0.7462`，Tail没有增强长视频优势；
- Real-anomaly standard Mean为`0.7960/0.7183`，crossfit Mean为
  `0.7953/0.7145`，OOF在该小子集没有改善；
- Uniform CVaR10的AUC下降，但AP由`0.7205`升至`0.7281`；
- Real-anomaly CVaR10的AP升至约`0.7314`，但AUC下降且只有65对样本，不能据此
  选择Tail。

因此长视频上的Feature-change正信号来自选窗本身，不来自位置级极端聚合。

## 6. 科学结论

1. **Feature-change保留为长视频候选。** 它是三种selector中总体最强，且在
   ≥16秒有限子集上有显著正增益，但需要独立长视频benchmark确认。
2. **Real-anomaly若继续使用，必须采用crossfit5。** OOF带来稳定小幅提升，证明
   matched in-sample calibration仍低估了selection bias；但其绝对性能仍不足。
3. **位置级Tail/CVaR停止。** 三selector、三个数据集、三个比例均没有形成主线，
   不继续搜索更多百分位或时空连通Tail，以免fake-guided调参。
4. **正式通用基线仍是Uniform Mean。** Feature-change尚未通过跨数据集Go gate；
   它适合作为预冻结的长视频扩展候选，而不是当前所有视频的默认方法。

## 7. 下一步

按原计划进入：

1. 对Uniform Mean、Feature-change Mean和Real-anomaly Crossfit Mean做real-only
   fusion；
2. 做target-domain、cross-domain和Universal Real Bank完整矩阵；
3. 做few-real 25/50/100/200与empirical/LW/OAS；
4. 完成Feature-change/Real-anomaly的scene-cut、motion、blur、brightness、duplicate
   shortcut审计；
5. 在冻结selector后使用独立长视频数据确认，不再改K、窗口长度、coarse FPS或阈值。

权威产物：

- `results/runs/caes_tail_matrix/`
- `results/analysis/caes_tail/`
- `results/caes/tail_fields_v1/`
- `results/caes/crossfit5_seed17/`
