# CAES 三选择器固定时长分层结果

> 日期：2026-09-04  
> Selector：Uniform、Feature-change、Real-anomaly  
> 固定区间：`[2,4)`、`[4,8)`、`[8,16)`、`[16,+∞)` 秒  
> 单个数据集/生成器 cell 至少需要20条real和20条fake

## 1. 目的与限制

该分析检验 adaptive selector 的收益是否随视频长度增加。时长边界在读取结果前按
2秒窗口的倍数固定，没有使用fake性能选边界。每个生成器仍与同一批FS0 real
视频配对，所有selector共享视频身份，并执行1,000次paired bootstrap。

不同区间可用的数据集和生成器不同，因此表中的 `Eligible-Macro` 只表示当前区间
满足样本门槛的数据集等权平均，不是完整三数据集 Macro-3。尤其是16秒以上仅有
GenVideo的Sora和WildScrape共65对，不能代替独立长视频benchmark。

## 2. 结果

| 时长 | 可用数据 | Uniform | Feature-change | Real-anomaly |
|---|---|---:|---:|---:|
| 2–4秒 | VideoFeedback，10 generators | 0.8621/0.8687 | 0.8600/0.8663 | 0.8596/0.8658 |
| 4–8秒 | ComGenVid，1 generator | 0.8986/0.9031 | 0.8998/0.9058 | 0.8932/0.8969 |
| 8–16秒 | ComGenVid+GenVideo，4 cells | 0.7824/0.8000 | 0.7682/0.7901 | 0.7546/0.7918 |
| ≥16秒 | GenVideo，2 generators | 0.7813/0.7205 | **0.8156/0.7559** | 0.7960/0.7183 |

每格为Eligible-Macro AUC/real-positive AP。

## 3. 相对Uniform的配对结果

| 时长 | Selector | AUC delta [95% CI] | AP delta [95% CI] |
|---|---|---:|---:|
| 2–4秒 | Feature-change | -0.0021 [-0.0025,-0.0016] | -0.0023 [-0.0029,-0.0018] |
| 2–4秒 | Real-anomaly | -0.0025 [-0.0030,-0.0020] | -0.0028 [-0.0035,-0.0021] |
| 4–8秒 | Feature-change | +0.0013 [-0.0034,+0.0063] | +0.0027 [-0.0020,+0.0077] |
| 4–8秒 | Real-anomaly | -0.0053 [-0.0112,+0.0005] | -0.0061 [-0.0115,-0.0008] |
| 8–16秒 | Feature-change | -0.0140 [-0.0320,+0.0015] | -0.0090 [-0.0217,+0.0055] |
| 8–16秒 | Real-anomaly | -0.0280 [-0.0537,-0.0015] | -0.0088 [-0.0305,+0.0142] |
| ≥16秒 | Feature-change | **+0.0340 [+0.0090,+0.0620]** | **+0.0337 [+0.0093,+0.0623]** |
| ≥16秒 | Real-anomaly | +0.0154 [-0.0155,+0.0519] | +0.0016 [-0.0346,+0.0362] |

## 4. 解释

Feature-change在≥16秒子集上出现幅度较大的稳定提升，支持“长视频中uniform K3
可能遗漏局部强变化证据”的机制假设。但结果并不随时长单调提高：8–16秒区间反而
下降，且≥16秒只有两个GenVideo生成器。因此当前证据只能支持继续实验，不能支持
把Feature-change直接设为长视频默认策略。

Real-anomaly在≥16秒AUC点估计提高约0.015，但置信区间跨零，AP几乎不变；在
8–16秒区间明显下降。它是否受in-sample selector reference影响，需要5-fold OOF
calibration结果回答。

## 5. 决策

1. 三种selector继续进入同一Tail矩阵，避免只为Feature-change选择有利实验。
2. 三种selector都完成standard/crossfit5控制；Uniform与Feature-change应严格不变。
3. 保留≥16秒结果作为预先固定的条件性证据，不把它宣传为跨域长视频结论。
4. 最终是否保留adaptive主线，必须同时参考Tail交互、OOF结果和独立长视频数据。

权威表位于 `results/analysis/caes_duration/`。
