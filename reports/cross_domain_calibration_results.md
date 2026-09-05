# Feature-change K3 跨真实域校准与 Universal Real Bank

## 协议

检测器固定为 Feature-change K3、STALL Global Spatial/T1 0.5/0.5、Local
same-grid D2 和最终 Global/Local 0.5/0.5。校准 bank 包括 ComGenVid、
VideoFeedback、GenVideo、GenVidBench，以及：

- Universal-3：三个开发域各等量贡献 100,000 个真实 D2 位置；
- Universal-4：上述四域各等量贡献 75,000 个真实 D2 位置。

每个目标视频只执行一次 DINO，再同时使用六套冻结 real bank 评分。每个 bank 的
Local Gaussian、窗口 CDF、effective-K 视频 CDF和阈值均不读取 evaluation real
或任何 fake。主指标为每生成器与真实视频平衡配对后再宏平均的 AUC/AP-real。

## 配对 AUC 矩阵

| Calibration bank | ComGenVid | VideoFeedback | GenVideo | GenVidBench | Macro-4 |
|---|---:|---:|---:|---:|---:|
| ComGenVid | **0.9030** | 0.8417 | 0.8507 | 0.8358 | 0.8578 |
| VideoFeedback | 0.8665 | **0.8501** | 0.8496 | 0.8093 | 0.8439 |
| GenVideo | 0.8540 | 0.8200 | **0.8781** | 0.8339 | 0.8465 |
| GenVidBench | 0.8549 | 0.8179 | 0.8680 | **0.8702** | 0.8528 |
| Universal-3 | 0.8911 | 0.8335 | 0.8688 | 0.8251 | 0.8546 |
| Universal-4 | 0.8852 | 0.8306 | 0.8717 | 0.8420 | **0.8574** |
| Target-domain diagonal | 0.9030 | 0.8501 | 0.8781 | 0.8702 | **0.8754** |

## 配对 AP-real 矩阵

| Calibration bank | ComGenVid | VideoFeedback | GenVideo | GenVidBench | Macro-4 |
|---|---:|---:|---:|---:|---:|
| ComGenVid | **0.9109** | 0.8542 | 0.8400 | 0.8343 | **0.8599** |
| VideoFeedback | 0.8635 | **0.8553** | 0.8314 | 0.7994 | 0.8374 |
| GenVideo | 0.8589 | 0.8275 | **0.8623** | 0.8351 | 0.8460 |
| GenVidBench | 0.8641 | 0.8253 | 0.8492 | **0.8708** | 0.8523 |
| Universal-3 | 0.8999 | 0.8441 | 0.8569 | 0.8277 | 0.8572 |
| Universal-4 | 0.8959 | 0.8414 | 0.8583 | 0.8433 | 0.8597 |
| Target-domain diagonal | 0.9109 | 0.8553 | 0.8623 | 0.8708 | **0.8748** |

Universal-4 相对 target-domain 的 Macro-4 AUC/AP 下降约 0.0180/0.0151。
四个目标域上的 paired bootstrap 区间均完全低于 0；其中 GenVidBench 下降最大，
为约 0.0282/0.0275。Universal-4 因而不能替代目标域 real bank 来获得最高排序性能。

## 固定 FPR 操作点

| Calibration | Target | 1%目标：实际FPR / fake recall | 5%目标：实际FPR / fake recall |
|---|---|---:|---:|
| Target-domain | ComGenVid | 1.34% / 27.62% | 16.93% / 78.53% |
| Target-domain | VideoFeedback | 3.60% / 33.00% | 7.60% / 50.73% |
| Target-domain | GenVideo | 0.51% / 18.92% | 7.09% / 60.38% |
| Target-domain | GenVidBench | 2.00% / 26.67% | 8.00% / 48.33% |
| Universal-4 | ComGenVid | 0.11% / 6.44% | 2.56% / 26.94% |
| Universal-4 | VideoFeedback | 1.60% / 11.73% | 6.00% / 39.40% |
| Universal-4 | GenVideo | 0.20% / 9.88% | 2.57% / 37.77% |
| Universal-4 | GenVidBench | 0.67% / 14.67% | 5.67% / 37.67% |

目标域 calibration 也不能自动保证 evaluation FPR，特别是 ComGenVid 的 5% 阈值
膨胀到 16.93%。Universal-4 在四域中明显改善阈值稳定性，1% 目标实际落在
0.11%--1.60%，5% 目标落在 2.56%--6.00%；代价是 fake recall 大幅降低。

## Universal-4 到 ViF-Bench

Universal-4 完全不包含 ViF real。ViF target-real calibration 与 Universal-4 的
生成器配对结果如下：

| Calibration | Macro AUC | Macro AP | 1%实际FPR / recall | 5%实际FPR / recall |
|---|---:|---:|---:|---:|
| ViF target real | 0.6043 | 0.6246 | 1.18% / 5.29% | 24.71% / 35.47% |
| Universal-4（0 ViF real） | 0.6026 | 0.6297 | 0.00% / 0.52% | 1.18% / 4.44% |

Universal-4 相对 target-real 的 AUC/AP 配对差为 -0.0018/+0.0048，两个 95% CI
均跨 0。它保持了相近的排序能力并显著减少真实误报，但几乎失去低 FPR 下的 fake
检出能力。因此 Universal bank 更适合作为保守 fallback，而不是当前 target-real
校准的直接替代。

## 数值与身份审计

跨域对角共 22,021 条视频。相对已有 target-domain Feature-change 分数：

- VideoFeedback 与 GenVidBench 达到浮点精度一致；
- ComGenVid 仅 5 条视频超过 `1e-9`，最大差一个 200-real CDF 步长 0.0025；
- GenVideo 仅 8 条视频超过 `1e-9`，最大差一个分支 CDF 步长 0.005；
- 四域 Pearson/Spearman 均显示为 1.0。

差异来自极少数 likelihood 位于经验 CDF 相邻秩边界，未改变总体结论。

## 结论

1. Local D2 参考具有明显真实域依赖，target-domain bank 仍提供最高 Macro 性能。
2. Universal-4 牺牲约 1.5--1.8 个百分点排序性能，换取显著更稳定的真实 FPR。
3. Universal-4 在完全未使用 ViF real 时仍只有约 0.60 AUC，说明主要瓶颈不是
   calibration 域，而是当前 Global/Local 证据难以区分 ViF 的高质量生成视频。
4. 论文应把方法准确定位为 target-real-calibrated training-free detector；
   Universal bank 是保守跨域部署选项，不能宣称 calibration-free 泛化已解决。

正式结果位于：

- `results/runs/cross_domain_feature_k3_equal_fusion/`
- `results/analysis/cross_domain_feature_k3_equal_fusion/`
- `results/runs/universal4_to_vifbench_feature_k3/`
- `results/analysis/universal4_to_vifbench/`
