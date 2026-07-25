# Alpha-STALLED 研究结果统一索引

本目录保存能够进入论文结论的轻量结果索引。完整逐窗口、逐视频、bootstrap
抽样和特征缓存保留在本地 `results/`，不作为 Git release 资产。

## 正式主方法

锁定 Alpha-STALLED U0 使用两个分支和统一配置：

```text
G_k = 0.5 * GlobalSpatial_k + 0.5 * GlobalT1_k
L_k = 0.1 * PatchSpatial_k + 0.9 * same-grid PatchD2_k
G_raw, L_raw = K=3 uniform 2-second window means
G, L = effective-K-matched real-only CDF calibration
S = 0.6 * G + 0.4 * L
```

Local 固定 DINOv3 ViT-L/16 final layer、原始 14x14 patch、region1、mean
aggregation；whitening、Gaussian likelihood 和 CDF 使用 float64。每个数据集
200 条独立真实视频用于校准，生成视频和评测真实视频不参与拟合、校准或权重选择。

| 数据集 | U0 AUC/real-positive AP |
|---|---:|
| ComGenVid | 0.8968/0.9064 |
| VideoFeedback | 0.8597/0.8689 |
| GenVideo | 0.8657/0.8416 |
| **Macro-3** | **0.8741/0.8723** |

相对 Original STALL Macro AUC/AP `0.8388/0.8428`，U0 提高
`+0.0353/+0.0295`；相对统一 K1 `0.8632/0.8636`，K=3 提高
`+0.0109/+0.0087`，AP 的 95% paired bootstrap CI 为
`[+0.0057,+0.0120]`。加入 Local 相对 K3 Global-only 提高 `+0.0237 AP`；
加入 Global 相对 K3 Local-only 提高 `+0.0431 AP`。

## 已确认边界

| 方向 | 结果 | 决策 |
|---|---:|---|
| K=5 / all-window | Macro AP 0.8672 / 0.8695 | 成本更高且不超过 K=3，拒绝 |
| bottom-2 / hybrid | 0.8684 / 0.8691 | lower-tail 过度惩罚，拒绝 |
| Joint Typicality J2/J3 | 0.8437 / 0.8602 | 低于固定线性融合，拒绝 |
| Spatial-mean residual | 0.8609，delta -0.0088 | CI 全负，4/20 生成器不下降，拒绝 |
| Fine+coarse | 0.8687 | 未超过 fine-only，拒绝 |
| late+final layer | 0.8698，delta 约 +0.0002 | CI 跨 0 且两个数据集下降，拒绝 |
| cross-layer min / motion gate | 0.8695 / 0.8692 | Macro AP 均下降，拒绝 |
| D3/D4、multi-lag、hard/soft | 均未超过 same-grid D2 | 不重复搜索 |

PatchSpatial 不是主要信号：统一 K1 中从 PatchD2-only 加入 0.1 PatchSpatial
使 Macro AP 下降约 0.0070。U0 保留 beta=0.1 是因为配置在最终审计前已冻结，
而不是根据敏感性结果事后选择。

## 校准与数值稳定性

- 25/50/100/200 条真实视频的五 seed Macro AP 均值为
  `0.8237/0.8487/0.8628/0.8680`；200 条的 seed 标准差为 `0.0022`。
- Local 比 Global 更依赖校准规模。方法定位是目标域真实视频无监督校准，
  不是 target-data-free universal detector。
- FP32 rank-1023 whitening 会产生 batch-shape 数值漂移；float64 score stage
  在 batch 1/4/8/16 下最大误差不超过 `2.27e-13`。
- Release 包含 600 条校准视频、21,421 条评测视频、58,496 个窗口，
  calibration/evaluation overlap 为 0；61 项 release validation 全部通过。
- 三份锁定 200-real Local 参数已随 `release/u0/params/` 发布；完整 release
  约 `57 MB`，不包含 DINO 特征缓存或原始视频。
- 跨域校准中 target-domain 对角线最终 AP 为 `0.8723`，单一源 off-domain
  均值为 `0.8489`；Local target/off-domain AP `0.8292/0.7425`，显著比 Global
  `0.8486/0.8475` 更依赖目标域。
- Pooled-200/600 Macro AP 为 `0.8628/0.8636`，没有替代目标域 200-real。
- OAS 将 Macro AP 只提高 `+0.000043`，CI 跨 0 且 seed 标准差略增，正式拒绝。
- 锁定后 GenVidBench 上 STALL/Clean K1/U0 AP 为 `0.8043/0.8369/0.8495`；
  U0 相对 STALL/K1 的 AP CI 分别为 `[+0.0297,+0.0634]` 和
  `[+0.0017,+0.0247]`，确认主增益可迁移到新生成器。
- 300-real 锁定注入并未证明通用异常定位：K3 对局部冻结的最终 score drop
  仅 `+0.0023`，对全帧冻结为 `-0.1667`，对合成切镜为 `-0.0264`；局部编辑
  patch-time AUPRC 仅 `0.0670--0.1558`。Local D2 是真实分布典型性证据，不能
  解释成对任意合成伪影都单调响应的语义定位器。
- 1,600-video 锁定鲁棒性子集中，Scenario-A 的 CRF23/drop10 AP 仅下降
  `0.0024/0.0024`，但 CRF35、resize、repeat25 和 4 FPS 下降
  `0.0196/0.0097/0.0226/0.0085`；matched CDF 未恢复严重退化。

## 历史结果分类

- `0.8737/0.8750` 及早期逐数据集高分结果存在 sample calibration leakage，
  只能作为审计记录，不能进入有效主表。
- `0.8694/0.8697` 没有样本泄漏，但 region/aggregation 使用目标 fake 指标选择，
  命名为 Historical dataset-specific tuned baseline。
- `0.8725/0.8722` 只统一了 temporal 分支，PatchSpatial 仍继承历史聚合，
  属于 pre-release audit result。
- 历史 clean single-window `0.8570/0.8600` 用于展示方法演进；统一 K1
  `0.8632/0.8636` 才是 K=3 覆盖的直接因果对照。

## 主要入口

- 锁定配置：`configs/alpha_stalled_u0_locked.yaml`
- 发布验证：`tools/verify_u0_locked_release.py`
- 最终逐视频分数：`release/u0/final_video_scores.csv`
- 协议与结果报告：`reports/u0_*.md`
- 当前实验注册表：`reports/u0_experiment_registry.csv`
