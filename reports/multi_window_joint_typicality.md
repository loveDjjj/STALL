# Alpha-STALLED Multi-window 与 Joint Typicality 实验报告

> **状态更新（2026-07-24）：** 本报告中的 K=3 MW2 `0.8694/0.8697`
> 使用目标数据集 fake AUC/AP 选择 region 和 aggregation。它不存在样本校准泄漏，
> 但只能作为 Historical dataset-specific tuned baseline，不能再作为严格统一主方法。替代 U0 协议见
> `reports/clean_universal_cross_layer_final.md`。

日期：2026-07-24

## 1. 最终结论

本轮实验只改动视频级时间覆盖和 Global-Local 联合融合，窗口级公式始终冻结：

```text
G_k = 0.5 * global_spatial_k + 0.5 * global_T1_k
L_k = 0.1 * patch_spatial_k + 0.9 * same_grid_patch_D2_k
S_k = 0.6 * G_k + 0.4 * L_k
```

最终保留的方案是 **K=3 均匀窗口 + Global/Local 分支分别均值 + 视频级真实重校准（MW2）**：

```text
G_raw(V) = mean_k(G_k)
L_raw(V) = mean_k(L_k)
G(V) = F_real,K(G_raw(V))
L(V) = F_real,K(L_raw(V))
S(V) = 0.6 * G(V) + 0.4 * L(V)
```

其中 `F_real,K` 只由对应数据集的 200 个独立真实校准视频建立；固定 K 实验按 `effective_K` 匹配，每个校准视频只贡献一个视频级值。

| 方法 | ComGenVid AUC/AP | VideoFeedback AUC/AP | GenVideo AUC/AP | Macro-3 AUC/AP |
|---|---:|---:|---:|---:|
| 原版 STALL | 0.8550/0.8606 | 0.8528/0.8640 | 0.8085/0.8036 | 0.8388/0.8428 |
| clean single-window Alpha-STALLED | 0.8699/0.8819 | 0.8631/0.8708 | 0.8382/0.8274 | 0.8570/0.8600 |
| **最终 K=3 MW2** | **0.8857/0.8996** | **0.8623/0.8684** | **0.8601/0.8410** | **0.8694/0.8697** |
| 历史 leakage-affected Alpha-STALLED | 0.9199/0.9231 | 0.8561/0.8687 | 0.8452/0.8332 | 0.8737/0.8750 |

最终方案相对 clean baseline 的 Macro-3 增益为 `+0.0123 AUC / +0.0096 AP`；相对原版 STALL 为 `+0.0306 / +0.0269`。历史版本仍高 `0.0043 AUC / 0.0053 AP`，但其 patch calibration 使用了测试真实视频，不能作为无泄漏改进声明。

Joint Typicality 全部拒绝。预注册主模型 J2 将 Macro AP 从 `0.8697` 降至 `0.8437`；无参数 conflict gate 也只有 `0.8487`。最终继续使用固定 `0.6/0.4` 线性融合。

## 2. 协议与完整性

- 评测交集固定为 21,421 个视频：ComGenVid 4,298，VideoFeedback 3,500，GenVideo 13,623。
- 校准集为每数据集 200 个独立真实视频，共 600 个；生成视频从未用于 CDF、协方差、阈值或参数拟合。
- 每个窗口严格使用 16 帧、名义 8 FPS、2 秒索引；不使用 1 秒回退，不复制帧。
- K=3/K=5 在完整允许起点区间均匀采样；重复映射的帧索引窗口去重。
- all-window 使用 2 秒非重叠窗口并丢弃不足 16 帧的尾部。
- 指标为 higher-is-real AUC/AP；对每个生成器与确定性平衡真实样本做 pairwise 评测，再做数据集内和 Macro-3 宏平均。
- Bootstrap 为 1,000 次配置配对、视频行配对、生成器分层重采样；Macro-3 先在数据集内宏平均，再对三个数据集等权。
- 所有 score CSV 使用 round-trip 浮点解析，避免 `1e-16` 误差打破理论并列；保存后重读的指标误差为 0。
- 单次 DINO `forward_features` 同时产生 CLS 与 patch token；真实缓存验证 Global/Patch embedding 最大误差均为 0。
- 50 个单元测试全部通过；另有实际 K1 窗口核对，Global 完全一致，Local 仅在校准阈值附近出现一个 `1/200=0.005` 的离散百分位跳动。

数据质量例外：`WildScrape/D325.mp4` 容器声明 194 帧，但 ffmpeg 实际只能解码索引 0--98。K=3/K=5 末窗口引用 frame 99，因此在 `configs/multi_window_exclusions.csv` 显式删除该窗口；视频仍保留，实际 K 分别变为 2 和 4。没有复制 frame 98。

另有 22 个 WildScrape 视频的元数据时长为 1.90--1.92 秒，但冻结协议已为其提供 16 个不同的合法帧索引。为保持完全相同评测交集，本轮保留这些边界样本并单列披露。

## 3. 时长与窗口覆盖

| 数据集 | 视频数 | 时长均值/中位数（秒） | 最小/最大（秒） | K=3 平均 K | K>=3 视频数 |
|---|---:|---:|---:|---:|---:|
| ComGenVid | 4,298 | 7.65/8.00 | 2.00/60.03 | 3.00 | 4,296 |
| VideoFeedback | 3,500 | 2.42/2.00 | 2.00/3.00 | 1.88 | 1,546 |
| GenVideo | 13,623 | 10.26/11.00 | 1.90/250.54 | 2.74 | 11,817 |

评测时长分箱为：`<2s` 22、`2-4s` 6,806、`4-6s` 3,715、`6-10s` 2,121、`10-20s` 7,199、`20s+` 1,558。

| 采样 | effective-K 分布摘要 | 评测窗口 | 唯一 DINO 帧 | 相对 K1 帧数 |
|---|---|---:|---:|---:|
| K1 | K=1: 21,421 | 21,421 | 342,736 | 1.00x |
| K=3 | K1/K2/K3: 3,689/73/17,659 | 56,812 | 787,690 | 2.30x |
| K=5 | K1/K2/K3/K4/K5: 3,689/72/11/13/17,636 | 92,098 | 1,100,359 | 3.21x |
| all non-overlap | K=1: 6,823；K>=3: 10,879；最大 K=125 | 86,096 | 1,377,536 | 4.02x |

包括 600 个校准视频后，K=3/K=5/all 分别为 58,496/94,862/88,544 个窗口。双 GPU 实测阶段 wall time 约为 K=3 85 分钟、增量 K=5 47 分钟、增量 all 62 分钟。K=5 精确复用 K=3 窗口；all 同时复用 K=3/K=5。

## 4. Multi-window 对比

### 4.1 K=3 完整消融

| 配置 | 含义 | ComGenVid | VideoFeedback | GenVideo | Macro-3 AUC/AP |
|---|---|---:|---:|---:|---:|
| MW0 | 冻结单窗口 | 0.8699/0.8819 | 0.8631/0.8708 | 0.8382/0.8274 | 0.8570/0.8600 |
| MW0R | 单窗口 + 新视频级重校准 | 0.8643/0.8747 | 0.8667/0.8728 | 0.8284/0.8173 | 0.8532/0.8549 |
| MW1 | mean(S_k) | 0.8807/0.8962 | 0.8557/0.8630 | 0.8726/0.8473 | 0.8696/0.8688 |
| **MW2** | **recal(mean G), recal(mean L)** | **0.8857/0.8996** | **0.8623/0.8684** | **0.8601/0.8410** | **0.8694/0.8697** |
| MW3 | Global mean + Local bottom-2 | 0.8867/0.9007 | 0.8647/0.8708 | 0.8526/0.8337 | 0.8680/0.8684 |
| MW4 | Global mean + Local mean/bottom-2 hybrid | 0.8856/0.8997 | 0.8635/0.8697 | 0.8572/0.8379 | 0.8687/0.8691 |

MW2 的 Macro AP 增益为 `+0.009627`，95% CI `[+0.007047,+0.012373]`；15/20 个生成器提升，最差数据集点估计下降 `-0.002438`。MW0R 反而下降 `-0.0051`，说明结果不是简单增加一层 CDF 所致。

MW2 与推荐的 lower-tail hybrid MW4 直接配对比较，MW4-MW2 的 Macro AP 为 `-0.000578`，95% CI `[-0.001048,-0.000194]`。Local lower-tail 没有优于统一均值，主要原因是 GenVideo 被过度惩罚。

### 4.2 K 数量与聚合选择

| 候选 | Macro-3 AUC/AP | AP 相对 K=3 MW2 | 95% CI |
|---|---:|---:|---:|
| **K=3 MW2** | **0.8694/0.8697** | 0 | - |
| K=3 MW4 | 0.8687/0.8691 | -0.000578 | [-0.001048,-0.000194] |
| K=5 MW2 | 0.8663/0.8672 | -0.002442 | [-0.003215,-0.001600] |
| K=5 MW4 | 0.8660/0.8664 | -0.003241 | [-0.004128,-0.002357] |
| all MW2 | 0.8712/0.8695 | -0.000148 | [-0.001317,+0.001260] |
| all MW4 | 0.8677/0.8648 | -0.004870 | [-0.006106,-0.003498] |

all MW2 的 AUC 略高，但 AP 与 K=3 MW2 统计持平，推理帧数为 1.75 倍，并且最大 K=125 超过真实校准视频覆盖范围，只能使用视频等权无条件 CDF，存在时长/K 分布混杂。没有证据支持这部分复杂度。

### 4.3 增益来自哪些生成器

K=3 MW2 的 AP 变化（相对 MW0）：

| 数据集 | 生成器 | Delta AP | 数据集 | 生成器 | Delta AP |
|---|---|---:|---|---|---:|
| GenVideo | Sora | +0.0309 | GenVideo | ModelScope | +0.0240 |
| GenVideo | Gen2 | +0.0225 | VideoFeedback | Text2Video-Zero | +0.0214 |
| ComGenVid | VEO3 | +0.0186 | GenVideo | Show_1 | +0.0169 |
| ComGenVid | Sora | +0.0167 | VideoFeedback | VideoCrafter2 | +0.0143 |
| GenVideo | Lavie | +0.0141 | GenVideo | MorphStudio | +0.0077 |
| VideoFeedback | LaVie-base | +0.0077 | VideoFeedback | AnimateDiff | +0.0059 |
| VideoFeedback | Fast-SVD | +0.0051 | GenVideo | WildScrape | +0.0032 |
| VideoFeedback | LVDM | +0.0025 | VideoFeedback | ModelScope | -0.0036 |
| GenVideo | Crafter | -0.0100 | VideoFeedback | SoRA-Clip | -0.0120 |
| VideoFeedback | Pika | -0.0201 | VideoFeedback | ZeroScope-576w | -0.0457 |

三个重点负迁移生成器中，Text2Video-Zero 明显改善，VideoCrafter2 改善，SoRA-Clip 进一步下降；multi-window 没有解决所有短视频生成器问题。

### 4.4 长度与稳定性诊断

K=3 MW2 的 Macro AP 变化按长度为：`2-4s -0.0024`、`4-6s +0.0143`、`6-10s -0.0101`、`10-20s +0.0239`、`20s+ +0.0177`。因此单窗口漏检的主要证据集中在 10 秒以上视频，但并非单调随时长增加，6--10 秒组存在负迁移。

`effective_K>=3` 子集从 `0.8674` 提升到 `0.8752` AP；`effective_K=1` 诊断也从 `0.8171` 到 `0.8288`，后者只能来自视频级分支重校准，不能解释为额外时间覆盖。因此结论是“长视频覆盖是主要收益来源之一，但不是唯一来源”，而不是“所有增益都来自长视频”。

K=3 每视频窗口标准差均值为 Global `0.0370`、Local `0.0361`、最终窗口分数 `0.0310`。K=5 增加到 `0.0392/0.0384/0.0327`，但性能下降，说明观测到更多波动不等于更有用的异常证据。

## 5. Joint Typicality

输入固定为 K=3 MW2 的每视频 `(G_video,L_video)`。每数据集对 200 个真实校准视频做 5-fold cross-fitting，并使用五个固定 seed `13,29,42,73,101`；Ledoit-Wolf shrinkage covariance 固定，无生成视频调参。

| 模型 | ComGenVid | VideoFeedback | GenVideo | Macro-3 AUC/AP |
|---|---:|---:|---:|---:|
| J0 冻结 0.6/0.4 | 0.8857/0.8996 | 0.8623/0.8684 | 0.8601/0.8410 | **0.8694/0.8697** |
| J0X cross-fit 0.6/0.4 | 0.8856/0.8996 | 0.8620/0.8683 | 0.8601/0.8410 | 0.8692/0.8696 |
| J1 对称 Mahalanobis | 0.8428/0.8518 | 0.7396/0.7197 | 0.7750/0.7553 | 0.7858/0.7756 |
| J2 单侧 lower-tail | 0.8831/0.8951 | 0.7931/0.8017 | 0.8526/0.8344 | 0.8429/0.8437 |
| J2G real-Q95 conflict gate | 0.8645/0.8784 | 0.8171/0.8327 | 0.8507/0.8349 | 0.8441/0.8487 |
| J3 Gaussian copula | 0.8950/0.9062 | 0.8287/0.8309 | 0.8622/0.8435 | 0.8620/0.8602 |

J0X 与 J0 几乎相同，证明失败不是 cross-fitting 本身造成。J2 的 Macro AP 差为 `-0.0259`，95% CI `[-0.0300,-0.0220]`；五 seed Macro AP 标准差仅 `0.0009`，属于稳定失败。

真实 OOF `Q95(|G-L|)` 定义的 conflict subset：

| 数据集 | 冲突视频（real/fake） | J0 错误率 | J2 错误率 | 结论 |
|---|---:|---:|---:|---|
| ComGenVid | 185（133/52） | 0.2811 | 0.2649 | 改善 |
| VideoFeedback | 97（63/34） | 0.3505 | 0.2990 | 改善 |
| GenVideo | 577（491/86） | 0.1490 | 0.2010 | 明显恶化 |

J2 对 SoRA-Clip、Text2Video-Zero、VideoCrafter2 的 AP 分别从 J0 的 `0.8041/0.7791/0.9464` 降至 `0.7274/0.6646/0.8838`。联合模型没有解决重点负迁移生成器。

失败原因是二维异常度压缩了线性分数保留的单调排序：J1 会惩罚过高真实分数；J2 在两个 margin 都高于中位数时产生大量零距离并失去排序；J3 虽改善 ComGenVid/GenVideo，仍在 VideoFeedback 大幅下降。200 个真实点不足以证明复杂联合边界优于稳定线性融合。

## 6. 六个问题的明确回答

1. **单窗口漏检是否主要发生在长视频？** 部分是。10--20 秒和 20 秒以上 AP 分别提升约 0.0239/0.0177，2--4 秒不升；但 6--10 秒下降，而且 K=1 组的重校准也有收益，所以不能归因于长视频覆盖这一项。
2. **Global mean + Local lower-tail 是否优于统一平均？** 否。K=3 MW2 AP 0.8697，高于 MW3 0.8684 和 MW4 0.8691；MW4-MW2 的配对 CI 全部小于 0。
3. **Multi-window 的增益来自哪些生成器？** 15/20 提升，主要来自 GenVideo Sora、ModelScope、Gen2，VideoFeedback Text2Video-Zero/VideoCrafter2，以及 ComGenVid 两个生成器；ZeroScope、Pika、SoRA-Clip 和 Crafter 下降。
4. **Joint fusion 是否真正改善 Global-Local conflict？** 只在 ComGenVid 和 VideoFeedback 的阈值错误率上改善，GenVideo 明显恶化；整体和重点生成器均失败，不能称为普遍改善。
5. **联合模型是否优于固定 0.6/0.4？** 否。最好的 J3 Macro AP 0.8602，仍低于 J0 0.8697；预注册 J2 为 0.8437。
6. **最终提升是否足以增加方法复杂度？** K=3 MW2 值得：+0.0096 Macro AP、15/20 生成器、CI 全正，推理帧约 2.30x。K=5、all-window、lower-tail 和 Joint 都不值得增加复杂度。

## 7. 最终方法与保留项

保留：冻结窗口级 Alpha-STALLED；每视频三个均匀 2 秒窗口；Global/Local 分别均值；effective-K 匹配的真实视频级 CDF；固定 `0.6/0.4` 融合。

拒绝：K=5、all-window 作为默认；Local bottom-2/hybrid；J1/J2/J3；conflict-gated J2；任何基于生成测试视频选择的权重或模型。

主要机器可读输出位于 `results/multi_window_joint_typicality/`：窗口分数、逐视频聚合、三套数据集/生成器指标、duration/effective-K 分组、1,000 次 bootstrap、五 seed OOF Joint 分数与 conflict 明细均已保存。
