# Stage 1：局部对应实验结果

> 完成日期：2026-09-02  
> 实现提交：`0ea2101`  
> 数据范围：ComGenVid、VideoFeedback、GenVideo 固定 21,421 条可用视频  
> 基线：当前正式 same-grid Local D2-only + Global STALL + K=3

## 1. 研究问题与预注册门槛

Stage 1 检验 H1：固定网格对应误差是否是当前 Local D2 的主要限制。四组实验严格共享数据、校准视频、seed、DINO 特征、窗口、Global、Local Gaussian likelihood、视频级 CDF 和 `0.6/0.4` 融合，仅改变 correspondence：

| ID | Correspondence | Confidence |
|---|---|---|
| C0 | same-grid | 无 |
| C1 | 3x3 hard local cosine match | 无 |
| C2 | 3x3 soft local match，`tau=0.07`，空间惩罚 `0.05` | 无 |
| C3 | 同 C2 | normalized entropy aggregation weight |

预注册继续条件：相对 C0，C3 的 Macro AUC 或 AP 至少 `+0.005`，且同一指标至少 2/3 数据集提升。若增益小于 `+0.002` 且不稳定，停止复杂 OT/matching。

## 2. 最终融合结果

| Variant | ComGenVid AUC/AP | VideoFeedback AUC/AP | GenVideo AUC/AP | Macro-3 AUC/AP | Delta vs C0 |
|---|---:|---:|---:|---:|---:|
| C0 same-grid | **0.9015/0.9110** | **0.8621/0.8687** | 0.8586/0.8391 | **0.8741/0.8729** | - |
| C1 hard | 0.8978/0.9080 | 0.8591/0.8667 | 0.8574/0.8355 | 0.8714/0.8701 | -0.0026/-0.0029 |
| C2 soft | 0.8993/0.9098 | 0.8558/0.8613 | **0.8615/0.8403** | 0.8722/0.8705 | -0.0018/-0.0024 |
| C3 soft+confidence | 0.9018/0.9110 | 0.8554/0.8605 | 0.8561/0.8335 | 0.8711/0.8683 | -0.0029/-0.0046 |

C2 只在 GenVideo 上提高 AUC/AP；C3 只在 ComGenVid AUC 上产生 `+0.00035` 的极小变化，VideoFeedback 和 GenVideo 均下降。因此没有任何候选满足“同一指标至少 2/3 数据集提升”。

## 3. Local-only 结果

| Variant | ComGenVid AUC/AP | VideoFeedback AUC/AP | GenVideo AUC/AP | Macro-3 AUC/AP |
|---|---:|---:|---:|---:|
| C0 same-grid | **0.8995/0.8970** | **0.7368/0.7539** | **0.8447/0.8179** | **0.8270/0.8229** |
| C1 hard | 0.8702/0.8725 | 0.7136/0.7298 | 0.8306/0.8049 | 0.8048/0.8024 |
| C2 soft | 0.8831/0.8843 | 0.7145/0.7241 | 0.8421/0.8173 | 0.8132/0.8086 |
| C3 soft+confidence | 0.8846/0.8844 | 0.7182/0.7283 | 0.8295/0.7997 | 0.8108/0.8041 |

Local-only 的下降远大于最终融合下降，说明不是固定 `0.6/0.4` 融合掩盖了 matching 收益。hard/soft alignment 本身削弱了当前 Local evidence。

## 4. 成对 bootstrap

按论文的生成器-真实视频平衡配对规则，在同一视频上执行 1,000 次成对重采样：

| Comparison | Macro AUC delta [95% CI] | Macro AP delta [95% CI] |
|---|---:|---:|
| C1 - C0 | -0.00261 [-0.00349, -0.00175] | -0.00280 [-0.00431, -0.00156] |
| C2 - C0 | -0.00182 [-0.00265, -0.00097] | -0.00241 [-0.00346, -0.00126] |
| C3 - C0 | -0.00292 [-0.00402, -0.00182] | -0.00456 [-0.00597, -0.00317] |

所有 Macro 区间均完全低于零。C2 在 GenVideo 的 AUC 增益为 `+0.00298 [0.00152,0.00444]`，但 VideoFeedback 同时下降 `-0.00632 [-0.00790,-0.00476]`，不构成跨数据集主线。

## 5. 部署指标

三个候选均未在 pooled AUC/AP 与低 FPR 上形成统一优势。例外包括：

- C3 将 ComGenVid Fake Recall@1% Real FPR 从 `0.1982` 提高到 `0.2550`，但其 0.1% FPR recall 仍为 0，且 Macro AUC/AP 下降。
- C2/C3 将 VideoFeedback 0.1% FPR recall 从 `0.0190` 小幅提高到 `0.0230/0.0237`，但该数据集 pairwise AUC/AP 明显下降。
- GenVideo 的 0.1%/1% FPR recall 在 C1-C3 均未超过 C0。

因此不能以单一 operating point 的局部例外推翻总体 gate。

## 6. 效率与存储

| Variant | calibration score time | evaluation score time | 记录峰值显存 |
|---|---:|---:|---:|
| C1 | 5,380.7 s | 5,324.4 s | 3.45 GiB |
| C2 | 5,496.9 s | 5,438.1 s | 3.45 GiB |
| C3 | 5,186.6 s | 5,128.5 s | 3.45 GiB |

三次运行各约 32 MB，合计约 96 MB；未创建新 feature cache。运行时间由 630 GB packed cache I/O 与 float64 Gaussian scoring 主导，当前数据不足以分辨 hard/soft 的微小纯计算差异。

## 7. 机制解释

结果否定的是“当前简单局部匹配能改善检测”，而不是证明现实视频不存在运动对应。可能原因包括：

1. DINOv3 `14x14` patch token 具有上下文语义，局部最高 cosine 不一定对应同一物体点。
2. 一对多/重复纹理会让每一帧独立 matching 跳到不同位置，反而产生不稳定的二阶速度。
3. same-grid D2 中包含的局部位移和边界穿越信号本身对生成视频有判别力；alignment 把这部分信号消除了。
4. C3 entropy 更偏向保留“易匹配”区域，可能下调遮挡、形变和边界区域，而这些正是生成异常集中位置。
5. C2 在 GenVideo 有小幅收益，说明数据域的运动/分辨率组成影响 matching 作用，进一步支持先做 conditional dynamics 与 shortcut control，而不是增加 matcher 复杂度。

## 8. 决策

**H1 被否定，correspondence 不进入主线。**

- 保留 C0 same-grid 作为 Stage 2 的基础。
- 不跑 radius=2、全局 NxN、optimal transport 或 learned matcher。
- correspondence 模块保留为已验证的负对照，不并入默认论文方法。
- Stage 2 检验 trajectory geometry 时，必须明确其与 ReStraV、SPLIT TTR 和 MotionPhys 的重叠；几何量只有在 sample efficiency 或 conditional likelihood 上提供新证据时才继续。

权威结果位于 `results/analysis/stage1_correspondence/`；三个 run 均保存各数据集 real-only Local 参数、calibration IDs、SHA256、逐窗口/逐视频分数和日志。
