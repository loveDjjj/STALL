# 2025-2026 Training-Free 生成视频检测相关工作审计

> 检索截止：2026-09-02。  
> 信息优先级：正式论文/会议页面 > arXiv 原文 > 官方代码。核心方法均同时阅读论文与官方实现；未公开代码的工作明确标记。  
> 本文中的 `training-free` 指检测阶段不使用真假视频训练额外分类器；使用真实视频统计或阈值校准不自动等于 calibration-free。

## 1. 术语口径

| 术语 | 本文定义 |
|---|---|
| Training-free | 不用 real/fake 标签训练额外 detector/classifier；允许冻结基础模型推理与非学习统计 |
| Fake-free fitting | 参数、密度、阈值和融合不使用生成视频 |
| Target-real-calibrated | 使用目标数据域真实视频建立统计或 CDF |
| Cross-real calibration | 在真实域 A 定阈值，在真实域 B 测真实 FPR/假视频 recall |
| Zero-shot | 文献含义不统一；本文避免单独使用，必须同时说明 fake training 与 real calibration |
| Same-grid | 相邻帧相同 patch 网格索引，不保证同一物体/内容对应 |

## 2. 方法总表

| Method | Year | Training-Free? | Fake Training? | Backbone | Global/Local | Temporal Cue | Statistical Model | Calibration | Dataset | Code | Overlap with us |
|---|---:|---|---|---|---|---|---|---|---|---|---|
| [D3](https://arxiv.org/abs/2508.00701) | ICCV 2025 | 是 | 否 | CLIP/XCLIP/DINOv2/CNN | Global | 相邻帧标量距离的一阶序列再差分 | D2 均值/标准差，无 likelihood | 无 real density | GenVideo、EvalCrafter、VideoPhy、VidProM | [官方](https://github.com/Zig-HS/D3) | 二阶时间变化高度重叠；数学对象不同 |
| [ReStraV](https://proceedings.neurips.cc/paper_files/paper/2025/hash/1d9a43752c2819e03967c5c1b708169c-Abstract-Conference.html) | NeurIPS 2025 | 否 | 是 | DINOv2-S/14 | Global token+patch 展平轨迹 | step distance、turning angle/curvature | 21-D descriptor + MLP | 真假训练集和 F1 阈值 | VidProM 等 | [官方](https://github.com/ChristianInterno/ReStraV) | 轨迹几何高度重叠；real-only likelihood 可形成差异 |
| [Over-Coherence](https://openaccess.thecvf.com/content/WACV2026/html/Brokman_Training-free_Detection_of_Text-to-video_Generations_via_Over-coherence_WACV_2026_paper.html) | WACV 2026 | 是 | 否 | CLIP ViT-L/14 | Global | 相邻帧 cosine sequence 的变化/极值 | 手工 criterion | 非密度；操作点依协议 | 23 generators、54.4K videos | [官方](https://github.com/FujitsuResearch/training-free-detection-of-text-to-video-generations-via-over-coherence) | 平滑/速度偏差与二阶 scalar cue 重叠 |
| [STALL](https://openaccess.thecvf.com/content/CVPR2026/html/Hayun_Training-free_Detection_of_Generated_Videos_via_Spatial-Temporal_Likelihoods_CVPR_2026_paper.html) | CVPR 2026 | 是 | 否 | DINOv3 ViT-L/16 | Global | 归一化 Global T1 | PCA whitening + Gaussian LL | VATEX real percentile | VideoFeedback、GenVideo、ComGenVid | [官方](https://github.com/OmerBenHayun/STALL) | 我们的 Global 直接继承；不能作为新增创新 |
| [SPLIT](https://arxiv.org/abs/2607.02886) | ECCV 2026 | 是 | 否 | XCLIP/CLIP/DINOv2/CNN | Local patch | TTR + LSMI | 手工乘法分数 | real-only threshold，cross-real transfer | FakeParts、GenVideo、ViF-Bench | [官方](https://github.com/mldljyh/SPLIT) | same-grid patch roughness、局部运动一致性、部分伪造、低 FPR 高度重叠 |
| [MotionPhys](https://arxiv.org/abs/2608.20770) | arXiv 2026 | 论文称轻量 | 待核实 | 光流/轨迹 | Sparse trajectories | 多尺度物理轨迹几何 | compact representation，细节待代码 | 待核实 | 多数据集 | 截止检索未发现官方代码 | “trajectory geometry/physical consistency” claim 高度重叠 |
| [LAVID](https://arxiv.org/abs/2502.14994) | 2025 | 是 | 否 | LVLM | Global reasoning | 视觉/时序语言推理 | prompt/inference criterion | 无 real density | 自建与公共集 | 见论文 | training-free 大类重叠，机制不同 |
| [Retrieval-Driven Attribution](https://arxiv.org/abs/2607.28955) | arXiv 2026 | 是 | 否/检索库 | 检索特征 | Global | 来源相似性 | retrieval ranking | reference bank | GenVidBench | 见论文 | 任务侧重 attribution，非主要碰撞 |
| [VidAudit](https://arxiv.org/abs/2606.31004) | arXiv 2026 | 评测框架 | 不适用 | 多 detector | 多类 | shortcut/control audit | 六控制协议 | 低 FPR 与跨域 | GenVidBench、AIGVDBench | [官方](https://github.com/KurbanIntelligenceLab/vidaudit) | 对我们的论证威胁极高：必须补协议审计 |
| [Dataset Biases and Shortcut Learning](https://arxiv.org/abs/2607.00948) | arXiv 2026 | 评测研究 | 不适用 | 多 detector | Motion-based | motion imbalance、FPS/重复帧偏差 | 重平衡与扰动审计 | 跨数据集 | 多 benchmark | 以论文为准 | 直接挑战 D2/K3 增益是否来自 motion shortcut |

## 3. 核心方法复原

### 3.1 STALL

STALL 用冻结 DINOv3 ViT-L/16 提取每帧 Global embedding `x_t`。校准阶段只使用真实 VATEX 视频：

1. Spatial：每个视频随机取一帧估计均值与经验 covariance。
2. Temporal：对全部 `normalize(x[t+1]-x[t])` 估计均值与 covariance。
3. 对 covariance 做特征分解，最多保留 `min(N-1,D)` 个正特征值方向，使用 `1/sqrt(lambda+1e-5)` 白化。
4. 在白化空间按标准高斯计算 log-likelihood。
5. Spatial 窗口取最大 likelihood，Temporal 窗口取最小 likelihood。
6. 两者分别映射到 VATEX real calibration percentile，再等权平均。

官方公式与代码一致。`max Spatial` 倾向于判断视频是否至少包含一帧高度真实的外观，`min Temporal` 捕获最异常的相邻帧转移；这两个选择也经过其论文的相关性/聚合消融。STALL 所称 training-free 是“不训练 detector、不使用 generated content”，不是“不需要真实参考库”。

对我们的约束：Global 分支是继承资产，不应列为贡献；多窗口目标域视频级再校准是我们的协议变化，必须单独披露。

### 3.2 D3

D3 官方实现的准确计算为：

```text
z[t]  = frozen encoder pooled/global embedding
d1[t] = cosine_similarity(z[t], z[t+1])
         或 ||z[t]-z[t+1]||_2
d2[t] = d1[t+1] - d1[t]
score  = mean(d2), std(d2)
```

官方评测使用 D2 的统计量直接区分真假，不拟合 real Gaussian likelihood，也不做 Local patch trajectory。它的 D2 是 scalar distance/similarity sequence 的差分，不是 feature vector 的中心二阶差分。

对我们的约束：可以准确声称“局部向量二阶似然不同于 D3 的全局标量距离二阶统计”，但不可以声称首次提出 second-order temporal detection，也不宜把牛顿/加速度直觉作为独占理论贡献。

### 3.3 ReStraV

ReStraV 将每帧 DINOv2 CLS 与 patch token 拼接并展平为表示轨迹 `Z_t`，计算：

- stepwise distance `||Z[t+1]-Z[t]||`；
- 相邻 velocity 的夹角 `acos(cos(delta[t],delta[t+1]))`；
- 前 7 个 distance、前 6 个 angle；
- distance 和 angle 各自的 mean/min/max/variance。

合计 `7 + 6 + 8 = 21` 维 descriptor。官方代码使用 real 和 fake 平衡样本训练两层 MLP，并在训练集上按 F1 选择阈值，因此它不是 training-free，也不是 fake-free。

对我们的启发与边界：curvature、turning angle 和 step distance 已经是明确先验。把同类几何量放入 real-only likelihood 可以改变学习协议，但不能把几何量本身声称为新发现。2026-08 的 MotionPhys 又进一步占据“物理轨迹、多尺度几何”叙事。

### 3.4 Over-Coherence

官方实现从 CLIP ViT-L/14 逐帧 embedding 构造相邻帧 cosine similarity：

```text
c[t] = cos(e[t], e[t+1])
criterion_legacy = max(c)
criterion = max(abs(c[t+1]-c[t]))
```

论文核心假设是生成视频可能表现出不自然的过度时间一致性，而不是所有 fake 都更粗糙。它还分析了不同 FPS、微小时间扰动和 blind setting。

对我们的约束：仅以“fake 存在异常速度/平滑度”为故事不足以构成新意；Local D2 也可能同时响应过粗糙与过平滑，因此 likelihood 的双尾行为应被显式分析，而不是只描述闪烁。

### 3.5 SPLIT

SPLIT 是当前最重要的 novelty collision。

#### TTR

对每个 fixed-grid patch trajectory `P[:,i,:]`：

```text
L1 = sum_t ||P[t+1]-P[t]||_2
L2_raw = sum_t ||P[t+2]-P[t]||_2
L2 = (L2_raw/2) * (T-1)/(T-2)
TTR_i = (log(L1+eps)-log(L2+eps))/log(2)
TTR = mean_i(TTR_i)
```

它不是我们的向量 D2 Gaussian likelihood，但同样是 same-grid patch trajectory 的两步时序粗糙度/路径-弦关系。

#### LSMI

先计算同网格 patch motion field：

```text
M[t,x,y] = P[t+1,x,y] - P[t,x,y]
grad_x = M[...,x,y] - M[...,x,y+1]
grad_y = M[...,x,y] - M[...,x+1,y]
LSMI = mean(||grad_x||) 与 mean(||grad_y||) 的平均
```

#### Fusion 与协议

```text
score = TTR^gamma * LSMI
gamma = 8
```

论文说明 gamma 在一个小 held-out validation split 选一次，然后固定。阈值只用 real calibration 设置到 0.1%/1%/5% FPR；还执行 ROVI 与 MSR-VTT 的 cross-real threshold transfer。SPLIT 在 FakeParts 上评估部分时空/空间编辑，在 GenVideo 与 ViF-Bench 上评估完整生成，并测试 blur、JPEG、翻转鲁棒性。

对我们的结论：patch roughness、same-grid motion、局部空间运动一致性、部分 fake、real-only threshold 和超低 FPR 已被覆盖。Correspondence-aware conditional real likelihood 仍有潜在差异，但必须用受控实验和精确数学定义证明，不可只换名称。

## 4. 2026 新工作带来的研究边界

### 4.1 MotionPhys

MotionPhys 直接以 sparse optical-flow trajectories 的物理一致性和多尺度几何做检测。截止检索日未找到可核验官方代码，因此暂不能断言其训练协议与 descriptor 细节。但它已经使“AI 视频违反轨迹几何/物理一致性”成为拥挤叙事。

结论：Trajectory Geometry 仍可做诊断，但不建议作为 Top-1 主线；除非我们的核心是“冻结 patch correspondence 下的 conditional real-only likelihood”，且证据显示该统计在严格 shortcut audit 后仍有效。

### 4.2 VidAudit 与 motion shortcut 审计

VidAudit 提出六类控制，包括统一重编码、时长泄漏过滤、real-vs-real dataset identity floor、matched harness、multi-seed/bootstrap 和真跨数据集。另一项 motion-bias 研究指出 FPS、重复帧、分辨率和 real/fake motion magnitude 不平衡可显著抬高 motion detector 表现。

结论：对当前方法而言，最大的投稿风险已经不只是“分数是否更高”，而是“分数是否真的测到生成痕迹”。至少应加入：

1. 统一编码/FPS 后复测；
2. motion-matched 或 motion-stratified evaluation；
3. real-vs-real domain classifier/floor；
4. 单帧扰动、重复帧、scene cut 压力测试；
5. 0.1% FPR 和跨真实域阈值转移。

## 5. 新数据集与 confirmation 选择

| Dataset | 规模/内容 | 价值 | 风险/成本 | 建议 |
|---|---|---|---|---|
| [AIGVDBench](https://openaccess.thecvf.com/content/CVPR2026/html/Ma_Your_One-Stop_Solution_for_AI-Generated_Video_Detection_CVPR_2026_paper.html) | 31 generators，超过 440K videos，T2V/I2V/V2V/商业模型 | 大范围 confirmation 与 shortcut audit | 全量存储/计算很高；需确认公开子集许可 | 先用官方 small/eval split，预注册后只跑最终候选 |
| [CoCoVideo-26K](https://openaccess.thecvf.com/content/CVPR2026/html/Feng_CoCoVideo_The_High-Quality_Commercial-Model-Based_Contrastive_Benchmark_for_AI-Generated_Video_Detection_CVPR_2026_paper.html) | 13 个商业生成器、26K、语义对齐 real/fake pairs | 很适合排除内容语义偏差 | 下载和许可需核查 | 优先级高，适合作为 untouched confirmation |
| [FakeParts](https://github.com/hi-paris/FakeParts) | 25K+ manipulated，加 real；时空/像素标注 | 检验 likelihood field 与 partial fake localization | 数据较大；SPLIT 已是强基线 | 仅在 tail/field 候选成立后下载 |
| [ViF-Bench/Skyra](https://openaccess.thecvf.com/content/CVPR2026/html/Li_Skyra_AI-Generated_Video_Detection_via_Grounded_Artifact_Reasoning_CVPR_2026_paper.html) | 3K 高质量、10+ 新生成器 | 新生成器、小规模 confirmation | 获取状态和许可需核查 | 优先于超大规模全量下载 |
| [FVBench](https://openaccess.thecvf.com/content/CVPR2026/html/Wang_FVBench_Benchmarking_Deepfake_Video_Detection_Capability_of_Large_Multimodal_Models_CVPR_2026_paper.html) | 120K+，42 synthesis/edit models | 完整生成与编辑统一压力测试 | 任务混杂、存储较高 | 作为后续广义 forgery 扩展，不是第一 confirmation |

## 6. 对当前主线的直接判断

1. **Local Spatial 不应恢复。** 已有内部受控结果为负，且 SPLIT 的 LSMI 已提供更直接的局部空间运动信号。
2. **same-grid D2 应降级为 baseline/special case。** 它仍是有效证据，但 novelty 空间被 D3、ReStraV、SPLIT 和 MotionPhys 显著压缩。
3. **Correspondence 只是一项可检验假设，不是预设贡献。** 如果 C3 相对 C0 不足 `+0.005` 且不跨 2/3 数据集稳定，立即停止复杂 matching。
4. **最有防守力的方向是条件真实动力学统计。** 重点不是再造一个 trajectory descriptor，而是检验 `p(local dynamics | motion state, correspondence confidence)` 是否比无条件 `p(D2)` 更稳定且更省真实校准样本。
5. **协议创新必须和方法创新并重。** 0.1% FPR、cross-real、motion-matched 和 untouched confirmation 已成为 2026 投稿的基本可信度要求。

## 7. 核心来源

- [STALL 论文（CVPR 2026）](https://openaccess.thecvf.com/content/CVPR2026/html/Hayun_Training-free_Detection_of_Generated_Videos_via_Spatial-Temporal_Likelihoods_CVPR_2026_paper.html)；[官方代码](https://github.com/OmerBenHayun/STALL)
- [D3 论文（ICCV 2025）](https://openaccess.thecvf.com/content/ICCV2025/html/Zheng_D3_Training-Free_AI-Generated_Video_Detection_Using_Second-Order_Features_ICCV_2025_paper.html)；[官方代码](https://github.com/Zig-HS/D3)
- [ReStraV 论文（NeurIPS 2025）](https://proceedings.neurips.cc/paper_files/paper/2025/hash/1d9a43752c2819e03967c5c1b708169c-Abstract-Conference.html)；[官方代码](https://github.com/ChristianInterno/ReStraV)
- [Over-Coherence 论文（WACV 2026）](https://openaccess.thecvf.com/content/WACV2026/html/Brokman_Training-free_Detection_of_Text-to-video_Generations_via_Over-coherence_WACV_2026_paper.html)；[官方代码](https://github.com/FujitsuResearch/training-free-detection-of-text-to-video-generations-via-over-coherence)
- [SPLIT 论文（ECCV 2026）](https://arxiv.org/abs/2607.02886)；[官方代码](https://github.com/mldljyh/SPLIT)
- [MotionPhys（2026）](https://arxiv.org/abs/2608.20770)
- [VidAudit（2026）](https://arxiv.org/abs/2606.31004)；[官方代码](https://github.com/KurbanIntelligenceLab/vidaudit)
- [Dataset Biases and Shortcut Learning（2026）](https://arxiv.org/abs/2607.00948)
