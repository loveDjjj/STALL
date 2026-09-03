# 2025-2026 视频帧/窗口选择相关工作审计

> 检索截止：2026-09-03。核心工作均阅读论文与官方仓库；研究对象以 long-video QA 为主，因此可迁移的是预算、覆盖和选择机制，不是 query relevance 目标。  
> CAES 暂定全称：Calibration-Aware Evidence Search，仅作内部名，实验成功前不作为论文方法名。

## 1. 术语表

| Canonical term | 定义 | 本项目决策 |
|---|---|---|
| Temporal selector | 在固定 K 个连续窗口预算下选择窗口位置 | 不称 keyframe sampler，避免把 frame 与 2 秒 window 混用 |
| Coarse scan | 低 FPS、仅 Global token 的全视频扫描 | 默认 1 FPS |
| Dense forensic window | 8 FPS、16 帧、2 秒的 Global+Local D2 检测窗口 | 与现有方法完全一致 |
| Real Temporal Forensic Signal | coarse transition 在 real-only reference 下的异常分位 | FS3-FS5 的选择信号 |
| Matched adaptive calibration | calibration real 与 test 使用完全相同 selector，并为每个 selector建立视频级 null | 所有 adaptive 实验的必要条件 |
| WindowManifest | 候选、选择分数、最终窗口和 selector identity 的独立产物 | 必须落盘 |

## 2. 方法总表

| Method | 原始任务 | Unit | Signal | Coverage | Diversity | Training-free | Query | Learned selector | 适配 CAES | Overlap risk |
|---|---|---|---|---|---|---|---|---|---|---|
| [AKS](https://openaccess.thecvf.com/content/CVPR2025/html/Tang_Adaptive_Keyframe_Sampling_for_Long_Video_Understanding_CVPR_2025_paper.html) | 长视频 QA | frame/bin | BLIP/CLIP/SeViLA query relevance | 递归 judge-and-split | 时间 bin 均衡 | 是 | 必须 | 否 | coverage 思想可迁移 | 中：固定预算 relevance+coverage 已存在 |
| [MDP3](https://openaccess.thecvf.com/content/ICCV2025/html/Sun_MDP3_A_Training-free_Approach_for_List-wise_Frame_Selection_in_Video-LLMs_ICCV_2025_paper.html) | Video-LLM | frame/segment | SigLIP query similarity + Gaussian kernel | 固定 segment DP | DPP log-det | 是 | 必须 | 否 | 可作为后续 diversity solver | 高：DPP/list-wise/sequentiality 不能 claim |
| [FOCUS](https://openreview.net/forum?id=1OQKqLFcbB) | 长视频 QA | clip-as-arm/frame | query similarity mean + Bernstein radius | coarse 探索所有 arms | uncertainty-driven exploration | 是 | 必须 | 否 | coarse-to-fine 可迁移 | 中高：bandit/explore-exploit 已被占据 |
| [WFS-SB](https://arxiv.org/abs/2603.00512) | 长视频 QA | semantic segment/frame | query-frame relevance curve | boundary segments 分预算 | MMR | 是 | 必须 | 否 | noisy score smoothing 可后续研究 | 高：wavelet/boundary/MMR 不可作第一轮 novelty |
| [DIG](https://arxiv.org/abs/2512.04000) | 长视频 QA | segment/frame | DINO content boundary + MLLM reward | global query 用 uniform | localized query 内均匀采样 | 部分，依赖预训练/服务模型 | 必须 | query/reward 模型 | DINO coarse boundary 是相关先验 | 中高：hybrid uniform/adaptive 已存在 |
| [KFS-Bench](https://openaccess.thecvf.com/content/WACV2026/html/Li_KFS-Bench_Comprehensive_Evaluation_of_Key_Frame_Sampling_in_Long_Video_WACV_2026_paper.html) | selection benchmark | frame/scene | CLIP query relevance | scene hit/balanced recall | clustering/CDF balance | 是 | 必须 | 否 | 证明 coverage 不能只看 Top-K | 高：balance/coverage 指标不是新意 |
| [Deepfake selection benchmark](https://doi.org/10.3390/app16115364) | 人脸 deepfake | face frame | uniform/motion/quality/shot/landmark | shot-aware/uniform | visual/landmark | selector多为非学习 | 否 | 部分质量模型 | 最接近 fixed detector control | 高：不能假设 motion selector 普遍最佳 |
| [VADTree](https://arxiv.org/abs/2510.22693) | training-free VAD | event node/segment | learned GEBD boundary + VLM/LLM priors | hierarchical tree | redundancy removal/clustering | detector training-free | anomaly priors | 使用预训练 GEBD | 提醒 adaptive anomaly sampling 已存在 | 高：coarse-to-fine anomaly search 宽泛 claim 已占据 |

## 3. AKS

### 原始机制

AKS 将 frame subset 目标写成 query relevance 总和与时间覆盖项的联合优化。覆盖通过递归二分时间轴近似：若某个 bin 的 relevance 不够集中则继续拆分；达到停止条件后按 bin 深度分配选帧数，并在 bin 内取 Top-K。官方实现使用约 1 FPS 的 BLIP/CLIP/SeViLA query-frame score，默认最大深度 5，并保存 selected frame indices。

### 可迁移思想

- relevance-only Top-K 会集中于短时间段，固定预算需要覆盖约束；
- recursive partition 是结构性 coverage，而不是 learned policy；
- selector 输出应与 detector 解耦并缓存。

### 不应照搬

- CAES 没有文本 query；不能把 real anomaly 生硬称为 query relevance；
- AKS 的阈值 `t1/t2/depth` 为 QA 场景设计，不应迁移；
- 第一轮 K=3 很小，递归树复杂度没有必要。

## 4. MDP3

MDP3 强调 relevance、list-wise diversity 和 sequentiality。官方实现先用 SigLIP 获取 query/image embedding，将候选按 32 帧 segment 分组；在每个 segment 内用多尺度 Gaussian kernel 构造 DPP，目标包含 query relevance 的 log 项和 selected-frame kernel 的 log determinant，再用 dynamic programming 在 segments 间分配 8 帧预算，并以前一个 segment 的少量选择作为条件。

可迁移的是“选择集合的价值不等于独立 frame score之和”和“时间 segment 应进入预算分配”。但 DPP/DP 对 K=3 forensic windows 过重，且其 novelty 与 MDP3 高度重叠。CAES 第一轮用 NMS 和三 strata 足以回答 relevance/coverage 问题；只有 FS 成功后才能把 DPP/MMR当 solver baseline。

## 5. FOCUS

FOCUS 把短时间 clip 当作 arm，先在每个 arm 采中心和随机点估计 reward mean/variance，再用 Bernstein-style optimistic upper bound 选 promising arms进行 fine sampling，最后结合 Top-score 与 arm allocation选固定数量 frame。官方默认 coarse 间隔约 16 秒、fine 1 秒，并完整保存 coarse/fine/arm sampling details。

它最值得迁移的是：

1. coarse scan 与 dense inspection 是两个预算层级；
2. 所有区域应先获得最低探索预算；
3. selector 必须保存完整决策轨迹与计算量。

不应第一轮照搬 bandit uncertainty。CAES 的视频仅数秒至几十秒、K=3，1 FPS 全扫描成本不足 1 GiB cache；复杂 bandit 不但收益有限，还会增加超参数与 novelty overlap。

## 6. WFS-SB

WFS-SB 对 noisy query-frame relevance curve 做 wavelet multi-resolution decomposition，从 detail signal 检测 semantic boundaries，将视频分成 coherent segments；segment importance由 duration、mean/max/variance relevance 加权，预算通过 softmax 分配，segment 内使用 MMR平衡 relevance 与视觉多样性。边界弱时官方 fallback 是一半 uniform、一半 Top-score。

可迁移的是“score curve 先平滑/分段，再做预算分配”和“边界弱时保留 uniform coverage”。但其 wavelet、四项 segment weight、peak prominence、temperature 和 MMR lambda 均是额外自由度。Stage FS 若未过 gate，禁止进入这一层；若成功，wavelet 只能用于 Real Temporal Forensic Signal 的 regime boundary，而不能声称新颖的 wavelet selector。

## 7. DIG

DIG 先把问题分为 global/localized。global query 直接 uniform；localized query 使用 DINOv2 以 2 samples/s 获取 frame features，对相邻 cosine difference 做 peak detection得到 content boundaries，再用 MLLM为代表帧分配 query reward，提取高 reward segments，最后在合并后的 segments 内均匀选 K 帧。

与 CAES 的重要差异：DIG 的 DINO变化只找 content boundary，真正选择依赖 query/MLLM reward；CAES拟使用仅真实视频拟合的 transition likelihood，不依赖语义问题或 fake。其 hybrid原则支持 FS5：当 forensic signal 不可靠时，coverage不能完全让位给 Top-K。

## 8. KFS-Bench

KFS-Bench 指出仅统计 selected frames命中相关场景的比例会奖励集中选择。它同时报告 Key Frame Rate、Scene Hit Rate、Balanced Scene Recall 和 Balanced Distribution Similarity，并组合为 UKSS。其 ASCS 根据 query relevance distribution的时间熵、质量熵和覆盖长度，在 similarity sampling 与 clustering sampling之间平衡。

CAES 不具备 ground-truth forensic scenes，不能直接计算 UKSS；但必须保存以下 coverage diagnostics：selected center span、stratum hit、pairwise temporal distance、候选分数与 coverage 的关系。只报 Top-K anomaly score不足以说明 selector质量。

## 9. Deepfake frame-selection benchmark

Applied Sciences 2026 在固定四个 pretrained detector下比较 12 种 selection和 2/4/8/16/32帧预算。关键结论不是“motion最好”，而是 selector与 detector存在交互：Uniform、Quality、Shot-aware、Diversity和 landmark策略各自在不同 detector上占优，复杂选择不一定带来成比例收益。

这直接支持 CAES 的受控设计：FS0-FS5 必须固定 detector、K、窗口长度、aggregation 和 fusion，只改变窗口位置；FS2 motion/change只能作为 heuristic baseline，不能预设为优于 uniform。

## 10. VADTree 与最近工作

VADTree 已提出 training-free generic video anomaly detection中的 adaptive coarse-to-fine segment sampling，使用预训练 GEBD边界、hierarchical tree、VLM/LLM anomaly priors和冗余消除。它与 CAES 在“有限预算主动寻找异常证据”的宽泛叙事高度重合。

CAES 仍可能成立的窄差异是：

- 任务为 AI-generated video detection而非开放世界事件异常；
- signal 为 frozen DINO transition的 real-only Gaussian likelihood；
- dense detector仍是 same-grid vector D2，不依赖VLM/LLM推理；
- 核心统计问题是 adaptive search-induced extreme bias，并对每个 selector构造 matched real video-level null。

截至检索日，没有发现同时具备“generated-video forensic window search + real-only selector likelihood + selector-matched null calibration”的公开工作。但这只是当前检索结论，不能写“首次”而不在投稿前再次更新检索。

## 11. 与 STALL、D3、SPLIT 的边界

- STALL 固定采样并对 Global Spatial/T1 做 real likelihood；它没有 adaptive window search，也没有 matched selector-specific video null。
- D3 使用全局 scalar second-order statistics；没有窗口预算搜索或 real likelihood。
- SPLIT 对 patch TTR/LSMI做 training-free detection和 real-only threshold；没有 coarse-to-dense窗口搜索，但已占据 patch roughness、partial fake和低 FPR叙事。

## 12. 最终研究判断

### KEEP

- 固定预算下的 relevance/coverage受控对照；
- single-pass coarse scan；
- Top-K、NMS、stratified 三个复杂度递增 selector；
- 完整 WindowManifest和选择成本；
- matched adaptive calibration。

### MODIFY

- 将“keyframe”统一改为“forensic window”；
- coarse signal用 real-only transition anomaly，不用 query similarity；
- 第一轮不使用 wavelet/DPP/bandit；
- FS3 primary用 window mean anomaly，max仅保存为诊断；
- FS5优先于 FS4作为最可能稳定方案，因为 K=3下覆盖比局部去重更重要。

### REJECT

- 从 QA代码复制阈值、lambda、temperature；
- 以 motion/change峰值默认代表 fake artifact；
- 只对 test adaptive、calibration uniform；
- 把 adaptive sampling或 coarse-to-fine本身作为 novelty claim；
- Stage FS未过 gate后继续 wavelet/DPP/FOCUS-style复杂化。

## 13. 核心来源

- [AKS paper](https://openaccess.thecvf.com/content/CVPR2025/html/Tang_Adaptive_Keyframe_Sampling_for_Long_Video_Understanding_CVPR_2025_paper.html), [official code](https://github.com/ncTimTang/AKS)
- [MDP3 paper](https://openaccess.thecvf.com/content/ICCV2025/html/Sun_MDP3_A_Training-free_Approach_for_List-wise_Frame_Selection_in_Video-LLMs_ICCV_2025_paper.html), [official code](https://github.com/sunh-23/MDP3)
- [FOCUS paper](https://openreview.net/forum?id=1OQKqLFcbB), [official code](https://github.com/NUS-HPC-AI-Lab/FOCUS)
- [WFS-SB paper](https://arxiv.org/abs/2603.00512), [official code](https://github.com/MAC-AutoML/WFS-SB)
- [DIG paper](https://arxiv.org/abs/2512.04000), [official code](https://github.com/Jialuo-Li/DIG)
- [KFS-Bench paper](https://openaccess.thecvf.com/content/WACV2026/html/Li_KFS-Bench_Comprehensive_Evaluation_of_Key_Frame_Sampling_in_Long_Video_WACV_2026_paper.html), [official code](https://github.com/NEC-VID/KFS-Bench)
- [Deepfake frame-selection benchmark](https://doi.org/10.3390/app16115364)
- [VADTree paper](https://arxiv.org/abs/2510.22693), [official code](https://github.com/wenlongli10/VADTree)
