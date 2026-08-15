# Alpha-STALLED：基于 STALL 的局部二阶时序证据扩展

> 项目来源：Omer Ben Hayun、Roy Betser、Meir Yossef Levi、Levi Kassel、Guy Gilboa  
> 论文：**Training-free Detection of Generated Videos via Spatial-Temporal Likelihoods**，CVPR 2026  
> 论文链接：[arXiv:2603.15026](https://arxiv.org/abs/2603.15026)  
> 原始项目：[STALL 官方实现](https://github.com/OmerBenHayun/STALL)

本文档整理当前仓库中的原版 STALL 复现、Patch-STALL/Alpha-STALLED 方法演进、完整模型流程、主实验与消融实验，并对现有代码的用途、冗余和科研有效性进行审计。

## 摘要

原版 STALL 使用 DINOv3 的帧级全局特征，分别建模单帧外观的空间似然和相邻帧变化的时序似然，再通过真实视频校准集把两类似然转换成百分位分数。该方法训练自由、只需要真实视频校准，但全局特征可能把面积较小、持续时间较短的局部生成伪影平均掉。

本项目在 STALL 上增加 patch 级空间与时序建模。实验表明，最有效的新信号不是单帧 patch 外观，也不是显式 patch 匹配，而是 **same-grid second-order temporal evidence**：在连续三帧的同一 patch 网格位置计算二阶差分，用于检测局部纹理闪烁、边缘跳变和运动不连续。将该 patch 分数与原版 STALL 全局分数固定融合后，三个数据集的 AUC 均超过原版基线。

当前仓库还实现了 persistence、sample fallback 和 split-minus selector。其内部结果进一步提高到 ComGenVid 0.9327/0.9357、VideoFeedback 0.8833/0.9184、GenVideo 0.8557/0.9869。但是，现有 fallback 实现会读取测试样本的 subset（真实/生成标签）和 fake source_model 来决定路由，并在整个测试批次上计算 rank，因此这些数字目前只能视为内部诊断上限，不能直接作为未知视频检测的公平主结果。

基于代码审计，本文建议把论文主线收敛为：

~~~text
原版 STALL 全局分支
+ patch same-grid second-order 分支
+ 固定 alpha 融合
~~~

把 persistence/fallback 保留为待修复的研究扩展。修复标签依赖和测试集 rank 依赖后，再决定是否晋升为最终方法。

## 1. 研究动机

### 1.1 原版 STALL 如何检测生成视频

原版 STALL 的基本假设是：真实视频在预训练视觉特征空间中的单帧分布和帧间变化分布具有相对稳定的统计规律，而生成视频会偏离这些规律。

对输入视频采样得到的帧序列，原方法使用 DINOv3 ViT-L/16 提取每帧全局特征：

~~~text
g_t in R^D, t = 1, ..., T
~~~

随后建立两个分支：

1. 全局空间分支：判断单帧全局特征 g_t 是否符合真实视频分布。
2. 全局时序分支：判断相邻帧特征差分 g_(t+1)-g_t 是否符合真实视频运动分布。

两个分支都在真实视频校准集上拟合 whitening 变换，并在白化空间中使用标准高斯对数似然：

~~~text
z = (x - mu) W
log p(z) = -0.5 * (D * log(2*pi) + ||z||^2)
~~~

视频级空间、时序统计分别映射到真实校准分布中的 percentile，最终得分为：

~~~text
global_score = 0.5 * global_spatial_percentile
             + 0.5 * global_temporal_percentile
~~~

分数方向统一为：

~~~text
higher score = more likely real
lower score  = more likely generated
~~~

### 1.2 原方法可以优化的地方

原版 STALL 的优点是结构简单、无需生成视频训练样本、泛化假设清楚。但其输入是每帧一个全局向量，存在三个局限：

1. **局部异常可能被全局池化稀释。** 生成视频中的手部、边缘、纹理或背景局部闪烁只占画面一小部分，整帧向量未必敏感。
2. **全局一阶差分难以描述局部动态不连续。** 局部纹理的突然变化可能不显著改变整帧语义，但会在 patch token 的二阶变化中体现。
3. **全局分数不能定位异常。** 即使判断视频异常，也难以说明问题集中在哪些帧和空间区域。

### 1.3 本项目的核心假设

本项目不替换原版全局分支，而是补充局部证据：

~~~text
Global evidence:
  视频整体外观与整体运动是否符合真实视频统计。

Local spatial evidence:
  某些局部区域的外观是否异常。

Local temporal evidence:
  某些局部区域随时间的变化是否不连续。
~~~

实验探索最终支持以下判断：局部空间证据单独较弱，局部二阶时序证据是主要增益来源，全局与局部证据具有互补性。

## 2. 整体框架

### 2.1 论文主线建议

~~~mermaid
flowchart TD
    A[输入视频] --> B[按 8 FPS 采样固定时长帧序列]
    B --> C[DINOv3 ViT-L/16]

    C --> D1[帧级全局特征]
    C --> D2[14 x 14 Patch Token 网格]

    D1 --> E1[全局空间似然]
    D1 --> E2[全局一阶时序似然]
    E1 --> F1[原版 STALL 全局分数]
    E2 --> F1

    D2 --> G1[Patch 空间似然]
    D2 --> G2[同网格二阶时序似然]
    G1 --> H1[局部 Patch 分数]
    G2 --> H1

    F1 --> I[固定 Alpha 融合]
    H1 --> I
    I --> J[最终真实性分数]
~~~

该版本的优点是：所有样本使用完全相同的计算路径，不需要知道样本真实标签或生成器来源，可以作为论文主方法和部署方法继续完善。

### 2.2 当前仓库的完整研究流程

~~~mermaid
flowchart TD
    A[视频或缓存特征] --> B1[Global STALL Score]
    A --> B2[Raw Patch Score]
    A --> B3[Patch Temporal Likelihood Map]

    B3 --> C[Persistence Feature]
    B1 --> D[Raw Base: 0.60 Global + 0.40 Patch]
    B2 --> D

    B1 --> E[Disagreement Gate]
    B2 --> E
    C --> E

    D --> F1[Universal Score]
    D --> F2[Split Score]
    E --> F1
    E --> F2

    F1 --> G[Split-minus Selector]
    F2 --> G
    G --> H[Frozen Final Score]
~~~

第二张图反映当前最高内部结果的实现，但其中 fallback 路径仍有标签依赖，不能直接作为部署流程。

## 3. 方法细节

### 3.1 视频采样与输入

原论文默认使用 8 FPS、2 秒窗口，即每个视频通常使用 16 帧。对于时长只有 1 秒的生成器，如 HotShot、MoonValley 或 Hotshot-XL，需要使用 1 秒窗口并与对应的 1 秒真实视频比较。

视频帧经过 DINOv3 标准预处理：缩放到 224 x 224，使用 ImageNet 均值和方差归一化。

### 3.2 DINOv3 全局与局部特征

DINOv3 ViT-L/16 对每帧输出：

~~~text
global token: [D] = [1024]
patch tokens: [P, D] = [196, 1024]
patch grid:   14 x 14
~~~

一个视频的特征形状为：

~~~text
global sequence: [T, D]
patch sequence:  [T, P, D]
~~~

全局 token 用于复现原版 STALL，patch tokens 用于局部扩展。两类特征来自同一次 DINOv3 前向，不需要训练新的视觉骨干。

### 3.3 原版全局空间分支

对每帧全局特征 g_t 使用真实校准集拟合的空间 whitening 参数：

~~~text
z_t^spat = (g_t - mu_spat) W_spat
ll_t^spat = log N(z_t^spat; 0, I)
~~~

当前原版配置对帧级空间 likelihood 使用 max 聚合，再映射到校准集 percentile：

~~~text
global_spatial_score = Percentile_real(max_t ll_t^spat)
~~~

### 3.4 原版全局时序分支

先计算相邻帧全局特征差分并做 L2 归一化：

~~~text
d_t = Normalize(g_(t+1) - g_t)
~~~

再使用独立的时序 whitening 参数计算似然：

~~~text
z_t^temp = (d_t - mu_temp) W_temp
ll_t^temp = log N(z_t^temp; 0, I)
~~~

当前原版配置使用 min 聚合，以捕获视频中最异常的一次全局变化：

~~~text
global_temporal_score = Percentile_real(min_t ll_t^temp)
~~~

最后得到原版全局分数：

~~~text
global_score = 0.5 * global_spatial_score
             + 0.5 * global_temporal_score
~~~

### 3.5 Patch 空间分支

对每个 patch token p_(t,i) 使用真实视频 patch 拟合的 whitening 参数：

~~~text
z_(t,i)^patch-spat = (p_(t,i) - mu_patch_spat) W_patch_spat
ll_(t,i)^patch-spat = log N(z_(t,i)^patch-spat; 0, I)
~~~

得到的 likelihood map 形状为 [T, P]。可使用两类聚合：

~~~text
mean:
  平均所有时空位置，稳定但容易稀释局部异常。

bottom-k mean:
  只平均 likelihood 最低的 k% 位置，对局部异常更敏感。
~~~

Patch spatial 单独使用在 ComGenVid 上仅为 0.8090/0.8190，明显低于原版 STALL，因此它在最终 patch 分数中只占较小权重。

### 3.6 Patch 同网格二阶时序分支

对连续三帧中相同网格位置 i 的 patch token 计算二阶差分：

~~~text
a_(t,i) = p_(t+2,i) - 2 * p_(t+1,i) + p_(t,i)
a_(t,i) = L2Normalize(a_(t,i))
~~~

二阶差分近似描述局部特征轨迹的“加速度”。如果局部纹理和结构随时间平滑变化，二阶项通常较稳定；如果出现闪烁、边缘跳变、突然形变或局部运动断裂，二阶项更容易偏离真实视频分布。

与 motion-hard/motion-soft 不同，该方法不在相邻帧之间搜索 patch 对应关系，因此没有额外的匹配误差。它牺牲了严格的物体跟踪，但在当前实验中显著更稳定。

经过时序 whitening 和高斯似然后得到：

~~~text
ll_(t,i)^patch-temp
patch_temporal_score = Percentile_real(Aggregate(ll^patch-temp))
~~~

### 3.7 Patch 区域与聚合

代码支持先把相邻 patch 平均成更大的区域，再计算时序特征：

~~~text
region = 1: 原始单 patch，14 x 14 网格
region = 2: 2 x 2 patch 区域池化
region = 3: 3 x 3 patch 区域池化并裁剪到可整除范围
~~~

当前各数据集的最佳已验证 patch 配置不同：

| 数据集 | Patch temporal | Region | 聚合 | Patch 内权重 |
|---|---|---:|---|---|
| ComGenVid | same-grid second-order | 3 | bottom-k 0.50 | spatial 0.10 / temporal 0.90 |
| VideoFeedback | same-grid second-order | 1 | mean | spatial 0.10 / temporal 0.90 |
| GenVideo | same-grid second-order | 2 | mean | spatial 0.10 / temporal 0.90 |

这说明局部异常的空间尺度和稀疏程度存在数据集差异。它也意味着当前 patch 配置不是严格的单一通用配置；若要声称零样本部署，应在独立验证集上冻结统一配置，或给出不使用测试标签的数据集属性选择规则。

Patch 最终分数为：

~~~text
patch_score = 0.10 * patch_spatial_score
            + 0.90 * patch_temporal_score
~~~

### 3.8 固定 Alpha 融合

最简单、路径完全一致的融合为：

~~~text
base_score = alpha * global_score + (1 - alpha) * patch_score
~~~

当前内部冻结比较使用：

~~~text
alpha = 0.60
base_score = 0.60 * global_score + 0.40 * patch_score
~~~

这里必须区分 raw-space 和 rank-space：

~~~text
raw-space:
  直接融合 global_score 和 patch_score。

rank-space:
  先把两个分数在整批测试样本上转换成 rank01，再融合。
~~~

当前 release baseline 已统一使用 `src/metrics.py` 的逐生成器 pairwise 平衡口径，并把输入分数复制到 `results/paper_scores/`。在该口径下，raw alpha=0.60 的可复算结果为：ComGenVid 0.9198/0.9211、VideoFeedback 0.8628/0.8750、GenVideo 0.8374/0.8283。早期报告中的 0.9219/0.9229、0.8626/0.9039、0.8451/0.9857 以及 0.98+ AP 结果来自不同 score 文件或非统一 AP 口径，不再作为 release baseline。

### 3.9 Persistence 证据

Persistence 不重新提取视觉特征，而是把 patch temporal likelihood map 从“若干独立低分点”变成结构化异常图。

首先，对每个时间位置和 patch 位置计算二阶时序 likelihood，并使用真实视频中同一 patch 位置的 likelihood 分布做 percentile 校准：

~~~text
patch temporal likelihood map
  -> per-position real percentile map
  -> low-percentile anomaly mask
~~~

低 percentile 表示该局部变化在真实视频中少见。随后提取视频级形态统计：

| 特征 | 含义 |
|---|---|
| anom_mass | 异常 cell 占全部时间-patch cell 的比例 |
| active_frame_frac | 至少包含一个异常 patch 的帧比例 |
| active_patch_frac | 至少在一个时刻异常的 patch 位置比例 |
| max_frame_mass | 单帧中异常 patch 比例的最大值 |
| max_patch_mass | 同一 patch 位置在时间上异常比例的最大值 |
| longest_patch_run | 同一 patch 位置连续异常的最长时间比例 |

当前冻结清单并未在三个数据集使用同一 persistence 字段：

| 数据集 | 当前 persistence 字段 |
|---|---|
| ComGenVid | anom_mass_thr0p2_real_pct_real |
| VideoFeedback | max_frame_mass_thr0p2_real_pct_real |
| GenVideo | temp_pct_mean_real_pct_real |

因此，现有 persistence 结果证明了“局部异常形态具有补充信息”，但尚未证明一个统一 persistence 定义能跨数据集工作。

### 3.10 Sample fallback 与 split-minus selector

当前代码先计算整批样本的分数排序：

~~~text
global_rank      = rank01(global_score)
patch_rank       = rank01(patch_score)
persistence_rank = rank01(persistence_score)
~~~

rank01 表示把样本在整批分数中的排序归一化到 [0,1]。然后计算：

~~~text
disagreement = abs(global_rank - patch_rank)
persistence_conf = abs(persistence_rank - 0.5) * 2

sample_gate = 1,
  if disagreement >= 0.20 and persistence_conf >= 0.30
otherwise 0
~~~

保守路径使用最多 0.15 的 persistence 权重：

~~~text
u = 0.15 * sample_gate
universal_score = (1-u) * base_score + u * persistence_rank
~~~

更强路径使用最多 0.25 的 persistence 权重，但当前实现还会根据真实/生成 subset 和 fake source 的 gate 比例决定权重：

~~~text
split_score = (1-w_split) * base_score
            + w_split * persistence_rank
~~~

最终 selector 为：

~~~text
if split_score - universal_score < 0:
    final_score = split_score
else:
    final_score = universal_score
~~~

冻结配置还设置 real_policy=split，即已知为真实的测试样本强制走 split 路径。该规则是当前最高内部结果的重要组成部分，也是它暂时不能用于未知单视频推理的根本原因。

## 4. 实验设置

### 4.1 数据集

| 数据集 | 真实来源 | 当前主结果中的生成来源 | 备注 |
|---|---|---:|---|
| ComGenVid | MSVD | Sora、VEO3，共 2 类 | 默认 2 秒窗口 |
| VideoFeedback | DiDeMo、Panda70M | 11 类 | 10 类使用 2 秒窗口；Hotshot-XL 使用 1 秒窗口 |
| GenVideo | MSR-VTT | 当前完成 8 类 | HotShot、MoonValley 的 Patch 分支尚未完成 |

原论文使用 VATEX 约 3.3 万个真实视频作为校准集，与三个测试数据集分离，不使用生成视频拟合 whitening 参数。

### 4.2 指标

使用 ROC-AUC 和 Average Precision（AP），方向均为“高分更真实”。原版代码按每个生成器做 real-vs-fake 比较，再平均各生成器指标。

需要注意：AUC 对正负样本比例不敏感，AP 对正类比例敏感。仓库中存在两套统计口径：

1. src/metrics.py：为每个生成器平衡真实与生成样本，接近论文复现口径。
2. 多数 tools 下的实验脚本：每个生成器与全部真实样本比较，不做平衡。

因此不同报告中的 AP 不能混合比较。本文主实验统一使用 src/metrics.py 的逐生成器 pairwise 平衡口径；非平衡 full-scope 结果不再放入论文主表。

### 4.3 公平比较协议

主实验沿用原论文的 zero-shot pairwise 协议：对每个生成器分别构造 real-vs-fake 比较，并对真实样本进行平衡采样。所有方法均报告 AUC 和 AP，最后对生成器结果做宏平均。Alpha-STALLED 使用不依赖测试标签的固定融合：

~~~text
Final Score = 0.60 × Global Score + 0.40 × Patch Score
~~~

当前 release baseline 中，VideoFeedback 统计 10 个两秒生成器；Hotshot-XL 的 1 秒 patch 分支需要单独补齐并重新生成 `results/paper_scores/` 后再并入。GenVideo 的 HotShot 和 MoonValley 尚未完成 patch 分支，因此在 Alpha-STALLED 列记为“—”。

## 5. 主实验：Zero-shot Detection

**表 1. 三个生成视频检测基准上的 zero-shot 结果。** AEROBLADE、RIGID、ZED、D3 和原版 STALL 的结果来自原论文 Table 1；Alpha-STALLED 使用本项目在相同 pairwise 平衡口径下复算的固定融合结果。每个单项中最佳结果使用**粗体**，第二名使用<u>下划线</u>。表中每个单元格均按 `AUC / AP` 排列。

### 5.1 完整对比表

#### VideoFeedback

| 生成模型 | AEROBLADE | RIGID | ZED | D3 (L2) | D3 (cos) | STALL | Alpha-STALLED |
|:--|:--:|:--:|:--:|:--:|:--:|:--:|:--:|
| AnimateDiff | 0.57 / 0.55 | 0.73 / 0.74 | 0.65 / 0.62 | 0.49 / 0.49 | 0.61 / 0.57 | <u>0.83 / 0.86</u> | **0.865 / 0.883** |
| Fast-SVD | 0.52 / 0.51 | 0.54 / 0.56 | 0.45 / 0.48 | 0.76 / 0.77 | 0.80 / 0.79 | <u>0.89 / 0.89</u> | **0.901 / 0.912** |
| LVDM | <u>0.88 / 0.90</u> | 0.65 / 0.57 | 0.76 / 0.70 | 0.42 / 0.49 | 0.31 / 0.41 | 0.86 / 0.89 | **0.900 / 0.913** |
| LaVie | 0.50 / 0.50 | 0.71 / 0.73 | 0.29 / 0.37 | 0.51 / 0.47 | 0.49 / 0.46 | <u>0.81 / 0.83</u> | **0.833 / 0.848** |
| ModelScope | 0.60 / 0.56 | 0.66 / 0.62 | 0.69 / 0.59 | 0.51 / 0.52 | 0.42 / 0.46 | <u>0.81 / 0.83</u> | **0.866 / 0.873** |
| Pika | 0.44 / 0.46 | 0.54 / 0.54 | 0.39 / 0.47 | <u>0.83 / 0.84</u> | 0.81 / 0.81 | 0.78 / 0.80 | **0.886 / 0.900** |
| Sora / SoRA-Clip | 0.65 / 0.62 | 0.43 / 0.44 | 0.56 / 0.62 | 0.62 / 0.56 | 0.67 / 0.58 | <u>0.81 / 0.82</u> | **0.829 / 0.833** |
| Text2Video-Zero | 0.67 / 0.63 | 0.70 / 0.68 | 0.55 / 0.49 | 0.15 / 0.33 | 0.22 / 0.36 | **0.83 / 0.83** | <u>0.761 / 0.778</u> |
| VideoCrafter2 | 0.60 / 0.58 | 0.80 / 0.76 | 0.53 / 0.50 | 0.69 / 0.71 | 0.80 / 0.79 | **0.93 / 0.94** | <u>0.910 / 0.927</u> |
| ZeroScope | <u>0.78</u> / 0.78 | 0.65 / 0.59 | 0.70 / 0.62 | 0.35 / 0.45 | 0.35 / 0.44 | <u>0.78 / 0.81</u> | **0.877 / 0.883** |
| Hotshot-XL | 0.20 / 0.34 | 0.51 / 0.58 | 0.44 / 0.45 | 0.64 / 0.67 | 0.60 / 0.62 | <u>0.79 / 0.80</u> | -- |
| **Average** | 0.58 / 0.58 | 0.63 / 0.62 | 0.54 / 0.54 | 0.54 / 0.57 | 0.55 / 0.57 | <u>0.83 / 0.85</u> | **0.863 / 0.875†** |

#### GenVideo

| 生成模型 | AEROBLADE | RIGID | ZED | D3 (L2) | D3 (cos) | STALL | Alpha-STALLED |
|:--|:--:|:--:|:--:|:--:|:--:|:--:|:--:|
| Crafter | 0.64 / 0.65 | 0.71 / 0.66 | 0.55 / 0.56 | 0.79 / **0.82** | 0.76 / 0.79 | <u>0.82 / 0.80</u> | **0.831** / <u>0.800</u> |
| Gen2 | 0.56 / 0.59 | 0.70 / 0.67 | 0.51 / 0.58 | <u>0.88 / 0.90</u> | <u>0.88 / 0.90</u> | <u>0.88</u> / 0.89 | **0.907 / 0.923** |
| Lavie | 0.58 / 0.59 | 0.77 / 0.76 | 0.39 / 0.42 | 0.68 / 0.68 | 0.67 / 0.68 | <u>0.85 / 0.84</u> | **0.859 / 0.860** |
| ModelScope | 0.60 / 0.60 | 0.62 / 0.59 | 0.61 / 0.57 | 0.63 / 0.64 | 0.60 / 0.63 | <u>0.78 / 0.78</u> | **0.834 / 0.839** |
| MorphStudio | 0.74 / 0.73 | 0.74 / 0.69 | 0.60 / 0.60 | 0.66 / 0.71 | 0.64 / 0.69 | <u>0.83</u> / **0.84** | **0.834** / <u>0.839</u> |
| Show 1 | 0.48 / 0.50 | 0.53 / 0.52 | 0.45 / 0.47 | 0.76 / <u>0.80</u> | 0.75 / 0.79 | <u>0.82 / 0.80</u> | **0.843 / 0.828** |
| Sora | 0.73 / 0.70 | 0.49 / 0.48 | 0.71 / 0.79 | 0.75 / 0.75 | 0.74 / 0.74 | <u>0.79 / 0.80</u> | **0.832 / 0.856** |
| WildScrape | 0.49 / 0.53 | 0.61 / 0.59 | 0.55 / 0.57 | 0.65 / **0.69** | 0.64 / **0.69** | <u>0.72 / 0.68</u> | **0.759** / <u>0.680</u> |
| HotShot-XL | 0.31 / 0.39 | <u>0.64 / 0.65</u> | 0.47 / 0.46 | 0.56 / 0.64 | 0.54 / 0.62 | **0.79 / 0.78** | -- |
| MoonValley | <u>0.75 / 0.78</u> | 0.72 / 0.66 | 0.63 / 0.72 | **0.81 / 0.82** | **0.81 / 0.82** | 0.72 / 0.75 | -- |
| **Average** | 0.59 / 0.61 | 0.65 / 0.63 | 0.55 / 0.57 | <u>0.72 / 0.74</u> | 0.70 / <u>0.74</u> | **0.80 / 0.80** | 0.837 / 0.828† |

#### ComGenVid

| 生成模型 | AEROBLADE | RIGID | ZED | D3 (L2) | D3 (cos) | STALL | Alpha-STALLED |
|:--|:--:|:--:|:--:|:--:|:--:|:--:|:--:|
| Sora | 0.72 / 0.67 | 0.53 / 0.55 | 0.58 / 0.59 | 0.68 / 0.65 | 0.68 / 0.65 | <u>0.84 / 0.85</u> | **0.909 / 0.909** |
| VEO3 | 0.67 / 0.62 | 0.62 / 0.63 | 0.52 / 0.55 | 0.79 / 0.76 | 0.79 / 0.78 | <u>0.86 / 0.87</u> | **0.931 / 0.933** |
| **Average** | 0.69 / 0.64 | 0.57 / 0.59 | 0.55 / 0.57 | 0.73 / 0.71 | 0.73 / 0.71 | <u>0.85 / 0.86</u> | **0.920 / 0.921** |

#### 跨基准汇总

| 汇总项 | AEROBLADE | RIGID | ZED | D3 (L2) | D3 (cos) | STALL | Alpha-STALLED |
|:--|:--:|:--:|:--:|:--:|:--:|:--:|:--:|
| All Benchmarks Average | 0.62 / 0.61 | 0.61 / 0.59 | 0.57 / 0.58 | <u>0.64 / 0.65</u> | <u>0.64 / 0.65</u> | **0.82 / 0.82** | 0.873 / 0.875† |

† Alpha-STALLED 的 VideoFeedback 平均值只覆盖当前 release score 中的 10 个两秒生成器，GenVideo 平均值只覆盖已有 Patch 分数的 8 个共同模型；All Benchmarks Average 因此也不参与原论文完整平均行的最佳/第二名排序。

### 5.2 主实验分析

Alpha-STALLED 在当前 VideoFeedback release baseline 的 10 个两秒生成器上达到 0.8628/0.8750，较原文 STALL 平均提高约 +0.033/+0.025；在 GenVideo 已完成的 8 个共同模型上达到 0.8374/0.8283；在 ComGenVid 上达到 0.9198/0.9211。逐生成器结果表明，局部二阶时序证据对 AnimateDiff、Pika、Gen2、Sora 和 VEO3 等模型具有明显补充作用，但在 Text2Video-Zero 和 VideoCrafter2 上仍不如原文 STALL，说明固定融合仍存在来源差异。

## 6. 消融实验

消融实验只控制三个核心部分：

1. **全局证据**：移除 Patch 分支，退化为原版 Global STALL。
2. **局部证据**：移除 Global 分支，只保留当前 Patch second-order 分数。
3. **局部时序定义**：保持其余设置不变，将二阶时序替换为空间、一阶、多间隔或显式匹配。

### 6.1 Global/Patch 组件消融

**表 2. Global/Patch 组件消融。** 性能和差值均按 `AUC / AP` 排列。

| Benchmark | Method | Controlled Component | Performance | Delta vs. STALL | Delta vs. Full Method |
|:--|:--|:--|:--:|:--:|:--:|
| ComGenVid | STALL | Original baseline | 0.850 / 0.860 | 0 / 0 | -0.070 / -0.061 |
| ComGenVid | w/o Patch | Global branch only | 0.853 / 0.855 | +0.003 / -0.005 | -0.067 / -0.066 |
| ComGenVid | w/o Global | Patch second-order only | **0.927 / 0.931** | +0.077 / +0.071 | +0.008 / +0.010 |
| ComGenVid | Alpha-STALLED | Global 0.60 + Patch 0.40 | <u>0.920 / 0.921</u> | +0.070 / +0.061 | 0 / 0 |
| VideoFeedback | STALL | Original baseline | 0.830 / 0.850 | 0 / 0 | -0.033 / -0.025 |
| VideoFeedback | w/o Patch | Global branch only | <u>0.847 / 0.861</u> | +0.017 / +0.011 | -0.015 / -0.014 |
| VideoFeedback | w/o Global | Patch second-order only | 0.823 / 0.830 | -0.007 / -0.020 | -0.040 / -0.045 |
| VideoFeedback | Alpha-STALLED | Global 0.60 + Patch 0.40 | **0.863 / 0.875** | +0.033 / +0.025 | 0 / 0 |
| GenVideo | STALL | Original baseline | 0.800 / 0.800 | 0 / 0 | -0.037 / -0.028 |
| GenVideo | w/o Patch | Global branch only | 0.796 / 0.790 | -0.004 / -0.010 | -0.041 / -0.038 |
| GenVideo | w/o Global | Patch second-order only | <u>0.807 / 0.809</u> | +0.007 / +0.009 | -0.030 / -0.019 |
| GenVideo | Alpha-STALLED | Global 0.60 + Patch 0.40 | **0.837 / 0.828** | +0.037 / +0.028 | 0 / 0 |

VideoFeedback 的当前 release 消融均值覆盖 10 个两秒生成器；Hotshot-XL 一秒分支尚未纳入 `results/paper_scores/`。GenVideo 仅统计已有 Patch 分数的 8 个共同生成器。原文 STALL 行来自论文 Table 1，其余消融行均按项目的 pairwise 平衡协议复算，具体 CSV 见 `results/paper_tables/ablation_summary.md`。

结果表明，VideoFeedback 和 GenVideo 都需要 Global 与 Patch 互补；任意删除一个分支都会降低完整方法性能。ComGenVid 的 Patch-only 分数高于固定融合，说明该数据集的局部异常更突出，但为了保持统一的跨数据集规则，主方法仍采用预先冻结的 alpha 融合。

### 6.2 局部时序定义消融

该实验在 ComGenVid 上固定 DINOv3、真实视频校准方式和 Patch 聚合流程，只改变局部时序特征定义。

**表 3. 局部时序定义消融。** 性能和差值均按 `AUC / AP` 排列。

| Method | Controlled Local Evidence | Performance | Delta vs. STALL | Delta vs. Full Method |
|:--|:--|:--:|:--:|:--:|
| STALL baseline | No Patch branch | 0.8532 / 0.8553 | 0 / 0 | -0.0666 / -0.0658 |
| Patch spatial only | Local static appearance | 0.8091 / 0.8194 | -0.0441 / -0.0359 | -0.1107 / -0.1017 |
| Same-grid lag-1 | First-order local change | 0.8556 / 0.8617 | +0.0024 / +0.0064 | -0.0642 / -0.0594 |
| Multi-lag | Multi-interval first-order change | 0.8546 / 0.8586 | +0.0014 / +0.0033 | -0.0652 / -0.0625 |
| Motion-hard | Hard-matched local change | 0.8430 / 0.8491 | -0.0102 / -0.0062 | -0.0768 / -0.0720 |
| Motion-soft | Soft-matched local change | 0.8181 / 0.8256 | -0.0351 / -0.0298 | -0.1017 / -0.0956 |
| Same-grid second-order | Second-order local change | **0.9273 / 0.9309** | +0.0741 / +0.0756 | +0.0075 / +0.0098 |
| Alpha-STALLED | Second-order Patch + Global fusion | <u>0.9198 / 0.9211</u> | +0.0666 / +0.0658 | 0 / 0 |

局部空间分支单独较弱；一阶和 multi-lag 只能带来有限提升；显式运动匹配会引入额外匹配噪声。Same-grid second-order 明显优于其他局部定义，说明生成视频中的局部闪烁、边缘跳变和动态不连续是当前方法的主要增益来源。ComGenVid 上 patch-only 高于融合，说明该数据集的局部异常信号很强；主方法仍保留全局--局部融合，是为了维持跨数据集的一致推理规则。

## 7. 当前方法的科研有效性审计

### 7.1 已经可以成立的结论

1. 原版 STALL 在本地得到基本一致的复现结果。
2. DINOv3 patch tokens 能提供全局特征之外的局部证据。
3. Same-grid second-order 明显优于 patch spatial、multi-lag 和 naive motion matching。
4. 固定 global/patch 融合在三个当前数据集上均提高 AUC。
5. Patch anomaly map 的持续性与集中度包含额外诊断信息。

### 7.2 目前不能直接写成最终论文结论的部分

#### 测试标签参与路由

tools/apply_sample_fallback_rule.py 中的 split 权重和 real_policy 根据 subset == real 分支处理。未知视频推理时并不知道该标签，因此当前 frozen_final_score 不是一个可直接部署的检测器输出。

#### Fake source 参与路由

代码先在已知 fake 行上按 source_model 统计 source_gate_mean，再决定该 source 是否启用更强 persistence 权重。真实部署既可能不知道生成器名称，也不能预先知道哪些行是 fake。

#### 测试批次 rank 依赖

rank01 在整批测试样本上计算。单个视频的结果依赖同批其他样本，无法独立推理，也属于 transductive evaluation。应改为使用独立真实校准集或训练/验证集上的固定经验分布。

#### 数据集特定 persistence 与 patch 配置

三个数据集使用不同 region、聚合方式和 persistence 字段。若这些参数由测试标签选择，则属于 benchmark-specific oracle。论文中必须说明选择协议，或重新冻结统一配置后评估。

#### AP 统计口径不统一

该问题已在 release baseline 中修复：`tools/eval_alpha_stalled.py`、`tools/fuse_scores.py` 和 `tools/eval_score_csv.py` 均调用 `src/metrics.py`，并使用逐生成器 pairwise 平衡口径。旧报告中的非平衡 full-scope AP 和 rank-space AP 只保留为历史审计，不再进入主表。

### 7.3 建议的框架简化

论文和系统实现建议分为两层：

~~~text
第一层：可部署主检测器
  Global STALL + Patch Second-Order + Fixed Alpha Fusion

第二层：待验证的可靠性扩展
  Label-free Persistence Calibration
~~~

不要在主框架中继续呈现 universal、split、split-minus 多条路径。当前 selector 对平均性能贡献极小，却显著增加解释成本和泄漏风险。

建议将 persistence 重写为单一、无标签残差校正：

~~~text
persistence_calibrated = CDF_real(persistence_feature)
confidence = abs(persistence_calibrated - 0.5) * 2
gate = f(abs(global_score - patch_score), confidence)

final_score = base_score
            + lambda * gate * (persistence_calibrated - base_score)
~~~

其中所有 CDF、阈值和 lambda 都必须在独立真实校准集或验证集上冻结，推理时只输入当前视频，不读取 subset、source_model 或其他测试视频分数。

## 8. 当前项目代码框架

### 8.1 代码调用关系

~~~mermaid
flowchart LR
    A[video_index.py] --> B[dataset_utils.py]
    B --> C[stall.py]
    C --> D[eval.py]

    B --> E[dataset_utils_patch.py]
    C --> F[stall_patch.py]
    E --> F
    F --> G[prefill_patch_cache.py]

    E --> H[create_patch_params.py]
    I[patch_matching.py] --> H
    H --> J[eval_patch_fast.py]

    J --> K[patch_calibrated_persistence_scores.py]
    D --> L[apply_sample_fallback_rule.py]
    J --> L
    K --> L

    L --> M[validation and audit tools]
~~~

### 8.2 主方法必需代码

| 文件 | 作用 | 建议 |
|---|---|---|
| src/stall.py | DINOv3、全局空间/时序似然、原版最终分数 | 保留，原版核心 |
| src/eval.py | 原版数据集推理与分数导出 | 保留 |
| src/metrics.py | 论文式 pairwise AUC/AP | 保留并作为唯一评估入口 |
| src/dataset_utils.py | CSV、视频与 global cache | 保留 |
| src/video_index.py | 建立采样索引 | 保留 |
| src/create_params.py | 拟合原版真实视频校准参数 | 保留 |
| src/stall_patch.py | 同时提取 global 与 patch token | 保留 |
| src/dataset_utils_patch.py | Patch cache 读写 | 保留 |
| src/patch_math.py | 轻量 numpy 数学工具，覆盖 bottom-k 与 percentile 等基础逻辑 | 保留 |
| src/create_patch_params.py | 拟合 patch whitening 与校准参数 | 保留 |
| src/eval_patch_fast.py | Same-grid 一阶/二阶 patch 快速评分 | 保留，当前 patch 核心 |
| src/patch_matching.py | 多种 patch temporal 定义 | 保留二阶核心；匹配实验可拆出 |
| tools/prefill_patch_cache.py | 批量生成 patch cache | 保留 |
| tools/eval_alpha_stalled.py | 主线 global/patch 融合与统一指标输出 | 保留 |
| tools/eval_score_csv.py | 已有 score CSV 的统一指标评估，用于组件和局部时序消融 | 保留 |
| tools/fuse_scores.py | 固定 alpha sweep，已调用统一 metrics | 保留 |
| tools/verify_alpha_stalled_release.py | 检查 release score、metrics、alpha sweep、配置和主校准文件一致性 | 保留 |

### 8.3 Persistence 研究链路

| 文件 | 作用 | 当前状态 |
|---|---|---|
| tools/patch_calibrated_persistence_scores.py | 从校准后的时序异常图提取 persistence 统计 | 有研究价值，保留 |
| tools/apply_sample_fallback_rule.py | raw base、gate、split/universal 与 selector | 需重写标签依赖后才能作为主方法 |
| tools/sample_level_fallback_sweep.py | 搜索 fallback 规则 | 仅研究/消融 |
| tools/validate_sample_fallback_rules.py | leave-one-source/dataset 规则验证 | 保留为研究验证 |
| tools/audit_fallback_vs_default_by_source.py | source 级对比 | 保留为诊断，不作为无泄漏证明 |
| tools/sample_fallback_promotion_gate.py | 内部 promotion gate | 保留，但 gate 前提需修正 |
| tools/verify_frozen_sample_fallback_manifest.py | 冻结配置一致性检查 | 保留 |

### 8.4 历史探索代码

以下类别不是“无用代码”，但不属于当前主方法，建议整体移动到 research_archive，避免主目录继续膨胀：

| 类别 | 代表文件 | 结论 |
|---|---|---|
| Motion matching | matching_、patch_motion_、patch_trajectory_ 系列 | 当前不如 same-grid second-order |
| Tail/morphology | patch_tail_、morphology_、hard_tail_ 系列 | 部分数据集有效，未形成统一默认 |
| Frequency/neighbor | patch_frequency_、patch_neighbor_ 系列 | 探索性信号，未晋升主线 |
| PatchField | patchfield_、apply_universal_patchfield_addon.py | 增益较小，已从当前方法说明删除 |
| Adaptive/reliability | adaptive_fusion.py、reliability_fusion.py | 未稳定超过固定融合 |
| Fresh/Hotshot scaffold | build/verify runbook 相关工具 | 工程验证脚手架，适合单独目录 |
| 旧版 patch evaluator | eval_patch.py、fast_multi、multiagg_fast | 保留复现，主线统一到 eval_patch_fast.py |

目前 tools 约有 212 个文件、scripts 约有 27 个文件。继续把一次性实验都放在同一目录，会让读者无法判断真正入口。建议目录重构为：

~~~text
STALL/
  src/
    baseline/           # 原版 STALL
    patch/              # patch 特征、参数与评分
    fusion/             # 固定融合和未来无标签校准
    evaluation/         # 唯一指标实现
  scripts/
    reproduce/          # 三个数据集复现命令
    ablations/          # 正式消融
  research_archive/
    matching/
    morphology/
    frequency/
    patchfield/
    routing/
  configs/
    alpha_stalled.yaml  # 冻结参数与数据集协议
  tests/
  results/
    paper_tables/
    archived_exploration/
~~~

### 8.5 生成文件与存储

当前主要目录规模约为：

| 目录 | 文件数 | 大小 | 判断 |
|---|---:|---:|---|
| cache | 约 119,410 | 694 GB | 运行缓存，不应纳入代码结构；建议 shard 化并记录可重建清单 |
| results | 约 1,436 | 1.7 GB | 正式表格与探索结果混杂，应分层归档 |
| precomputed | 约 70 | 572 MB | 大量 sweep 参数，只保留正式冻结参数在主目录 |
| debug_outputs | 约 25 | 49 MB | 可归档或清理 |
| logs | 约 15 | 652 KB | 含旧 PID 文件，不属于方法资产 |

logs 下的 PID 文件当前内容为 3 或 4，明显不是可靠的活跃任务记录，建议在确认没有对应任务后删除。本文档不自动删除任何现有实验资产。

### 8.6 当前工程缺口

1. 自动化测试已开始建立：`tests/test_alpha_stalled_core.py` 覆盖融合、指标、分数方向、二阶差分、patch pooling、bottom-k 和 percentile；仍缺少 DINOv3/patch cache 端到端测试。
2. 评估口径已统一到 `src/metrics.py`；旧 full-scope 或 rank-space AP 只作为历史审计保留。
3. 当前 release 主入口是已有 global/patch score CSV 的融合与评估；从原始视频到最终分数的端到端 CLI 仍需要在 patch cache 和 DINOv3 环境下补齐测试。
4. 主配置已收敛到 `configs/alpha_stalled.yaml`；仍需进一步明确 dataset-specific patch 配置的选择协议。
5. `results/paper_scores/`、`results/paper_tables/` 和 `results/paper_sweeps/` 已从 `.gitignore` 中解禁；历史探索结果仍应继续归档。
6. 大量一次性工具已迁入 `research_archive/`，但历史结果目录还需要进一步分层标记“当前”“废弃”“上限实验”。

## 9. 推荐的正式实验组织

### 9.1 主实验

主表应只包含推理时不使用测试标签和 source identity 的方法：

~~~text
Original STALL
Patch second-order only
Global + Patch fixed alpha
修复后的 label-free persistence calibration（完成后再加入）
~~~

### 9.2 必需消融

1. Global spatial、global temporal 及其组合。
2. Patch spatial only、lag-1、multi-lag、second-order。
3. Same-grid 与 motion-hard/motion-soft。
4. Region 1/2/3。
5. Mean 与不同 bottom-k 比例。
6. Patch 内 spatial/temporal 权重。
7. Global/patch alpha sweep，并区分 validation-selected 与测试集 oracle。
8. 无 persistence、单一 persistence 校正、旧 fallback 上限。
9. 统一配置跨数据集迁移，而不是每个数据集独立选最佳配置。

当前 release 已重新生成 Global/patch alpha sweep，路径为 `results/paper_sweeps/alpha_sweep_summary.md`。结果显示，单数据集 oracle 最优 alpha 分别为 ComGenVid 0.20、VideoFeedback 0.80、GenVideo 0.65；这说明 alpha 选择本身具有数据集依赖。除非后续定义独立验证集选择协议，论文主线仍应使用预先冻结的 alpha，而不能把测试集最优 alpha 写成部署规则。

### 9.3 泛化与可信度实验

1. 在第四个全新数据集上预注册配置后评估。
2. Leave-one-generator-out 选择参数，并在被留出的生成器上测试。
3. 报告每个 source 的变化，避免平均值掩盖回退。
4. 对视频长度、运动强度、压缩质量和分辨率分层分析。
5. 对固定 FPR 下的 TPR 做 bootstrap 置信区间。
6. 对异常帧/patch 可视化进行定性验证，但不把可视化当成性能证据。

## 10. 推荐的下一步实现顺序

1. **补齐 release score 覆盖。** VideoFeedback Hotshot-XL 的 1 秒 patch 分支尚未纳入当前 `results/paper_scores/`；GenVideo 的 HotShot 和 MoonValley patch 分支仍缺失。
2. **冻结并记录最终配置来源。** 当前三个数据集的 patch region/aggregation 不同，应在论文中明确这是预先冻结配置、开发集选择，还是当前实验证据下的 dataset-specific release baseline。
3. **重写 fallback。** 删除 subset、fake source_model 和 test-batch rank；只允许当前视频分数与独立校准分布参与计算。
4. **重新做 persistence 消融。** 先选择一个跨数据集统一 feature，再比较是否稳定超过 fixed alpha。
5. **扩展自动化测试与 release verifier。** 当前已覆盖融合公式、CSV one-to-one merge、统一指标入口、分数方向、二阶差分、patch region pooling、percentile 和 bottom-k，并新增 release 资产一致性检查；后续还应覆盖 patch cache 磁盘评分、DINOv3 特征提取和单样本推理独立性。
6. **完成参数与结果归档。** 当时生成的资产清单现归档于 `research_archive/docs/pre_u0_release/asset_manifest.md`；`results/` 保留该阶段的 score、metrics、alpha sweep 和论文图表资产。debug/重复 `.npz` 仍由 `.gitignore` 排除，除非后续需要补充历史消融复现。

## 11. 结论

本项目最扎实的创新点是：在原版 STALL 的全局空间-时序似然框架中，引入 DINOv3 patch token 的同网格二阶时序似然。该信号能够捕获全局表征容易忽略的局部动态不连续，并在三个数据集上通过固定融合提高 AUC。

Persistence 异常形态进一步显示出增益潜力，但当前 fallback 的高分结果依赖测试标签、生成器分组和测试批次 rank，尚不满足训练自由未知视频检测的推理条件。最合理的框架简化不是删除 patch 主分支，而是把方法收敛为“全局 STALL + 局部二阶证据 + 固定融合”，再把 persistence 改造成单路径、无标签、基于独立真实校准集的可靠性校正。

项目已经积累了丰富的实验资产，当前重构已把评估协议、主实验 score、消融 score、alpha sweep 和主要资产清单收敛到 release baseline。剩余工作集中在缺失短视频分支、参数选择协议和 fallback 无泄漏重写；完成这些步骤后，才能形成一套结构清楚、结果可复现、论证可经受审查的 Alpha-STALLED 方法。

## 12. 主要复现资产

| 类别 | 内容 | 路径 |
|---|---|---|
| 主线 release | 冻结配置 | configs/alpha_stalled.yaml |
| 主线 release | 主实验 score | results/paper_scores/ |
| 主线 release | 主实验 metrics | results/paper_tables/alpha_stalled_main_summary.md |
| 主线 release | 组件与局部时序消融 | results/paper_tables/ablation_summary.md |
| 主线 release | 固定 alpha sweep | results/paper_sweeps/alpha_sweep_summary.md |
| 主线 release | 短视频 patch 覆盖缺口 | results/paper_tables/patch_coverage_gaps.md |
| pre-U0 release | 资产清单 | research_archive/docs/pre_u0_release/asset_manifest.md |
| pre-U0 release | 复现审计 | research_archive/docs/pre_u0_release/reproducibility_audit.md |
| 历史审计 | 原版复现对照 | results/stall_repro_comparison.md |
| 历史代码 | 非主线路径与诊断工具 | research_archive/ |
