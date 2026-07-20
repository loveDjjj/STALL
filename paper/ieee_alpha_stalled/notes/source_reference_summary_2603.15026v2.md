# `2603.15026v2.pdf` 阅读摘要

源文件：`references/source_papers/2603.15026v2.pdf`

## 主文第 2 节：相关工作定位

- 原文把生成视频检测区分于 deepfake 检测：任务目标是检测完全生成的视频，而不是检测真实视频中的局部篡改。
- 图像检测器可以提供空间证据，但逐帧处理会忽略跨帧时序一致性。
- 监督式视频检测器依赖生成器标签和训练数据，在未见生成器上有泛化风险。
- D3 是重要的训练自由视频检测方法，使用二阶时序特征，但主要依赖时序证据。
- STALL 的定位是用真实视频校准的概率似然同时建模空间外观和全局时序转移。

对本文写作的影响：

- Alpha-STALLED 的创新不能写成“首次使用二阶时序”，因为 D3 已有二阶时序思想。
- 更准确的表述是：把局部 patch 二阶时序证据纳入 STALL 的真实视频似然校准框架，并与全局 STALL 分数形成两级融合。

## 主文第 3 节：预备知识

- 原文使用白化变换，把冻结视觉表征映射到零均值、单位协方差空间。
- 在白化空间内使用标准高斯 log-likelihood 作为真实分布典型性的代理。
- 原文强调归一化时序差分比原始差分更接近高斯假设。

对本文写作的影响：

- 方法部分应保留“白化 + 高斯似然 + 百分位校准”的统计链条。
- Patch 分支应被写成同一统计链条的局部扩展，而不是新的监督分类器。

## 主文第 4 节：STALL 方法

- STALL 对全局帧特征计算空间似然，对归一化相邻帧差分计算时序似然。
- 空间分支和时序分支先各自聚合，再通过真实视频百分位映射到同一尺度。
- 原文使用真实视频校准分布，而不是生成样本训练。

对本文写作的影响：

- Alpha-STALLED 的全局分支应明确继承 STALL。
- 融合部分应避免把全局、局部空间、局部时序写成扁平三分支比例；更合理的是先构造局部分数，再做全局--局部融合。

## 主文第 5 节与附录：实验体系

原文实验包括：

- 三 benchmark 主结果与外部 image/video baseline 对比。
- 空间/时序组件消融。
- temporal derivative order、frame-level aggregation、FPS、step size、video length 消融。
- calibration set size、calibration source、backbone encoder、图像扰动、时序扰动实验。
- D3 baseline protocol audit，指出类别不平衡和 GenVideo 真实视频 FPS 帧复制会影响 AP。
- inference time 和 memory 分析。
- qualitative examples。

当前 Alpha-STALLED 已对应完成：

- 三数据集 global-only / patch-only / Alpha-STALLED 主表。
- ComGenVid 局部时序定义消融与 D=2/3/4 对照。
- region、aggregation、bottom-k、beta、alpha 敏感性。
- 生成器宏平均 paired bootstrap。
- D3 protocol audit，但没有外部 D3 重跑。
- runtime/storage audit 和小样本 video-stage benchmark。
- patch anomaly map 与关键帧解释。

仍不应夸大的缺口：

- 尚未完成同协议外部 baseline 重跑。
- 尚未完成大规模 backbone、image/temporal perturbation、calibration set size/source 的完整补充实验。
