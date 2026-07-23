# 真实校准数量与 alpha 微调实验草稿

## 一句话论点

在 Alpha-STALLED 中，局部 patch likelihood 不是一个可任意叠加的外观分支；它需要足够且有代表性的真实视频校准，并且其最优融合权重随数据集发生变化。

## 术语表

| 术语 | 统一写法 | 说明 |
|---|---|---|
| Alpha-STALLED | `\method` / Alpha-STALLED | 全局 STALL 与局部二阶 patch 分支的融合方法 |
| Global-only | global-only STALL | 原版 STALL 全局分支复现 |
| Patch-only | local second-order patch branch | 使用真实视频校准的局部二阶 patch 分支 |
| real calibration size | 真实视频校准数量 | 拟合 patch likelihood 参数的真实视频数量 |
| α | global fusion weight | `score = α global + (1-α) patch` 中的全局分支权重 |
| LOGO α | leave-one-generator-out alpha selection | 用其他生成器选择 α，再评估留出生成器；该协议弱于独立验证集，强于测试集 oracle |

## 推荐实验段落草稿

为检验局部 patch likelihood 是否受真实视频校准规模限制，我们在原始主实验数据集和外部小规模数据集上构建了互不泄漏的真实视频校准池与 holdout 评测集，并在固定 patch 结构下扫描校准真实视频数量。结果显示，真实校准数量不足时，局部分支通常接近随机排序：GenVidBench、AIGVDBench、ComGenVid、GenVideo 和 VideoFeedback-small 在 25 或 50 个真实视频时均未形成稳定的 patch-only 优势。随着真实校准数量增加，patch-only AUC 在多个数据集上明显上升，例如 GenVideo 从 n=50 的 0.5204 提升到 n=200 的 0.8137，VideoFeedback-small 从 n=50 的 0.6153 提升到 n=200 的 0.7761，AIGVDBench-resplit 从 n=200 的 0.5983 提升到 n=300 的 0.6588。这一趋势表明，局部二阶时序统计需要足够真实视频来估计稳定的 patch-level 白化分布。

真实视频数量并不产生严格单调收益。ComGenVid 在 n=800 时低于 n=400，GenVideo 在 n=500 时低于 n=200，VideoFeedback-small 在 n=500 时也低于 n=200。该现象说明，真实校准集规模是局部分支可用性的必要条件，但校准集的来源代表性、质量噪声和目标真实域覆盖同样影响最终泛化。因此，本文不把更多真实视频简化为单调提升假设，而将其表述为一个带有分布边界的校准条件。

进一步地，固定融合权重并不是所有数据集上的最优选择。AEGIS-hard 的最优点接近 patch 主导，ComGenVid 和 GenVideo 更偏向中等全局权重，VideoFeedback-small 则需要更强 global 主导。为降低测试集 oracle 选择的偏差，我们进一步做了 leave-one-generator-out alpha 选择：每次用其他生成器选择 α，再评估留出的生成器。该协议下，ComGenVid 在 n=200 时仍取得 0.8703/0.8854 的 AUC/AP，GenVideo 在 n=200 时取得 0.8484/0.8380，VideoFeedback-small 在 n=200 时取得 0.8649/0.8729，均高于各自 global-only。相反，本表中的 AIGVDBench-resplit 默认 region3 bottom-k 设置在 n=300 时固定 α=0.60 和 best α=0.65 虽分别达到 0.7033/0.7146 与 0.7038/0.7158，但 LOGO 选择降至 0.6801/0.6812，低于 global-only 的 0.6975/0.7070；其 paired bootstrap 相对 global-only 的 95% CI 也跨 0。后续 region3 mean 和 Vidu 第三生成器扩展已经显示 AIGVDBench 可通过 aggregation 调整获得更强 fixed-alpha 增益，因此这里的 bottom-k 行应被解读为聚合方式边界，而不是 AIGVDBench 的最终结论。因此，本文将固定 α 视为冻结默认配置，将验证集选择 α 作为潜在部署适配策略；测试集 best/oracle α 只表示诊断上限，不写成严格部署协议。

## 表格摘要

| 数据集 | best n | Global AUC/AP | Patch AUC/AP | Fixed AUC/AP | LOGO AUC/AP | Best AUC/AP | 解释边界 |
|---|---:|---:|---:|---:|---:|---:|---|
| AEGIS-hard | 100 | 0.6023/0.5998 | 0.7061/0.6967 | 0.6701/0.6444 | 0.7664/0.7927 | 0.7664/0.7927 | 外部强正结果，patch 主导 |
| ComGenVid | 200 | 0.8550/0.8606 | 0.8034/0.8208 | 0.8668/0.8790 | 0.8703/0.8854 | 0.8704/0.8855 | 原始数据集稳定正结果 |
| GenVideo | 200 | 0.8085/0.8036 | 0.8137/0.8058 | 0.8423/0.8305 | 0.8484/0.8380 | 0.8492/0.8394 | 原始数据集强正结果，n=500 回落 |
| VideoFeedback-small | 200 | 0.8528/0.8640 | 0.7761/0.7908 | 0.8631/0.8708 | 0.8649/0.8729 | 0.8682/0.8757 | 受存储限制的小规模原始补充 |
| GenVidBench-Pair1 | 100 | 0.7921/0.8043 | 0.5611/0.5554 | 0.7981/0.8049 | --/-- | 0.8027/0.8113 | 外部小幅正结果 |
| AIGVDBench-small | 200 | 0.7070/0.7108 | 0.5679/0.5609 | 0.7096/0.7134 | --/-- | 0.7097/0.7133 | 旧协议弱/诊断，validation transfer 失败 |
| AIGVDBench-resplit | 300 | 0.6975/0.7070 | 0.6588/0.6489 | 0.7033/0.7146 | 0.6801/0.6812 | 0.7038/0.7158 | 默认 bottom-k 弱正；region3 mean/Vidu extension 另见补充证据 |

## Claim-evidence map

- Claim: patch 分支依赖足够真实校准数据。Evidence: `real_calib_size_cross_dataset_curve.csv` 中多数数据集 25/50 接近随机，100/200/300 后改善。Status: supported.
- Claim: 更多真实校准数据不是严格单调收益。Evidence: ComGenVid n=800 < n=400, GenVideo n=500 < n=200, VideoFeedback-small n=500 < n=200。Status: supported.
- Claim: 可调 α 比固定 α 更适合作为部署适配策略。Evidence: 最优 α 从 AEGIS 的 0.05 到 VideoFeedback-small 的 0.80，存在明显数据集差异；LOGO α 在 ComGenVid、GenVideo 和 VideoFeedback-small 上仍超过 global-only。Status: supported as cross-generator diagnostic; independent validation remains stronger.
- Claim: LOGO α 在所有数据集上都稳健优于 global-only。Evidence: AIGVDBench-resplit n=300 的 LOGO AUC/AP 为 0.6801/0.6812，低于 global-only 的 0.6975/0.7070。Status: not supported; 应写成边界案例。
- Claim: AIGVDBench 默认 bottom-k 协议已经强显著超过 global-only。Evidence: resplit n=300 均值为正但 bootstrap CI 跨 0。Status: not supported; 应写成聚合方式边界。若写 AIGVDBench 显著提升，应引用 region3 mean 与 Vidu extension 的独立证据。

## 生成资产

- `figures/results/real_calib_size_curves.*`
- `figures/results/alpha_gain_summary.*`
- `tables/real_calib_alpha_summary.tex`
