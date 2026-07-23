# Alpha-STALLED 当前研究目标完成度审计

更新时间：2026-07-23

本文件对应当前持续目标：检查可调 `alpha` 是否能让 Alpha-STALLED 超越原版 STALL；在新增数据集上做小规模验证；分析真实校准数据数量是否限制 patch 分支；根据结果继续优化新数据集泛化能力。

> 归档说明：本审计记录了清理前的完整实验路径。2026-07-23 起，结论级权威
> 索引为 `results/research_summary/README.md` 和 `experiment_metrics.csv`；未提交的
> 逐视频融合、bootstrap 抽样、临时 split/prefill 与失败探索目录已删除。表中旧路径
> 仅表示证据来源历史，不再承诺作为 release 资产存在。

## 结论状态

当前目标尚未完全完成。已有证据足以支持“固定融合不是普适最优，真实校准数量影响 patch 分支，可调 alpha 在多个数据集上超过 global-only STALL”。AIGVDBench-resplit 的结论已经从“单纯失败边界”更新为“聚合方式边界”：默认 region3 bottom-k 设置下 LOGO 失败，但 region3 mean 在固定 alpha=0.60 下稳定超过 global-only；加入 Vidu 第三生成器后，三源宏平均 fixed alpha=0.60 相对 global-only 的 bootstrap ΔAUC/ΔAP CI 均大于 0；继续把同一三源协议的 real-only calibration pool 从 300 扩到 400/500/600 后，fixed alpha=0.60 从 0.7272/0.7326 提升到 0.7417/0.7519，进一步支持“更多真实校准有帮助但不严格单调”；继续加入 CogVideoX1.5 后，四源 fixed alpha=0.60 仍为正，但 CogVideoX1.5 暴露 patch-only 负迁移和 LOGO alpha 失败。进一步的三分支拆分诊断显示，单一整体 patch alpha 可能不是最优参数化，空间 patch 与时序 patch 分开融合是下一步优化方向；跨数据集快速验证显示，AIGVDBench 得到的保守三分支候选在 real 去重后的 GenVidBench 上小幅超过 fixed alpha，在 AEGIS 上与 fixed alpha 基本持平但超过 global。原始三数据集的后验三分支也均超过 alpha=0.60。最新 leave-one-source-out paired bootstrap 进一步说明，alpha-only 选参和三分支选权重都能稳定超过 global-only；外部集合上三分支相对 alpha-only 仍有额外 AP 增益，但原始集合上三分支相对 alpha-only 几乎无实质增益。因此三分支仍不能写成默认部署协议，alpha-only 验证选参反而是更稳妥的部署适配线。

## 目标要求逐项审计

| 目标要求 | 当前证据 | 状态 | 可写结论 | 注意边界 |
|---|---|---|---|---|
| 固定融合不只用 `alpha=0.60/0.40`，扫描可调 alpha | `results/paper_sweeps/`; `results/journal_experiments/real_calib_size_cross_dataset_summary/`; `results/journal_experiments/logo_alpha_selection/` | 已完成主线扫描与 LOGO 诊断 | 最优 alpha 明显随数据集变化；固定 alpha 应写成 release 默认值，不应写成普适最优 | best/oracle alpha 使用 holdout 选择，只能作为诊断上限；LOGO 仍不是完全独立验证集 |
| 看是否能超越原版 STALL/global-only | `paper/ieee_alpha_stalled/tables/real_calib_alpha_summary.tex`; `figures/results/alpha_gain_summary.pdf`; `results/journal_experiments/aigvdbench_small/resplit_real300/variant_grid/`; `results/journal_experiments/aigvdbench_small/resplit_real300/vidu_extension/`; `results/journal_experiments/aigvdbench_small/resplit_real300/cogvideox15_extension/` | 部分完成 | AEGIS、ComGenVid、GenVideo、VideoFeedback-small 在对应最佳真实校准规模下，LOGO 或 best alpha 超过 global-only；AIGVDBench-resplit 在 region3 mean + fixed alpha=0.60 下超过 global-only 且 bootstrap CI 大于 0；加入 Vidu 后三源 fixed-alpha 增益更强；加入 CogVideoX1.5 后四源 fixed-alpha 仍为正 | GenVidBench 只有小幅正结果；AIGVDBench 的 LOGO 在四源下失败，说明跨生成器 alpha 选择仍有明显边界 |
| 新增数据集小规模验证 | `results/journal_experiments/genvidbench_small/`; `results/journal_experiments/aigvdbench_small/`; `results/journal_experiments/aegis_full/` | 已完成三个外部/新增集合的轻量验证，并新增 AIGVDBench Vidu 与 CogVideoX1.5 生成器扩展 | GenVidBench 小幅正，AEGIS 强正，AIGVDBench 在 mean 聚合变体下 fixed-alpha 正；Vidu 上 patch-only 明显强于 global-only；CogVideoX1.5 是 patch 负迁移压力源 | GenVidBench 当前是 Pair1 小子集；AIGVDBench 的跨生成器 alpha 选择仍不完全稳定 |
| 下载/接入新数据集后持续分析 | `cache/patch_embeddings/`; `results/journal_experiments/*_small/`; `results/journal_experiments/aegis_full/`; `results/journal_experiments/genvidbench_small/genvidbench_local_expansion_audit.md` | 已完成可控子集接入和结果归档；已审计 GenVidBench 本地扩展条件 | 工程上已经验证新增数据集可跑 STALL/global 和 patch 分支；GenVidBench Pair1 的 MS/Pika 可用，T2VZ 是 8fps 协议边界 | 全量 GenVidBench/AIGVDBench/GenWorld 尚未完成；GenVidBench Pair2 本地只有 label，没有视频压缩包，需要额外下载 |
| 真实数据是否不够多 | `real_calib_size_cross_dataset_curve.csv`; `real_calib_size_curves.pdf`; `results/journal_experiments/aigvdbench_small/resplit_real300/vidu_extension/real_calib_size_region3_mean/`; `results/journal_experiments/aigvdbench_small/resplit_real600/vidu_extension/real_calib_size_region3_mean/` | 已完成跨数据集分析，并在 AIGVDBench region3 mean + Vidu 三源优化协议下复验；新增 real600 扩展 | 25/50 个真实校准视频通常不足；100/200/300 后 patch-only 才更稳定；AIGVDBench 三源优化协议下 n=300 已明显工作，继续到 400/500/600 后 fixed alpha 进一步提高，n=600 达到 0.7417/0.7519 | 真实数量不是严格单调因素，真实域代表性和质量同样重要；n=500/600 的 patch AUC 已出现平台期/轻微回落 |
| 在原始数据集验证真实数量影响 | `results/journal_experiments/original_real_calib_size/` | 已完成 ComGenVid、GenVideo、VideoFeedback-small | 原始数据集上也出现真实校准数量影响，支持不是外部数据集偶然现象 | VideoFeedback-small 是受存储限制的小规模 holdout |
| 在新增数据集验证真实数量影响 | `genvidbench_pair1_real_calib_size_sensitivity_summary.csv`; `aigvdbench_small_real_calib_size_sensitivity_1s_summary.csv`; `aigvdbench_resplit_real300_*`; `aegis_real_calib_size_patch_fusion_summary.csv`; `aigvdbench_resplit_real300_plus_vidu_region3_mean_real_calib_sweep.csv` | 已完成 | 外部数据集也显示 patch 分支依赖真实校准规模；AIGVDBench 加入 Vidu 且改用 region3 mean 后，n=300 才形成强 patch/fusion | AIGVDBench 的 Open-Sora 仍是困难来源；真实数量不是唯一因素 |
| 找到让新数据集能力超过原版 STALL 的方法 | AEGIS、GenVidBench、AIGVDBench 结果 | 部分完成 | AEGIS 强正；GenVidBench 小幅正；AIGVDBench-resplit 的 region3 mean + fixed alpha=0.60 正且 bootstrap CI 大于 0；Vidu 第三生成器进一步支持 mean 聚合可迁移 | 后续需要验证 AIGVDBench mean 聚合是否能迁移到更大子集或更多生成器 |

## 当前最强证据

| 数据集 | 最佳校准 real 数 | Global AUC/AP | Fixed AUC/AP | LOGO AUC/AP | Best AUC/AP | 当前判断 |
|---|---:|---:|---:|---:|---:|---|
| AEGIS-hard | 100 | 0.6023/0.5998 | 0.6701/0.6444 | 0.7664/0.7927 | 0.7664/0.7927 | 外部强正，patch 主导 |
| ComGenVid | 200 | 0.8550/0.8606 | 0.8668/0.8790 | 0.8703/0.8854 | 0.8704/0.8855 | 原始数据集稳定正 |
| GenVideo | 200 | 0.8085/0.8036 | 0.8423/0.8305 | 0.8484/0.8380 | 0.8492/0.8394 | 原始数据集强正 |
| VideoFeedback-small | 200 | 0.8528/0.8640 | 0.8631/0.8708 | 0.8649/0.8729 | 0.8682/0.8757 | 小规模原始补充，正 |
| GenVidBench-Pair1 | 100 | 0.7921/0.8043 | 0.7981/0.8049 | -- | 0.8027/0.8113 | 外部小幅正；另有 pooled validation-selected alpha 小幅正 |
| AIGVDBench-small | 200 | 0.7070/0.7108 | 0.7096/0.7134 | -- | 0.7097/0.7133 | 旧小协议弱/诊断 |
| AIGVDBench-resplit | 300 | 0.6975/0.7070 | 0.7033/0.7146 | 0.6801/0.6812 | 0.7038/0.7158 | 边界案例；LOGO 失败 |
| AIGVDBench-resplit region3 mean | 300 | 0.6975/0.7070 | 0.7162/0.7278 | 0.7079/0.7087 | 0.7195/0.7173 | 新增优化；fixed alpha bootstrap CI 大于 0，LOGO CI 跨 0 |
| AIGVDBench-resplit region3 mean + Vidu | 300 | 0.6994/0.7008 | 0.7272/0.7326 | 0.7348/0.7357 | 0.7414/0.7423 | 三生成器扩展；fixed alpha bootstrap CI 大于 0，LOGO AUC CI 大于 0、AP CI 边界 |
| AIGVDBench-resplit region3 mean + Vidu + CogVideoX1.5 | 300 | 0.6673/0.6633 | 0.6782/0.6832 | 0.6519/0.6519 | 0.6790/0.6841 | 四生成器扩展；fixed alpha bootstrap CI 大于 0，但 CogVideoX1.5 patch-only 低于随机，LOGO 失败 |
| AIGVDBench 四源三分支诊断 | 300 | 0.6673/0.6633 | 0.6816/0.6911 | -- | 0.6907/0.7101 | 诊断性结果；保守三分支 fixed 权重为 0.50/0.15/0.35，test 后验最优为 0.05/0.05/0.90；source-level LOSO bootstrap 对四源平均超过 global-only，但 CogVideoX1.5 留出仍负迁移 |
| GenVidBench Pair1 三分支 probe | 100 | 0.7921/0.8043 | 0.8008/0.8087 | -- | 0.8014/0.8116 | real 去重口径；固定三分支候选 0.50/0.15/0.35 小幅高于 fixed alpha=0.60；source-level LOSO 仅小幅且不稳定 |
| AEGIS 三分支 probe | 100 | 0.6004/0.6207 | 0.6668/0.6652 | -- | 0.7627/0.8022 | 固定三分支候选与 fixed alpha=0.60 基本持平，但显著高于 global；后验最优强偏 spatial |
| ComGenVid 三分支 probe | release | 0.8532/0.8552 | 0.9193/0.9227 | -- | 0.9298/0.9326 | 后验三分支超过 alpha=0.60；固定三分支 AUC 接近 alpha=0.60，AP 略高；source-level LOSO 两个 source 均超过 alpha=0.60 |
| GenVideo 三分支 probe | release | 0.8108/0.9829 | 0.8441/0.9856 | -- | 0.8472/0.9859 | 后验三分支略高于 alpha=0.60；固定三分支接近 alpha=0.60；source-level LOSO 主要超过 global，但相对 alpha=0.60 有正有负 |
| VideoFeedback 三分支 probe | release | 0.8507/0.8978 | 0.8545/0.9008 | -- | 0.8698/0.9080 | 后验三分支高于 alpha=0.60；固定三分支低于 alpha=0.60；source-level LOSO 平均略高于 alpha=0.60，但多个 source 仍为负 |

## 当前方法学解释

1. `alpha` 不是一个可以跨数据集固定解释为最优的常数。当前证据支持三层表述：release 可冻结一个默认值；部署/扩展时优先使用独立验证源选择 alpha-only 融合权重；three-branch 权重选择主要作为诊断和外部数据集潜在优化方向，而不是默认方法。
2. patch 分支的有效性依赖真实视频校准数量。真实数量过少时，白化似然估计不稳定，patch-only 经常接近随机。
3. 真实数量增加不是充分条件。AIGVDBench-resplit 说明即使校准 real 提高到 300，默认 bottom-k 聚合下跨生成器选择仍可能失败。后续 region/aggregation 网格显示 mean 聚合显著优于 bottom-k；其中 region3 mean + fixed alpha=0.60 的 ΔAUC/ΔAP bootstrap CI 均大于 0。进一步分数分布诊断显示，mean 聚合不是让 patch-only 全面优于 global，而是把 Open-Sora 上的 fixed-alpha 负迁移转为小幅正迁移，并保留 Pika 上的较大增益。加入 Vidu 后，patch-only 在 Vidu 上达到 0.7832/0.7817，高于 global-only 的 0.7032/0.6883，使三源 fixed-alpha 宏平均增益扩大到 ΔAUC/ΔAP +0.0277/+0.0318。最新 Vidu 三源 real-size sweep 进一步显示：在同一 region3 mean 设置下，n=25/50 的 patch-only 完全退化为 0.5000/0.5000，n=100 仍接近随机，n=200 开始有效，n=300 才达到 patch-only 0.7138/0.7074 和 fixed 0.7272/0.7326。继续加入 CogVideoX1.5 后，global-only 降到 0.6673/0.6633，fixed alpha=0.60 仍提升到 0.6782/0.6832，且 bootstrap CI 为正；但 CogVideoX1.5 单源 patch-only 仅 0.4560/0.4764，LOGO alpha 降到 0.6519/0.6519。三分支拆分后，保守 fixed 权重 0.50/0.15/0.35 在 AIGVDBench 四源达到 0.6816/0.6911，并在 real 去重后的 GenVidBench Pair1 达到 0.8008/0.8087，高于 fixed alpha=0.60 的 0.7981/0.8049；原始三数据集上，后验三分支也均超过 alpha=0.60。新增 source-level LOSO paired bootstrap 显示，外部集合中 alpha-only 选参和三分支选权重都稳定超过 global-only；三分支相对 alpha-only 在 global_min_040/050 下仍有正 CI，但最保守 global_min_060 的 AUC CI 跨 0、AP 为正。原始三数据集上 alpha-only 选参已解释大部分收益，三分支相对 alpha-only 只有约 +0.0005 AUC，AP 约为 0 或略负。因此问题不只是 real 数量不足，还包括局部异常聚合方式、空间/时序分支权重、生成器差异以及 alpha/weight 选择协议。
4. 当前论文应强调“局部二阶证据带来可验证的补充信息”，而不是声称“所有外部基准都稳定超过 STALL”。

## 下一轮实验优先级

### P0：低成本、直接补强当前主线

1. 已审计 GenVidBench-Pair1 扩展条件。当前已有 pooled validation-selected alpha：heldout test mean AUC/AP 0.8186/0.8348，高于 global 0.8154/0.8330，bootstrap CI 大于 0；但提升很小，且与统一 real-calib 表协议不完全一致。本地 Pair2 只有 label 文件，没有视频压缩包；T2VZ 在 8fps 协议下无有效 fake index。若继续扩大 GenVidBench，需要额外下载 Pair2 或定义单独的 4fps stress test。
2. 已完成 AIGVDBench-resplit region3 mean 扩展：新增下载 `vidu.zip` 和 `Cogvideox1.5.zip`，各抽取 300 个候选 fake，均满足 8fps/1s 协议；使用 Vidu fake200 加入 Open-Sora/Pika fake200 后，fixed alpha=0.60 相对 global-only 的 paired bootstrap ΔAUC/ΔAP CI 均大于 0；继续加入 CogVideoX1.5 后，fixed alpha=0.60 仍为正，但 LOGO alpha 失败。下一步若继续扩展，可再加入 `ClosedSource/Opensora.zip` 或更大 fake 子集。
3. 已完成四源三分支拆分诊断、外部跨数据集 probe、原始三数据集 probe，并新增不使用 test 后验的 source-level LOSO 权重选择、alpha-only LOSO 对照和 paired bootstrap：证据文件为 `results/journal_experiments/aigvdbench_small/resplit_real300/cogvideox15_extension/diagnostics/aigvdbench_four_source_three_branch_diagnostic.md`、`results/journal_experiments/three_branch_fusion_probe/three_branch_probe_summary.md`、`results/journal_experiments/three_branch_fusion_probe/original_three_branch_probe_summary.md`、`results/journal_experiments/three_branch_fusion_probe/three_branch_loso_bootstrap_macro_summary.csv`、`results/journal_experiments/three_branch_fusion_probe/alpha_loso_bootstrap_macro_summary.csv` 与对应 `original_*` 文件。GenVidBench probe 已修正为 real 去重口径。当前结论是三分支验证式选权重能稳定超过 global-only，但相对 alpha-only 的额外价值主要体现在外部集合 AP；下一步若继续三分支，应使用真正独立 validation split 或重新设计约束，而不是继续扩大 test 后验 sweep。
4. 已完成：检查 region3 mean 与 region3 bottom-k 的 patch 分数分布和固定融合诊断。证据文件为 `results/journal_experiments/aigvdbench_small/resplit_real300/variant_grid/diagnostics/aigvdbench_region3_aggregation_diagnostic.md` 和同目录分布图。当前解释是：bottom-k 在 Open-Sora 上放大 patch 负迁移，而 mean 聚合改善 global/patch 互补性。
5. 已完成：AIGVDBench Vidu 三源 region3 mean 的真实校准数量扫描。证据文件为 `results/journal_experiments/aigvdbench_small/resplit_real300/vidu_extension/real_calib_size_region3_mean/aigvdbench_resplit_real300_plus_vidu_region3_mean_real_calib_sweep.md`。该结果直接支持“真实校准数量不足会限制优化后 patch 分支”的论点。
6. 已完成：AIGVDBench Vidu 三源 real600 扩展。证据文件为 `results/journal_experiments/aigvdbench_small/resplit_real600/vidu_extension/real_calib_size_region3_mean/aigvdbench_resplit_real600_plus_vidu_region3_mean_real_calib_extension.md`。该实验额外从远端 `Real.zip` 抽取 300 个未使用真实视频，保持同一 real200 holdout 和三源 fake200 测试集不变；fixed alpha=0.60 在 n=400/500/600 分别为 0.7358/0.7443、0.7410/0.7500、0.7417/0.7519，均高于 n=300 的 0.7272/0.7326。

### P1：中等成本、可能提升新数据集结果

1. AIGVDBench 继续扩展 mean 聚合设置，例如 region2/3 的更多真实校准规模、不同 patch 空间/时序权重，以及更大 fake 子集；优先用已有 patch cache 或只重算评分，不重新提取特征。
2. GenVidBench 扩大到第二个 pair 或更大轻量子集，确认当前小幅正结果是否稳定。
3. 对真实校准集做来源代表性分析：真实视频时长、分辨率、运动强度分桶，比较不同真实子集对 patch-only 和 fusion 的影响。

### P2：高成本，审稿明确要求时再做

1. 全量 GenVidBench/AIGVDBench/GenWorld 扩展。
2. 外部 baseline 同协议重跑。
3. Backbone ablation、扰动鲁棒性、global temporal order sweep。

## 当前不应写入主文的说法

- 不应写“固定 alpha=0.60 是最优融合比例”。
- 不应笼统写“所有 AIGVDBench 协议上都显著超过原版 STALL”；可以具体写“resplit-real300 的 region3 mean + fixed alpha=0.60 在 Open-Sora/Pika 两源、加入 Vidu 的三源协议、以及继续加入 CogVideoX1.5 的四源协议下均超过 global-only，但四源结果同时暴露 patch 负迁移和 LOGO alpha 失败”。
- 不应把 best/oracle alpha 写成部署协议。
- 不应把“真实校准数量越多越好”写成单调规律。
- 不应把 LOGO alpha 等同于完全独立外部验证集。
