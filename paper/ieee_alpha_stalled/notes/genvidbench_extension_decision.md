# GenVidBench 小子集扩展结果写作决策

## 结论

暂不将 GenVidBench 小子集结果写入论文主结果表。当前证据不足以支撑固定
Alpha-STALLED 融合在 GenVidBench 上相对原始 STALL 的稳定提升。后续补充的
pooled validation-selected alpha 形成了小幅正证据，但幅度很小，更适合放入附录或
rebuttal，而不是主结果表。

## 实验协议

- 数据集：GenVidBench Pair1。
- 真实来源：`vript`。
- 有效伪造来源：`ms`、`pika`。
- 不纳入有效指标的来源：`t2vz`，原因是候选视频基本为 4 fps，低于当前
  STALL `video_index.py --target-fps 8` 协议。
- 校准集：199 个有效 `vript` real，仅用于 real-only 校准。
- 测试集：每个有效生成器使用 300 个 `vript` real 与 300 个 fake。
- 指标：项目统一的 pairwise balanced AUC/AP，分数方向为“越高越接近真实视频”。
- 本地 Pair2 只有 label 文件，未发现 Pair2 视频压缩包；直接扩展 Pair2 需要额外下载。

## 关键结果

| generator | STALL global AUC/AP | Patch AUC/AP | Alpha-STALLED AUC/AP |
|---|---:|---:|---:|
| `ms` | 0.811 / 0.824 | 0.685 / 0.657 | 0.808 / 0.807 |
| `pika` | 0.774 / 0.785 | 0.661 / 0.672 | 0.781 / 0.790 |
| macro mean | 0.792 / 0.804 | 0.673 / 0.665 | 0.794 / 0.799 |

paired bootstrap 中，固定 Alpha-STALLED 相对 STALL global 的宏平均
Delta AUC 为 +0.0024，95% CI 为 [-0.0065, +0.0107]；Delta AP 为
-0.0043，95% CI 为 [-0.0155, +0.0067]。区间跨 0，不能写成显著或稳定提升。

补充的 pooled validation-selected alpha 结果为：selected alpha=0.95，heldout test
mean AUC/AP 为 0.8186/0.8348，高于对应 global 的 0.8154/0.8330；bootstrap
Delta AUC 95% CI 为 [+0.0007, +0.0062]，Delta AP 95% CI 为
[+0.0001, +0.0037]。这可以写成“可调 alpha 在 GenVidBench-Pair1 上形成小幅外部正证据”，
但不能替代主表里的稳定提升证据。

## 写作边界

若后续需要使用该结果，只建议放在附录或补充材料的“外部小子集压力测试”中，
而不是主结果表。可写的结论是：

1. GenVidBench Pair1 上，原始 STALL 的 VATEX 参数在 `ms` 与 `pika` 上仍有中等判别力；
2. 局部 patch likelihood 在两个生成器上有非随机判别力，但固定融合没有形成稳定收益；
3. pooled validation-selected alpha 能给出很小但 bootstrap 为正的收益；
4. `t2vz` 暴露了外部 benchmark 子源与 STALL 固定 8 fps 采样假设之间的协议边界。

不可写的结论是：

1. Alpha-STALLED 在 GenVidBench 上优于原始 STALL；
2. Alpha-STALLED 达到或超过 GenVidBench 官方训练式模型；
3. 当前小子集结果能代表完整 GenVidBench。
4. 通过降低 `target_fps` 纳入 `t2vz` 后仍与原 STALL/VATEX 参数同协议。

## 与公开模型比较

GenVidBench 官方论文及 AAAI 2026 版本报告的是大规模、训练式
cross-source/cross-generator 协议。其官方表格在训练式评测下报告了
DeMamba Top-1 85.47 / AUROC 99.28、VideoSwin Top-1 80.39 / AUROC 96.05、
MViTv2 Top-1 80.45 / AUROC 90.29 等结果。MAST、VidAudit 等公开或相关结果也多
属于训练式、全量或审计设置。它们可以作为背景说明 GenVidBench 的难度和当代检测器
范围，但不能与本文 600 条/生成器、training-free、real-only calibration 小子集放入
同一个性能表。

参考源：

- GenVidBench arXiv HTML: `https://arxiv.org/html/2501.11340v2`
- GenVidBench HuggingFace dataset: `https://huggingface.co/datasets/jian-0/GenVidBench`

## 相关本地文件

- 结论级归档：`results/research_summary/README.md`
- 关键指标：`results/research_summary/experiment_metrics.csv`
- 原始逐视频、alpha sweep 和 bootstrap 明细已在 2026-07-23 仓库清理中删除。
