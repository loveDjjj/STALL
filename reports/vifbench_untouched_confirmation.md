# ViF-Bench 未见生成器冻结确认

## 协议

本实验在查看 ViF-Bench 检测分数前冻结以下方法：1 FPS Feature-change、K=3、
2 秒 8 FPS dense 窗口、STALL Global Spatial/T1、Local same-grid D2，以及
0.5 Global + 0.5 Local 融合。数据来自
[ViF-Bench 官方仓库](https://huggingface.co/datasets/JoeLeelyf/ViF-Bench)，
`source_videos.zip` SHA256 为
`41e79dff9f8ff16f7bcddd99eb18e4019b52f5ed9fe26a6a0fa0739b085907a8`。

165 条可用 real 按 `seed=17` 的文件名 SHA256 排序，前 80 条用于 real-only
calibration，其余 85 条用于 evaluation。只保留与这 85 条 evaluation real
同文件名语义配对的 fake，共 1,531 条、19 个生成器。与 calibration real
配对的 1,448 条 fake 被删除，另有 14 条无法匹配官方 real 文件名的 fake 被删除。

19 个生成器与三个开发集没有名称完全相同的来源。`pika-v2` 与 `sora-2` 属于已见
Pika/Sora 家族的新版本，因此“未见生成器”结论应同时报告严格名称和模型家族边界。

## 结果

主指标采用每个生成器与 real 配对后再宏平均的 AUC/real-positive AP。

| Selector | Macro AUC | Macro AP | 相对 Uniform AUC/AP |
|---|---:|---:|---:|
| Uniform K3 | **0.6094** | **0.6250** | -- |
| Random K3（seed 17） | 0.6101 | 0.6188 | +0.0007/-0.0061 |
| Feature-change K3 | 0.6043 | 0.6246 | -0.0051/-0.0003 |

Feature-change 相对 Uniform 的配对 bootstrap AUC 差值均值为 -0.0052，95% CI
[-0.0091, -0.0010]；AP 差值区间为 [-0.0069, +0.0052]。AUC 显著下降，
AP 没有可检测差异。19 个生成器中只有 4 个 AUC 提升、8 个 AP 提升。

Random 相对 Uniform 的 AUC 区间为 [-0.0028,+0.0040]，没有可靠变化；AP
区间为 [-0.0098,-0.0008]，显著下降。Random 因而只能作为无方向选择的空对照，
不能解释 Feature-change 在 GenVidBench 上的正增益，也没有改善 ViF-Bench。

Feature-change 相对 Random 的 AUC 差为 -0.0058，95% CI
[-0.0105,-0.0012]；AP 差为 +0.0047，CI [-0.0016,+0.0111]。因此在
ViF-Bench 上，Feature-change 的排序性能甚至显著低于随机窗口，而 AP 没有可靠差异。

Random 改变了 1,591/1,616 条视频的窗口集合，平均集合 Jaccard 仅 0.254；
Feature-change 改变 1,145/1,616 条，平均 Jaccard 为 0.466。Random 的窗口变化
更大却没有 AUC 收益，说明结果不能用“覆盖不同位置自然更好”解释。

## 分支诊断

| Selector / score | Macro AUC | Macro AP |
|---|---:|---:|
| Uniform Global-only | 0.6075 | 0.5979 |
| Uniform Local-only | 0.5613 | 0.5819 |
| Uniform Final | **0.6094** | **0.6250** |
| Feature-change Global-only | 0.6037 | 0.5968 |
| Feature-change Local-only | 0.5680 | 0.5875 |
| Feature-change Final | 0.6043 | 0.6246 |

Feature-change 对 Local D2 有小幅正作用，但使 Global 下降，融合后净结果为负。
因此 GenVidBench 上约 +0.0265/+0.0260 的收益不能推广为跨生成器普遍规律。

## Real-only 阈值迁移

| Selector | calibration目标FPR | calibration经验FPR | evaluation实际real FPR | fake recall |
|---|---:|---:|---:|---:|
| Uniform | 0.1%/1% | 0.0% | 2.35% | 9.73% |
| Feature-change | 0.1%/1% | 0.0% | 1.18% | 5.29% |
| Uniform | 5% | 5% | 30.59% | 41.93% |
| Feature-change | 5% | 5% | 24.71% | 35.47% |
| Random | 5% | 5% | 29.41% | 41.48% |

80 条 calibration real 无法分辨 0.1% 与 1% 的经验分位点。更重要的是，5% 阈值
在 evaluation real 上发生约 20--26 个百分点的 FPR 膨胀，表明 ViF-Bench 的
calibration/evaluation real 即使来自同一官方集合，参考统计仍存在明显漂移。

## 结论

该 untouched confirmation 否定了“Feature-change 在所有新生成器上稳定提高检测”
的强结论，也表明当前方法在高质量新生成器上整体检测能力有限。可以保留的更窄结论
是：Feature-change 在 GenVideo 和 GenVidBench 有效，但收益具有数据域依赖；Local
D2 与 Global 的互补性在 ViF-Bench 很弱。

本结果不得用于回调 Feature-change、权重或 K。下一步应以跨真实域矩阵解释参考库
敏感性，并把 Uniform 与 Feature-change 都保留到最终论文，而不是只报告正结果。

### 不使用 ViF real 的补充结果

由 ComGenVid、VideoFeedback、GenVideo 和 GenVidBench real 构建、完全不包含
ViF real 的 Universal-4 bank 在 ViF 上达到 0.6026/0.6297 配对 Macro AUC/AP。
相对 ViF target-real calibration 的差值为 -0.0018/+0.0048，两个 bootstrap
区间均跨 0。但 Universal-4 在 1% real FPR 目标下只有 0.52% fake recall，在 5%
目标下也只有 4.44%，说明相近 AUC 并不意味着可用的低误报检测能力。

## 产物

- `data/manifests/confirmation/vifbench_protocol.json`
- `results/runs/caes_confirmation_vifbench/`
- `results/analysis/vifbench_confirmation/`
- `results/runs/universal4_to_vifbench_feature_k3/`
- `results/analysis/universal4_to_vifbench/`
