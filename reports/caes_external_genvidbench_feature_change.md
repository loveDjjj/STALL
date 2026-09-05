# Feature-change 的 GenVidBench Pair-1 外部确认

## 实验目的

本实验在完成三个开发集上的 CAES 探索后，冻结 Feature-change 的全部规则，
在第三方 GenVidBench Pair-1 上检验其外部域迁移能力。外部数据不用于选择
coarse FPS、候选步长、窗口数、融合权重或任何阈值。

## 固定协议

| 项目 | 固定设置 |
|---|---|
| 外部数据 | GenVidBench Pair-1 |
| 真实校准 | 199 条独立 VRiPT，只使用真实视频 |
| 评测 | 300 条 VRiPT real + 300 条 ModelScope fake |
| Dense 窗口 | 8 FPS、2 秒、16 帧、最多 K=3 |
| Uniform | 原外部主实验的均匀 K=3，分数逐视频原样复用 |
| Feature-change | 冻结的 1 FPS Global feature-change，0.5 秒候选步长，Top-K |
| 检测器 | Global STALL + Local D2，固定 0.60/0.40 融合 |
| 校准 | 每个 selector 在同一 199 条真实视频上执行 matched real calibration |
| 指标 | AUC、real-positive AP，视频身份配对 bootstrap 1000 次 |

新 run 中 Uniform 的 600 条视频与原正式外部 run 身份完全一致，Global、Local 和
Final 三列分数的最大绝对差异均为 0。这确保下述差值只来自窗口选择。

## 结果

| 窗口选择 | AUC | AP-real | TPR@1% FPR | FPR@95% TPR |
|---|---:|---:|---:|---:|
| Uniform K=3 | 0.8386 | 0.8430 | 0.0933 | 0.4867 |
| Random K=3（seed 17） | 0.8417 | 0.8439 | 0.1000 | 0.5033 |
| Feature-change K=3 | **0.8652** | **0.8690** | **0.1200** | **0.4467** |
| Feature-change vs Uniform | **+0.0265** | **+0.0260** | +0.0267 | -0.0400 |

严格配对 bootstrap 给出的 AUC 差值均值为 +0.0266，95% CI
[+0.0154, +0.0379]；AP-real 差值均值为 +0.0251，95% CI
[+0.0163, +0.0346]。两个区间均不跨 0。

Random 相对 Uniform 仅提高 +0.0031 AUC / +0.0010 AP，配对 bootstrap
区间分别为 [-0.0050, +0.0109] 和 [-0.0058, +0.0073]，均跨 0；其
FPR@95% TPR 还由 0.4867 恶化至 0.5033。Feature-change 相对 Random
提高 +0.0235 AUC / +0.0250 AP，对应 bootstrap 区间为
[+0.0127, +0.0351] 和 [+0.0151, +0.0349]。因此外部提升不能由随机更换
窗口解释，而来自 feature-change 信号提供的定向选择。

Random 改变了 284/600 条视频的窗口集合，甚至多于 Feature-change 的 247 条，
但没有获得可靠增益。这进一步排除了“窗口变化数量更多自然更好”的解释。

Feature-change 改变了 247/600（41.2%）条评测视频的窗口集合；平均窗口集合
Jaccard 为 0.6577。评测集中 330 条视频短于 4 秒，53 条不少于 16 秒；由于
303/600 条视频只有一个有效窗口，本结果是在大量短视频限制下取得的。

## 结论与边界

Feature-change 在三个开发集上的增益主要来自 GenVideo，而在这个冻结的外部域上
同时获得约 2.6 个百分点的 AUC/AP 提升，且低 FPR 指标方向也一致。因此，它不再
只是开发集上的微小偶然增益，而是当前最值得保留的自适应窗口策略。

结果目录中的 `gate_decision.csv` 仍调用开发阶段统一的 2/3 数据集门槛，因此在只有
一个外部数据集时会显示 `passes_go_gate=False`；该字段对本实验不适用，应以配对
bootstrap 区间和冻结外部协议作为判断依据。

该结果仍不能称为严格的“未见生成器泛化”：GenVidBench Pair-1 的假视频来源是
ModelScope，而 ModelScope 也出现在开发数据中。它证明的是未参与 selector 调参的
外部视频身份和真实域迁移。下一步若要形成更强投稿结论，仍需在包含未见生成器、
且长视频占比更高的 untouched benchmark 上做一次完全冻结的确认。

## 可复现入口

```bash
bash scripts/run_caes_external_feature_change.sh --device cuda:1
bash scripts/run_caes_external_feature_change.sh --selector random --device cuda:1
conda run --no-capture-output -n stall \
  python scripts/analyze_caes_external_selectors.py --overwrite
```

正式产物位于 `results/runs/caes_external_genvidbench_feature_change/`。
