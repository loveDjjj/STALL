# 原视频关键帧解释案例

本目录在已有 patch anomaly case visualization 基础上补充原视频关键帧。关键帧选择规则为：
先在同网格二阶 patch temporal anomaly 曲线中找到 top-10% patch anomaly 最高的时间步，
再抽取对应二阶差分中心帧。该选择只用于事后解释，不参与训练、推理或调参。

| case | source/file | keyframe | global | patch | alpha | 解释标签 | 写作观察 |
|---|---|---:|---:|---:|---:|---|---|
| T2V-Zero / hard generated | annotated/Text2Video-Zero/`1000945.mp4` | 13 (1.62s) | 0.9293 | 0.9042 | 0.9193 | 残余难例 | 该生成视频在 global 与 patch 分支下均保持较高真实百分位，说明局部时序异常不足以抵消整体真实感；可作为生成器高保真样本的上界案例。 |
| T2V-Zero / patch raises score | annotated/Text2Video-Zero/`1004886.mp4` | 1 (0.12s) | 0.0324 | 0.8972 | 0.3783 | 负迁移案例 | patch 分支显著抬高生成视频分数，说明局部二阶时序证据在该样本上更接近真实校准分布；该例适合说明融合需要保留全局分支约束。 |
| VideoCrafter2 / patch-global conflict | annotated/VideoCrafter2/`2001966.mp4` | 6 (0.75s) | 0.0013 | 0.9104 | 0.3649 | 分支冲突 | global 分支判为明显异常而 patch 分支给出较高真实百分位，显示局部运动统计与全局外观证据不一致；该例用于说明 patch 证据的互补性和边界。 |
| LaVie-base / patch-global conflict | annotated/LaVie-base/`4001104.mp4` | 3 (0.38s) | 0.0234 | 0.9839 | 0.4076 | 分支冲突 | global 分支判为明显异常而 patch 分支给出较高真实百分位，显示局部运动统计与全局外观证据不一致；该例用于说明 patch 证据的互补性和边界。 |
| AnimateDiff / patch-global conflict | annotated/AnimateDiff/`5000419.mp4` | 2 (0.25s) | 0.0166 | 0.9082 | 0.3733 | 分支冲突 | global 分支判为明显异常而 patch 分支给出较高真实百分位，显示局部运动统计与全局外观证据不一致；该例用于说明 patch 证据的互补性和边界。 |
| Panda70M / real false-positive | real/Panda70M/`p106966.mp4` | 12 (1.50s) | 0.0048 | 0.0014 | 0.0035 | 真实误伤 | 真实视频在 global 与 patch 分支上均处于低百分位，通常应归入真实域偏移、低质量或低运动边界；若论文讨论具体视觉原因，应人工复核原片段。 |

## 使用边界

- 图和表可以支撑“如何选择 failure / boundary cases”的可复现说明。
- 当前 `draft_observation` 是基于分数关系和关键帧的审稿写作草稿；若主文声称具体语义原因，例如低运动、压缩伪影或文本-视频错配，应由作者人工观看原视频后确认。
- 该图优先放入补充材料；主文若空间有限，可只保留 2–3 个代表案例。
