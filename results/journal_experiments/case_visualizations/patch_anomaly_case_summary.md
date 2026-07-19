# Patch anomaly case visualization

本目录基于 `failure_case_candidates.csv` 选取代表案例，并从已有 patch cache 与真实视频校准参数生成 patch-level anomaly map。
案例只用于事后解释，不参与推理、训练或超参数选择。

| case | dataset | subset/source | filename | global | patch | alpha | 说明 |
|---|---|---|---|---:|---:|---:|---|
| T2V-Zero / hard generated | videofeedback | annotated/Text2Video-Zero | `1000945.mp4` | 0.9293 | 0.9042 | 0.9193 | Alpha-STALLED 仍最难识别的生成视频。 |
| T2V-Zero / patch raises score | videofeedback | annotated/Text2Video-Zero | `1004886.mp4` | 0.0324 | 0.8972 | 0.3783 | 生成视频被 Alpha-STALLED 打得更像真实，可能削弱检测。 |
| VideoCrafter2 / patch-global conflict | videofeedback | annotated/VideoCrafter2 | `2001966.mp4` | 0.0013 | 0.9104 | 0.3649 | patch 分支显著高于 global，适合检查局部证据是否误导融合。 |
| LaVie-base / patch-global conflict | videofeedback | annotated/LaVie-base | `4001104.mp4` | 0.0234 | 0.9839 | 0.4076 | patch 分支显著高于 global，适合检查局部证据是否误导融合。 |
| AnimateDiff / patch-global conflict | videofeedback | annotated/AnimateDiff | `5000419.mp4` | 0.0166 | 0.9082 | 0.3733 | patch 分支显著高于 global，适合检查局部证据是否误导融合。 |
| Panda70M / real false-positive | videofeedback | real/Panda70M | `p106966.mp4` | 0.0048 | 0.0014 | 0.0035 | Alpha-STALLED 最容易误伤的真实视频。 |

图中空间 map 为同网格二阶时序 likelihood 的高异常区域汇总：先取负 log-likelihood 作为 anomaly，
再对时间维取 90 分位并做每视频 robust 归一化。时序曲线为每个时间步 top-10% patch anomaly 的均值。
因此，颜色/曲线越高表示该视频在局部二阶时序上越偏离真实视频校准分布。

当前选择覆盖 VideoFeedback 中 paired bootstrap 显示稳定负迁移的 Text2Video-Zero 与 VideoCrafter2，
并加入 LaVie-base/AnimateDiff 的 patch-global conflict 以及 Panda70M 真实误伤样本，用于说明 patch 分支的收益边界。
