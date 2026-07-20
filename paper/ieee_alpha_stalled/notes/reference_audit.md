# 参考文献审计

本表记录当前稿件中主要参考文献的用途和来源。CV/ML 核心文献优先使用 arXiv、CVF open access、ECCV/ICCV/CVPR 官方页面或项目论文页；不为了满足 Nature/CNS 偏好而替换成不直接支撑本文技术主张的综述或新闻材料。

| BibTeX key | 用途 | 来源/链接 | 支撑等级 |
|---|---|---|---|
| `benhayun2026stall` | 核心基线、方法协议、实验体系、D3 protocol audit 参照 | `references/source_papers/2603.15026v2.pdf`; https://arxiv.org/abs/2603.15026 | 强支撑 |
| `simeoni2025dinov3` | 冻结视觉编码器与全局/patch token 来源 | https://arxiv.org/abs/2508.10104 | 强支撑 |
| `zheng2025d3` | 训练自由生成视频检测、二阶时序相关工作、外部 baseline 协议边界 | https://openaccess.thecvf.com/content/ICCV2025/html/Zheng_D3_Training-Free_AI-Generated_Video_Detection_Using_Second-Order_Features_ICCV_2025_paper.html | 强支撑 |
| `chen2024genvideo` | GenVideo benchmark 与生成视频检测背景 | https://arxiv.org/abs/2405.19707 | 强支撑 |
| `he2024videoscore` | VideoFeedback/VideoScore benchmark 背景 | https://arxiv.org/abs/2406.15252 | 强支撑 |
| `wang2019vatex` | VATEX 真实视频校准来源背景 | CVF ICCV 2019 open access | 强支撑 |
| `xu2016msrvtt` | MSR-VTT 真实视频来源背景 | CVF CVPR 2016 open access | 强支撑 |
| `chen2011msvd` | MSVD/MSR-VTT 等真实视频来源相关背景 | ACL Anthology/原论文 | 背景支撑 |
| `chen2024panda70m` | Panda70M 真实视频来源背景 | CVF CVPR 2024 open access | 背景支撑 |
| `ricker2024aeroblade`, `cozzolino2024zed`, `he2024rigid` | 训练自由图像检测相关工作 | CVF/ECCV/arXiv 页面 | 背景支撑 |
| `betser2025whitenedclip`, `betser2026conditional`, `rachmil2025whitening` | 白化似然和激活空间统计背景 | 论文/预印本 | 部分支撑 |

## 需要后续人工确认的引用

- 若目标期刊要求 DOI，应在最终投稿前用 Crossref 或 DBLP 补齐 DOI、卷期和出版社字段。
- `betser2025whitenedclip`、`betser2026conditional`、`rachmil2025whitening` 当前主要来自 STALL 原文参考文献和 arXiv/会议元数据，建议投稿前再次核对最终出版信息。
- 如果后续加入外部 baseline 主表，需要为 AIGVDet、T2VE、ZED、RIGID、AEROBLADE、D3 的具体实现版本和 checkpoint 来源单独建引用/脚注。
