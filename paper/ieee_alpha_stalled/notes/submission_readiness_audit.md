# 投稿准备度审计

当前 `paper/ieee_alpha_stalled/` 已经具备中文双栏论文初稿、图表、参考文献、实验总览和核心参考文献 PDF，但还不是最终投稿包。以下项目需要在确定目标期刊后补齐。

## 已完成

- IEEE 双栏 LaTeX 工作区。
- 中文正文：摘要、引言、相关工作、方法、实验、消融、讨论、结论。
- 主图和结果图：方法流程、bootstrap、逐生成器热图、temporal order、cross-dataset frozen hyperparameter、关键帧解释。
- 参考文献：核心 BibTeX 条目与 `2603.15026v2.pdf`。
- 实验总结：主线结果、消融、敏感性、D3 protocol audit、duration/window、runtime/storage、coverage gaps、failure/keyframe cases。
- 静态检查：input、figure、citation key、label 重复检查通过。

## 需要作者提供或确认

| 项目 | 当前状态 | 为什么需要 |
|---|---|---|
| 作者姓名、单位、邮箱 | `main.tex` 中仍为占位符 | 投稿必需 |
| 目标期刊或会议 | 未指定 | 决定摘要格式、页数、参考文献格式、图表数量和是否需要声明页 |
| 论文题名 | 当前为工作题名 | 需根据目标期刊和作者偏好定稿 |
| 外部 baseline 是否必须补跑 | 当前只做 D3 protocol audit | 若目标期刊要求 SOTA 表，需要重跑 D3/AEROBLADE/RIGID/ZED/T2VE/AIGVDet |
| 校准源策略 | 当前 patch release 使用 target benchmark real-only 统计 | 若目标期刊强调严格外部校准，需要补 calibration source 消融 |
| 数据/代码可用性声明 | 已写入 `sections/08_declarations.tex`，但链接、许可证和 release tag 仍待作者确认 | 期刊投稿通常必需 |
| 伦理/滥用声明 | 已写入 `sections/08_declarations.tex` | 生成视频检测论文建议补充 |
| 利益冲突与作者贡献 | 已写入占位声明 | 需要作者确认，不能由工具推断 |
| 基金和致谢 | `main.tex` 中为占位符 | 投稿必需 |

## 建议下一步

1. 在 Overleaf 使用 XeLaTeX 编译，记录所有报错和版面问题。
2. 作者确认目标期刊和页数限制。
3. 若目标期刊需要英文稿，将当前中文结构作为底稿，再进行英文重写，而不是逐句翻译。
4. 若先投中文期刊，优先补齐作者信息、声明、图表脚注和外部 baseline 边界说明。
5. 若审稿预期强，优先补 D3-DINOv3 同协议重跑，其次补 calibration source 代表实验。
