# Alpha-STALLED IEEE 中文初稿

这是 Alpha-STALLED 论文的可迭代 LaTeX 工作区，格式沿用 `IEEE_Conference_Template` 的双栏模板，但只保留实际写作需要的文件。

## 文件结构

- `main.tex`：论文入口、宏包、题名、章节输入和参考文献入口。
- `sections/`：中文正文分节。
- `tables/`：主结果、组件消融、二阶时序对照、统计稳定性等表格。
- `figures/`：论文图片。
  - `method/`：方法流程图。
  - `results/`：主结果、敏感性、解释性图片。
  - `supplementary/`：优先放入补充材料的图片。
- `references.bib`：BibTeX 参考文献。
- `references/source_papers/2603.15026v2.pdf`：STALL 原文，本稿的核心参考文献。
- `notes/`：写作计划、图表清单、实验到论点映射和参考文献阅读摘要。
  - `submission_readiness_audit.md` 记录投稿前仍需作者确认或补充的项目。

## 可复现表格来源

- `tables/validation_fusion_loso.tex` 是已完成 LOSO 诊断的冻结论文表格；结论级数据和协议边界已汇总到 `results/research_summary/`。大体量 bootstrap 抽样明细和一次性表格构建脚本不属于 release。

## Overleaf 设置

1. 上传整个 `ieee_alpha_stalled/` 目录。
2. Menu → Compiler → 选择 `XeLaTeX`。
3. 主文件选择 `main.tex`。
4. 如果参考文献第一次不显示，按 Overleaf 的 Recompile 再编译一次。

## 当前写作边界

- 主结果使用 `results/paper_tables/alpha_stalled_main_summary.md` 和 `results/paper_tables/ablation_summary.md` 的 pairwise balanced AUC/AP。
- 方法部分不绑定具体融合比例；只定义先构造局部分数，再与全局 STALL 分数进行评测前冻结融合。
- 当前稿件将 alpha-only validation/LOSO 选参写作更稳妥的部署适配策略；three-branch LOSO 只作为诊断和外部数据集潜在优化方向，不作为默认方法。
- D3 只写 protocol audit，不声称已经完成外部 baseline 重跑。
- 关键帧解释案例只作为事后解释，不参与训练、推理或调参。
- 数据/代码/伦理/利益冲突/作者贡献声明已经放入 `sections/08_declarations.tex`，未知事实保留为 `AUTHOR_INPUT_NEEDED`。
