# 目标完成度审计

审计日期：2026-07-20

本审计对应用户目标：在 `STALL` 下建立论文写作工作区，使用 `IEEE_Conference_Template` 格式，中文双栏，结合当前实验与 `2603.15026v2.pdf`，利用 nature-skill 思路打磨可投稿期刊论文，不在本机编译 LaTeX。

## 逐项核对

| 目标要求 | 当前证据 | 状态 |
|---|---|---|
| 在 `STALL` 下建立论文写作部分 | `paper/README.md`; `paper/ieee_alpha_stalled/README.md` | 已完成 |
| 格式沿用 `IEEE_Conference_Template` | `paper/ieee_alpha_stalled/IEEEtran.cls`; `main.tex` 使用 `\documentclass[conference]{IEEEtran}` | 已完成 |
| 中文、双栏 LaTeX 稿件 | `main.tex`; `sections/00_abstract.tex` 到 `sections/08_declarations.tex` | 已完成 |
| 方法部分靠近 `2603.15026v2` 原文思路，同时突出本文创新 | `sections/01_introduction.tex`; `sections/02_related_work.tex`; `sections/03_method.tex`; `notes/source_reference_summary_2603.15026v2.md` | 已完成 |
| 避免固定三分支融合比例，改为局部分数与全局分数融合 | `sections/03_method.tex` 的 `\slocal(V)` 与 `S(V)=\alpha\sglobal(V)+(1-\alpha)\slocal(V)` | 已完成 |
| 总结当前所有已完成实验 | `notes/current_experiments_summary.md`; `tables/experiment_inventory.tex`; `sections/04_experiments.tex`; `sections/05_ablation_analysis.tex` | 已完成 |
| 纳入 `2603.15026v2.pdf` 作为参考文件 | `references/source_papers/2603.15026v2.pdf` | 已完成 |
| 图片放入论文工作区并优化格式 | `figures/method/`; `figures/results/`; `figures/supplementary/`; PDF/SVG/PNG 多格式；主文关键帧图裁剪为 `keyframe_patch_anomaly_cases_main.pdf` | 已完成 |
| 参考文献仔细调研并建立 BibTeX | `references.bib`; `notes/reference_audit.md` | 已完成，最终 DOI/卷期需目标期刊前再确认 |
| 加入投稿常见声明和初投稿材料 | `sections/08_declarations.tex`; `notes/initial_submission_materials.md`; `notes/submission_readiness_audit.md` | 已完成，占位事实需作者确认 |
| 使用 conda 环境 `stall` 做验证 | 静态审计命令使用 `conda run --no-capture-output -n stall python` | 已完成 |
| 本机不编译 LaTeX | 未运行 LaTeX 编译；README 指导 Overleaf 使用 XeLaTeX | 已遵守 |
| 提交并推送到 GitHub | 当前工作分支已推送到 GitHub；具体 HEAD 以远端分支为准 | 已完成/本轮补充 |

## 保留的作者输入

以下项目不能由工具推断，已保留为待补充项：

- 作者姓名、单位、邮箱、ORCID。
- 目标期刊、文章类型、最终题名和短题名。
- 基金、致谢、利益冲突和作者贡献。
- 数据仓库、代码 release tag、许可证和访问条件。
- 是否需要补跑外部 baseline 或严格外部校准源消融。

这些不是当前结构与论文初稿任务的实现缺口，而是最终正式投稿前的作者事实输入。

## 最终验证口径

- 文件存在性：`find paper/ieee_alpha_stalled -maxdepth 3 -type f`。
- Git 状态：`git status --short --branch`。
- 静态 LaTeX 审计：检查 `\input`、`\includegraphics`、`\cite`、label 重复和 BibTeX key。
- 远端状态：`origin/refactor/alpha-stalled-reproducible-release` 与当前工作分支保持同步，本审计文件应随 HEAD 变化更新。
