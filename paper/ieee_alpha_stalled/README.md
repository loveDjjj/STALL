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

## 可复现结果来源

- 锁定协议：`configs/alpha_stalled_u0_locked.yaml`。
- 发布清单、锁定 Local 参数与逐视频分数：`release/u0/`。
- 协议、核心消融、校准规模和数值稳定性报告：`reports/u0_*.md`。
- 结论级指标索引：`results/research_summary/`。
- 主文正式结果表：`tables/main_results.tex`、`tables/component_ablation.tex`、`tables/u0_calibration_stability.tex` 和 `tables/u0_failed_directions.tex`。

## Overleaf 设置

1. 上传整个 `ieee_alpha_stalled/` 目录。
2. Menu → Compiler → 选择 `XeLaTeX`。
3. 主文件选择 `main.tex`。
4. 如果参考文献第一次不显示，按 Overleaf 的 Recompile 再编译一次。

## 当前写作边界

- 主结果是固定 21,421 视频交集上的 U0：Macro-3 AUC/real-positive AP `0.8741/0.8723`。
- 方法固定 `G_k=0.5G_s+0.5G_{t1}`、`L_k=0.1L_s+0.9L_{D2}`、K=3 分支均值、effective-K 真实重校准和 `S=0.6G+0.4L`。
- 生成视频和测试真实视频不得用于 whitening、CDF、阈值、权重或模型选择；历史 leakage-affected `0.8737/0.8750` 只能作为审计记录。
- K=5/all-window、bottom-2/hybrid、residual、多尺度、中间层、Joint Typicality、三分支和动态融合均不是正式方法。
- 历史 clean single-window `0.8570/0.8600` 只用于方法演进展示；正式 K=3 增益以统一核心 K1 `0.8632/0.8636` 为因果对照。
- 外部集、鲁棒性和可控注入只按预锁定协议报告，不允许据此回调 U0。
- 数据/代码/伦理/利益冲突/作者贡献声明位于 `sections/08_declarations.tex`，未知事实保留为 `AUTHOR_INPUT_NEEDED`。
