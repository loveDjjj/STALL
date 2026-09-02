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

- 唯一基础配置：`configs/benchmark.yaml`。
- 正式无 Spatial 主结果：`results/runs/alpha_stall_full_d2_k3_no_spatial_refit/`。
- D1/D2、K1/K3 与 Spatial 跨 run 统计：`results/runs/ablation_refit_comparisons/`。
- GenVidBench 外部结果与统计：`results/runs/alpha_stall_external_genvidbench*` 和 `results/runs/external_genvidbench_comparisons/`。
- 主文当前表格：`tables/main_results.tex`、`tables/component_ablation.tex`、`tables/temporal_ablation.tex`、`tables/window_coverage_ablation.tex`、`tables/spatial_ablation.tex` 和 `tables/u0_external_validation.tex`。

## Overleaf 设置

1. 上传整个 `ieee_alpha_stalled/` 目录。
2. Menu → Compiler → 选择 `XeLaTeX`。
3. 主文件选择 `main.tex`。
4. 如果参考文献第一次不显示，按 Overleaf 的 Recompile 再编译一次。

## 当前写作边界

- 主结果是固定 21,421 条视频交集上的无 Spatial D2-only 方法：Macro-3 AUC/real-positive AP `0.8741/0.8729`。
- 方法固定 `G_k=0.5G_s+0.5G_{t1}`、`L_k=L_{D2}`、K=3 分支均值、effective-K 真实重校准和 `S=0.6G+0.4L`。
- 生成视频和测试真实视频不得进入 whitening、CDF 或阈值拟合，locked run 不再据此调参；但 `alpha/beta/K` 和总体结构曾在三个开发基准上查看生成结果后冻结，因此它们不是 untouched confirmation sets。历史 leakage-affected `0.8737/0.8750` 只能作为审计记录。
- Local Spatial、K=5/all-window、bottom-2/hybrid、residual、多尺度、中间层、Joint Typicality、三分支和动态融合均不是正式方法。
- 正式 K1/K3 因果对照使用无 Spatial 的 `full_d2_k1_no_spatial_refit` 与 `full_d2_k3_no_spatial_refit`；Macro AUC/AP 增益为 `+0.0060/+0.0024`。
- 旧 U0、region、校准 reserve、K5/all-window 和 45,185 fake 结果只作为历史资产，在无 Spatial 方法下重跑前不得写入当前摘要或主结论。
- 外部集、鲁棒性和可控注入只按预锁定协议报告，不允许据此回调 U0。
- 数据/代码/伦理/利益冲突/作者贡献声明位于 `sections/08_declarations.tex`，未知事实保留为待补充项。
