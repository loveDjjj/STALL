# STALL 论文写作区

本目录用于存放 Alpha-STALLED 论文相关材料。它与代码、缓存和实验结果目录解耦，便于后续根据新增实验持续修改正文、图表和参考文献。

当前主工作区：

- `ieee_alpha_stalled/`：基于 `IEEE_Conference_Template` 精简得到的中文双栏 LaTeX 初稿。

组织原则：

- 正文只引用已经在 `results/` 中审计过的实验，不把诊断上限、测试集 rank 路由或生成器来源选择写成主方法。
- 主文优先围绕全局 STALL 与局部 patch 二阶时序证据的互补性展开。
- 外部 baseline 当前只做协议审计，不写成已经完成重跑；若审稿要求，再按 `results/journal_experiments/d3_protocol_audit/` 的同索引、同帧、pairwise balanced 协议补跑。
- 本机不编译 LaTeX；Overleaf 建议选择 XeLaTeX。
