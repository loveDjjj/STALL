# Alpha-STALLED 写作计划

## 当前版本

- 已建立 IEEE 双栏中文 LaTeX 初稿。
- 已将正文拆分为 abstract、introduction、related work、method、experiments、ablation、discussion、conclusion。
- 已放入主图、主表、核心参考文献和 `2603.15026v2.pdf`。

## 下一轮优化顺序

1. 在 Overleaf 用 XeLaTeX 编译，记录所有 LaTeX 报错、overfull box 和图片位置问题。
2. 根据编译结果调整图宽、表格压缩和双栏浮动位置。
3. 补充作者、单位、基金、数据/代码可用性声明。
4. 若目标是期刊而不是会议，确定目标期刊后再调整摘要结构、章节粒度和参考文献格式。
5. 如果审稿或投稿需要外部 baseline，优先按 D3 protocol audit 重跑 D3-DINOv3；AEROBLADE/RIGID/ZED/T2VE/AIGVDet 另建独立 baseline 表。

## 术语表

| 规范术语 | 含义 | 避免写法 |
|---|---|---|
| Alpha-STALLED | 本文方法 | alpha stalled, Alpha STALL |
| STALL | 原文全局真实视频似然检测器 | 原始方法、baseline 方法（首次出现后可说明） |
| 全局分数 | STALL 全局空间--时序分数 | 全局概率、全局判别器 |
| 局部分数 | patch 空间和 patch 二阶时序先组合后的分数 | 局部空间和局部时序直接与全局三分支并列 |
| 同网格二阶时序 | 固定 DINOv3 patch 网格上的二阶差分 | 光流、patch 跟踪 |
| pairwise balanced AUC/AP | 每个生成器与等量真实视频比较后平均 | unbalanced AP |
