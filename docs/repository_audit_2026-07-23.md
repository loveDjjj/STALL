# Alpha-STALLED 仓库审计与清理记录

审计日期：2026-07-23

## 仓库与环境

- Git 仓库：`/data/OneDay/STALL_project/STALL`
- 分支：`refactor/alpha-stalled-reproducible-release`
- 远端：`origin=loveDjjj/STALL`，`upstream=OmerBenHayun/STALL`
- Conda 环境：`stall`
- Python：3.10.20
- PyTorch：2.11.0+cu128
- NumPy / pandas / scikit-learn：2.2.6 / 2.3.3 / 1.7.2

仓库没有 `.codegraph/` 索引，因此本次审计按仓库文档、Git 状态和源码直接完成。

## 目录边界

| 类型 | 路径 | 处理 |
|---|---|---|
| 核心代码 | `src/`, `tools/`, `scripts/`, `tests/` | 提交 |
| 冻结配置与小型校准参数 | `configs/`, `precomputed/` | 仅提交主线资产 |
| Release 分数与表格 | `results/paper_*` | 提交 |
| 稳定研究结论 | `results/research_summary/` | 提交 |
| 历史已提交实验 | `results/journal_experiments/` | 保留；新增输出默认忽略 |
| 数据、embedding 与 patch cache | `datasets/`, `cache/`, `dinov3/` | 本地保留，不提交 |
| 日志与 debug 输出 | `logs/`, `debug_outputs/` | 本地临时文件，不提交 |

审计时本地体量约为：`datasets/` 115 GB、`cache/` 323 GB、`dinov3/` 1.2 GB、
`results/` 499 MB。前三类是运行输入或缓存，不能依据 Git 清理需求误删。

## 清理前 Git 数据

- 未跟踪文件 698 个，其中 `results/` 659 个、`tools/` 28 个、`paper/` 11 个。
- 未跟踪结果约 323 MB，主要由逐视频融合输出、bootstrap 抽样明细、重复 split、
  prefill/runlist、smoke 结果和多个失败探索目录构成。
- 这些结果来自不同样本、窗口、校准集和聚合协议，不能直接合并为附件要求的
  统一基线；其结论已汇总到 `results/research_summary/`。

## 版本控制策略

1. 逐视频源分数只有在无法低成本重建且属于冻结协议时才提交。
2. 融合网格、bootstrap 抽样明细、临时 split/prefill 清单和 smoke 输出不提交。
3. 每组实验最多保留一个结论说明和一个紧凑指标表；跨实验结论统一进入
   `results/research_summary/`。
4. 一次性探索脚本在结论稳定后删除；正式评测器、协议构建器、验证器和论文资产
   构建器保留。
5. 新实验默认被 `.gitignore` 排除，需经过协议审查后显式提升为 release 资产。

## 后续任务边界

用户附件要求下一阶段使用同一视频 ID、real/fake 数量、DINOv3 cache、窗口、FPS、
calibration set 和缺失规则重建 B0-B8。现有分散结果只用于发现问题和确定优先级，
不能替代该统一协议实验。阶段 0 的公式与实现审计应先于任何大规模重跑。
