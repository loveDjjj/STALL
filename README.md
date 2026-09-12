# Alpha-STALLED

最新补实验已完成：独立新真实视频池、拟合位置数匹配及固定观察下的重编码测试均已验收，数据与边界已加入[论文正文](docs/MANUSCRIPT_REVISION_PLAN_zh.md)。主线参数未自动替换；新池与编码协议见[执行合同](docs/REFERENCE_CONFIRMATION_zh.md)。

2026-09-11：已按用户要求停止Looped训练并切回`paper/alpha-stalled`，保留所有已有训练成果和可复用Patch缓存。下一步围绕真实参考Local D2的独立信息与稳定性补证据，详见[研究复核与实验计划](docs/REAL_REFERENCE_D2_RESEARCH_PLAN_zh.md)；新实验尚未启动。

当前论文主线：**目标域适配Global + Local归一化D2**，Feature-change K3，Global/Local等权融合。
最新主实验及全部消融已补齐23单元，并对齐原文动态等级筛选：[完整结果](results/paper_complete/RESULTS_zh.md)、[论文正文](docs/MANUSCRIPT_REVISION_PLAN_zh.md)、[复现差异说明](docs/MANUSCRIPT_PROTOCOL_NOTES_zh.md)。当前完整协议Macro-3为**0.874472 / 0.877075**；以下旧数字是20单元回归锚点，模型未改变。
冻结开发配对Macro AUC/AP-real为 **0.881462 / 0.882444**。这是已有结果，不代表新域保证；模型需要独立目标真实拟合集，不能称零目标域参考。

前五组论文实验已完成当前代码重跑：见[完整结果](results/paper/RESULTS_zh.md)及[验收记录](results/runs/paper_completion_audit/verification.json)。包含逐生成器表、源组置信区间、独立阈值与成本测试。

## 安装与运行

从本目录运行，现有服务器使用conda环境`stall`。新环境可按`environment.yml`安装；GPU驱动、DINO源码/权重和数据另需准备，不能仅安装依赖就复现完整实验。

```bash
bash scripts/run_main.sh --help
bash scripts/run_main.sh score --dataset comgenvid --limit 2 --dry-run
bash scripts/run_main.sh --set 'runtime.devices=[cuda:0,cuda:1]' score \
  --dataset comgenvid --output results/runs/new_scores
bash scripts/run_main.sh evaluate --run-dir results/runs/new_scores \
  --pairs data/manifests/active/comgenvid/pairs.csv
```

完整命令、fit/cache/video、CDF、导出、恢复、组件表和bootstrap见[运行手册](docs/RUNBOOK_zh.md)。旧源码和报告不再保留为运行入口。
新run禁止覆盖；修改配置或源码后不沿用旧断点。主方法不修改DINO，不使用fake拟合Gaussian/CDF。

## 唯一活跃结构

```text
configs/paper.yaml         唯一基础配置
scripts/run.py            统一命令入口
scripts/run_main.sh       环境与参数透传
src/                      特征、选择、参考、评分、调度、数据与评价
tests/                    主线功能与数值回归
docs/                     当前方法、数据、运行与修改规范
data/manifests/active/    明确fit/CDF/threshold/evaluation/pairs角色
precomputed/target_reference/     冻结五域目标参考包
results/reference/       当前baseline与精选历史对照表
results/runs/            后续新实验输出
cache/                   保留的可复用特征、选窗、位置场与参数缓存
```

论文所需原视频与可复用缓存保留，缓存已按用途统一命名；当前目录和补充数据说明见[数据职责](docs/DATA_zh.md)，精确搬移映射见data/catalog/path_migration.json。此前清理记录paper_asset_pruning.json保留迁移前身份。
旧run、报告和重复快照已清理。历史配置、做法和逐子集数值只维护[这一份总结](docs/All_Branches_Experiments_and_Data_Summary_zh.md)，后续用新主线重跑。

## 规范入口

- [论文主线与实验矩阵](docs/PAPER_MAINLINE_zh.md)
- [数据职责](docs/DATA_zh.md)
- [仓库规范](docs/REPOSITORY_RULES_zh.md)
- [历史实验总结](docs/All_Branches_Experiments_and_Data_Summary_zh.md)
- [AI修改入口](AGENTS.md)

旧B3、U0、Universal、软CDF等数字不能混入新主线表。AP-real与AP-fake分开，pilot与全量分开，原版STALL与同窗口Global对照分开。
`paper/`中的历史稿尚未改为最新方法；正式稿以论文主线文档和新实验结果为依据，不从旧图直接推断当前实现。
