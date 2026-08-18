# `results/` 数据说明

当前 `results/` 只保留轻量结论索引。权威逐视频分数和冻结 release 资产位于
`release/u0/`。

## `research_summary/`

| 文件 | 说明 |
|---|---|
| `README.md` | 当前研究结果、协议边界和保留工具的统一索引 |
| `experiment_metrics.csv` | 关键数据集/协议的 AUC、AP 与结论级状态 |
| `run_manifests/` | 结论级运行的配置、命令、选择边界和 artifact 哈希索引 |
| `cache_inventory.json` | 当前 cache 全覆盖、生命周期和只读布局指纹快照 |
| `data_catalog.json` | 三个 canonical index、60,949 个身份及 locked U0 覆盖快照 |
| `tool_dependency_inventory.json` | 顶层 Python 工具 AST 依赖、分类和迁移状态快照 |

## 更新原则

- 主实验证据以 `release/u0/` 和 `reports/` 为准。
- `results/` 只保留可快速浏览的机器可读索引。
- 历史探索结果不再放在 `results/` 树中，统一写入 `docs/EXPLORATION_LOG.md`。
