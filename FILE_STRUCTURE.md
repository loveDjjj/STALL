# Alpha-STALLED 文件架构

本文档是当前仓库的目录导航。正式实验协议、结果状态和治理规则分别以
`docs/CURRENT_PROTOCOL.md`、`docs/RESULT_STATUS.md` 和
`docs/PROJECT_ORGANIZATION.md` 为准；历史目录说明不能覆盖这三份文档。

## 当前事实来源

| 路径 | 作用 |
|---|---|
| `configs/alpha_stalled_u0_locked.yaml` | `u0_locked_v1` 的机器可读冻结配置 |
| `configs/config_registry.yaml` | 全部配置资产的身份、生命周期与唯一当前权威声明 |
| `configs/parameter_assets.yaml` | 已提交参数文件的哈希、NPZ契约和生命周期声明 |
| `environment.yml` / `environment.lock.yml` | 精确直接依赖与Linux/CUDA 12.8完整解析锁 |
| `release/u0/` | 锁定校准/评测清单、哈希和发布身份 |
| `release/README.md` | 主发布与外部确认release的角色和目录索引 |
| `docs/CURRENT_PROTOCOL.md` | 当前协议的人类可读说明 |
| `docs/RESULT_STATUS.md` | 正式、消融、诊断、拒绝和无效结果账本 |
| `reports/u0_experiment_registry.csv` | 结论级实验的机器可读注册表 |
| `results/research_summary/` | 统一指标、数据/cache/工具库存和验证快照 |

发生冲突时，优先级为：锁定配置与 release 哈希、逐视频分数、机器可读指标和
注册表、实验报告、论文表格、历史文档。

## 顶层目录

| 目录 | 当前职责 | 提交边界 |
|---|---|---|
| `src/alpha_stalled/` | U0 采样、分支、校准、聚合、指标、资产和治理共享实现 | 提交 |
| `src/` | 原版 STALL 兼容实现和底层特征/cache 代码 | 提交 |
| `tools/` | 正式 CLI、受控实验入口、验证器及历史兼容包装器 | 提交；生命周期见工具库存 |
| `configs/` | 正式、扩展、历史协议与治理 YAML | 提交；不得静默修改 locked 配置 |
| `release/` | 不可变发布清单和哈希 | 提交；新版本新建目录 |
| `docs/` | 当前协议、结果状态、治理和数据获取说明 | 提交 |
| `reports/` | 实验报告、审计和准入结论 | 提交轻量证据 |
| `results/` | run 输出、统一汇总及历史论文资产 | 按 `.gitignore` 和 manifest 管理 |
| `paper/` | 从已登记结果生成的论文源码和表格 | 提交；禁止只手改指标 |
| `scripts/` | 可恢复命令编排，不承载算法公式 | 提交 |
| `tests/` | 单元、协议、资产和文档回归测试 | 提交 |
| `assets/` | 仓库级静态图片资产 | 提交已引用或明确保留的轻量文件 |
| `notebooks/` | 原版 STALL 交互式演示 | 提交；不作为正式指标实现 |
| `supplementary/` | 原版 STALL 论文样本 manifest | 提交；与 U0 release manifest 区分 |
| `research_archive/` | 冻结历史探索、旧发布文档和归档实现 | 可提交；默认不被正式主线导入 |
| `precomputed/` | 旧 STALL/历史 patch 校准资产 | 按协议状态保留，不等同当前 release |
| `cache/` | 可重建特征和中间结果 | 不提交；受 cache contract/inventory 管理 |
| `datasets/` | 原始视频 | 不提交；canonical 身份见 data catalog |
| `dinov3/` | 本地第三方源码和权重 | 不提交 |
| `logs/`、`debug_outputs/` | 运行日志和临时诊断 | 不提交；不能作为完成证据 |

## 当前 U0 代码边界

`src/alpha_stalled/` 是可复用事实实现。关键模块包括：

- `u0_protocol.py`、`sampling.py`：锁定身份和确定性 K=3 窗口；
- `global_branch.py`、`local_branch.py`：Global T1/Spatial 与 Local D2/Spatial；
- `calibration.py`、`whitening.py`：独立真实校准和 effective-K ECDF；
- `aggregation.py`、`u0_scoring.py`、`metrics.py`：视频聚合、融合和 Macro 指标；
- `artifacts.py`、`release_io.py`、`release_index.py`、`run_manifest.py`：发布资产与运行证据；
- `data_catalog.py`、`cache_inventory.py`、`config_registry.py`、
  `parameter_assets.py`、`tool_dependencies.py`、`environment_lock.py`：项目库存治理。

`tools/` 中部分命令是历史兼容包装器，文件存在不表示它属于当前方法。每个工具的
`lifecycle`、维护决策、证据路径和归档目标统一记录在
`configs/tool_dependencies.yaml` 与 `reports/tool_dependency_inventory.md`。

## 数据、缓存和结果

- canonical 数据身份由 `configs/data_catalog.yaml` 声明，实测快照位于
  `results/research_summary/data_catalog.json`；
- cache 分组和保留策略由 `configs/cache_inventory.yaml` 声明，缓存不是发布证据；
- 参数身份由 `configs/parameter_assets.yaml` 声明；当前U0参数与本机ignored sweep
  不得混用；
- 新实验写入 `results/runs/<experiment_id>/`，并保存 `run_capture.json`、
  `run_manifest.json`、逐视频分数和指标；
- 旧 `results/paper_scores/` 属于 pre-U0 score-CSV 发布链，不能替代 locked U0；
- 旧发布结构文档已保存在 `research_archive/docs/pre_u0_release/`。

## 验证入口

当前 release 必须使用：

```bash
conda run --no-capture-output -n stall \
  python tools/verify_environment_lock.py --check-current
conda run --no-capture-output -n stall \
  python tools/verify_u0_locked_release.py
conda run --no-capture-output -n stall \
  python tools/verify_u0_pipeline_reconstruction.py
conda run --no-capture-output -n stall \
  python tools/verify_config_registry.py
conda run --no-capture-output -n stall \
  python tools/verify_project_documentation.py
conda run --no-capture-output -n stall \
  python -m unittest discover -s tests
```

`tools/verify_alpha_stalled_release.py` 仅是 pre-U0 score-CSV 资产的兼容入口，
不验证 locked manifests、原始窗口重建或 U0 哈希。
