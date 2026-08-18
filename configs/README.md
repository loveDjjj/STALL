# 配置状态

| 配置 | protocol/角色 | 状态 |
|---|---|---|
| `config_registry.yaml` | 全部配置资产的机器可读身份和生命周期注册表 | 当前治理声明源 |
| `alpha_stalled_u0_locked.yaml` | `u0_locked_v1` strict-20 主协议 | 当前锁定主方法 |
| `multi_window_exclusions.csv` | 已知不可解码窗口排除 | 多个锁定协议共享输入 |
| `run_manifests/alpha_stalled_u0_locked.yaml` | locked U0 run manifest spec | 历史重建 provenance |
| `run_manifests/captured_experiment.yaml.template` | 新实验 capture-backed manifest 字段模板 | 模板，不直接执行 |
| `cache_inventory.yaml` | 本机 cache 分组、生命周期和清理优先级 | 非破坏性治理声明 |
| `data_catalog.yaml` | canonical index、数据根目录与locked release manifests | 数据身份声明源 |
| `parameter_assets.yaml` | 正式、外部和历史参数文件的哈希、NPZ契约与本机sweep边界 | 参数身份声明源 |
| `tool_dependencies.yaml` | 顶层Python工具AST许可边与隔离入口 | CLI依赖治理声明 |

选择配置前先阅读 `docs/CURRENT_PROTOCOL.md` 与 `docs/RESULT_STATUS.md`。不得通过
修改 `alpha_stalled_u0_locked.yaml` 更新实验；任何协议变化都应创建带新
`protocol_version` 的 YAML、独立 manifest、输出目录和结果登记。

`config_registry.yaml` 必须完整覆盖本目录中除 README 外的 YAML、CSV 和 template；
只允许一个 `authority: current`，当前必须是 `alpha_stalled_u0_locked.yaml`。
历史方法配置必须携带显式 `config_identity`，新增配置不得只靠文件名或注释表达
状态。验证命令：

```bash
conda run --no-capture-output -n stall \
  python tools/verify_config_registry.py
```

`run_manifests/` 中的 YAML 不定义算法，而是定义运行身份、选择边界、命令和需要
哈希的 artifact。用 `tools/build_run_manifest.py` 生成 JSON，禁止手工填写 bytes
或 SHA-256。新实验必须先由 `tools/capture_experiment_run.py` 生成
`results/runs/<experiment_id>/run_capture.json`，spec 只引用该 capture，不能事后
手填 `captured` 时间和命令。使用 `tools/verify_run_capture.py` 独立检查运行记录，
再由 `tools/build_run_manifest.py` 生成带 artifact 哈希的完整 manifest。

`cache_inventory.yaml` 不定义算法，也不把大缓存纳入 release。它声明每个缓存组
的 producer、consumer、可重建性、cache-key 缺口和 `P0`--`P3` 保留级别。
`tools/build_cache_inventory.py` 生成实测快照和报告，
`tools/verify_cache_inventory.py` 只读验证当前布局；两者均不删除文件。

特征 cache 的 encoder/entry 身份不写入本目录的静态 YAML，而由
`src/alpha_stalled/cache_contract.py` 从实际 checkpoint、DINO Git 状态、预处理与
batching 生成 root JSON contract。现有缓存仍按 `cache_inventory.yaml` 标记为
legacy，不允许生成静态 YAML 后反向“认证”。

`data_catalog.yaml` 不复制逐帧索引或视频内容。它声明三个开发基准的 canonical
index、数据根目录及 locked U0 calibration/evaluation manifests，由
`tools/build_data_catalog.py` 生成 `results/research_summary/data_catalog.json` 和
`reports/data_catalog.md`，统一回答完整 index 规模、来源/时长覆盖、release 成员
关系和本机文件缺失情况。

```bash
conda run --no-capture-output -n stall \
  python tools/build_data_catalog.py --check
conda run --no-capture-output -n stall \
  python tools/verify_data_catalog.py
```

派生 calibration/holdout CSV 是实验资产，不得另行声明为 canonical dataset；正式
split 身份仍由 release manifest 决定。

`parameter_assets.yaml` 登记所有已提交的 Global/Local NPZ 参数，并验证字节数、
SHA-256、数组形状、calibration count 与 aggregation 语义。未登记的本机
`patch_params_*.npz` 只能作为被忽略的可重建 sweep 资产，不能成为 release 输入。

```bash
conda run --no-capture-output -n stall \
  python tools/verify_parameter_assets.py
```

`tool_dependencies.yaml` 不是鼓励 CLI 互相复用，而是锁定尚未迁移的历史依赖。
每条 retained edge 必须属于一个 retained family；family 必须登记完整工具端点、
证据文件、归档目标与移动门槛，生成清单会记录证据文件的大小和 SHA-256。
同一政策还必须把每个顶层工具唯一归入formal release、paper evidence、governance、
historical frozen、research utility或compatibility生命周期，并声明新工作与归档政策。
research utility和compatibility还必须逐工具声明维护决策。归档候选必须无入站引用且
登记内容寻址证据；物理归档后必须保留受验证的兼容包装器，并记录原始、包装器与归档
实现哈希；仍被引用的候选必须先声明替代方案和目标路径。
新共享逻辑必须进入 `src/alpha_stalled/`；使用
`tools/build_tool_dependency_inventory.py --check` 和
`tools/verify_tool_dependencies.py` 检查未登记边、导入符号、循环及正式入口隔离。
