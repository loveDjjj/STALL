# 项目组织与实验治理规范

本文档定义后续代码、数据、缓存和实验结果的归类规则。它不改变任何锁定算法
或结果；当前协议仍以 `docs/CURRENT_PROTOCOL.md` 和 locked release 为准。

## 1. 事实来源层级

从高到低：

```text
locked YAML + release hashes/manifests
-> final per-video scores
-> machine-readable metrics/experiment registry
-> experiment reports
-> generated paper tables and manuscript text
-> historical README/archive notes
```

低层文档与高层资产冲突时必须修正低层文档，禁止反向修改锁定资产来匹配手工
写入的论文数字。

## 2. 当前目录职责

| 目录 | 当前职责 | 新增内容规则 |
|---|---|---|
| `src/` | 可复用特征、数学、评分和指标逻辑 | 不写实验路径和目标数据集 oracle |
| `tools/` | 当前 CLI、实验分析和 release 工具 | 新脚本必须声明协议/实验 ID |
| `configs/` | 正式、扩展和历史协议 YAML | 每个 YAML 必须声明状态和 protocol ID |
| `scripts/` | 可恢复的命令编排 | 不实现算法或指标公式 |
| `release/` | 不可变发布资产 | 只能创建新 release，不能覆盖旧 release |
| `results/` | run 输出、轻量结论和历史论文资产 | 每个新 run 使用独立目录 |
| `reports/` | 协议、结果、准入和失败原因 | 不作为唯一机器可读证据 |
| `paper/` | 从已登记结果生成的论文资产 | 不允许只在 LaTeX 手改指标 |
| `assets/` | 仓库级静态图片 | 只保留被引用或有明确用途的轻量资产 |
| `notebooks/` | 交互式 demo | 不作为正式评分或指标事实来源 |
| `supplementary/` | 原版 STALL 样本 manifest | 不得与 `release/u0/` 的锁定清单混用 |
| `cache/` | 可重建特征和中间结果 | 不进入 Git，必须由完整 cache key 区分 |
| `datasets/` | 原始视频 | 只读，不进入 Git |
| `logs/`, `debug_outputs/` | 运行日志和临时诊断 | 不作为结论或成功运行的唯一证据 |
| `research_archive/` | 已冻结的历史探索 | 默认不被主线导入 |

`release/README.md` 必须索引每个发布目录；每个发布目录还必须用自己的 README
递归索引全部资产，并明确区分核心锁定文件、补充诊断和外部确认。release 根目录
不得直接出现资产文件，新协议必须创建独立子目录。`tools/verify_project_documentation.py`
会拒绝未索引目录、未索引文件以及直接落在 release 根目录的产物。

### 2.1 文档生命周期

- `README.md`、`FILE_STRUCTURE.md` 和 `docs/` 顶层治理文档描述当前状态；
- `docs/restructure/` 只保留旧外部路径的短兼容入口，不再存放历史正文；
- 完整 pre-U0 发布材料冻结在 `research_archive/docs/pre_u0_release/`；
- 历史正文可以保留当时路径和术语，但必须在开头标明历史身份；
- 新文档不得引用兼容入口作为当前证据，必须直接引用 locked 配置、release 资产、
  当前治理文档或登记后的实验报告。

## 3. 方法和协议命名

- 方法名统一为 `Alpha-STALLED`。
- `u0_locked_v1` 是当前 strict-20 release protocol ID。
- `duration_aware_23source_v1` 是独立覆盖扩展。
- `LSTL` 不再作为另一个方法名；历史报告中它表示完整 Local-D2 U0。
- 配置或结果名称必须包含影响可比性的协议限定，如 K1/K3、strict20/full23、
  AP_real/AP_fake、Macro-3/All-23。

改变数据、split、窗口、backbone、分支、校准、数值规则或指标定义时，必须创建
新 protocol ID，禁止静默修改 locked YAML。

## 4. 新实验最小结构

新实验先写入：

```text
results/runs/<experiment_id>/
  run_capture.json
  run_manifest.json
  per_video_scores.csv
  dataset_metrics.csv
  generator_metrics.csv
  decision.json
```

需要逐窗口结果时增加 `per_window_scores.csv`；需要 bootstrap 时增加
`bootstrap_deltas.csv`。大文件可留在忽略目录，但 `run_manifest.json` 必须记录
路径、字节数和 SHA-256。

`run_manifest.json` 使用 `alpha_stalled_run_manifest_v1`，按职责分组：

```text
identity: experiment_id, protocol_id, parent_experiment_id, status
provenance: mode, git commit, dirty flag, environment, commands, timestamps, gaps
factors: changed, frozen
selection: generated-for-fit, generated-for-selection, selection scope
data: calibration/evaluation/overlap counts, score/AP/Macro semantics
metrics: Macro AUC/AP
artifacts: inputs, intermediates, outputs with role/path/bytes/SHA-256
outcome: state, failures
```

未来运行必须使用 `tools/capture_experiment_run.py` 启动，在命令执行前写入
`run_capture.json`，原生记录带时区的开始/结束时间、exact argv、Git commit/dirty
status、环境和退出码。输出目录固定为 `results/runs/<experiment_id>/` 且不可复用。
完成后，YAML spec 通过 `provenance.capture_path` 引用 capture；builder 自动将其作为
输入 artifact 哈希。历史运行只能使用 `reconstructed`，并显式记录
`provenance_gaps` 和 canonical reproduction commands；禁止补写无法证明的原始命令
或时间。共享实现位于 `src/alpha_stalled/run_capture.py` 和 `run_manifest.py`，入口为
`tools/capture_experiment_run.py`、`build_run_manifest.py` 与
`verify_run_capture.py`、`verify_run_manifest.py`。

## 5. 实验状态

只使用 `docs/RESULT_STATUS.md` 定义的状态。`invalid_leakage`、`superseded` 和
`rejected` 资产可以保留，但不得由默认论文生成流程读取。准入失败必须保存
失败原因，不能通过继续调权重或堆叠其他失败组件恢复成主方法。

机器可读注册表 `reports/u0_experiment_registry.csv` 使用两类互不替代的字段：

- `status`：由本节定义的证据生命周期，决定是否允许进入正式结论；
- `paper_status`：展示位置或历史角色，不得用来绕过 `status`。

`experiment_id` 必须稳定且唯一，`parent_experiment_id` 必须指向注册表中已有实验，
`result_path/report_path` 必须是仓库内可解析的相对路径。当前 registry 由
`src/alpha_stalled/experiment_registry.py` 和
`tools/verify_experiment_registry.py` 校验；新增结论级实验不得只追加报告文字。

## 6. 数据与 split

统一字段：

```text
video_id, dataset, source_model, subset, label, protocol_split
video_path, filename, duration_seconds
sampling_mode, effective_k, window_id, frame_indices
```

现有 `subset=annotated` 表示生成视频；新 schema 应同时保存明确的二值 `label`
和 `label_semantics`，但迁移时不得直接改写旧 CSV。release 应保存身份 hash、
文件大小和内容 hash/稳定指纹，避免同名文件内容变化。

## 7. Cache key

特征/分数缓存至少由以下字段共同确定：

```text
checkpoint SHA-256 + DINO commit + layer
preprocessing + frame-index SHA-256 + token type
frame grouping/batch protocol + numerical implementation
region + temporal definition + aggregation + calibration parameter SHA-256
```

历史 K1 patch cache 不得仅因 shape 相同而复用于正式 K3；不同 batch grouping
也不能默认视为等价。缓存元数据缺失时按不可复用处理。

新 cache root 已由 `src/alpha_stalled/cache_contract.py` 实现两级契约：root contract
锁定 checkpoint SHA、DINO commit、输出层、预处理、batch grouping 和 extractor
source hash；每个 `.pt.meta.json` 锁定源视频 SHA、精确帧索引、tensor 描述和 cache
SHA。`auto` 对空目录启用 strict contract，对已有无 contract 的 `.pt` 根目录发出
legacy 警告；`strict` 拒绝为历史文件补写身份；`legacy` 不能绕过已存在的 contract。

当前十个 global/patch 根目录都早于该机制，仍是 legacy。contract-aware 主入口是
`src/eval.py`、`tools/prefill_patch_cache.py`、`src/create_patch_params.py` 和
`src/eval_patch_fast.py`。新 patch 参数保存 feature-cache contract SHA，评分时必须与
缓存 root 完全一致；旧参数与旧缓存都只标记为 `legacy_uncontracted`。直接调用
`torch.load` 的历史研究工具只能复现 legacy 实验，不能证明新实验缓存兼容性。
完整边界见 `reports/feature_cache_contract.md`。

```bash
conda run --no-capture-output -n stall \
  python tools/verify_feature_cache_contract.py \
  --cache-root cache/patch_embeddings/<new_contract_root> \
  --verify-cache-hashes
```

### 7.1 缓存库存与生命周期

`configs/cache_inventory.yaml` 是当前缓存分组和保留政策的声明源；
`results/research_summary/cache_inventory.json` 是实测文件数、逻辑字节数、扩展名
分布与布局指纹快照；`reports/cache_inventory.md` 是面向人工审查的决策表。

清理优先级固定为：

```text
P0_keep                  小型协议或 smoke 资产，保留
P1_keep_active           当前研究仍消费，活跃保留
P2_review                历史或昂贵资产，删除前逐项复核
P3_safe_delete_candidate 无 release/当前 consumer，但仍须人工批准
```

库存构建器要求 `cache/` 中每个文件恰好归属一个互不重叠的 group。索引内容做
SHA-256；数百 GiB 特征缓存只记录相对路径和字节数的布局 SHA-256，避免验证过程
完整读取所有 tensor。任何 `P3` 声明都必须同时满足非 release 依赖和
`safe_delete_candidate`，工具没有删除功能。

```bash
conda run --no-capture-output -n stall \
  python tools/build_cache_inventory.py --check

conda run --no-capture-output -n stall \
  python tools/verify_cache_inventory.py
```

`--skip-layout` 只检查声明结构，不能作为本机缓存仍与快照一致的证据。新增缓存
目录时必须先更新 YAML 并重新生成 JSON/报告；不允许为了通过验证而忽略目录。

### 7.2 数据身份总账

`configs/data_catalog.yaml` 是 canonical dataset 声明源；
`results/research_summary/data_catalog.json` 是由三个 canonical index 和 locked U0
manifests 生成的机器可读快照；`reports/data_catalog.md` 是来源、时长与 split 覆盖表。

固定身份使用 `dataset|subset|source_model|filename` 的 SHA-256。当前 60,949 个
canonical 身份无重复，600 个 calibration 和 21,421 个 evaluation 身份全部可回溯
到且仅回溯到一个 canonical index，本机缺失源视频为 0。38,928 个未进入 locked U0
的身份仍保留为可用数据，不得解释为错误或被拒绝样本。

派生 calibration/holdout CSV 只属于生成它的实验；不能与 canonical index 并列成为
新的数据集定义。catalog 对 index 和 release manifest 做内容 SHA-256，并检查视频
文件存在性，但为了避免每次扫描 115 GiB 数据，不对所有视频字节做内容哈希。

```bash
conda run --no-capture-output -n stall \
  python tools/build_data_catalog.py --check

conda run --no-capture-output -n stall \
  python tools/verify_data_catalog.py
```

### 7.3 参数资产身份

`configs/parameter_assets.yaml` 是已提交 Global/Local 参数的机器总账，覆盖当前
locked U0、外部确认和历史兼容三种生命周期。每个登记文件锁定路径、字节数、
SHA-256、NPZ数组集合/形状/dtype、calibration count、whitening rank 和 Local
aggregation语义；当前U0的4个路径与哈希还必须同时匹配locked配置及release哈希索引。

`precomputed/` 中未登记的 `patch_params_*.npz` 和 `debug_patch_params_*.npz` 只允许
作为被Git忽略的本机可重建sweep。它们即使shape兼容，也不能成为当前协议、论文主结果
或release输入。`release/*/params/*.npz` 则必须全部逐文件登记，不允许存在来源不明的
参数。确定性报告写入 `reports/parameter_asset_inventory.md`。

```bash
conda run --no-capture-output -n stall \
  python tools/verify_parameter_assets.py
```

## 8. 配置身份与生命周期

`configs/config_registry.yaml` 是配置资产的机器可读总账，必须完整覆盖 `configs/`
下除 README 外的 YAML、CSV 和 template。每项资产均声明角色、生命周期、协议引用、
可变性和权威级别；只允许 `configs/alpha_stalled_u0_locked.yaml` 具有
`authority: current`，且它必须是不可变的 `current_locked` 配置。覆盖扩展、历史配置、
release 输入、运行证据、治理声明和模板不能冒充当前方法。

历史方法配置还必须在文件内部携带与注册表一致的 `config_identity`。新增协议应创建
新文件和新 `protocol_id`，不能修改 locked U0 或仅靠文件名、注释表达状态。
`tools/verify_config_registry.py` 检查精确文件覆盖、协议引用、内部身份和唯一当前权威；
总体验证器也执行同一检查，并要求 `configs/README.md` 索引全部配置资产。

### 8.1 运行环境身份

根目录 `environment.yml` 是精确版本的直接依赖声明；
`environment.lock.yml` 是当前验证平台 Linux x86-64、CUDA 12.8 的完整解析锁，固定
全部 Conda build 和项目 pip 依赖闭包。两者用途不同：前者便于维护直接依赖，后者
用于严格复现，不得用一次新的无约束求解覆盖旧锁。

`src/alpha_stalled/environment_lock.py` 对 `src/` 和 `tools/` 执行 AST 导入扫描，当前
15个外部顶层模块必须全部映射到直接依赖；直接声明必须与锁中版本一致，锁中Conda项
必须包含build string，pip项必须使用精确 `==`。`tools/verify_environment_lock.py`
还可将活动环境逐包与锁比较并运行 `pip check`。对于锁中明确同时出现的
`setuptools`/`tzdata`，pip覆盖安装可能移除同名Conda元数据；验证器始终检查有效pip
版本，并在Conda记录仍存在时继续检查其版本和build。其他Conda包不得缺失。本机额外
下载/排版辅助pip包可以存在，但不会进入项目依赖闭包，也不能据此修改release结果。

```bash
conda run --no-capture-output -n stall \
  python tools/verify_environment_lock.py --check-current
```

## 9. 工具脚本迁移边界

历史上正式 U0 曾形成以下反向依赖：

```text
analyze_u0_locked -> build_multi_order_baselines
verify_u0_locked_release -> analyze_u0_locked + build_multi_order_baselines
```

这些正式依赖现已清除：scorer、analyzer 和 verifier 均只依赖
`src/alpha_stalled/` 或 `src/` 正式模块，不再导入其他 `tools/` CLI，并由
`tests/test_u0_protocol.py` 的 AST 边界检查锁定。D3 专用工具仍可依赖历史
`build_multi_order_baselines.py` 的 D3 定义，但通用 metrics 不再经它转发。
完整边界见 `reports/u0_protocol_code_boundary.md`。

其他历史研究脚本仍有裸模块交叉依赖，因此禁止直接批量移动。安全迁移分四阶段：

当前依赖不是靠文本搜索管理。`src/alpha_stalled/tool_dependencies.py` 对全部顶层
Python 工具执行 AST 分析；`configs/tool_dependencies.yaml` 明确登记许可边、导入
符号、实验类别、迁移状态和原因。确定性快照写入
`results/research_summary/tool_dependency_inventory.json`，可读表写入
`reports/tool_dependency_inventory.md`。当前 126 个工具中有 8 条已登记边、无循环，
50 个正式、治理或已迁移历史入口保持零内部工具依赖。剩余边被划分为4个retained
family；每个family严格覆盖其边端点，并内容寻址关键报告、指标、论文图表，记录未来
归档目标和移动门槛。任何新边、删除但未更新的边、符号变化、循环、证据漂移、族归属
遗漏或隔离入口反向依赖都会使 `tools/verify_tool_dependencies.py` 失败。

全部126个入口同时具有唯一生命周期：`formal_release=10`、`paper_evidence=45`、
`governance=16`、`historical_frozen=38`、`research_utility=1`、
`compatibility=16`。正式发布和治理入口必须是CLI依赖叶节点；现存内部边必须保持在
同一生命周期。研究辅助结果只有经过新protocol、实验登记和证据审计后才能升级为
论文证据，冻结历史入口不得用于继续搜索主方法参数。

对`research_utility`和`compatibility`，治理器扫描根目录文档以及`configs/`、
`docs/`、`paper/`、`release/`、`reports/`、`research_archive/`、
`results/research_summary/`、`scripts/`、`src/`、`tests/`和`tools/`中的文本引用，
排除普通实验结果目录、政策本身、
生成清单和工具自身。当前维护决策为`compatibility_wrapper=14`、
`retain_referenced=3`。十四个历史工具已完成可逆物理归档：实现位于
`research_archive/tools/journal_experiments/`或`research_archive/tools/pre_release_assets/`，
旧命令路径只作兼容转发。清单记录
移动前、包装器、归档实现和结果证据SHA-256。仍被引用的工具只有在兼容路径稳定或
调用方迁移完成后才能移动。

`audit_patch_likelihood_assumptions`使用已废弃的ComGenVid region-3配置，生成的表和
图未被当前U0正文包含，因此作为历史实现归档，不能作为locked U0证据。唯一仍处于
`research_utility`生命周期的入口是活跃patch cache生产者`prefill_patch_cache`。
旧`verify_alpha_stalled_release`只验证pre-U0的dataset-specific score CSV和alpha sweep；
当前release必须使用`verify_u0_locked_release`的manifest、raw reconstruction和哈希检查。

1. **入口治理（当前阶段）**：统一 README、协议、状态、配置和报告索引。
2. **共享包提取**：将 sampling、manifest、calibration、metrics、bootstrap、I/O
   提取到 `src/alpha_stalled/`，保留原函数的回归测试。窗口 sampling 和 release
   I/O 已完成提取；旧 `analyze_multi_window_feasibility`、
   `build_u0_release_manifests` 与 `score_multi_window` 继续保留兼容出口。可直接转发的
   API 保持同一函数对象；retry 包装保留历史 mock 入口。
   U0 分片统一使用 `release_io.video_id_shard`；旧 multi-window 的五列身份
   哈希是不同的历史协议，不与 U0 分片合并。
   Global 参考分布、raw Gaussian 参数和 U0 `+inf` T1 CDF 已分别集中到
   `parameters.py` 与 `calibration.py`。共享包统一采用显式子模块导入，包级
   `__init__` 无导入副作用，以免轻量 manifest/finalize 工具隐式加载
   OpenCV 或 PyTorch。
float64 Gaussian 评分、D1/D2、零差分掩码和 right-inclusive ECDF 已集中到
`alpha_stalled/whitening.py`；`src/stable_whitening.py` 仅保留兼容转发。
历史score CSV的三列身份读取、严格Global/Local连接、固定融合和pairwise指标已集中到
`alpha_stalled/score_csv.py`；`eval_alpha_stalled.py`、`eval_score_csv.py`和
`fuse_scores.py`只保留CLI参数、结果写出与sweep展示。
   正式 region1/mean Local D1/D2、raw scoring 和固定 beta 融合已集中到
   `alpha_stalled/local_branch.py`。`patch_matching.py` 与 `patch_math.py` 只服务
   generic fitting、审计和历史实验，不得由 locked U0 新增反向依赖；边界见
   `reports/local_branch_code_boundary.md`。
   正式 Global Spatial/T1 max/min raw scoring 和固定等权融合已集中到
   `alpha_stalled/global_branch.py`；`stall.py` 保留原版检测器和特征提取兼容性，
   D3/volatility 仍属于独立历史实验。完整边界见
   `reports/global_branch_code_boundary.md`。
   DINOv3 路径、标准预处理、加载和 STALL/PatchSTALL 共享模型缓存已集中到
   `alpha_stalled/backbone.py`。进程复用 key 不能替代 checkpoint 内容 SHA、DINO
   commit 和 extractor source hash；两层身份边界见
   `reports/backbone_code_boundary.md` 与 `reports/feature_cache_contract.md`。
   全量、任意索引和 locked-window 原生帧解码已集中到
   `alpha_stalled/video_io.py`。strict cache 使用 `require_all=True`，因此 sidecar
   记录的每个帧索引都必须存在；原版静默省略缺帧的行为只由兼容入口保留。边界见
   `reports/video_io_code_boundary.md`。
   U0 的窗口级四分量校准、`G_k/L_k/S_k` 与视频级 `0.6G+0.4L` 也由
   `calibration.py` 单点定义；D1/D2 消融复用相同 Local/最终融合函数。
   pairwise-balanced AUC/AP、generator-pair、dataset/Macro-3 表、paired bootstrap、
   分数方向和真实来源均衡采样已集中到
   `alpha_stalled/metrics.py`；`src/metrics.py` 只做原版与 archive 的兼容转发，
   正式 `src/` 和 `tools/` 不再从兼容层导入。
   Macro-3 cluster bootstrap、扩展 binary metrics、cluster row repeat 和稳定 seed
   也已集中到 `alpha_stalled/metrics.py`；历史 multi-window/metric-audit CLI 只保留
   同一函数对象的兼容出口。U0 K1 四分量 raw calibration 和 K3 effective-K candidate
   calibration 已集中到 `alpha_stalled/u0_analysis.py`，核心/二阶消融不再导入评分或
   分析 CLI。历史 multi-window 的五列身份、窗口读取、原生帧解码、retry、sharding、
   resume key 和 calibrated score row 由 `alpha_stalled/legacy_window_scoring.py`
   集中定义；`score_multi_window.py` 只保留运行编排及 mock-compatible retry 包装。
   名称中的 `legacy` 是协议边界，locked U0 不得反向依赖该模块。duration-aware
   23-source 的数据身份、校准规模和任务构建由
   `alpha_stalled/duration_aware_protocol.py` 定义，联合窗口解码和score-row身份由
   `duration_aware_scoring.py` 定义，full-23 cohort与固定50% prevalence指标由
   `duration_aware_metrics.py` 定义；对应构建、评分和分析CLI不再互相导入。
   U0 calibration-size、cross-domain、independent-real和OAS实验共用的真实reserve设计、
   candidate命名、raw part读取及两级校准由
   `alpha_stalled/u0_calibration_experiments.py` 单点定义；缓存K1四分量raw评分则复用
   `u0_scoring.score_raw_components`。历史multiscale/clean-universal的窗口身份与
   target-K参考、Joint Typicality的lower-tail校准聚合集中到
   `historical_window_analysis.py`；Local-D2 residual的数据集和strict交集契约集中到
   `legacy_local_d2_protocol.py`。五轮迁移累计将内部工具边从45条降低到8条，当前
   已无`migrate_to_shared`边，剩余边全部是显式冻结的历史实验族。
   历史multi-window分析入口可读取完整原始分片或release保留的相邻合并CSV；若只
   存在部分分片则立即失败，不会用合并产物静默补齐不完整运行。
   locked manifests、raw shard 完整性、600 条 K1 reference、帧索引核对和
   effective-K reference 选择已集中到 `alpha_stalled/u0_protocol.py`；正式三个 CLI
   仅保留参数解析、流水线编排和产物写出。严格去重解码、逐视频 DINO batching 与
   四分量 raw scoring 由 `alpha_stalled/u0_scoring.py` 定义；窗口级四分量校准、
   视频均值和 effective-K CDF 由 `alpha_stalled/u0_analysis.py` 定义。其他工具不得
   再从正式 scorer/analyzer CLI 反向导入这些函数。
   本机 retained raw shards 的强重建证据由只读
   `tools/verify_u0_pipeline_reconstruction.py` 提供；它不覆盖 locked validation JSON。
   effective-K 的 midpoint/endpoint 位置规则和逐视频均值由
   `aggregation.py` 定义；effective-K 对应的 G/L 视频 CDF 继续由
   `calibration.py` 统一实现。数据集、split 与校准成员筛选仍由协议包装器显式完成。
   expected shard、checkpoint part 与 resume ID 的确定性读取由
   `artifacts.py` 实现；重复键、预期行数和 split 完整性仍由各协议入口检查。
3. **薄 CLI**：旧工具改为只解析参数并调用包函数，同时保留兼容入口。
4. **实验归档**：只有对应retained family的证据、兼容包装和run manifest满足其
   `archive_gate` 后，才能将整族拒绝/历史 CLI 移至配置声明的
   `research_archive/tools/<family>`；禁止逐文件移动或打断论文复现路径。

每一步必须先证明 locked release verifier、核心测试和历史需要保留的复现入口
没有变化，再进行下一步。

## 10. 建议目标结构

```text
src/alpha_stalled/
  backbone.py
  sampling.py
  global_branch.py
  local_branch.py
  whitening.py
  calibration.py
  u0_protocol.py
  u0_scoring.py
  u0_analysis.py
  aggregation.py
  metrics.py
  manifests.py
  release_io.py
  video_io.py
  parameters.py
  artifacts.py
  experiment_registry.py
  run_capture.py
  cache_contract.py
  legacy_window_scoring.py

experiments/
  main_u0/
  second_order/
  duration_aware/
  calibration/
  robustness/
  rejected/
  legacy/

results/
  runs/<experiment_id>/
  research_summary/

release/
  u0/
  duration_aware_23source/
```

这是迁移目标而不是要求一次性改名。当前最重要的是保持 release 路径和脚本兼容。

## 11. 每次提交前检查

```bash
conda run --no-capture-output -n stall \
  python tools/verify_environment_lock.py --check-current

conda run --no-capture-output -n stall \
  python tools/verify_project_documentation.py

conda run --no-capture-output -n stall \
  python tools/verify_u0_locked_release.py

conda run --no-capture-output -n stall \
  python tools/verify_u0_pipeline_reconstruction.py

conda run --no-capture-output -n stall \
  python tools/verify_experiment_registry.py

conda run --no-capture-output -n stall \
  python tools/verify_config_registry.py

conda run --no-capture-output -n stall \
  python tools/verify_run_manifest.py

conda run --no-capture-output -n stall \
  python tools/verify_cache_inventory.py

conda run --no-capture-output -n stall \
  python tools/verify_data_catalog.py

conda run --no-capture-output -n stall \
  python tools/verify_parameter_assets.py

conda run --no-capture-output -n stall \
  python tools/verify_tool_dependencies.py

conda run --no-capture-output -n stall \
  python -m unittest tests/test_project_documentation.py \
                     tests/test_experiment_registry.py \
                     tests/test_config_registry.py \
                     tests/test_release_index.py \
                     tests/test_run_capture.py \
                     tests/test_run_manifest.py \
                     tests/test_cache_inventory.py \
                     tests/test_data_catalog.py \
                     tests/test_environment_lock.py \
                     tests/test_u0_locked_release.py \
                     tests/test_u0_metric_protocol.py \
                     tests/test_whitening_batch_invariance.py
```

新实验还必须执行对应的协议、split、公式和指标测试，不能用上述轻量检查替代。
