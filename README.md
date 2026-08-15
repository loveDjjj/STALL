# Alpha-STALLED / STALL

本仓库基于原版 STALL，实现并评测训练自由的生成视频检测方法
Alpha-STALLED。当前正式版本是统一锁定协议 `u0_locked_v1`：原版 STALL
Global 分支与 DINOv3 patch 同网格二阶时序 Local 分支共享一次 backbone
前向，并通过真实视频经验分布校准后固定融合。

> **权威入口：** 当前方法以
> [`configs/alpha_stalled_u0_locked.yaml`](configs/alpha_stalled_u0_locked.yaml)
> 和 [`release/u0/`](release/u0/) 为准。早期 `paper_scores`、dataset-specific
> region/aggregation 及 `0.9198/0.9211` 等结果属于历史协议，不是当前有效主结果。

## 当前协议

```text
G_k = 0.5 * GlobalSpatial_k + 0.5 * GlobalT1_k
L_k = 0.1 * PatchSpatial_k + 0.9 * same-grid PatchD2_k

G_raw(V) = mean_k(G_k)
L_raw(V) = mean_k(L_k)
G(V), L(V) = effective-K-matched target-real ECDF calibration
S(V) = 0.6 * G(V) + 0.4 * L(V)
```

- DINOv3 ViT-L/16 final layer 23，输入 `224x224`，patch 网格 `14x14`。
- 每个有效窗口为 2 秒、8 FPS、16 个互异帧。
- 在完整允许起点范围确定性均匀选择至多三个窗口；相同窗口去重。
- 严格主协议不使用 1 秒 fallback、不复制帧、不使用不足 16 帧的尾部。
- Local 使用 region 1 和 mean aggregation；region 1 表示不做 raw-token 空间池化。
- 每个数据集使用 200 条与评测身份互斥的真实视频拟合 Local 和视频级 CDF。
- 生成视频不参与 whitening、Gaussian 参数、CDF 或阈值拟合；locked run 不再
  根据评测生成视频调整配置。`alpha/beta/K` 和总体结构曾在这三个开发基准上
  查看生成视频结果后冻结，因此它们不是 untouched confirmation sets。
- 分数越高越接近真实视频；锁定主表 AP 以真实视频为正类。

完整定义、数据划分和复现入口见：

- [`FILE_STRUCTURE.md`](FILE_STRUCTURE.md)：当前目录职责、事实来源和提交边界。
- [`docs/CURRENT_PROTOCOL.md`](docs/CURRENT_PROTOCOL.md)：当前协议的唯一可读说明。
- [`docs/RESULT_STATUS.md`](docs/RESULT_STATUS.md)：正式、审计、拒绝和无效结果分类。
- [`docs/PROJECT_ORGANIZATION.md`](docs/PROJECT_ORGANIZATION.md)：代码、数据、缓存和实验治理规则。
- [`configs/README.md`](configs/README.md)：配置文件状态与适用范围。
- [`reports/README.md`](reports/README.md)：实验报告分类入口。
- [`results/research_summary/README.md`](results/research_summary/README.md)：结论级实验索引。
- [`scripts/README.md`](scripts/README.md)：正式、论文证据、覆盖扩展和历史 launcher 分类。
- [`paper/ieee_alpha_stalled/`](paper/ieee_alpha_stalled/)：当前论文工作区。

## 锁定结果

指标是生成器级 pairwise-balanced AUC/real-positive AP。数据集内先对生成器
等权平均，再对三个数据集等权得到 Macro-3。

| 数据集 | 评测视频 | 生成器 | AUC | AP_real |
|---|---:|---:|---:|---:|
| ComGenVid | 4,298 | 2 | 0.896818 | 0.906412 |
| VideoFeedback | 3,500 | 10 | 0.859718 | 0.868912 |
| GenVideo | 13,623 | 8 | 0.865682 | 0.841574 |
| **Macro-3** | **21,421** | **20** | **0.874072** | **0.872299** |

相对固定交集上的 Original STALL `0.8388/0.8428`，Macro AUC/AP 提高
`+0.0353/+0.0295`。相对完全相同核心的统一 K=1 `0.8632/0.8636`，K=3
提高 `+0.0109/+0.0087`。

注意：原 STALL 论文 Table 1 使用生成视频为 AP 正类。项目中必须显式写成
`AP_fake` 或 `AP_real`；两种 AP 不可直接混用。`Macro-3` 也不等于 23 个
生成器直接等权的 `All-23`。

## Release 资产

完整角色和文件索引见 `release/README.md`。`release/u0/` 是当前 strict-20 主发布；
`release/u0_external_genvidbench/` 是外部确认，不能与主结果合并解释。

| 路径 | 作用 |
|---|---|
| `configs/alpha_stalled_u0_locked.yaml` | 锁定协议、参数、哈希和预期指标 |
| `release/u0/calibration_manifest.json` | 600 条真实校准视频 |
| `release/u0/evaluation_manifest.json` | 21,421 条评测视频 |
| `release/u0/frame_indices.json` | 每个视频的锁定窗口和帧索引 |
| `release/u0/params/` | 三个数据集的 region1/mean Local 参数 |
| `release/u0/final_video_scores.csv` | 权威逐视频最终分数 |
| `release/u0/reproduction_metadata.json` | 环境、原始 shard 和指标元数据 |
| `release/u0/config_and_checkpoint_hashes.json` | 配置、模型、参数和输出 SHA-256 |
| `release/u0/validation.json` | 61 项 release 检查结果 |

## 环境

`environment.yml` 固定仓库直接依赖；`environment.lock.yml` 进一步冻结当前验证过的
Linux x86-64、CUDA 12.8 完整 Conda build 和 pip 依赖闭包。常规开发环境可用前者，
严格复现应创建独立锁定环境：

```bash
conda env create -f environment.yml
conda activate stall

conda env create -f environment.lock.yml
conda activate stall-locked-linux-64-cu128
```

以下命令只读检查直接声明、精确锁、`src/`/`tools/` 的15个外部导入归属、当前
Conda build/pip版本和 `pip check`；允许本机另装的非项目pip辅助工具，但它们不会
进入锁定复现环境：

```bash
conda run --no-capture-output -n stall \
  python tools/verify_environment_lock.py --check-current
```

从视频重新提取特征需要本地 DINOv3 ViT-L/16 代码和权重：

```text
dinov3/weights/dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth
```

具体 checkpoint、DINO commit、预处理和 SHA-256 已锁定在配置和 release
元数据中，不应根据本机现有模型自动替换。

## 验证现有 Release

以下命令会验证 manifest、身份隔离、帧索引、原始 shard、参数/输出哈希、
逐视频分数公式和最终指标：

```bash
conda run --no-capture-output -n stall \
  python tools/verify_u0_locked_release.py
```

本机保留 `results/u0_locked_reproduction/raw` 与 `calibration_raw` 时，可进一步从
58,496 个 raw 窗口只读重建全部 21,421 行最终分数，并逐列、逐值、逐 dtype 精确比较：

```bash
conda run --no-capture-output -n stall \
  python tools/verify_u0_pipeline_reconstruction.py
```

验证 README、协议文档和结果状态账本是否仍与锁定配置/release 一致：

```bash
conda run --no-capture-output -n stall \
  python tools/verify_project_documentation.py
```

验证实验 ID、协议、状态、父子关系、证据路径和 locked U0 登记：

```bash
conda run --no-capture-output -n stall \
  python tools/verify_experiment_registry.py
```

验证 `configs/` 全部资产的协议身份、生命周期和唯一当前权威配置：

```bash
conda run --no-capture-output -n stall \
  python tools/verify_config_registry.py
```

验证正式参数文件的内容哈希、NPZ结构、语义和本机sweep隔离：

```bash
conda run --no-capture-output -n stall \
  python tools/verify_parameter_assets.py
```

验证 locked U0 运行 provenance 和 28 个 input/intermediate/output artifact：

```bash
conda run --no-capture-output -n stall \
  python tools/verify_run_manifest.py
```

验证本机缓存目录是否仍与声明的 13 组生命周期、文件数和布局指纹一致：

```bash
conda run --no-capture-output -n stall \
  python tools/verify_cache_inventory.py
```

缓存库存是研究资产治理，不是 locked release 的运行依赖；命令只读扫描，任何
清理都要求人工明确批准。

轻量核心测试：

```bash
conda run --no-capture-output -n stall \
  python -m unittest tests/test_u0_locked_release.py \
                     tests/test_u0_metric_protocol.py \
                     tests/test_whitening_batch_invariance.py \
                     tests/test_experiment_registry.py \
                     tests/test_config_registry.py \
                     tests/test_parameter_assets.py \
                     tests/test_release_index.py \
                     tests/test_run_manifest.py \
                     tests/test_cache_inventory.py \
                     tests/test_project_documentation.py
```

## 从空结果目录重算 U0

正式重算分为 K=3 窗口评分、K=1 校准参考评分、视频级分析和 release 验证。
两个 shard 可分别分配到两张 GPU；输出目录必须是新目录，不能覆盖
`release/u0/`。

```bash
mkdir -p results/u0_reproduction_new

bash scripts/run_u0_locked_shard.sh 0 0 results/u0_reproduction_new
bash scripts/run_u0_locked_shard.sh 1 1 results/u0_reproduction_new

bash scripts/run_u0_locked_calibration_shard.sh 0 0 \
  results/u0_reproduction_new
bash scripts/run_u0_locked_calibration_shard.sh 1 1 \
  results/u0_reproduction_new

conda run --no-capture-output -n stall \
  python tools/analyze_u0_locked.py \
  --raw-dir results/u0_reproduction_new/raw \
  --calibration-raw-dir results/u0_reproduction_new/calibration_raw \
  --num-shards 2 \
  --output-dir results/u0_reproduction_new/analysis \
  --release-dir results/u0_reproduction_new/release
```

历史 K1 `paper_scores` 重建流程仍保留用于实验考古，但不能用于重建当前
locked U0。相关旧资产和命令见 `results/README.md` 与
`scripts/reproduce/README.md`，使用时必须同时注明协议状态。

## 代码结构

| 路径 | 作用 |
|---|---|
| `src/stall.py` | 原版STALL检测器、DINO Global提取与历史API兼容实现 |
| `src/alpha_stalled/backbone.py` | DINO路径、预处理、模型加载和Global/Patch共享进程缓存 |
| `src/alpha_stalled/global_branch.py` | 正式Global Spatial/T1 raw评分和固定0.5/0.5融合 |
| `src/stall_patch.py` | 同一次 DINO forward 提取 global/patch token |
| `src/alpha_stalled/whitening.py` | float64 Gaussian评分、D1/D2和右闭合ECDF的共享唯一实现 |
| `src/stable_whitening.py` | 历史实验与外部调用的兼容转发层 |
| `src/alpha_stalled/local_branch.py` | 正式region1/mean Local D1/D2、raw评分和固定beta融合 |
| `src/alpha_stalled/sampling.py` | K1/K3/K5/non-overlap共享确定性采样原语 |
| `src/alpha_stalled/release_io.py` | video ID、U0稳定分片、路径、SHA-256和原子JSON写入 |
| `src/alpha_stalled/video_io.py` | 全量/索引/锁定窗口解码；strict缺帧失败与legacy兼容边界 |
| `src/alpha_stalled/calibration.py` | U0窗口四分量CDF及Global/Local固定融合公式 |
| `src/alpha_stalled/u0_protocol.py` | locked manifests、raw shards、K1校准引用和effective-K协议校验 |
| `src/alpha_stalled/u0_scoring.py` | locked去重解码、逐视频DINO batching和四分量raw scoring |
| `src/alpha_stalled/u0_analysis.py` | 窗口四分量校准、视频均值与effective-K视频CDF |
| `src/alpha_stalled/aggregation.py` | effective-K窗口选择与逐视频等权均值 |
| `src/alpha_stalled/artifacts.py` | expected shard、checkpoint part与resume ID读取 |
| `src/alpha_stalled/parameters.py` | 冻结Global/Local Gaussian raw参数加载 |
| `src/alpha_stalled/experiment_registry.py` | 实验身份、状态、lineage和证据路径校验 |
| `src/alpha_stalled/run_manifest.py` | 运行 provenance、artifact bytes/SHA-256 契约 |
| `src/alpha_stalled/run_capture.py` | 新实验 exact argv、Git状态、时间和退出码采集 |
| `src/alpha_stalled/cache_inventory.py` | 缓存生命周期、覆盖率和只读布局审计 |
| `src/alpha_stalled/cache_contract.py` | 新特征缓存的encoder/root/逐文件身份契约 |
| `src/alpha_stalled/data_catalog.py` | canonical数据身份、来源覆盖和locked release成员关系审计 |
| `src/alpha_stalled/parameter_assets.py` | Global/Local参数哈希、NPZ结构、语义与生命周期治理 |
| `src/alpha_stalled/tool_dependencies.py` | tools AST依赖图、隔离入口、许可边和循环检查 |
| `src/alpha_stalled/environment_lock.py` | 直接/精确环境、外部导入归属和当前包版本校验 |
| `src/alpha_stalled/legacy_window_scoring.py` | pre-release多窗口身份、解码、resume和评分兼容层 |
| `src/patch_matching.py` | 实验性region、matching、residual、multi-lag和D3/D4定义 |
| `src/patch_math.py` | 历史bottom-k聚合兼容 helper；不属于locked U0 |
| `src/alpha_stalled/metrics.py` | pairwise-balanced AUC/AP 的共享唯一实现 |
| `src/metrics.py` | 原版及历史脚本的兼容转发层；新代码不再导入 |
| `tools/score_u0_locked_windows.py` | 正式 U0 逐窗口 raw scoring |
| `tools/analyze_u0_locked.py` | 窗口校准、视频聚合、effective-K CDF 和指标 |
| `tools/verify_u0_locked_release.py` | 锁定 release 完整性验证 |
| `tools/verify_u0_pipeline_reconstruction.py` | 从retained raw shards精确重建并比较最终分数 |
| `tools/verify_experiment_registry.py` | 机器可读实验注册表与 locked U0 一致性验证 |
| `tools/build_run_manifest.py` / `tools/verify_run_manifest.py` | 构建和验证运行清单 |
| `tools/capture_experiment_run.py` | 在新目录运行命令并原生记录 captured provenance |
| `tools/verify_run_capture.py` | 独立验证 capture 身份、路径、时间和退出状态 |
| `tools/build_cache_inventory.py` / `tools/verify_cache_inventory.py` | 构建和验证缓存库存；不执行删除 |
| `tools/build_data_catalog.py` / `tools/verify_data_catalog.py` | 构建和验证canonical index与release身份总账 |
| `tools/verify_config_registry.py` | 验证全部配置资产身份、生命周期和唯一当前权威配置 |
| `tools/verify_parameter_assets.py` | 验证正式参数哈希、NPZ契约及本机sweep边界 |
| `tools/verify_feature_cache_contract.py` | 验证strict cache contract与逐文件sidecar |
| `tools/build_tool_dependency_inventory.py` / `tools/verify_tool_dependencies.py` | 构建和验证CLI内部依赖政策 |
| `tools/verify_environment_lock.py` | 验证精确环境锁及当前Conda/pip环境漂移 |
| `reports/` | 实验报告和协议审计 |
| `results/research_summary/` | 可以进入结论的轻量机器可读索引 |
| `research_archive/` | 历史探索，不属于默认检测器 |

`datasets/`、`cache/`、DINO 权重和大体量逐窗口结果不进入 Git。当前缓存声明见
`configs/cache_inventory.yaml`，实测快照和可读决策分别见
`results/research_summary/cache_inventory.json` 与 `reports/cache_inventory.md`。
三个 canonical index、60,949 个数据身份及 locked U0 成员覆盖由
`configs/data_catalog.yaml` 声明，机器快照和可读来源表分别见
`results/research_summary/data_catalog.json` 与 `reports/data_catalog.md`；使用
`tools/verify_data_catalog.py` 检查 index、release manifests 和本机文件覆盖。
`configs/config_registry.yaml` 完整登记 `configs/` 下的13个非README资产，并将
`alpha_stalled_u0_locked.yaml` 锁定为唯一当前权威配置；历史、扩展、模板和治理配置
不能仅靠文件名表达状态，使用 `tools/verify_config_registry.py` 验证。
`configs/parameter_assets.yaml` 进一步内容寻址8个已提交参数文件：当前U0使用4个、
外部确认使用1个、历史兼容保留3个。其数组形状、calibration count和aggregation
语义由 `tools/verify_parameter_assets.py` 检查；其他本机patch sweep仍是ignored资产。
`configs/tool_dependencies.yaml` 则登记剩余历史 `tools -> tools` 依赖；当前 AST
快照覆盖 126 个 Python 工具、8 条显式许可边、4 个 retained family 和 50 个零内部
依赖入口。每个 retained family 都登记工具端点、内容哈希证据、归档目标和移动门槛，详见
`results/research_summary/tool_dependency_inventory.json` 与
`reports/tool_dependency_inventory.md`。新增工具间导入、符号变化或循环都会被
`tools/verify_tool_dependencies.py` 拒绝。
全部126个入口还被唯一分类为：10个`formal_release`、45个`paper_evidence`、
16个`governance`、38个`historical_frozen`、1个`research_utility`和16个
`compatibility`。正式发布与治理入口禁止导入其他CLI；跨生命周期依赖同样会失败。
对1个research utility和16个compatibility入口还执行确定性引用扫描：14个历史工具
的实现已移动到`research_archive/tools/journal_experiments/`
或`research_archive/tools/pre_release_assets/`，原命令由`compatibility_wrapper`
保留；其余3个为`retain_referenced`。清单同时记录移动前
实现、当前包装器、归档实现及结果证据哈希。
历史 `score_multi_window.py` 已经是薄运行入口；其五列身份、窗口 JSON、重试解码、
checkpoint key 和旧 calibrated scorer 统一由
`src/alpha_stalled/legacy_window_scoring.py` 定义。该模块只用于复现 pre-release
multi-window 实验，不属于 `u0_locked_v1`。
现有特征缓存仍是 legacy；新缓存的严格身份规则和迁移边界见
`reports/feature_cache_contract.md`。新 patch whitening/CDF 参数会绑定特征缓存
contract SHA，评分器拒绝跨缓存身份复用；历史参数和历史缓存仍可在明确的 legacy
模式下复现旧实验。
新增实验应先保存在独立输出目录，确认协议与结论后再将轻量指标和决策摘要登记
到研究索引。

新实验不得直接 `mkdir` 后裸跑。先用 `tools/capture_experiment_run.py` 写入
`results/runs/<experiment_id>/run_capture.json`，完成后再从 capture 构建
run-local manifest；具体流程见 `scripts/experiments/README.md` 与
`configs/run_manifests/README.md`。

## 原版 STALL

原版 Global-only 入口仍为：

```bash
conda run --no-capture-output -n stall python src/eval.py \
  --real-dir /path/to/real \
  --fake-dir /path/to/fake
```

原版 Global 参数为 `precomputed/stall_params_vatex_dino_v3.npz`。原版 STALL
固定随机窗口与当前 K=3 均匀窗口不是同一采样协议，比较时应使用项目中已经
构建的同身份 factorial 审计，而不是直接拼接不同论文表格。

## 引用

```bibtex
@inproceedings{hayun2026trainingfreedetectiongeneratedvideos,
  title     = {Training-free Detection of Generated Videos via Spatial-Temporal Likelihoods},
  author    = {{Ben Hayun}, Omer and Betser, Roy and Levi, Meir Yossef and Kassel, Levi and Gilboa, Guy},
  booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  year      = {2026},
  eprint    = {2603.15026},
  archivePrefix = {arXiv},
  primaryClass  = {cs.CV}
}
```
