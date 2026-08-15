# 报告索引

报告用于解释机器可读结果，不能单独覆盖 locked config、release manifest 或
逐视频分数。当前结果状态以 `docs/RESULT_STATUS.md` 为统一入口。

## 正式主线

| 报告 | 内容 |
|---|---|
| `u0_release_reproduction.md` | 从空目录重算、环境、hash、61项验证 |
| `u0_protocol_provenance_audit.md` | 主结果与历史协议来源 |
| `u0_metric_protocol_audit.md` | AP方向、pairwise、Macro和bootstrap |
| `u0_numerical_stability.md` | whitening、batch invariance、CDF tie和CPU/GPU数值审计 |
| `u0_core_ablation.md` | Global/Local、K1/K3和权重消融 |
| `alpha_stalled_u0_final_package.md` | U0发布包总览 |

## 校准、外部确认与部署边界

| 报告 | 内容 |
|---|---|
| `u0_calibration_size_and_seed.md` | 真实校准规模与seed稳定性 |
| `duration_aware_calibration_size_optimization.md` | 23-source扩展的real-only校准规模选择 |
| `u0_cross_dataset_calibration.md` | target/off-domain真实校准 |
| `second_order_and_independent_calibration.md` | Local D1/D2及独立校准复测 |
| `u0_locked_external_validation.md` | 锁定后GenVidBench确认 |
| `u0_robustness.md` | 编码、resize、帧扰动和4FPS |
| `u0_localization_and_injection.md` | 注入响应与定位解释边界 |

## 覆盖扩展

| 报告 | 内容 |
|---|---|
| `duration_aware_23source_full.md` | 1秒/2秒兼容的23-source结果 |
| `original_k1_full23_factorial.md` | 原窗口K1与当前K3因子分解 |
| `full_coverage_paper_protocol.md` | 全45,185 fake及固定先验AP |
| `original_stall_protocol_and_coverage_audit.md` | 原STALL数据、窗口和指标审计 |

## 失败或历史探索

| 报告 | 内容/状态 |
|---|---|
| `multi_window_joint_typicality.md` | K5/all/lower-tail/Joint；历史K3 comparator |
| `local_residual_multiscale_final.md` | residual拒绝 |
| `unified_multiscale_intermediate_layers.md` | coarse/layer融合拒绝 |
| `global_local_multi_order_final.md` | 高阶/多分支历史探索 |
| `global_local_d3_audit.md` | D3受控实验的阶段0实现与协议诊断 |
| `clean_universal_cross_layer_final.md` | pre-U0统一配置与跨层验证历史结论 |
| `local_residual_multiscale_audit.md` | residual/multiscale重复方向和实现审计 |
| `u0_oas_covariance_candidate.md` | OAS增益低于门槛 |
| `unified_multiscale_intermediate_layer_audit.md` | region与中间层实验前置审计 |

## 项目治理

| 报告 | 内容 |
|---|---|
| `cache_inventory.md` | 13组缓存的文件数、容量、生命周期、cache-key缺口和人工清理决策 |
| `data_catalog.md` | 三个canonical index的60,949个身份、来源/时长覆盖和U0成员关系 |
| `parameter_asset_inventory.md` | 8个正式/外部/历史参数的哈希、NPZ契约及本机ignored sweep边界 |
| `feature_cache_contract.md` | 新缓存root/entry身份契约、legacy边界和严格验证命令 |
| `local_branch_code_boundary.md` | 正式Local API与matching/region/residual历史实验边界 |
| `global_branch_code_boundary.md` | 正式Global Spatial/T1与D3/volatility历史实验边界 |
| `backbone_code_boundary.md` | DINO路径、预处理、共享模型缓存与strict cache身份边界 |
| `video_io_code_boundary.md` | 全量、索引与锁定窗口解码语义，以及strict缺帧失败边界 |
| `u0_protocol_code_boundary.md` | U0 scorer/analyzer/verifier依赖方向、共享指标与协议边界 |
| `tool_dependency_inventory.md` | 126个Python工具的六类生命周期、17项维护决策、8条冻结边、4个证据哈希retained family和零循环审计 |
| `score_csv_code_boundary.md` | 历史score CSV读取、固定融合、alpha sweep和pairwise指标的共享实现边界 |
| `u0_paper_readiness_audit.md` | 论文声明、证据覆盖和剩余投稿工作审计 |

缓存报告由 `configs/cache_inventory.yaml` 和实际 `cache/` 目录确定性生成。它不会
删除缓存；`P3_safe_delete_candidate` 也必须经过人工明确批准。

机器可读总注册表是 `u0_experiment_registry.csv`；结论级紧凑指标位于
`results/research_summary/experiment_metrics.csv`。注册表的 `status` 只允许使用
`docs/RESULT_STATUS.md` 定义的生命周期状态；旧 `paper_status` 仅表示论文展示角色。
每条结论级记录必须有稳定 `experiment_id`、`protocol_id`、父实验、证据路径和
拒绝/准入原因。使用以下命令验证：

```bash
conda run --no-capture-output -n stall \
  python tools/verify_experiment_registry.py
```

新增报告还必须写明数据身份、AP正类、Macro定义、参数选择是否使用生成视频，
以及准入/拒绝状态。`tools/verify_project_documentation.py` 要求 `reports/` 顶层的
每份 Markdown 报告都在本索引中出现；专题子目录应由其自己的 README 管理。
