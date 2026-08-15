# U0 Locked Release

本目录对应 `u0_locked_v1`。核心协议是统一 `region1/mean`、K=3、2秒窗口、16帧、
8 FPS、effective-K 独立真实重校准和固定 `alpha=0.6`、`beta=0.1`。当前正式
Macro-3 AUC/AP 为 `0.8740724396350226/0.8722992273320395`。

## 核心锁定资产

| 资产 | 作用 |
|---|---|
| `calibration_manifest.json` | 每个数据集200个独立真实校准视频 |
| `evaluation_manifest.json` | 21,421个锁定评测视频 |
| `frame_indices.json` | 58,496个评测窗口及600个K1校准参考的帧索引 |
| `params/comgenvid_region1_mean.npz` | ComGenVid Local参数 |
| `params/videofeedback_region1_mean.npz` | VideoFeedback Local参数 |
| `params/genvideo_region1_mean.npz` | GenVideo Local参数 |
| `final_video_scores.csv` | 21,421行锁定逐视频分数 |
| `reproduction_metadata.json` | 环境、raw shard哈希和重建元数据 |
| `config_and_checkpoint_hashes.json` | 配置、checkpoint和输入资产完整性索引 |
| `validation.json` | 61项发布验证的冻结结果 |
| `missing_videos.json` | 锁定清单中的缺失/排除记录 |

核心验证：

```bash
conda run --no-capture-output -n stall \
  python tools/verify_u0_locked_release.py

conda run --no-capture-output -n stall \
  python tools/verify_u0_pipeline_reconstruction.py
```

## 后续补充资产

这些文件依附于 U0，但不改变主方法、主指标或核心 release 哈希。其结论必须通过
实验注册表和对应报告解释，不能因文件位于 `u0/` 就视为主结果的一部分。

| 资产 | 角色 |
|---|---|
| `calibration_reserve_manifest.json` | 未进入locked calibration/evaluation的真实候选池 |
| `calibration_split_membership.csv` | calibration、evaluation和reserve成员关系审计 |
| `cross_calibration_banks.json` | 跨数据集真实校准诊断所用bank |
| `independent_remaining_real_manifest.json` | 严格独立真实校准复测的剩余真实划分 |
| `injection_plan.json` | 局部注入诊断计划 |
| `injection_subset_manifest.json` | 注入诊断的冻结子集 |
| `robustness_perturbation_plan.json` | 编码、resize和帧扰动定义 |
| `robustness_subset_manifest.json` | 鲁棒性实验冻结子集 |

核心文件不可原地改写。新主协议必须使用新的 `release/<protocol_id>/` 目录；补充
实验则应保留自己的 experiment ID、结果目录和报告，不得覆盖本目录已有文件。
