# Alpha-STALLED release 范围

本文档把 `results/alpha_stalled_project_manuscript_zh.md` 中的方法主线转化为
可复现代码 release 的边界。

## 主检测器

公开主线为：

```text
原版 STALL 全局分支
+ patch 同网格二阶时序分支
+ 固定协议的 global/local 融合
```

检测器推理时不得使用测试批次 rank、`subset` 标签、生成器身份或来源特定路由。

## 必需实验

| 类别 | 要求 | 需要保留的证据 |
|---|---|---|
| 主实验 | ComGenVid、VideoFeedback、GenVideo 上的 zero-shot pairwise balanced AUC/AP | global、patch、fused score CSV 和 metrics CSV |
| 组件消融 | STALL only、patch second-order only、global+patch | 同一批 score 文件和 `src/metrics.py` 输出 |
| 局部时序消融 | patch spatial、lag-1、multi-lag、motion-hard、motion-soft、same-grid second-order | `results/paper_scores/` 中的消融 score |
| Alpha sweep | 固定 global/patch alpha sweep | `results/paper_sweeps/` |
| 审计边界 | persistence/fallback 等诊断上限，不属于主检测器 | `research_archive/` 和清单说明 |

## 主线代码

| 路径 | 作用 |
|---|---|
| `src/stall.py` | DINOv3 加载和原版全局 STALL 打分 |
| `src/eval.py` | 原版 STALL CLI |
| `src/metrics.py` | 唯一接受的论文指标实现 |
| `src/video_index.py` | 数据集索引和固定窗口帧选择 |
| `src/create_params.py` | 原版 STALL 校准拟合 |
| `src/stall_patch.py` | global + patch token 提取 |
| `src/dataset_utils_patch.py` | patch cache 读写辅助 |
| `src/patch_math.py` | patch 校准和测试共用的 numpy 辅助函数 |
| `src/create_patch_params.py` | patch 白化和校准拟合 |
| `src/eval_patch_fast.py` | 基于 patch cache 的同网格打分 |
| `tools/prefill_patch_cache.py` | patch cache 生成 |
| `tools/eval_alpha_stalled.py` | global/patch 融合和指标 |
| `tools/fuse_scores.py` | alpha sweep |
| `configs/alpha_stalled.yaml` | 冻结方法和评测协议 |
| `tools/verify_alpha_stalled_release.py` | release 资产一致性检查 |

## 归档出主线

以下类别保存在 `research_archive/`，不作为默认检测器：

| 类别 | 示例 | 原因 |
|---|---|---|
| Motion matching | `matching_*`, `patch_motion_*`, `patch_trajectory_*` | 主审计中未超过同网格二阶 |
| Tail/morphology | `patch_tail_*`, `morphology_*`, `hard_tail_*` | 探索信号，非冻结方法 |
| Frequency/neighbor | `patch_frequency_*`, `patch_neighbor_*` | 探索信号族 |
| PatchField | `patchfield_*`, `apply_universal_patchfield_addon.py` | 已从当前方法叙事移除 |
| Adaptive/reliability | `adaptive_fusion.py`, `reliability_fusion.py` | 稳定性不如固定 alpha |
| Fallback/routing | `apply_sample_fallback_rule.py`, selector 工具 | 存在标签/来源/rank 泄漏风险 |
| Fresh/Hotshot 脚手架 | build/verify runbook | 操作脚手架，非核心方法 |

## 数据策略

release 保留：

- 论文需要的小型结果表；
- 复现已报告分数所需的小型 `.npz` 校准文件；
- 描述 cache 位置和重建方法的清单。

release 不纳入：

- 原始视频数据集；
- DINOv3 权重；
- 逐视频 embedding cache；
- debug patch shard；
- 日志和 PID 文件。
