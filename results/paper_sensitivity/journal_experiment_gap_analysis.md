# 期刊版补充实验与可视化分析

本报告基于现有 `results/paper_scores/` 和 `results/paper_sweeps/` 生成，
不重新提取 DINOv3 特征，不使用测试批次 rank、生成器标签路由或真实/生成标签参与推理。

## 已有基础消融是否足够

当前基础版消融已经覆盖：

- 三数据集 global-only、patch-only、Alpha-STALLED 组件消融；
- ComGenVid 上 patch 空间、lag-1、multi-lag、motion-hard、motion-soft、同网格二阶时序消融；
- 三数据集 global/patch 融合权重 alpha sweep。

因此，不建议重复做同类基础消融。期刊版更应该补强以下证据：

1. 超参数边界：alpha、patch 内部 beta、patch region、bottom-k、temporal run length；
2. 稳定性：逐生成器增益/退化、跨数据集 best-alpha 漂移、随机采样置信区间；
3. 失败模式：短视频覆盖缺口、patch 分支相对 global 的负迁移来源；
4. 可解释可视化：分数分布、每生成器 delta heatmap、超参数曲线、代表性 patch anomaly map。

## 本次新增的无重算特征分析

### patch-only beta sweep

| 数据集 | Best beta | Best AUC / AP | beta=0.10 AUC / AP |
|---|---:|---:|---:|
| ComGenVid | 0.10 | 0.9273 / 0.9309 | 0.9273 / 0.9309 |
| VideoFeedback | 0.15 | 0.8231 / 0.8319 | 0.8228 / 0.8302 |
| GenVideo | 0.25 | 0.8082 / 0.8138 | 0.8072 / 0.8092 |

### fixed-alpha fusion beta sweep

| 数据集 | Best beta | Best AUC / AP | beta=0.10 AUC / AP |
|---|---:|---:|---:|
| ComGenVid | 0.00 | 0.9205 / 0.9220 | 0.9198 / 0.9211 |
| VideoFeedback | 0.10 | 0.8628 / 0.8750 | 0.8628 / 0.8750 |
| GenVideo | 0.30 | 0.8396 / 0.8311 | 0.8374 / 0.8283 |

## 逐生成器增益/退化结论

下表列出 Alpha-STALLED 相对 global-only 的 AUC 变化范围。大于 0 表示融合提升，
小于 0 表示 patch 分支对该生成器产生负迁移。

| 数据集 | 最小 ΔAUC | 最大 ΔAUC | 平均 ΔAUC | 负迁移生成器数 |
|---|---:|---:|---:|---:|
| ComGenVid | +0.0638 | +0.0695 | +0.0666 | 0 |
| VideoFeedback | -0.0749 | +0.0991 | +0.0153 | 3 |
| GenVideo | +0.0078 | +0.1266 | +0.0413 | 0 |

## Bootstrap 置信区间

本次新增 `n_boot` 次逐生成器 bootstrap，分别对 global-only、patch-only 和
Alpha-STALLED 的 AUC/AP 估计 95% CI；同时使用同一次重采样计算 paired ΔAUC/ΔAP CI。
该分析不改变主指标，只用于判断提升是否稳定。

| 数据集 | Alpha-STALLED AUC 95% CI 中位宽度 | Global-only AUC 95% CI 中位宽度 |
|---|---:|---:|
| ComGenVid | 0.0174 | 0.0244 |
| VideoFeedback | 0.0185 | 0.0182 |
| GenVideo | 0.0433 | 0.0487 |

Paired bootstrap 下，Alpha-STALLED 相对 global-only 的 ΔAUC 概况：

| 数据集 | ΔAUC 均值范围 | 95% CI 完全大于 0 的生成器数 | 95% CI 完全小于 0 的生成器数 |
|---|---:|---:|---:|
| ComGenVid | +0.0638 to +0.0692 | 2 | 0 |
| VideoFeedback | -0.0709 to +0.0924 | 6 | 2 |
| GenVideo | +0.0091 to +0.0839 | 5 | 0 |

## 失败样本候选

已生成 `failure_case_candidates.csv`，用于后续做代表性视频或 patch anomaly map。
这些候选只用于事后审计，不参与模型推理。

| 类别 | 样本数 | 用途 |
|---|---:|---|
| `generated_alpha_still_high` | 60 | Alpha-STALLED 仍最难识别的生成视频。 |
| `generated_patch_global_conflict` | 60 | patch 分支显著高于 global，适合检查局部证据是否误导融合。 |
| `generated_score_increased_by_alpha` | 60 | 生成视频被 Alpha-STALLED 打得更像真实，可能削弱检测。 |
| `real_alpha_low` | 60 | Alpha-STALLED 最容易误伤的真实视频。 |
| `real_score_decreased_by_alpha` | 60 | 真实视频被 Alpha-STALLED 打得更像生成，可能削弱真实召回。 |

## 生成的图和表

- `results/paper_sensitivity/beta_sensitivity_summary.csv`：beta 平均指标曲线；
- `results/paper_sensitivity/beta_sensitivity_per_model.csv`：beta 逐生成器指标；
- `results/paper_sensitivity/component_per_generator_delta.csv`：global、patch、Alpha-STALLED 逐生成器差值；
- `results/paper_sensitivity/alpha_score_distribution_summary.csv`：Alpha-STALLED 分数分布统计；
- `results/paper_sensitivity/journal_experiment_runbook.md`：后续实跑补充实验命令模板；
- `results/paper_sensitivity/bootstrap_ci_summary.csv`：逐生成器 AUC/AP bootstrap 置信区间；
- `results/paper_sensitivity/paired_bootstrap_delta_summary.csv`：方法差异的 paired bootstrap ΔAUC/ΔAP 置信区间；
- `results/paper_sensitivity/failure_case_candidates.csv`：后续案例可视化和失败样本审计候选；
- `results/paper_sensitivity/failure_case_candidates_summary.md`：失败/边界案例候选摘要；
- `results/paper_figures/alpha_sensitivity_curves.svg`：alpha 敏感性曲线；
- `results/paper_figures/beta_sensitivity_patch_only.svg`：patch-only beta 敏感性；
- `results/paper_figures/beta_sensitivity_fused_alpha0p60.svg`：固定 alpha 下 beta 敏感性；
- `results/paper_figures/per_generator_auc_delta_heatmap.svg`：逐生成器 AUC delta heatmap；
- `results/paper_figures/alpha_score_distribution_panel.svg`：真实/生成分数分布；
- `results/paper_figures/bootstrap_alpha_minus_global_auc_ci.svg`：Alpha-STALLED 相对 global-only 的 paired bootstrap AUC 增益区间；
- `results/paper_figures/comgenvid_bottomk_sensitivity.svg`：ComGenVid bottom-k 聚合比例敏感性曲线；
- `results/paper_figures/comgenvid_region_sensitivity.svg`：ComGenVid patch region size 敏感性曲线；
- `results/paper_figures/genvideo_region_sensitivity.svg`：GenVideo patch region size 敏感性曲线；
- `results/paper_figures/videofeedback_region_sensitivity.svg`：VideoFeedback patch region size 敏感性曲线；
- `results/paper_figures/genvideo_aggregation_sensitivity.svg`：GenVideo mean 与 bottom-k aggregation 敏感性曲线；
- `results/paper_figures/videofeedback_aggregation_sensitivity.svg`：VideoFeedback aggregation 基线点，待补 bottom-k 对照。

## 已完成的实跑补充实验

本轮完成了 ComGenVid / region=3 / bottom-k 敏感性实跑。该实验只改变
patch 分支的低分局部证据聚合比例，用于说明主配置附近的超参数边界，
不改变 release 默认配置。

| 数据集 | region | bottom-k | 平均 AUC / AP | 状态 |
|---|---:|---:|---:|---|
| ComGenVid | 3 | 0.10 | 0.9245 / 0.9285 | journal_full_eval |
| ComGenVid | 3 | 0.15 | 0.9262 / 0.9300 | journal_full_eval |
| ComGenVid | 3 | 0.20 | 0.9273 / 0.9309 | journal_full_eval |
| ComGenVid | 3 | 0.25 | 0.9282 / 0.9316 | journal_full_eval |
| ComGenVid | 3 | 0.30 | 0.9292 / 0.9323 | journal_full_eval |
| ComGenVid | 3 | 0.35 | 0.9298 / 0.9327 | journal_full_eval |
| ComGenVid | 3 | 0.40 | 0.9304 / 0.9330 | journal_full_eval |
| ComGenVid | 3 | 0.50 | 0.9312 / 0.9334 | journal_full_eval |

当前 8 个完成点呈单调上升趋势，bottom-k=0.50 取得最高平均 AUC/AP。
该结果说明 ComGenVid 上的 patch 分支并非只依赖极少数最低分局部片段；
扩大低分区域聚合范围仍能保留检测收益。论文表述中应将其作为敏感性证据，
而不是事后重选主配置。

同时完成了 ComGenVid / mean aggregation / same-grid second-order 的
region size 敏感性实跑：

| 数据集 | region | aggregation | 平均 AUC / AP | 状态 |
|---|---:|---|---:|---|
| ComGenVid | 1 | mean | 0.9088 / 0.9075 | journal_full_eval |
| ComGenVid | 2 | mean | 0.9230 / 0.9227 | journal_full_eval |
| ComGenVid | 3 | mean | 0.9299 / 0.9288 | journal_full_eval |

该结果支持 ComGenVid 使用更大的局部空间支持域。region=3 相对 region=1
平均 AUC 提升 +0.0211，相对 region=2 提升 +0.0069。与 bottom-k
sweep 对照后可以看到，region=3 mean 已接近 bottom-k 主线附近性能，但
bottom-k=0.50 仍取得更高平均 AUC/AP（0.9312 / 0.9334）。因此，
ComGenVid 的增益不是单独来自更大 region，也不是单独来自 bottom-k；
更合理的解释是，大 region 提供稳定的局部二阶时序支持域，而 bottom-k
进一步突出生成视频中更异常的低似然局部证据。

本轮还完成了 GenVideo / mean aggregation / same-grid second-order 的
region size 敏感性实跑：

| 数据集 | region | aggregation | 平均 AUC / AP | 状态 |
|---|---:|---|---:|---|
| GenVideo | 1 | mean | 0.7976 / 0.7985 | journal_full_eval |
| GenVideo | 2 | mean | 0.8072 / 0.8092 | journal_full_eval |
| GenVideo | 3 | mean | 0.7893 / 0.7909 | journal_full_eval |

当前结果支持 GenVideo 主线使用 region=2：相对 region=1，平均 AUC 提升
+0.0097；相对 region=3，平均 AUC 提升 +0.0179。该曲线说明过大的局部邻域
会稀释二阶时序差异，而过小邻域又可能缺少足够空间上下文。

进一步完成了 VideoFeedback / mean aggregation / same-grid second-order 的
region size 敏感性实跑：

| 数据集 | region | aggregation | 平均 AUC / AP | 状态 |
|---|---:|---|---:|---|
| VideoFeedback | 1 | mean | 0.8228 / 0.8302 | journal_full_eval |
| VideoFeedback | 2 | mean | 0.7941 / 0.7946 | journal_full_eval |
| VideoFeedback | 3 | mean | 0.7533 / 0.7551 | journal_full_eval |

VideoFeedback 的最优点出现在 region=1，且随着局部邻域扩大，平均 AUC/AP
持续下降。该结果与 GenVideo 上 region=2 最优的结论不同，说明 region size
反映的是数据集局部时序证据的空间尺度，而不是一个可凭经验全局固定的装饰性参数。
论文中应将 region size sweep 表述为方法边界和鲁棒性分析：Alpha-STALLED 的
创新点在于将全局 STALL 校准与 patch-level 二阶时序证据结合；region size
决定局部证据的空间支持域，过大的区域可能平滑掉小范围运动不一致，过小的区域则可能
缺少足够上下文。

此外完成了 GenVideo / region=2 / same-grid second-order 下的
aggregation 敏感性实跑。该实验将 mean aggregation 作为 region=2 主线基线，
并补充两个 bottom-k ratio 对照点：

| 数据集 | region | aggregation | bottom-k | 平均 AUC / AP | 状态 |
|---|---:|---|---:|---:|---|
| GenVideo | 2 | bottomk_mean | 0.20 | 0.7652 / 0.7571 | journal_full_eval |
| GenVideo | 2 | bottomk_mean | 0.50 | 0.7795 / 0.7746 | journal_full_eval |
| GenVideo | 2 | mean | 1.00 | 0.8072 / 0.8092 | region_mean_baseline |

结果显示两个 bottom-k 点均低于 mean baseline，说明 GenVideo 上局部二阶时序证据
更适合以整体局部网格分布的均值形式进入 patch 分支，而不是只聚焦最低分局部区域。
这与 ComGenVid 上 bottom-k=0.50 略优的趋势不同，支持将 aggregation 写作
数据集局部证据结构的边界分析，而不是把 bottom-k 作为普适默认。

## 仍建议补充的实跑实验

优先级 P0：

1. **aggregation 敏感性**：三数据集 region=1/2/3 mean 已补齐。
   ComGenVid 已完成 region=3 下 bottom-k ratio =
   0.10/0.15/0.20/0.25/0.30/0.35/0.40/0.50；
   GenVideo 已完成 region=2 下 bottom-k ratio = 0.20/0.50，结果支持 mean
   aggregation；VideoFeedback 仍需要 bottom-k 或 aggregation 对照，以证明 mean
   aggregation 选择不是偶然。
2. **patch 可解释案例图**：基于 `failure_case_candidates.csv` 选取真实/生成代表视频，回到 patch cache 或原视频绘制 patch anomaly map。

优先级 P1：

3. **duration/window 敏感性**：1s/2s/3s/4s，尤其解释 HotShot/MoonValley/Hotshot-XL 的短视频边界。
4. **cross-dataset frozen hyperparameter**：用一个数据集选出的 alpha/beta/region，在其他数据集冻结评测，区分 oracle sweep 和可泛化配置。
5. **runtime 和存储开销**：global-only、patch cache prefill、patch-only eval、fusion 的时间和 cache 规模。

优先级 P2：

6. **paired bootstrap 扩展到生成器平均指标**：当前已输出逐生成器 paired ΔAUC/ΔAP；如果手稿需要一个总体显著性结论，可进一步对生成器平均指标做 paired bootstrap。
7. **失败样本人工审计**：从候选表检查是否来自低运动、短时长、压缩伪影或真实视频域偏移。

## 图表规范

本次图遵循期刊/Nature-leaning 的基础规范：优先 SVG 矢量图，PNG 作为预览；
使用色盲友好配色；每个面板只回答一个问题；图题和轴标签直接说明变量含义；
敏感性曲线明确标出冻结超参数位置，避免把 oracle sweep 误写成主方法选择协议。
