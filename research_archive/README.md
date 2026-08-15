# 研究归档

本目录保存从 Alpha-STALLED 公开主线中移出的探索代码。保留这些文件是为了
复现实验考古和后续排查，但它们不属于 `configs/alpha_stalled.yaml` 描述的
默认无泄漏检测器。

## 目录说明

| 目录 | 内容 |
|---|---|
| `tools/routing/` | sample fallback、split/universal selector、source-aware routing 和 selector audit |
| `tools/patchfield/` | PatchField 实验和 patchfield correction sweep |
| `tools/frequency/` | patch frequency、neighbor coherence、temporal-frequency 信号 |
| `tools/morphology_tail/` | morphology、likelihood-tail、residual-tail 和 hard-tail probe |
| `tools/motion_matching/` | patch matching、motion-hard/soft 和 matching-confidence 实验 |
| `tools/adaptive_reliability/` | adaptive fusion、reliability fusion 和 persistence 变体 |
| `tools/score_exploration/` | score ensemble、three-score fusion 和 token dynamics probe |
| `tools/fresh_hotshot/` | fresh validation、demo duration 和 HotShot 操作脚手架 |
| `tools/cache_engineering/` | cache shard benchmark 和 cache 工程工具 |
| `tools/diagnostics/` | source audit、候选决策表和报告导出 |
| `tools/journal_experiments/` | 已产出结果的一次性journal诊断实现；`tools/`保留兼容命令 |
| `tools/pre_release_assets/` | U0锁定前的score-CSV release资产检查；不属于当前release |
| `src/` | 旧 patch evaluator 和 multi-aggregation 变体 |
| `scripts/` | 历史一次性实验启动脚本 |
| `docs/` | 早期实现计划和冻结的历史发布文档 |
| `docs/pre_u0_release/` | dataset-specific score-CSV 发布阶段的结构、范围、资产和复现审计 |

## 提升回主线的规则

归档文件只有同时满足以下条件，才应重新移回公开主线：

1. 支持论文主检测器或明确命名的官方消融。
2. 推理时不使用测试标签、生成器身份或测试批次 rank；若使用，必须明确标为
   oracle/诊断上限。
3. 调用 `src/metrics.py` 的统一指标实现。
4. 在 `scripts/reproduce/` 或 `scripts/ablations/` 中有公开命令。
5. 至少有 smoke test 或静态 verifier。

## 归档后的主线边界

公开 `tools/` 目录应限制为：

| 路径 | 作用 |
|---|---|
| `tools/eval_alpha_stalled.py` | Alpha-STALLED 全局/patch 融合和指标 |
| `tools/eval_score_csv.py` | 已有 score CSV 和消融的统一指标入口 |
| `tools/fuse_scores.py` | 固定 alpha sweep |
| `tools/prefill_patch_cache.py` | patch cache 生成 |
| `tools/inspect_dinov3_tokens.py` | DINOv3 token 检查工具 |
| `tools/render_manuscript_tables.js` | 手稿表格渲染 |
| `tools/summarize_metrics_average_rows.py` | 指标 Average 行汇总 |

公开 `src/` 目录保留原版 STALL 路径和当前 patch fast 路径。旧 evaluator
如 `eval_patch.py`、`eval_patch_fast_multi.py` 和 `eval_patch_multiagg_fast.py`
继续归档，除非之后被删除或提升为官方消融入口。
