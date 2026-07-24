# Alpha-STALLED 研究结果统一索引

本目录把 2026-07-23 前分散在本地 `journal_experiments/` 的结论压缩为可审查的
研究状态。详细数值见 `experiment_metrics.csv`；逐视频融合网格、bootstrap 抽样
明细、prefill/split/runlist 和 smoke 输出不属于 release 资产。

## 已确认结果

| 主题 | 证据 | 当前判断 |
|---|---|---|
| Release global+patch | ComGenVid 0.9198/0.9211，VideoFeedback 0.8628/0.8750，GenVideo 0.8374/0.8283（AUC/AP） | 当前稳定默认仍为两分支 global+local patch |
| 真实校准数量 | 多数外部与原始 holdout 在 25/50 real 时 patch 接近随机，100-300 后改善 | 校准数量是必要条件，但收益不单调，代表性同样重要 |
| 可调 alpha | 最优 global 权重从 AEGIS 的 0.05 到 VideoFeedback-small 的 0.80 | 固定 0.60 是 release 默认，不是普适最优；oracle 只作上界 |
| AIGVDBench 聚合 | region3 mean 缓解 bottom-k 负迁移；三源 fixed 0.60 达到 0.7272/0.7326 | aggregation mismatch 是明确边界，不能泛称所有外部集稳定提升 |
| GenVideo raw D3-style | 2s raw D3 0.8429/0.8628；global+D3 0.8950/0.9011 | 全局二阶信息很强，必须保留并做真实校准 |
| VideoFeedback-small raw D3-style | 2s→1s raw D3 0.5245/0.5637；GenVideo 三分支权重明显负迁移 | raw D3 不能进入默认方法 |
| 简单 gate/cap | 最佳跨 GenVideo/VideoFeedback 规则只有约 0.001 worst-case AP 改善 | 不继续扩展简单 gate、clip、cap 或非线性融合 |
| Local volatility / 高阶 | 小样本局部 volatility 增益不足；D=3/4 低于 same-grid D=2 | 保留 same-grid D=2，后续只研究局部二阶残差 |
| K=3 MW2 | ComGenVid 0.8857/0.8996，VideoFeedback 0.8623/0.8684，GenVideo 0.8601/0.8410，Macro-3 0.8694/0.8697 | 当前无泄漏冻结基线；相对 clean single-window AP +0.0096 |
| K=5 / all-window | Macro AP 0.8672/0.8695；成本高于 K=3 | 不进入默认方法 |
| Joint Typicality | 最好 J3 Macro AP 0.8602，低于固定线性融合 0.8697 | 全部拒绝，保留 0.6/0.4 |
| Spatial-mean residual D2 | K=3 Macro 0.8626/0.8609，AP 相对 R0 -0.0088，4/20 生成器不下降 | 拒绝；停止共同运动 residual 路线 |
| 统一 region 多尺度 | 最佳 MS1 Macro 0.8687/0.8695，AP 相对基线 -0.0001，CI 跨 0 | MS1-MS4 全部拒绝；数据集特定 region 只作为参考上界 |
| DINO 中间层 D2 | 最佳 H4（layer 17/23 等权）Macro 0.8721/0.8698，AUC +0.0028、AP 仅 +0.0002 | 9/20 生成器不下降且两个数据集 AP 约 -0.005；拒绝，保留 final layer |

## 方法决策

当前主线收敛为两分支 K=3 MW2：

```text
Global = mean of calibrated STALL window scores
+ Local = mean of calibrated spatial + same-grid D2 window scores
```

Global 分支只保留 STALL 空间和一阶时序；Local 分支保留 patch 空间与 same-grid
D=2。每视频取三个均匀 2 秒窗口，分支分别均值后按 effective-K 用独立真实视频
重校准。全局 D3、raw volatility、motion hard/soft、更高阶、多 lag、简单 gate/cap
和 Joint Typicality 只保留为失败消融结论。

## 协议边界

- 现有结果混合 1s、2s、2s→1s fallback、不同真实校准规模和不同 fake 子集。
- DINOv3 operator-controlled 结果不等同于 D3 官方 XCLIP-B/16 结果。
- 所有 best/oracle 权重都是诊断上界；主结果必须使用固定或独立验证选择。
- 后续 B0-B8 必须在同一视频 ID、窗口、FPS、cache、calibration 和缺失规则下重跑。

## 保留的实现入口

核心评测与 D3 协议工具保留在 `tools/`：

- `eval_global_second_order_volatility.py`
- `prepare_genvideo_d3_exact_protocol.py`
- `extract_d3_frames_from_runlist.py`
- `eval_d3_exact_from_frames.py`
- `run_d3_exact_genvideo_batch.py`
- `summarize_d3_exact_genvideo_metrics.py`
- `audit_d3_exact_readiness.py`
- `eval_genvideo_d3_comparable_protocol.py`
- `compare_d3_smoke_with_existing_scores.py`
- `analyze_d3_window_cache_variants.py`

论文资产构建器和仍被稿件引用的验证分析入口也保留。Multi-window 的通用采样、
打分和统计入口已进入主线；与已暂停方向绑定的一次性 gating、nonlinear、
third-branch 和外部下载/扫描脚本不再进入提交。
