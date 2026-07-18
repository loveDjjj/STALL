# Patch 覆盖缺口

本文件记录若干短视频来源未纳入当前 Alpha-STALLED release baseline 的原因。

| 数据集 | 来源 | 状态 | 证据 | Release 决策 | 原因 |
|---|---|---|---|---|---|
| VideoFeedback | Hotshot-XL | 历史 local variant 未保留在 release results 中 | 清理后的 `results/` 树不保留该诊断资产 | 从当前 release baseline 排除 | 目标为 1 秒视频，但兼容性 gate 将可用 patch 分数标记为 `DURATION_MISMATCH_LOCAL_VARIANT_ONLY`，因为它使用的是 2 秒参数/校准设置。 |
| GenVideo | HotShot-XL | 缺少 patch 分数 | — | 从当前 release baseline 排除 | 当前结果树中没有 GenVideo HotShot-XL patch score 资产。 |
| GenVideo | MoonValley | 缺少 patch 分数 | — | 从当前 release baseline 排除 | 当前结果树中没有 GenVideo MoonValley patch score 资产。 |

因此，当前 release baseline 中 VideoFeedback 只报告
`results/paper_scores/videofeedback_*.csv` 覆盖的 10 个 2 秒来源；GenVideo 只
报告 `results/paper_scores/genvideo_*.csv` 覆盖的 8 个来源。
