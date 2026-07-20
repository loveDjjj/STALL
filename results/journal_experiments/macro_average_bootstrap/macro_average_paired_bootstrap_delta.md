# 生成器宏平均 paired bootstrap ΔAUC/ΔAP

本分析只读取 `results/paper_scores/`，不重新运行模型。
每个 bootstrap 轮次内，先对每个生成器分别做 balanced real/fake 重采样并计算 AUC/AP，
再对生成器取宏平均，最后报告方法差值的 95% CI。

| dataset | comparison | ΔAUC mean | ΔAUC 95% CI | ΔAP mean | ΔAP 95% CI | 判定 |
|---|---|---:|---:|---:|---:|---|
| ComGenVid | alpha_minus_global | +0.0669 | [+0.0618, +0.0714] | +0.0659 | [+0.0609, +0.0708] | AUC CI > 0 |
| ComGenVid | alpha_minus_patch | -0.0076 | [-0.0109, -0.0046] | -0.0099 | [-0.0135, -0.0066] | AUC CI < 0 |
| ComGenVid | patch_minus_global | +0.0745 | [+0.0669, +0.0815] | +0.0759 | [+0.0681, +0.0835] | AUC CI > 0 |
| VideoFeedback | alpha_minus_global | +0.0153 | [+0.0127, +0.0179] | +0.0137 | [+0.0117, +0.0157] | AUC CI > 0 |
| VideoFeedback | alpha_minus_patch | +0.0394 | [+0.0380, +0.0410] | +0.0442 | [+0.0420, +0.0466] | AUC CI > 0 |
| VideoFeedback | patch_minus_global | -0.0241 | [-0.0279, -0.0203] | -0.0305 | [-0.0344, -0.0266] | AUC CI < 0 |
| GenVideo | alpha_minus_global | +0.0344 | [+0.0266, +0.0433] | +0.0329 | [+0.0256, +0.0408] | AUC CI > 0 |
| GenVideo | alpha_minus_patch | +0.0302 | [+0.0238, +0.0356] | +0.0196 | [+0.0105, +0.0280] | AUC CI > 0 |
| GenVideo | patch_minus_global | +0.0043 | [-0.0077, +0.0177] | +0.0133 | [-0.0003, +0.0285] | AUC CI crosses 0 |

## 关键结论

- ComGenVid: Alpha-STALLED 相对 global-only ΔAUC=+0.0669 (+0.0618, +0.0714)，宏平均 AUC 提升在 95% CI 下为正。
- VideoFeedback: Alpha-STALLED 相对 global-only ΔAUC=+0.0153 (+0.0127, +0.0179)，宏平均 AUC 提升在 95% CI 下为正。
- GenVideo: Alpha-STALLED 相对 global-only ΔAUC=+0.0344 (+0.0266, +0.0433)，宏平均 AUC 提升在 95% CI 下为正。
