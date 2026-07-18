# Alpha-STALLED 消融汇总

本文件所有行均由 `tools/eval_score_csv.py` 重新计算。该工具使用
`src/metrics.py` 的 pairwise balanced real-vs-generator 评测。结果报告为
`AUC / AP`，且分数越高表示越接近真实视频。

## Global/Patch 组件消融

| Benchmark | 组件 | Score CSV | 分数列 | 平均 AUC / AP |
|---|---|---|---|---:|
| ComGenVid | 仅 Global | `results/paper_scores/comgenvid_global.csv` | `final_score` | 0.8532 / 0.8553 |
| ComGenVid | 仅 Patch | `results/paper_scores/comgenvid_patch_second_order.csv` | `patch_final_score` | 0.9273 / 0.9309 |
| ComGenVid | Alpha-STALLED | `results/paper_scores/comgenvid_alpha_stalled.csv` | `final_score` | 0.9198 / 0.9211 |
| VideoFeedback | 仅 Global | `results/paper_scores/videofeedback_global.csv` | `final_score` | 0.8475 / 0.8614 |
| VideoFeedback | 仅 Patch | `results/paper_scores/videofeedback_patch_second_order.csv` | `patch_final_score` | 0.8228 / 0.8302 |
| VideoFeedback | Alpha-STALLED | `results/paper_scores/videofeedback_alpha_stalled.csv` | `final_score` | 0.8628 / 0.8750 |
| GenVideo | 仅 Global | `results/paper_scores/genvideo_global.csv` | `final_score` | 0.7961 / 0.7901 |
| GenVideo | 仅 Patch | `results/paper_scores/genvideo_patch_second_order.csv` | `patch_final_score` | 0.8072 / 0.8092 |
| GenVideo | Alpha-STALLED | `results/paper_scores/genvideo_alpha_stalled.csv` | `final_score` | 0.8374 / 0.8283 |

## ComGenVid 局部时序定义消融

| 局部证据 | Score CSV | 分数列 | 平均 AUC / AP |
|---|---|---|---:|
| 仅 patch 空间 | `results/paper_scores/comgenvid_patch_spatial.csv` | `patch_final_score` | 0.8091 / 0.8194 |
| 同网格 lag-1 | `results/paper_scores/comgenvid_patch_lag1.csv` | `patch_final_score` | 0.8556 / 0.8617 |
| Multi-lag | `results/paper_scores/comgenvid_patch_multilag.csv` | `patch_final_score` | 0.8546 / 0.8586 |
| Motion-hard | `results/paper_scores/comgenvid_patch_motionhard.csv` | `patch_final_score` | 0.8430 / 0.8491 |
| Motion-soft | `results/paper_scores/comgenvid_patch_motionsoft.csv` | `patch_final_score` | 0.8181 / 0.8256 |
| 同网格二阶 | `results/paper_scores/comgenvid_patch_second_order_ablation.csv` | `patch_final_score` | 0.9273 / 0.9309 |

## 说明

- 这些数值取代早期一次性脚本生成的表格；早期脚本没有始终使用统一的
  real/generated 平衡口径。
- Persistence、fallback 和 source/rank selector 仍为诊断上限资产，
  不纳入本消融汇总。
