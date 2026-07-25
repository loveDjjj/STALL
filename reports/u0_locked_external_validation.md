# Locked U0 external GenVidBench validation

The method and manifests were locked before scoring. Local whitening and all CDFs use 199 disjoint VRIPT real calibration videos; no generated video enters fitting, calibration, or parameter selection. Evaluation contains 300 disjoint real, 300 MS, and 300 Pika videos.

| Method | Pairwise AUC | Real-positive AP | Fake-positive AP |
|---|---:|---:|---:|
| Original STALL | 0.7921 | 0.8043 | 0.7663 |
| Clean single-window | 0.8270 | 0.8369 | 0.8021 |
| Locked U0 K3 | 0.8410 | 0.8495 | 0.8203 |

## By generator

| Generator | Method | AUC/AP |
|---|---|---:|
| ms | Original STALL | 0.8106/0.8239 |
| ms | Clean single-window | 0.8391/0.8420 |
| ms | Locked U0 K3 | 0.8384/0.8456 |
| pika | Original STALL | 0.7736/0.7847 |
| pika | Clean single-window | 0.8150/0.8317 |
| pika | Locked U0 K3 | 0.8436/0.8535 |

## Paired bootstrap

| Comparison | Metric | Delta 95% CI |
|---|---|---:|
| locked_u0_minus_clean_k1 | ap | +0.0126 [+0.0017, +0.0247] |
| locked_u0_minus_clean_k1 | auc | +0.0139 [-0.0007, +0.0294] |
| locked_u0_minus_original_stall | ap | +0.0460 [+0.0297, +0.0634] |
| locked_u0_minus_original_stall | auc | +0.0488 [+0.0310, +0.0672] |

## Protocol notes

- Evaluation effective-K distribution: `{1: 303, 2: 2, 3: 595}`.
- Text2Video-Zero is excluded: its native 4 FPS videos cannot provide 16 distinct frames in two seconds under the locked 8 FPS protocol.
- Duration groups, calibration-real motion-tertile groups, per-video scores, and failure cases are retained as machine-readable CSV files.
