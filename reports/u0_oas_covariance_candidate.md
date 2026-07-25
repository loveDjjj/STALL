# U0 OAS covariance candidate

S1 changes only the Local D2 covariance estimator from the locked effective-rank whitening to parameter-free OAS shrinkage. PatchSpatial, features, K=3, CDFs, alpha, beta, and all calibration videos remain fixed.

## Locked-split result

| Configuration | ComGenVid | VideoFeedback | GenVideo | Macro-3 |
|---|---:|---:|---:|---:|
| S0 | 0.8968/0.9064 | 0.8597/0.8689 | 0.8657/0.8416 | 0.8741/0.8723 |
| S1 | 0.8969/0.9065 | 0.8599/0.8691 | 0.8656/0.8415 | 0.8741/0.8723 |

## Independent-reserve seed stability

| Seed | S0 Macro AUC/AP | S1 Macro AUC/AP | Delta AP |
|---:|---:|---:|---:|
| 17 | 0.8695/0.8662 | 0.8696/0.8661 | -0.0001 |
| 29 | 0.8718/0.8684 | 0.8717/0.8683 | -0.0001 |
| 43 | 0.8742/0.8705 | 0.8743/0.8705 | +0.0001 |
| 71 | 0.8730/0.8695 | 0.8730/0.8696 | +0.0000 |
| 101 | 0.8675/0.8652 | 0.8676/0.8652 | +0.0000 |

S0/S1 five-seed Macro AP standard deviations are `0.002210/0.002245`.

## Covariance and numerics

Locked-split OAS shrinkage range is `0.00385515` to `0.00554455`; all OAS models have effective rank 1024. Maximum observed condition number is `2.99e+03`.
Batch 1/4/8/16 maximum raw error is `0` with `0` rank changes.

## Admission status before external validation

- Macro AP delta: `+0.000043` (required >= +0.003).
- Dataset AP deltas: comgenvid `+0.000046`, videofeedback `+0.000157`, genvideo `-0.000074` (each required >= -0.003).
- Generator AP non-decline: `13/20` (required >= 12/20).
- Seed AP std S0/S1: `0.002210/0.002245` (must not increase).
- The 1,000-iteration paired cluster-bootstrap gate is appended after scoring; external validation is required only if all internal gates pass.

## Paired cluster bootstrap

| Metric | Delta | 95% CI |
|---|---:|---:|
| auc | +0.000047 | [-0.000107, +0.000193] |
| fake_positive_ap | -0.000078 | [-0.000341, +0.000141] |
| real_positive_ap | +0.000043 | [-0.000089, +0.000174] |

## Internal decision

OAS is rejected by the predeclared internal gates; S0 remains the locked method and no further covariance search is allowed.
