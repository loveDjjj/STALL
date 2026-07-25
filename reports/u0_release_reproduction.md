# Alpha-STALLED U0 locked release reproduction

Date: 2026-07-24

## Authoritative result

The fully unified U0 was recomputed from source videos in an initially empty
result directory. It does not read historical final-score files. Both Local
components use final DINOv3 layer 23, region 1, and mean aggregation.

| Dataset | Evaluation videos | Generators | AUC | Real-positive AP |
|---|---:|---:|---:|---:|
| ComGenVid | 4,298 | 2 | 0.896818 | 0.906412 |
| VideoFeedback | 3,500 | 10 | 0.859718 | 0.868912 |
| GenVideo | 13,623 | 8 | 0.865682 | 0.841574 |
| **Macro-3** | **21,421** | **20** | **0.874072** | **0.872299** |

This is the authoritative locked metric. Relative to original STALL it is
`+0.0353 AUC / +0.0295 AP`; relative to the clean single-window detector it is
`+0.0171 / +0.0123`; relative to the historical dataset-specific K3 comparator
it is `+0.0047 / +0.0026`.

## Locked computation

```text
G_k = 0.5 GlobalSpatial_k + 0.5 GlobalT1_k
L_k = 0.1 PatchSpatial_k + 0.9 PatchD2_k
G_raw, L_raw = mean over up to three deterministic 2-second windows
G, L = effective-K-matched real-video ECDFs
S = 0.6 G + 0.4 L
```

PatchD2 is `P[t+2]-2P[t+1]+P[t]`, followed by feature-axis L2 normalization,
real-only centering/whitening, Gaussian likelihood, and mean aggregation. The
score stage and CDF use float64; CDF sorting is stable mergesort and ties are
right-inclusive. Each dataset contributes 200 disjoint real calibration videos
and zero generated calibration videos.

Local first-level ECDFs use 600 separately scored locked K1-current windows,
one per calibration video. These are not substituted with a K3 window: only
74/600 K1 references coincide with any K3 window. Video-level ECDFs then use
one K3 aggregate per eligible calibration video, matched to evaluation
`effective_K`.

## Reproduction audit

- Manifests contain 600 calibration and 21,421 evaluation videos with zero overlap.
- K3 traversal contains 22,021 videos and 58,496 windows; all frame-index lists match the lock.
- K1 traversal contains exactly 600 real reference windows with `effective_K=1`.
- Missing videos, duplicate keys, decode failures, and non-finite required raw scores: zero.
- Raw artifacts: 18 MB K3 shards, 220 KB K1 shards, and 27 MB checkpoints.
- Dual-GPU K3 traversal took approximately 52 minutes wall time; K1 references took under one minute.
- Several CUDA scorer processes exited with code 139 only after all checkpoints were written. A Torch-free
  finalizer independently verified IDs, counts, and frame indices before producing those shards.
- `tools/verify_u0_locked_release.py` passed all 61 release checks.
- The repository test suite passed 85 tests after the reproduction.

Release score SHA-256 before the configuration-metadata refresh:
`3f4e2501bcca42cf1f38a1f03583950f37f650afc11affb013b42f131a98b839`.
The final hash is recorded in `release/u0/config_and_checkpoint_hashes.json`.

## Difference from pre-release U0

The pre-release temporal-unified result was `0.872499/0.872243`. It replaced
PatchD2 with region1+mean but inherited dataset-specific PatchSpatial scores.
The locked run recomputes both PatchSpatial and PatchD2 with region1+mean. The
resulting Macro change is `+0.001573 AUC / +0.000056 AP`: AP remains within the
declared `0.0002` tolerance, while the larger AUC change is fully explained by
the protocol correction rather than whitening or batch drift. Dataset AUC/AP
changes are ComGenVid `-0.001820/-0.002811`, VideoFeedback
`-0.002539/+0.000501`, and GenVideo `+0.009079/+0.002479`.

## Remaining package work

The U0 structure and weights are now closed. Metric-orientation audit,
calibration sensitivity, cross-dataset real calibration, OAS admission,
robustness/localization, and locked external validation may measure U0 but may
not change region, aggregation, layer, alpha, beta, K, or sampling.
