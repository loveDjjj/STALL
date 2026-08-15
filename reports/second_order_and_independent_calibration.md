# Second-order temporal ablation and independent real calibration

## Purpose and locked protocol

This experiment changes only the Local temporal representation: same-grid patch D1 versus D2. Encoder, final DINOv3 layer 23, 2 s/16-frame windows, nominal 8 FPS, deterministic K=3 sampling, region1, mean aggregation, Local Spatial, Global, alpha=0.6, beta=0.1, real identities, whitening estimator, right-inclusive empirical CDF, and effective-K video calibration remain fixed.

D1 and D2 use the same fitting algorithm and the same locked 200 real videos per dataset, but each temporal representation fits its own whitening transform and K1 CDF. Reusing the D2 whitening matrix for D1 would not be a valid controlled comparison.
The D1 parameter files copy `mu_patch_spat`, `W_patch_spat`, and the PatchSpatial calibration array exactly from the locked D2 files; only temporal parameters are refit. All three copied Spatial arrays pass elementwise equality checks.

## Implementation audit

- Locked final configuration: `configs/alpha_stalled_u0_locked.yaml` (Clean Universal U0/LSTL).
- Patch D1: `src/alpha_stalled/whitening.py::l2_normalized_patch_first_order`.
- Patch D2: `src/alpha_stalled/whitening.py::l2_normalized_second_order`.
- D1 fitting: `tools/fit_u0_local_d1_params.py`.
- Recoverable scalar-only D1 scoring: `tools/score_u0_local_d1_windows.py`.
- The locked 200-real identities per dataset are declared by `release/u0/calibration_manifest.json`; their K1/K3 frame indices are frozen in `release/u0/frame_indices.json`.
- Independent calibration identities are selected by source-stratified seeded SHA-256 rank in `tools/build_u0_calibration_reserve.py` and recorded in `release/u0/calibration_split_membership.csv`.
- Local whitening is fit by `src/create_patch_params.py::build_patch_params`; K1 right-inclusive window CDF and effective-K video CDF are applied by `tools/analyze_u0_calibration_sensitivity.py::calibrate_candidate` and `tools/analyze_u0_core_ablation.py::calibrate_k3_candidate`.
- Formal D1 extraction uses the release frame batch size 32, groups unique frames per video, and disables compact K1-cache reuse for both K1 CDF references and K3 windows. A batch-size-8 pilot was rejected and is not used in any reported result.
- Analysis and split audit: `tools/analyze_second_order_independent_calibration.py`.
- Independent all-remaining-real analysis: `tools/analyze_independent_real_complement.py`.
- D2 complete-Local regression max error: `0`; LSTL final regression max error: `0`.
- No full K=3 patch-token cache was written; only parameters, per-window scalar scores, and analysis tables are retained.
- Existing patch-token caches contain historical K1 windows only; no cache matches all formal K3 frame indices and release batch-32 grouping, so formal K3 D1 is extracted from source videos.
- Resource policy: K3 patch tokens are discarded after scalar scoring. At launch `/data` had about 248 GB free. GPU 1 was occupied by an external service (about 29.5/32.6 GB), so formal DINO work used GPU 0 only; shard count changes scheduling, not the per-video batch-32 feature definition.

## D1 versus D2

A/B isolate the pure calibrated Local Temporal score (PatchD1 versus PatchD2), excluding PatchSpatial and Global. C/D are the complete model: they retain `0.1 PatchSpatial + 0.9 PatchD1/D2`, then add the unchanged calibrated Global branch with the locked 0.6/0.4 fusion.

| Variant | ComGenVid AUC/AP | VideoFeedback AUC/AP | GenVideo AUC/AP | Macro-3 AUC/AP | Macro delta AUC/AP |
|---|---:|---:|---:|---:|---:|
| Local first-order only | 0.8846/0.8848 | 0.7675/0.7712 | 0.8188/0.7952 | 0.8237/0.8171 | -- |
| Local second-order only | 0.8907/0.8908 | 0.7783/0.7870 | 0.8279/0.8066 | 0.8323/0.8281 | +0.0086/+0.0111 |
| Full model with first-order | 0.8966/0.9055 | 0.8563/0.8645 | 0.8635/0.8385 | 0.8721/0.8695 | -- |
| LSTL | 0.8968/0.9064 | 0.8597/0.8689 | 0.8657/0.8416 | 0.8741/0.8723 | +0.0019/+0.0028 |

### Per-dataset second-order deltas

| Dataset | Local D2-D1 AUC/AP | LSTL-Full D1 AUC/AP |
|---|---:|---:|
| ComGenVid | +0.0060/+0.0060 | +0.0002/+0.0009 |
| VideoFeedback | +0.0108/+0.0158 | +0.0034/+0.0045 |
| GenVideo | +0.0090/+0.0114 | +0.0022/+0.0030 |
| Macro-3 | +0.0086/+0.0111 | +0.0019/+0.0028 |

### Detailed four-variant tables

The delta columns are populated only for the corresponding second-order row; first-order rows are the reference.

#### ComGenVid

| Variant | AUC | AP | Delta AUC vs corresponding first-order | Delta AP vs corresponding first-order |
|---|---:|---:|---:|---:|
| Local first-order only | 0.8846 | 0.8848 | -- | -- |
| Local second-order only | 0.8907 | 0.8908 | +0.0060 | +0.0060 |
| Full model with first-order | 0.8966 | 0.9055 | -- | -- |
| LSTL | 0.8968 | 0.9064 | +0.0002 | +0.0009 |

#### VideoFeedback

| Variant | AUC | AP | Delta AUC vs corresponding first-order | Delta AP vs corresponding first-order |
|---|---:|---:|---:|---:|
| Local first-order only | 0.7675 | 0.7712 | -- | -- |
| Local second-order only | 0.7783 | 0.7870 | +0.0108 | +0.0158 |
| Full model with first-order | 0.8563 | 0.8645 | -- | -- |
| LSTL | 0.8597 | 0.8689 | +0.0034 | +0.0045 |

#### GenVideo

| Variant | AUC | AP | Delta AUC vs corresponding first-order | Delta AP vs corresponding first-order |
|---|---:|---:|---:|---:|
| Local first-order only | 0.8188 | 0.7952 | -- | -- |
| Local second-order only | 0.8279 | 0.8066 | +0.0090 | +0.0114 |
| Full model with first-order | 0.8635 | 0.8385 | -- | -- |
| LSTL | 0.8657 | 0.8416 | +0.0022 | +0.0030 |

#### Macro-3

| Variant | AUC | AP | Delta AUC vs corresponding first-order | Delta AP vs corresponding first-order |
|---|---:|---:|---:|---:|
| Local first-order only | 0.8237 | 0.8171 | -- | -- |
| Local second-order only | 0.8323 | 0.8281 | +0.0086 | +0.0111 |
| Full model with first-order | 0.8721 | 0.8695 | -- | -- |
| LSTL | 0.8741 | 0.8723 | +0.0019 | +0.0028 |

### Paired bootstrap

1,000 paired cluster resamples preserve D1/D2 score pairing and use video IDs, never windows, as sampling units. Following `tools/audit_u0_metric_protocol.py`, real-video cluster counts are drawn once from the dataset-level union and projected into every generator pair, while generated videos are resampled within generator.

| Comparison | Metric | Delta | 95% CI |
|---|---|---:|---:|
| local_d2_minus_d1 | ap | +0.0111 | [+0.0083, +0.0132] |
| local_d2_minus_d1 | auc | +0.0086 | [+0.0059, +0.0112] |
| lstl_minus_full_d1 | ap | +0.0028 | [+0.0019, +0.0038] |
| lstl_minus_full_d1 | auc | +0.0019 | [+0.0008, +0.0030] |

## Independent real calibration

The locked main result already uses 200 real calibration videos per dataset with zero identity overlap with evaluation. The additional experiment changes only the independent 200-real calibration bank while keeping the 21,421 evaluation identities fixed. This isolates calibration-bank variation; it is a stability test, not a repair of an observed leak.

Each dataset/seed refits Local D2 whitening from its selected 200 reserve real videos, rebuilds the K1 PatchSpatial/PatchD2 CDFs, and rebuilds effective-K video CDFs from the corresponding reserve K3 scores. The predeclared split seed is also the deterministic 300k-token reservoir seed for that fit, so the reported standard deviation measures full calibration-pipeline variation rather than identity variation alone. Global STALL parameters remain the fixed source-domain parameters declared by U0. Generated videos never enter fitting or calibration.

Primary statistics use the requested three fixed seeds 17/29/43. Seeds 71/101 are retained as an extended five-seed audit. Every `std` below is the sample standard deviation across calibration splits (`ddof=1`), not a bootstrap confidence interval; the two uncertainty analyses answer different questions.

| Dataset | 3-seed AUC mean +/- std | 3-seed AP mean +/- std | 5-seed AUC mean +/- std | 5-seed AP mean +/- std |
|---|---:|---:|---:|---:|
| ComGenVid | 0.8980 +/- 0.0019 | 0.9076 +/- 0.0015 | 0.8964 +/- 0.0038 | 0.9065 +/- 0.0027 |
| VideoFeedback | 0.8502 +/- 0.0028 | 0.8589 +/- 0.0019 | 0.8489 +/- 0.0028 | 0.8578 +/- 0.0021 |
| GenVideo | 0.8674 +/- 0.0042 | 0.8385 +/- 0.0041 | 0.8683 +/- 0.0038 | 0.8395 +/- 0.0039 |
| Macro-3 | 0.8718 +/- 0.0024 | 0.8684 +/- 0.0021 | 0.8712 +/- 0.0027 | 0.8680 +/- 0.0022 |

### Per-seed results

| Seed | ComGenVid | VideoFeedback | GenVideo | Macro-3 |
|---:|---:|---:|---:|---:|
| 17 | 0.8970/0.9065 | 0.8473/0.8570 | 0.8642/0.8352 | 0.8695/0.8662 |
| 29 | 0.8967/0.9070 | 0.8530/0.8608 | 0.8658/0.8373 | 0.8718/0.8684 |
| 43 | 0.9002/0.9093 | 0.8503/0.8589 | 0.8721/0.8432 | 0.8742/0.8705 |
| 71 | 0.8982/0.9077 | 0.8481/0.8569 | 0.8726/0.8440 | 0.8730/0.8695 |
| 101 | 0.8901/0.9021 | 0.8458/0.8556 | 0.8666/0.8379 | 0.8675/0.8652 |

### Locked versus independent

| Dataset | Locked AUC/AP | Independent 3-seed mean AUC/AP | Delta AUC/AP |
|---|---:|---:|---:|
| ComGenVid | 0.8968/0.9064 | 0.8980/0.9076 | +0.0011/+0.0012 |
| VideoFeedback | 0.8597/0.8689 | 0.8502/0.8589 | -0.0095/-0.0100 |
| GenVideo | 0.8657/0.8416 | 0.8674/0.8385 | +0.0017/-0.0030 |
| Macro-3 | 0.8741/0.8723 | 0.8718/0.8684 | -0.0022/-0.0039 |

Every fixed-evaluation split contains 200 calibration real identities, the fixed locked test-real identities, no generated calibration videos, and zero overlap. File hashes are recorded in `split_audit.csv`.

### Strict all-remaining-real diagnostic

This second view follows the literal `200 calibration real + all eligible remaining real` protocol. It combines the fixed evaluation-real set, the historical locked-calibration set, the independent reserve complement, and every additional raw-index real video satisfying the same strict 2 s rule. Only VideoFeedback has such an additional pool (3,080 videos). The test-real counts are 1,498/3,880/9,784 for ComGenVid/VideoFeedback/GenVideo, paired with all 3,400/3,000/5,639 locked generated videos. It changes the test-real pool, so its delta from the locked table mixes calibration-bank and test-set sampling effects and is not used as the primary causal comparison.

| Dataset | Paper protocol AUC mean +/- std | Paper protocol AP mean +/- std | Pooled-all AUC mean +/- std | Pooled-all AP mean +/- std |
|---|---:|---:|---:|---:|
| ComGenVid | 0.9045 +/- 0.0026 | 0.9106 +/- 0.0023 | 0.9050 +/- 0.0026 | 0.8367 +/- 0.0039 |
| VideoFeedback | 0.8256 +/- 0.0037 | 0.8430 +/- 0.0026 | 0.8486 +/- 0.0030 | 0.8827 +/- 0.0017 |
| GenVideo | 0.8589 +/- 0.0032 | 0.8342 +/- 0.0034 | 0.8717 +/- 0.0047 | 0.9021 +/- 0.0036 |
| Macro-3 | 0.8630 +/- 0.0020 | 0.8626 +/- 0.0023 | 0.8751 +/- 0.0025 | 0.8739 +/- 0.0028 |

Pooled-all AP is intentionally not compared with the balanced paper AP: real prevalence is 1,498/4,898, 3,880/6,880, and 9,784/15,423, so AP has a different class-prior baseline. AUC is much less sensitive to this prevalence change.

Strict all-remaining-real split identities and hashes are stored under the backward-compatible path `splits_complement/` and in `independent_complement_split_audit.csv`; every overlap count is zero.

Per-seed paper-protocol all-remaining-real results:

| Seed | ComGenVid | VideoFeedback | GenVideo | Macro-3 |
|---:|---:|---:|---:|---:|
| 17 | 0.9041/0.9093 | 0.8220/0.8400 | 0.8573/0.8325 | 0.8611/0.8606 |
| 29 | 0.9021/0.9091 | 0.8294/0.8447 | 0.8568/0.8320 | 0.8628/0.8619 |
| 43 | 0.9072/0.9132 | 0.8253/0.8442 | 0.8626/0.8381 | 0.8650/0.8652 |

## Modified files

- `src/alpha_stalled/whitening.py`: exact same-grid Local D1 primitive.
- `tools/fit_u0_local_d1_params.py`: independent D1 whitening fit on the locked real-only calibration identities.
- `tools/score_u0_local_d1_windows.py`: recoverable K1/K3 scalar scorer.
- `tools/analyze_independent_real_complement.py`: three-seed strict all-remaining-real evaluation.
- `tools/build_independent_remaining_real_manifest.py`: exhaustive strict-2s real-pool supplement audit.
- `tools/score_u0_calibration_candidates.py`: adds recoverable scoring for the supplemental real manifest.
- `tools/analyze_second_order_independent_calibration.py`: unified metrics, bootstrap, split audit, paper table, and report.
- `tests/test_whitening_batch_invariance.py`: explicit D1/D2 formula regression test.
- `tests/test_second_order_independent_calibration.py`: independent-split and metric-protocol integrity tests.
- `scripts/run_second_order_ablation.sh`: reproducible execution wrapper.

## Interpretation

- Standalone Local D2 changes Macro AUC/AP by `+0.0086/+0.0111` relative to Local D1.
- In the full model, D2 changes Macro AUC/AP by `+0.0019/+0.0028`; the paired AP interval is `[+0.0019,+0.0038]`.
- Local D2 improves AP for `19/20` generators; complete LSTL improves `18/20`. The only complete-model AP decreases are videofeedback/Pika `-0.0015`, videofeedback/ModelScope `-0.0001`.
- The strict experiment supports the claim that second-order Local temporal modeling improves the full locked method over an otherwise identical first-order version.
- Current release manifests and all independent splits show zero calibration/test identity overlap. The residual risk is calibration-domain and finite-sample sensitivity, not observed identity leakage.

## Paper artifacts

- `paper/ieee_alpha_stalled/tables/local_d1_d2_ablation.tex`: paper-ready D1/D2 table.
- `results/second_order_independent_calibration/dataset_metrics.csv`: per-dataset and Macro metrics.
- `results/second_order_independent_calibration/generator_metrics.csv`: all 20 generators.
- `results/second_order_independent_calibration/bootstrap_deltas.csv`: paired confidence intervals.
- `results/second_order_independent_calibration/independent_seed_metrics.csv`: raw seed results.
- `results/second_order_independent_calibration/splits/`: reproducible calibration/test identity files.
- `results/second_order_independent_calibration/independent_complement_seed_metrics.csv`: strict all-remaining-real raw seed results under both metric protocols.
- `results/second_order_independent_calibration/splits_complement/`: strict all-remaining-real split files (backward-compatible directory name).

## Commands

```bash
# Independent-real reserve and fixed split membership (already materialized for this run).
conda run --no-capture-output -n stall python tools/build_u0_calibration_reserve.py
conda run --no-capture-output -n stall python tools/build_independent_remaining_real_manifest.py
# Refit Local whitening for every dataset/seed/size; use SHARD in [0, N).
conda run --no-capture-output -n stall python tools/fit_u0_calibration_sensitivity.py --shard-index SHARD --num-shards N
# Score each split once for all real-only calibration candidates.
conda run --no-capture-output -n stall python tools/score_u0_calibration_candidates.py --split SPLIT --dataset DATASET --shard-index SHARD --num-shards N --extract-device cuda:GPU --score-device cuda:GPU --frame-batch-size 32
# SPLIT must include evaluation, reserve, locked_calibration, and independent_remaining_real for the strict all-remaining-real diagnostic.
# This run reused complete evaluation/reserve scalar caches and newly scored only locked_calibration plus independent_remaining_real.
# This run used N=4 for independent_remaining_real VideoFeedback scoring.
conda run --no-capture-output -n stall python tools/analyze_u0_calibration_sensitivity.py
conda run --no-capture-output -n stall python tools/analyze_independent_real_complement.py

# Controlled Local D1/D2 ablation.
conda run --no-capture-output -n stall python tools/fit_u0_local_d1_params.py --dataset DATASET --device cuda:0
conda run --no-capture-output -n stall python tools/score_u0_local_d1_windows.py --dataset DATASET --sampling calibration_k1 --shard-index SHARD --num-shards 2 --extract-device cuda:GPU --score-device cuda:GPU --frame-batch-size 32 --no-reuse-k1-cache
# Formal K3 shard counts: ComGenVid=2, VideoFeedback=2, GenVideo=4.
conda run --no-capture-output -n stall python tools/score_u0_local_d1_windows.py --dataset DATASET --sampling k3 --shard-index SHARD --num-shards N --extract-device cuda:GPU --score-device cuda:GPU --frame-batch-size 32 --no-reuse-k1-cache
conda run --no-capture-output -n stall python tools/analyze_second_order_independent_calibration.py --iterations 1000 --workers 3
# Convenience wrapper for the D1 portion: scripts/run_second_order_ablation.sh
```
