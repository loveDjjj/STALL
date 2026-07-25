# Alpha-STALLED U0 numerical stability audit

Date: 2026-07-24

## Decision

The release score stage must use **N2**, which converts cached patch features,
the fitted mean, and the whitening operator to float64 before whitening and
Gaussian-likelihood evaluation. Each window is evaluated with the same
two-dimensional GEMM shape, so an outer I/O batch cannot change its reduction
tree. The DINO backbone and cached patch tokens remain float32.

N2 passes the numerical gate on the independent cached-feature audit:

- batch sizes 1, 4, 8, and 16 produce exactly identical raw likelihoods,
  percentiles, final scores, ranks, AUC, and AP;
- the maximum final-score error is `0`, below the required `1e-8`;
- CPU/GPU float64 raw-score error is at most `2.27e-13`, with no percentile
  changes;
- N1, which only disables TF32, has the same batch-shape-dependent FP32 raw
  drift as N0 and is not an adequate fix.

This report establishes the release numerical primitive. It does **not** claim
that the complete 21,421-video K=3 release reproduction has already finished.
The existing final-layer patch-token cache contains the original two-second K1
window, while the additional K=3 windows were retained as scalar research
scores rather than complete tokens. The required clean-from-empty K=3 traversal
and final metric comparison are therefore part of the immediately following
locked release reproduction.

## Problem reproduced

The historical U0 implementation evaluated a high-dimensional whitened
Gaussian likelihood in float32. On the full 7,076-window VideoFeedback
cross-layer run, changing the CUDA batch shape from 4 to 8 caused:

| Operator | Raw max error | Changed percentiles | Changed eval videos | Final max error | AUC/AP change |
|---|---:|---:|---:|---:|---:|
| layer23 region1 | 84.38 | 12 | 3 | 0.0676 | +0.000073/+0.000116 |
| layer17 region1 | 96.40 | 12 | 3 | 0.0648 | +0.000072/+0.000128 |

Float64 rechecks of the six affected layer/window scores agreed with batch4
percentiles within `3e-8`; their raw values were within `0.00140` of batch4 but
up to `96.40` from batch8. This identifies a float32 GEMM batch-shape effect in
the ill-conditioned whitening computation, not a video-content difference or a
method gain. Fixing batch size to 4 would hide rather than remove the issue.

## Audit protocol

`tools/audit_u0_numerical_stability.py` uses existing K1 two-second final-layer
patch-token caches and does no DINO extraction. It selects a deterministic,
hash-balanced diagnostic sample with 16 videos per available real/fake group:

| Dataset | Real | Fake groups | Total videos |
|---|---:|---:|---:|
| ComGenVid | 16 | 2 x 16 | 48 |
| VideoFeedback | 16 | 10 x 16 | 176 |
| GenVideo | 16 | 8 x 16 | 144 |
| **Total** | **48** | **20 generators** | **368** |

The fixed Global and PatchSpatial components are merged from existing K1
scores. Only the Local Temporal numerical implementation changes. The audit
compares four outer batch sizes (`1,4,8,16`) in three modes:

The later release-provenance audit found that those fixed pre-release
PatchSpatial scores inherited historical dataset-specific aggregation
(bottom-20 on ComGenVid and mean elsewhere). This does not alter the N2 Local
Temporal batch-invariance conclusion, but it means the 368-video audit cannot
establish a fully unified detector by itself. The locked clean reproduction
therefore recomputes Global, mean PatchSpatial, and mean PatchD2 raw scores from
source videos and rebuilds all real-only CDFs.

- **N0:** current float32 batched whitening/Gaussian likelihood;
- **N1:** N0 plus `cuda.matmul.allow_tf32=False`,
  `cudnn.allow_tf32=False`, and matmul precision `highest`;
- **N2:** float64 mean, whitening, GEMM, squared norm, reduction, raw score, and
  CDF sorting, with fixed per-window GEMM shape.

N3 is reserved for the later predeclared OAS candidate and is not part of the
numerical repair decision.

## Batch invariance

Each row compares the indicated batch size with batch size 1 in the same mode.
Times sum one measured score pass for each of the three datasets and exclude
cache loading and metric analysis.

| Mode | Batch | Raw max abs error | Raw max relative error | Percentile changes | Final changes | Rank changes | AUC/AP delta | Score time (s) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| N0 | 4 | 1.22e-4 | 8.54e-8 | 0 | 0 | 0 | 0/0 | 0.368 |
| N0 | 8 | 2.44e-4 | 1.64e-7 | 0 | 0 | 0 | 0/0 | 0.315 |
| N0 | 16 | 2.44e-4 | 1.66e-7 | 0 | 0 | 0 | 0/0 | 0.273 |
| N1 | 4 | 1.22e-4 | 8.54e-8 | 0 | 0 | 0 | 0/0 | 0.364 |
| N1 | 8 | 2.44e-4 | 1.64e-7 | 0 | 0 | 0 | 0/0 | 0.308 |
| N1 | 16 | 2.44e-4 | 1.66e-7 | 0 | 0 | 0 | 0/0 | 0.271 |
| **N2** | **4** | **0** | **0** | **0** | **0** | **0** | **0/0** | **2.292** |
| **N2** | **8** | **0** | **0** | **0** | **0** | **0** | **0/0** | **2.307** |
| **N2** | **16** | **0** | **0** | **0** | **0** | **0** | **0/0** | **2.306** |

N0 and N1 differ from N2 at batch size 1 by maximum/median raw errors
`1.95e-4/4.50e-5`. Those differences do not cross a 200-real CDF boundary in
this diagnostic sample, so all three modes happen to have identical sampled
metrics. That is not evidence that FP32 is safe: the larger full
VideoFeedback audit above already contains CDF and metric changes.

N2 is slower in this cache-only microbenchmark because it intentionally fixes
the per-window GEMM shape and uses float64. The measured approximately two
seconds for 368 windows is negligible relative to DINO video decoding and
feature extraction. A full release run will report end-to-end overhead.

## Covariance diagnostics

Eigenvalues are reconstructed from the fitted whitening operator. A dimension
omitted by the fitted operator is treated as a dropped zero-eigenvalue
direction; no new variance threshold is selected.

| Dataset | Dimension | Effective rank | Dropped | Min/max retained eigenvalue | Condition number |
|---|---:|---:|---:|---:|---:|
| ComGenVid | 1024 | 1024 | 0 | 2.54e-10 / 1.13e-2 | 4.44e7 |
| VideoFeedback | 1024 | 1023 | 1 | 6.13e-7 / 1.09e-2 | 1.78e4 |
| GenVideo | 1024 | 1024 | 0 | 9.48e-11 / 9.02e-3 | 9.52e7 |

The large condition numbers make float64 appropriate even though the most
visible historical failure occurred in rank-1023 VideoFeedback. N2 preserves
the existing fitted rank and operator; it does not change features, covariance,
PCA threshold, calibration videos, or the detector formula.

## Deterministic CDF and tie policy

All release CDF references are converted to float64 and sorted with stable
mergesort. The empirical CDF is right-inclusive:

```text
F_real(x) = count(reference <= x) / N
```

Exact ties therefore include every equal calibration score. NaN and infinity
are rejected. Tests cover duplicate reference values, input-order invariance,
non-finite rejection, outer-batch invariance, and the currently locked U0
reference metrics.

## CPU/GPU check

Two cached windows per dataset were independently evaluated with the same N2
implementation on CPU and CUDA:

| Dataset | Raw max abs error | Percentile changes |
|---|---:|---:|
| ComGenVid | 2.27e-13 | 0 |
| VideoFeedback | 0 | 0 |
| GenVideo | 2.27e-13 | 0 |

The CPU/GPU comparison is deliberately small and is a numerical diagnostic,
not a substitute for full result reproduction.

## Verification and artifacts

Executed in the `stall` conda environment:

```bash
conda run --no-capture-output -n stall python \
  tools/audit_u0_numerical_stability.py \
  --per-group 16 --modes N0 N1 N2 \
  --output-dir results/u0_numerical_stability

conda run --no-capture-output -n stall python -m unittest \
  tests.test_cdf_tie_policy \
  tests.test_whitening_batch_invariance \
  tests.test_u0_score_reproducibility
```

The six mandatory tests pass. The conda environment does not currently contain
the `pytest` module, so the tests were executed through Python's standard
`unittest` runner rather than an unrelated user-level pytest entry point.

Machine-readable outputs:

- `results/u0_numerical_stability/pilot_manifest.csv`
- `results/u0_numerical_stability/per_video_scores.csv`
- `results/u0_numerical_stability/mode_metrics.csv`
- `results/u0_numerical_stability/batch_invariance_summary.csv`
- `results/u0_numerical_stability/mode_comparison_summary.csv`
- `results/u0_numerical_stability/covariance_diagnostics.csv`
- `results/u0_numerical_stability/cpu_gpu_float64.csv`
- `results/u0_numerical_stability/metadata.json`

## Remaining gate

N2 is accepted as the release score-stage implementation. The stable U0 metric
is not yet declared: the locked release stage must regenerate all K=3 window
scores without reading old final-score files, rebuild the effective-K real
video CDFs, and compare the resulting 21,421-video Macro-3 AUC/AP with the
current reference `0.8725/0.8722`. A difference larger than `0.0002` must be
explained and the stable recomputation becomes authoritative.
