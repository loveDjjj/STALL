# Alpha-STALLED U0 protocol and provenance audit

Date: 2026-07-24

Git state audited: commit `794f7e087398f436abea2cea38820d8708022c7c` plus the
current uncommitted Clean Universal U0 worktree. The machine-readable registry is
`reports/u0_experiment_registry.csv`.

## Decision

The pre-release temporal-unified result is Macro-3 AUC/AP
`0.872499/0.872243` on the fixed 21,421-video intersection. A release audit
subsequently found that its analyzer replaced PatchD2 with region1+mean but
retained PatchSpatial from the historical score file. ComGenVid therefore
inherited bottom-20 PatchSpatial aggregation while VideoFeedback and GenVideo
used mean. The following four classes must remain separate:

| Class | Macro-3 AUC/AP | Sample-calibration leakage | Fake-informed development | Paper role |
|---|---:|---:|---:|---|
| Historical leakage-affected B8 | 0.8737/0.8750 | yes | yes | excluded |
| Historical dataset-specific tuned baseline | 0.8694/0.8697 | no | yes, including target-specific region/aggregation | audit/supplement only |
| Pre-release temporal-unified U0 | 0.8725/0.8722 | no | historical PatchSpatial aggregation remained dataset-specific | valid audit result, not fully unified release |
| Locked fully unified U0 | **0.8741/0.8723** | no | inherited architecture/weights were developed on the three benchmarks | locked main method; clean reproduction validated |

The `0.8694/0.8697` result is not an Oracle upper bound. It is a historical
dataset-specific tuned baseline: its 200-real calibration split is disjoint, but
ComGenVid, VideoFeedback, and GenVideo use fake-selected
`region3+bottom20`, `region1+mean`, and `region2+mean`, respectively.

## Locked U0 protocol

For window `k`:

```text
G_k = 0.5 GlobalSpatial_k + 0.5 GlobalT1_k
A[t,i] = P[t+2,i] - 2 P[t+1,i] + P[t,i]
D2[t,i] = A[t,i] / max(||A[t,i]||_2, eps)
P_k = ECDF_real200(mean GaussianLikelihood(whiten(PatchToken(layer23))))
T_k = ECDF_real200(mean GaussianLikelihood(whiten(D2(layer23, region1))))
L_k = 0.1 P_k + 0.9 T_k
```

Each video uses up to three deterministic uniform two-second windows, 16 frames
per window at nominal 8 FPS. One-second fallback and duplicated frames are
forbidden; duplicate frame-index windows are removed and recorded through
`effective_K`. Video aggregation is:

```text
G_raw = mean_k(G_k)
L_raw = mean_k(L_k)
G = ECDF_real200,effective-K(G_raw)
L = ECDF_real200,effective-K(L_raw)
S = 0.6 G + 0.4 L
```

Every dataset uses 200 independent real calibration videos, zero generated
calibration videos, and zero calibration/evaluation video overlap. The locked
run applies mean aggregation to both PatchSpatial and PatchD2, with region 1,
layer 23, alpha 0.6, beta 0.1, and K=3 identical across datasets. This corrects
the pre-release analyzer's inherited PatchSpatial aggregation; the correction
is protocol-driven and is not selected from generated-video performance.

## Hyperparameter provenance

### Alpha 0.6

Alpha 0.6 predates U0. The single-window study explicitly evaluated fixed alpha,
leave-one-dataset-out selection, worst-case selection, and target diagnostics.
Fixed alpha 0.6 and LODO selected `P0/0.6` on ComGenVid and GenVideo, while the
diagnostic per-dataset optima varied substantially. The old release sweeps show
best-AP alpha 0.20 on ComGenVid, 0.75 on VideoFeedback, and 0.60 on GenVideo.
Therefore 0.6 is a fixed cross-dataset development choice, not a value proven
without examining generated-video metrics.

### Beta 0.1

Beta 0.1 was retained because the single-window P0 ablation
`0.1 PatchSpatial + 0.9 PatchD2` raised Macro AP from `0.8040` to `0.8162`,
improved all 20 generator comparisons, and had no dataset AP decline. It is
likewise a development-set choice. U0 does not reselect beta per dataset and
the final core ablation must recompute the requested beta sensitivity under the
strict region1+mean+K3 protocol.

### K=3

K=3 was chosen after the multi-window study. Relative to the single-window
detector, branch-wise mean plus effective-K real-video recalibration increased
Macro AP by `0.009627`, with 95% paired-bootstrap CI
`[0.007047,0.012373]` and 15/20 generator improvements. K=5 reduced AP by
`0.002442`; all-window was statistically tied in AP while requiring more
inference and introducing K/duration calibration confounding. K=3 is therefore
a development-backed fixed choice, not a hyperparameter selected independently
of the three evaluation benchmarks.

### Region 1, mean, layer 23

The historical dataset-specific region and aggregation choices are inadmissible
as the universal main method. The pre-release U0 predeclared region1+mean for
PatchD2 before U0-U5, but its implementation did not independently recompute
PatchSpatial. The locked configuration applies the declared mean aggregation to
both Local components. Layer 23 is the final backbone layer; intermediate layers
and cross-layer combinations subsequently failed their admission gates. No
generated video enters whitening, Gaussian parameters, window CDF, or video CDF.

## Development and evaluation roles

The defensible claim is:

> Locked U0 uses no generated video for model fitting or calibration, does not
> use dataset-specific fake labels to choose region or aggregation, and keeps
> one architecture and one hyperparameter configuration across ComGenVid,
> VideoFeedback, and GenVideo.

The following stronger claims are not supported:

- that the complete research process never inspected target generated videos;
- that alpha, beta, and K were fixed before all three benchmarks were examined;
- that the three current datasets are untouched confirmation sets;
- that U0 is a target-data-free universal zero-shot detector.

The three datasets served both development and evaluation roles. The appropriate
method positioning is **a training-free generated-video detector with a unified
architecture and hyperparameters, using unsupervised target-domain real-video
calibration**. A new dataset evaluated only after locking U0 is required for a
confirmation claim.

## Historical experiment index

| Experiment | Macro-3 AUC/AP | Status |
|---|---:|---|
| Original STALL B2 | 0.8388/0.8428 | main baseline |
| Clean single-window | 0.8570/0.8600 | main ablation |
| Historical leakage B8 | 0.8737/0.8750 | excluded |
| Historical dataset-specific tuned K3 | 0.8694/0.8697 | supplement/audit |
| Pre-release temporal-unified U0 | 0.8725/0.8722 | audit result; PatchSpatial inherited |
| Locked fully unified U0 | 0.8741/0.8723 | locked main method; clean stable reproduction passed 61 checks |
| K5 / all-window | 0.8663/0.8672; 0.8712/0.8695 | failed complexity ablations |
| Joint Typicality J3 | 0.8620/0.8602 | failed ablation |
| Spatial-mean residual R1 | 0.8626/0.8609 | failed ablation |
| Unified-region MS1 | 0.8687/0.8695 | failed; not equivalent to U0 |
| Intermediate-layer H4 | 0.8721/0.8698 | failed transfer/admission |
| Cross-layer C1 / C2 | 0.8707/0.8695; 0.8695/0.8692 | failed admission |

Global D3 results in this project are operator-controlled **D3-inspired global
volatility diagnostics**, not an official D3 reproduction. Hard/soft matching,
multi-lag, patch D3/D4, residuals, region multiscale, intermediate layers,
Joint Typicality, K5/all-window optimization, lower-tail aggregation, and
cross-layer gates are indexed as historical or failed experiments and must not
be rerun.

## Evidence and remaining confirmation work

Primary evidence:

- `reports/clean_universal_cross_layer_final.md`
- `reports/multi_window_joint_typicality.md`
- `reports/global_local_multi_order_final.md`
- `reports/local_residual_multiscale_final.md`
- `reports/unified_multiscale_intermediate_layers.md`
- `results/paper_tables/ablation_summary.md`
- `results/paper_sweeps/alpha_sweep_summary.md`

The float64 score primitive is batch-invariant, and the clean-from-empty locked
reproduction establishes authoritative Macro-3 AUC/real-positive AP
`0.874072/0.872299`. Metric orientation, calibration stability, cross-domain
real calibration, and locked external validation remain required for the final
paper package; they do not reopen the U0 configuration.
