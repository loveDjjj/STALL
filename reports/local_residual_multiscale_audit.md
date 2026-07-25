# Local residual and multiscale audit

> **Status update (2026-07-24):** references to the historical `0.8694/0.8697`
> baseline below mean the historical dataset-specific tuned comparator. Its
> calibration split is disjoint, but region/aggregation were fake-label selected.

Date: 2026-07-24

## Decision

The historical comparator is K=3 MW2, with Macro-3 AUC/AP `0.8694/0.8697`.
This audit eliminates two proposed experiments as duplicates:

- coordinate-median residual D2 was already evaluated as L2 and reduced Macro AP from
  `0.8040` to `0.7655` in the leakage-free single-window local study;
- raw-token 2x2 pooling followed by D2 is exactly the existing `region=2` implementation.
  Region 1/2/3 has already been evaluated on all three datasets, so Stage 3 must not be
  rerun.

Spatial-mean residual D2 and intermediate-layer D2 have not been evaluated. Only the
spatial-mean residual is eligible for the next full experiment. Intermediate layers remain
conditional on an earlier candidate passing its admission gate.

## Historical comparator and evidence

For window `k`:

```text
G_k = 0.5 * global_spatial_k + 0.5 * global_T1_k
L_k = 0.1 * patch_spatial_k + 0.9 * same_grid_patch_D2_k
```

K=3 MW2 averages each branch over deterministic uniform two-second windows, recalibrates
the two video-level branch values with disjoint real videos matched by `effective_K`, then
uses `S=0.6G+0.4L`. The fixed evaluation intersection contains 4,298 ComGenVid, 3,500
VideoFeedback, and 13,623 GenVideo videos. Each dataset has 200 disjoint real calibration
videos; generated videos are not used for whitening, CDFs, thresholds, or weights.

The main existing ablations are:

| Direction | Existing result | Decision |
|---|---|---|
| clean single-window Alpha-STALLED | Macro `0.8570/0.8600` | superseded by K=3 MW2 |
| K=3 MW2 | Macro `0.8694/0.8697`, AP delta `+0.0096`, CI fully positive | historical dataset-specific tuned comparator |
| K=5 / all-window | Macro AP `0.8672/0.8695` | reject extra cost |
| window bottom-2 / hybrid | K=3 AP `0.8684/0.8691` | reject lower-tail aggregation |
| Joint J1/J2/J3 | best AP `0.8602` vs linear `0.8697` | reject joint fusion |
| Local lag-1 / multi-lag | ComGenVid AP `0.8617/0.8586` vs D2 `0.9309` | reject |
| Motion hard / soft | ComGenVid AP `0.8491/0.8256` | reject |
| Patch D3 / D4 | ComGenVid AP `0.8936/0.8949` vs D2 `0.9309` | reject |
| Global raw/calibrated D3 | severe VideoFeedback transfer loss; no calibrated gain retained | reject |
| Third branch / volatility | generator and dataset transfer failure | reject |
| gate / clip / cap | about `0.001` best worst-case AP gain | reject complexity |

Evidence is in `reports/global_local_multi_order_final.md`,
`reports/multi_window_joint_typicality.md`, `results/paper_tables/ablation_summary.md`,
`results/journal_experiments/`, and `results/research_summary/README.md`.

## Code-path audit

### 1. Exact D2 definition

`src/eval_patch_fast.py::temporal_features` first applies optional raw-token region pooling,
then computes:

```text
A[t,i] = P[t+2,i] - 2 P[t+1,i] + P[t,i]
T[t,i] = A[t,i] / max(||A[t,i]||_2, eps)
```

The NumPy reference in `src/patch_matching.py::same_grid_finite_difference` uses the same
coefficients. Therefore the finite difference is exact. However, the frozen local model is
not always a 196-token D2: it uses region 3/1/2 for ComGenVid/VideoFeedback/GenVideo. The
corresponding D2 token counts are 16 (14x14 cropped to 12x12, then 4x4), 196, and 49.
An unpooled `[T-2,196,1024]` acceleration is valid for common-motion diagnosis, but is not
the actual scored tensor for ComGenVid or GenVideo.

### 2. Operation order

Temporal processing is:

```text
raw patch tokens
-> optional non-overlapping region mean
-> raw D2 or residual subtraction
-> L2 normalization along feature dimension
-> centering with real-only temporal mean
-> real-only whitening
-> standard-normal Gaussian log likelihood
-> frozen patch/time aggregation
-> window-level real percentile
```

The local spatial branch separately applies centering/whitening/likelihood directly to final
patch tokens. No whitening occurs before D2, residual subtraction, or L2 normalization.

### 3. Token content and order

The cache stores `x_norm_patchtokens` only, shape `[T,196,1024]`. CLS is stored separately
as `global`; four DINOv3 storage/register tokens are returned separately by the backbone and
are not in the patch tensor. At 224x224 with patch size 16, DINO flattens its `[14,14]`
patch grid in row-major order. The project uses `patch_id = row * grid_width + col`, matching
that flattening. Actual cache payloads for all three datasets were inspected and contain
`patch [16,196,1024]`, `global [16,1024]`, `grid_size [14,14]`, and 16 distinct frame
indices.

### 4. Meaning of old region experiments

Old `region=1/2/3` is option C: raw token pooling before D2. The implementation reshapes
`[T,196,D]` to `[T,14,14,D]`, crops the bottom/right edge to the largest divisible grid,
averages non-overlapping regions, and only then computes D2. It is neither likelihood-score
pooling nor D2-feature pooling. Consequently proposed coarse 2x2 token pooling is already
covered by `region=2` and must not be rerun.

Existing region-mean AP values are:

| Dataset | region 1 | region 2 | region 3 |
|---|---:|---:|---:|
| ComGenVid | 0.9075 | 0.9227 | **0.9288** |
| VideoFeedback | **0.8302** | 0.7946 | 0.7551 |
| GenVideo | 0.7985 | **0.8092** | 0.7909 |

These older sensitivity numbers use their documented historical evaluation protocol; they
establish implementation coverage and scale preference, not a new K=3 improvement claim.

### 5. Cache completeness

The compact cache preserves every final-layer patch token for each cached 16-frame window,
not only an aggregate. It is sufficient for single-window residual refitting. It does not
contain the additional K=3 windows: the multi-window run retained their scalar scores and
frame indices, so K=3 residual diagnostics require DINO extraction for the missing windows.
The existing 361 GB patch cache must not be overwritten.

### 6. Intermediate layers

The DINOv3 ViT-L/16 has 24 blocks, four storage tokens, and a public
`get_intermediate_layers` path. Requested zero-based layers are mid 11, late 17, final 23.
A one-forward traversal test returned three tensors of `[1,196,1024]`; the layer-23 tensor
matched `forward_features()['x_norm_patchtokens']` with maximum absolute error `0.0`.
Intermediate extraction is technically feasible while excluding CLS/storage tokens, but the
current cache contains final-layer tokens only. A full intermediate-layer experiment would
require new extraction and separate real-only calibration for every layer.

Checkpoint SHA-256:
`8aa4cbddda325040fc78db2c272754af6ebe8ff2c55f6ec4f1964d8890f66035`.

## Duplicate-direction matrix

| Proposed item | Already exists? | Action |
|---|---|---|
| spatial-mean residual | no | eligible as R1 |
| coordinate-median residual | yes, L2 after the frozen region pooling | summarize; do not rerun |
| CLS/global-motion subtraction | yes, L1 | reject; Macro AP `0.5186` |
| intermediate-layer D2 | no | only after an earlier candidate passes |
| calibrated fine/coarse fusion | no exact fusion, but fine/coarse components and scale sweep exist | do not run because Stage 3 duplication rule is triggered |
| raw-token 2x2 coarse D2 | yes, region 2 | do not rerun |

## Stage decisions

1. Run Stage 1 common-motion diagnostics on unpooled 196-token accelerations, clearly
   labeling them as diagnostics.
2. In Stage 2, retain the historical R0 comparator, do not rerun R2, and test only new R1
   spatial-mean residual with independent real-only whitening/CDF. R3 is permitted only if
   R1 is competitive enough to justify calibrated score fusion.
3. Stop Stage 3 as duplicate work. Do not run new fine/coarse extraction or fusion.
4. Do not start intermediate layers unless R1 passes the predeclared full admission gate.
5. Preserve beta `0.1` until a temporal candidate passes; otherwise Stage 5 is unnecessary.
