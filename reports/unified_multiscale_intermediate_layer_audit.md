# Unified region and intermediate-layer audit

> **Status update (2026-07-24):** the historical 3/1/2 region configuration is an
> historical dataset-specific tuned baseline because its choices used fake AUC/AP. The
> clean universal successor fixes region 1 and mean aggregation everywhere.

Date: 2026-07-24

## Historical tuned protocol

The reference is the leakage-free K=3 MW2 detector in
`configs/alpha_stalled_multi_window.yaml`. Its Macro-3 AUC/AP is
`0.8694/0.8697` on the fixed 21,421-video intersection. Global and Local
window formulas, K=3 sampling, effective-K calibration, beta `0.1`, and alpha
`0.6` were held fixed in this study.

## Region configuration

The current temporal region and aggregation are dataset-specific:

| Dataset | Raw-token region | Temporal aggregation |
|---|---:|---|
| ComGenVid | 3 | bottom-20% mean |
| VideoFeedback | 1 | mean |
| GenVideo | 2 | mean |

The settings were selected after comparing generated-video AUC/AP in the old
region and aggregation sensitivity experiments. Region AP optima were region 3
for ComGenVid (`0.9288`), region 1 for VideoFeedback (`0.8302`), and region 2
for GenVideo (`0.8092`). Bottom-k versus mean was likewise selected from fake
evaluation metrics. Consequently, the historical K=3 result is a useful
dataset-specific reference upper bound, but it is not evidence for a single
region configuration that transfers unchanged across datasets.

The old region implementation pools raw patch tokens before temporal
differencing. Region 2 maps 14x14 raw tokens to 7x7 by non-overlapping 2x2
averaging, then computes same-grid D2. It is therefore already the requested
coarse-token operator; no single-scale feature extraction is repeated here.

The old region 1/2/3 parameter files are not admissible for the new unified
comparison: inspection found that their window CDFs used all available real
videos (for example, 1,698 ComGenVid real scores), whereas the historical K=3 model
uses the disjoint 200-real calibration split. The old files remain historical
sensitivity evidence only. The unified study therefore reuses the existing raw
K1 patch-token caches, but refits region 1/2/3 whitening,
Gaussian-likelihood, and window-CDF parameters on exactly the fixed 200 real
calibration videos. This refit performs no DINO extraction.

Each K=3 frame then receives one final-layer DINO extraction, after which all
region scores are computed from the same token tensor. Scale fusion occurs only
between calibrated temporal percentiles, with identical fixed weights on all
datasets.

## Intermediate layers

DINOv3 ViT-L/16 exposes 24 blocks. The predeclared zero-based layers are 11,
17, and 23. `get_intermediate_layers(..., norm=True)` can return all three patch
token tensors in one backbone traversal. Prior probing established:

- each layer has 196 ordered patch tokens and excludes CLS/register tokens;
- the token grid is row-major 14x14 at every layer;
- layer 23 matches the existing final normalized patch token with maximum
  absolute error `0.0`;
- all layers share the exact same image preprocessing and checkpoint.

Each layer requires its own real-only temporal whitening, mean, Gaussian
likelihood reference, and window CDF. Cross-layer combinations are fixed equal
averages of calibrated temporal percentiles; raw features are never concatenated
and generated samples never fit parameters or weights.

## Decision boundary

MS0 is retained only as the dataset-specific reference. MS1-MS4 are the valid
unified configurations. H0-H5 use one common layer definition and common fixed
fusion weights across all datasets. Beta pruning is triggered only if a unified
candidate improves Macro AP by at least `0.004` and satisfies the remaining
admission criteria.
