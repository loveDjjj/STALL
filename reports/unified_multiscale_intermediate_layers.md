# Alpha-STALLED unified multiscale and intermediate-layer study

> **Status update (2026-07-24):** the `0.8694/0.8697` MS0/H0 reference below
> is dataset-specific and fake-label selected. It remains a historical tuned reference,
> not the strict universal baseline. See
> `reports/clean_universal_cross_layer_final.md` for the later temporal-unified
> audit and `configs/alpha_stalled_u0_locked.yaml` for the corrected release.

Date: 2026-07-24

## Executive decision

No candidate was admitted relative to the historical K=3 MW2 tuned comparator
with Macro-3 AUC/AP `0.8694/0.8697`. The later temporal-unified audit reported
U0 `0.8725/0.8722`, but its PatchSpatial inheritance was corrected only in the
subsequent locked clean reproduction.

- The best fully unified region configuration is MS1 (region 1), with
  `0.8687/0.8695`; its AP delta is `-0.00013` and its 95% paired-bootstrap CI
  crosses zero.
- The best layer configuration is H4, an equal average of independently
  calibrated layer-17 and layer-23 temporal percentiles. It reaches
  `0.8721/0.8698`: Macro AUC improves by `+0.00276`, but Macro AP improves by
  only `+0.00016`, with CI `[-0.00176,+0.00217]`.
- H4 lowers VideoFeedback and GenVideo AP by `0.00529` and `0.00502`, and only
  9/20 generators do not decline. It fails the AP, worst-dataset, generator,
  and bootstrap gates.
- No layer candidate reaches the predeclared `+0.004` Macro AP trigger, so
  beta `0/0.05/0.2` is not run. Beta remains `0.1` and alpha remains `0.6`.

## Held-fixed protocol and leakage audit

The experiment uses the exact 21,421-video evaluation intersection and the
fixed 200-real calibration split per dataset. Every K=3 window has 16 indexed
frames at nominal 8 FPS, with no one-second fallback or duplicated frames.
Global, PatchSpatial, beta, alpha, video aggregation, and effective-K matched
video CDFs remain frozen.

The current baseline uses dataset-specific Local Temporal settings:

| Dataset | Region | Aggregation |
|---|---:|---|
| ComGenVid | 3 | bottom-20% mean |
| VideoFeedback | 1 | mean |
| GenVideo | 2 | mean |

The historical region sensitivity tables selected these settings from
generated-video AUC/AP: the region AP optima were region 3 on ComGenVid,
region 1 on VideoFeedback, and region 2 on GenVideo. MS0 is therefore a
dataset-specific reference upper bound, not a unified transferable setting.

The old region parameter files were also inadmissible for this study because
their CDFs used all available real videos (for example, 1,698 ComGenVid real
scores). All nine region parameter sets were rebuilt from the existing raw K1
patch cache using exactly the fixed 200 calibration videos. This required no
DINO feature extraction. The rebuilt baseline-region parameter arrays match
the frozen K=3 L0 arrays exactly, with maximum error `0.0`.

No evaluation real video or generated video enters whitening, likelihood CDFs,
video CDFs, weights, or thresholds.

## Unified region multiscale

For each final-layer patch tensor, region 1/2/3 independently performs raw-token
pooling, same-grid D2, L2 normalization, real-only whitening, Gaussian
likelihood, frozen dataset aggregation, and a 200-real window CDF. Fusion is
only between calibrated temporal percentiles:

```text
MS0 = current dataset-specific region reference
MS1 = T_region1
MS2 = T_region2
MS3 = 0.5 T_region1 + 0.5 T_region2
MS4 = (T_region1 + T_region2 + T_region3) / 3
```

MS1-MS4 use the same region combination and fixed weights on all datasets.

| Config | ComGenVid AUC/AP | VideoFeedback AUC/AP | GenVideo AUC/AP | Macro-3 AUC/AP |
|---|---:|---:|---:|---:|
| MS0 | 0.8857/0.8996 | 0.8623/0.8684 | 0.8601/0.8410 | **0.8694/0.8697** |
| MS1 | 0.8875/0.9012 | 0.8623/0.8685 | 0.8563/0.8390 | 0.8687/0.8695 |
| MS2 | 0.8891/0.9021 | 0.8520/0.8561 | 0.8601/0.8411 | 0.8671/0.8664 |
| MS3 | 0.8888/0.9023 | 0.8586/0.8638 | 0.8581/0.8400 | 0.8685/0.8687 |
| MS4 | 0.8881/0.9019 | 0.8558/0.8599 | 0.8583/0.8399 | 0.8674/0.8672 |

MS1 is the only unified candidate with 12/20 non-declining generators and no
dataset AP loss beyond `0.004`. It still fails because Macro AP decreases by
`0.00013` and its Macro AP CI is `[-0.00162,+0.00119]`. MS2-MS4 have larger
VideoFeedback losses. Equal calibrated scale fusion therefore does not improve
the dataset-specific reference.

## DINO intermediate layers

DINOv3 ViT-L/16 has 24 blocks. Layers 11, 17, and 23 are returned together by
one `get_intermediate_layers(..., norm=True)` traversal. Each layer has 196
ordered patch tokens after CLS/register removal and uses the same transform and
checkpoint (`sha256` prefix `8aa4cbddda325040`). Layer 23 matches
`forward_features` tokens with maximum error `0.0` on all three calibration
probes.

Each layer has an independent 200-real temporal mean, whitening matrix,
Gaussian likelihood, and window CDF. No cross-layer raw feature concatenation
or shared whitening is used. Layer identifiers and equal fusion weights are
identical across all datasets; the layer study deliberately inherits the
frozen dataset-specific region/aggregation so that only the layer changes.

```text
H0 = T_23
H1 = T_11
H2 = T_17
H3 = 0.5 T_11 + 0.5 T_23
H4 = 0.5 T_17 + 0.5 T_23
H5 = (T_11 + T_17 + T_23) / 3
```

| Config | ComGenVid AUC/AP | VideoFeedback AUC/AP | GenVideo AUC/AP | Macro-3 AUC/AP |
|---|---:|---:|---:|---:|
| H0 | 0.8857/0.8996 | 0.8623/0.8684 | 0.8601/0.8410 | **0.8694/0.8697** |
| H1 | 0.8916/0.8925 | 0.8213/0.8337 | 0.8498/0.8277 | 0.8542/0.8513 |
| H2 | 0.9039/0.9062 | 0.8436/0.8501 | 0.8441/0.8272 | 0.8638/0.8612 |
| H3 | 0.8971/0.9027 | 0.8480/0.8551 | 0.8578/0.8367 | 0.8676/0.8648 |
| H4 | 0.9040/0.9104 | 0.8582/0.8631 | 0.8542/0.8360 | **0.8721/0.8698** |
| H5 | 0.9026/0.9063 | 0.8481/0.8542 | 0.8544/0.8340 | 0.8684/0.8648 |

H4 exhibits a real AUC-only tradeoff. Its Macro AUC delta is `+0.00276`, with
95% CI `[+0.00061,+0.00505]`, driven by ComGenVid. Its AP evidence does not
transfer:

| Scope | H4-H0 AP delta | 95% CI |
|---|---:|---:|
| ComGenVid | +0.01079 | [+0.00669,+0.01513] |
| VideoFeedback | -0.00529 | [-0.00745,-0.00278] |
| GenVideo | -0.00502 | [-0.00832,-0.00185] |
| Macro-3 | +0.00016 | [-0.00176,+0.00217] |

Thus an intermediate late layer helps ComGenVid but is not a transferable
Local Temporal replacement.

## Generator and motion diagnosis

H4's largest AP improvements are Pika `+0.0190`, ZeroScope-576w `+0.0185`,
WildScrape `+0.0152`, Gen2 `+0.0141`, VEO3 `+0.0140`, and Fast-SVD `+0.0092`.
Its largest losses are Text2Video-Zero `-0.0443`, Lavie `-0.0195`,
MorphStudio `-0.0193`, LaVie-base `-0.0186`, GenVideo Sora `-0.0136`,
ModelScope `-0.0126`, Crafter `-0.0119`, LVDM `-0.0108`, and SoRA-Clip
`-0.0103`.

For the five required focus generators:

| Generator | MS1 AP delta | H4 AP delta |
|---|---:|---:|
| ZeroScope-576w | approximately 0 | +0.0185 |
| Pika | +0.00001 | +0.0190 |
| SoRA-Clip | +0.00006 | -0.0103 |
| Crafter | +0.0044 | -0.0119 |
| Text2Video-Zero | +0.0005 | -0.0443 |

Motion stratification explains part of H4's transfer failure:

| Fake motion group | H0 Macro AUC/AP | H4 Macro AUC/AP | AP delta |
|---|---:|---:|---:|
| Low | 0.9153/0.9243 | 0.9329/0.9397 | +0.0154 |
| Mid | 0.8590/0.8737 | 0.8616/0.8781 | +0.0044 |
| High | 0.7245/0.7515 | 0.6632/0.6991 | -0.0523 |

H4 improves low/mid-motion generated videos but strongly raises errors on
high-motion fakes. Among 411 baseline false negatives at threshold 0.5, H4
reduces the still-false-negative count to 372, but this threshold improvement
does not compensate for its cross-generator ranking losses.

MS1 is much milder: it improves low-motion fake AP by `0.0016`, lowers
mid-motion AP by `0.0035`, and lowers high-motion AP by `0.0046`.

## Admission

| Candidate | Macro AP delta | Worst dataset AP | Non-decline | Macro AP CI | Admit |
|---|---:|---:|---:|---:|---|
| MS1 | -0.00013 | -0.00207 | 12/20 | [-0.00162,+0.00119] | no |
| MS2 | -0.00324 | -0.01232 | 9/20 | [-0.00414,-0.00215] | no |
| MS3 | -0.00098 | -0.00465 | 9/20 | [-0.00193,+0.00008] | no |
| MS4 | -0.00244 | -0.00846 | 8/20 | [-0.00338,-0.00146] | no |
| H1 | -0.01837 | -0.03470 | 6/20 | [-0.02199,-0.01449] | no |
| H2 | -0.00851 | -0.01834 | 8/20 | [-0.01153,-0.00535] | no |
| H3 | -0.00483 | -0.01326 | 6/20 | [-0.00721,-0.00269] | no |
| H4 | +0.00016 | -0.00529 | 9/20 | [-0.00176,+0.00217] | no |
| H5 | -0.00486 | -0.01424 | 9/20 | [-0.00715,-0.00236] | no |

The required gates are Macro AP `>= +0.004`, worst dataset AP `>= -0.004`,
at least 12/20 non-declining generators, and a positive Macro AP CI lower
bound. No candidate passes all gates.

## Integrity, cost, and numerical stability

- The combined output has 58,496 windows including calibration and 21,421
  evaluation videos. Keys are unique, frame-index lists match the frozen K=3
  file one-to-one, all scores are finite, and layer-23 percentiles match the
  frozen D2 score with maximum error `1.11e-16`.
- The final full pass shares decoding and one three-layer DINO traversal between
  MS and H. Dual RTX 5090 wall times were about 13.8 minutes for ComGenVid,
  4.6 minutes for VideoFeedback, and 28.5 minutes for GenVideo, or about 46.9
  minutes across sequential dataset phases.
- Intermediate real calibration caches occupy 8.4 GB: 0.5 GB ComGenVid, 6.3 GB
  VideoFeedback, and 1.6 GB GenVideo. Nine layer parameter files occupy 73 MB;
  nine region parameter files occupy 73 MB; combined raw score CSVs occupy
  19 MB.
- VideoFeedback's rank-1023 whitening is numerically sensitive to GEMM batch
  shape. A preliminary batch-8 run changed three evaluation window percentiles.
  Re-running with the frozen batch size 4 restored every final-layer percentile
  exactly. Only the exact batch-4 output is used in this report.

## Final answer

Unified region fusion does not beat the dataset-specific reference. A late
intermediate layer contains useful ComGenVid and low-motion evidence, but the
benefit does not transfer to VideoFeedback, GenVideo, high-motion fakes, or a
majority of generators. The extra calibration/cache complexity is not
justified. This experiment therefore retains final-layer Local D2, beta `0.1`, and fixed
`0.6 Global + 0.4 Local` without adding a scale or layer branch. The later universal audit
fixes region1+mean as U0.
