# Alpha-STALLED U0 final experiment package

Date: 2026-07-25

## Executive result

The only locked main method is the fully unified, numerically stable U0:

```text
G_k = 0.5 * GlobalSpatial_k + 0.5 * GlobalT1_k
L_k = 0.1 * PatchSpatial_k + 0.9 * same_grid_PatchD2_k
G_raw(V) = mean_k(G_k)
L_raw(V) = mean_k(L_k)
G, L = effective-K-matched real-video ECDFs
S(V) = 0.6 * G(V) + 0.4 * L(V)
```

Each video uses up to three deterministic uniform 2-second windows, each containing 16
distinct frames at nominal 8 FPS. Local uses final-layer DINOv3 ViT-L/16 14x14 patch
tokens, region 1, same-grid D2, and mean aggregation. The score stage and CDF are float64.
Every dataset has 200 disjoint real calibration videos; generated videos and evaluation
reals never enter whitening, CDFs, thresholds, weights, or parameter fitting.

| Dataset | Videos | Generators | AUC | Real-positive AP |
|---|---:|---:|---:|---:|
| ComGenVid | 4,298 | 2 | 0.896818 | 0.906412 |
| VideoFeedback | 3,500 | 10 | 0.859718 | 0.868912 |
| GenVideo | 13,623 | 8 | 0.865682 | 0.841574 |
| **Macro-3** | **21,421** | **20** | **0.874072** | **0.872299** |

Relative to Original STALL (`0.8388/0.8428`), U0 gains `+0.0353 AUC / +0.0295 AP`;
the AP 95% paired cluster-bootstrap CI is `[+0.0241,+0.0358]`. Relative to the unified
K1 implementation (`0.8632/0.8636`), K3 gains `+0.0109/+0.0087`, with AP CI
`[+0.0057,+0.0120]`.

## 1. Protocol and historical audit

Three historical result classes must not be mixed:

| Class | Macro AUC/AP | Status |
|---|---:|---|
| Leakage-affected historical result | 0.8737/0.8750 | Invalid for method selection or comparison |
| Historical dataset-specific tuned K3 | 0.8694/0.8697 | No sample leakage, but region/aggregation selected with target fake metrics |
| Pre-release temporal-unified U0 | 0.8725/0.8722 | PatchD2 unified; PatchSpatial still inherited historical aggregation |
| **Locked fully unified U0** | **0.8741/0.8723** | **Main method** |

Alpha `0.6`, beta `0.1`, K `3`, layer `23`, region `1`, and mean aggregation are frozen
across all datasets. The three internal benchmarks were used during development, so they
are not untouched confirmation sets. U0 is accurately positioned as a training-free method
with a unified structure and target-domain real-only unsupervised calibration, not as a
target-data-free universal detector.

## 2. Numerical stability and release reproduction

FP32 whitening at rank 1023 could change VideoFeedback raw likelihood by about 96 when the
outer batch shape changed. N2 converts cached FP32 features to float64 before centering,
whitening, matrix multiplication, Gaussian likelihood, and CDF sorting. Batch sizes
1/4/8/16 then agree to at most `2.27e-13`; CPU/GPU checks are at the same scale and final
AUC/AP are batch-invariant.

The release was rebuilt from source videos into an empty result directory. It contains 600
real calibration videos, 21,421 evaluation videos, 58,496 windows, zero overlap, zero
decode/scoring failures, locked frame indices, hashes, and 21,421 unique final scores.
`tools/verify_u0_locked_release.py` passes all 61 checks. Dual RTX 5090 K3 traversal took
approximately 52 minutes; K1 calibration references took under one minute.

## 3. Metric protocol

The headline AP is **real-positive AP**: label real as 1 and score with `S`. The release also
reports fake-positive AP using fake label 1 and anomaly score `1-S`; these values are not
numerically interchangeable. Results are reported as generator-pairwise macro, unique-video
pooled, and 20-generator macro metrics. Bootstrap resamples video IDs within dataset and
generator strata; windows are never independent sampling units.

At the headline generator-pairwise Macro-3 scope, AUC/fake-positive AP/real-positive AP are
`0.8741/0.8642/0.8723`; balanced accuracy at threshold 0.5 is `0.6911`, fake TPR at
1%/5% real FPR is `0.1985/0.4514`, and EER is `0.2006`. Unique-video pooled AP is
prevalence-sensitive and therefore is not compared numerically with balanced pairwise AP.

## 4. Core ablation

| Configuration | Macro AUC/AP |
|---|---:|
| Original STALL | 0.8388/0.8428 |
| K1 GlobalSpatial | 0.8083/0.8277 |
| K1 GlobalT1 | 0.8149/0.8045 |
| K1 calibrated Global | 0.8385/0.8402 |
| K1 PatchSpatial | 0.6997/0.6973 |
| K1 PatchD2 | 0.8245/0.8214 |
| K1 calibrated Local | 0.8191/0.8144 |
| K3 Global-only | 0.8485/0.8486 |
| K3 Local-only | 0.8357/0.8292 |
| Unified Global+Local K1 | 0.8632/0.8636 |
| **Locked U0 K3** | **0.8741/0.8723** |

Adding Local to K3 Global improves Macro AP by `+0.0237`, CI `[+0.0191,+0.0285]`;
adding Global to K3 Local improves it by `+0.0431`, CI `[+0.0350,+0.0513]`. K3 is the
second independent gain, `+0.0087 AP` over unified K1. It improves ComGenVid/GenVideo AP
by `+0.0186/+0.0119` but lowers VideoFeedback by `-0.0045`.

PatchSpatial is not the main Local evidence. Adding its frozen 0.1 weight to K1 D2 changes
Macro AP by `-0.0070`, CI `[-0.0088,-0.0054]`. Beta remains 0.1 because U0 was locked
before sensitivity evaluation, not because post-hoc beta selection supports it.

Relative to Original STALL, U0 improves 16/20 generators. Largest AP gains are GenVideo
ModelScope `+0.0710`, Sora `+0.0618`, Gen2 `+0.0575`, and ComGenVid VEO3 `+0.0528`;
largest declines are Text2Video-Zero `-0.0336`, SoRA-Clip `-0.0096`, and Crafter
`-0.0060`. Relative to unified K1, U0 improves 15/20 generators.

## 5. Calibration size and seed stability

The reserve is disjoint from both the locked calibration and evaluation sets. Five fixed
seeds use source-stratified, nested real-only samples.

| Reals per dataset | Macro AUC mean +/- std | Macro AP mean +/- std |
|---:|---:|---:|
| 25 | 0.8280 +/- 0.0060 | 0.8237 +/- 0.0050 |
| 50 | 0.8495 +/- 0.0027 | 0.8487 +/- 0.0026 |
| 100 | 0.8631 +/- 0.0054 | 0.8628 +/- 0.0052 |
| 200 | 0.8712 +/- 0.0027 | 0.8680 +/- 0.0022 |

Under the declared rule of retaining 95% of the N=200 improvement over STALL, 200 is the
smallest tested sufficient size. Local Macro AP rises from 0.5193 at N=25 to 0.8175 at
N=200, while Global changes much less, so Local is the calibration-sensitive branch.

## 6. Rejected directions

| Candidate | Macro result | Decision |
|---|---:|---|
| K5 mean | 0.8663/0.8672 | Higher cost, below K3 |
| All non-overlap mean | 0.8712/0.8695 | AP tied, duration-dependent cost |
| K3 bottom-2 / hybrid | AP 0.8684 / 0.8691 | Below mean; over-penalizes GenVideo |
| Joint Typicality J2 / J3 | AP 0.8437 / 0.8602 | Below fixed 0.6/0.4 |
| Spatial-mean residual | 0.8626/0.8609 | Delta AP -0.0088, CI fully negative |
| Fine+coarse token | 0.8685/0.8687 | Does not beat fine-only |
| Late+final layer | 0.8721/0.8698 | About +0.0002 AP, CI crosses zero |
| Cross-layer min / motion gate | AP 0.8695 / 0.8692 | Both decrease Macro AP |
| D3/D4, multi-lag, hard/soft matching | Below same-grid D2 | Historical rejection retained |
| Global D3, three-branch, gate/clip/cap | Unstable or negative transfer | Not part of U0 |

The common-motion ratio is not dataset-invariant, and residual subtraction improves only
4/20 generators. Old region 2 already performs raw 14x14-to-7x7 token pooling before D2,
so coarse D2 was not accidentally omitted. No failed component is stacked into U0.

## 7. Cross-dataset calibration

Only the real calibration bank changes; U0 structure, weights, evaluation videos, and metric
protocol are fixed.

| Real bank | ComGenVid | VideoFeedback | GenVideo | Macro-3 |
|---|---:|---:|---:|---:|
| ComGenVid-200 | 0.8968/0.9064 | 0.8367/0.8510 | 0.8537/0.8365 | 0.8624/0.8646 |
| VideoFeedback-200 | 0.8683/0.8726 | 0.8597/0.8689 | 0.8648/0.8377 | 0.8643/0.8597 |
| GenVideo-200 | 0.8572/0.8679 | 0.8098/0.8277 | 0.8657/0.8416 | 0.8442/0.8457 |
| Pooled-200 | 0.8855/0.8966 | 0.8324/0.8477 | 0.8662/0.8440 | 0.8614/0.8628 |
| Pooled-600 | 0.8887/0.8998 | 0.8365/0.8497 | 0.8640/0.8413 | 0.8631/0.8636 |

The target-domain diagonal mean final AP is `0.8723`; mean single-source off-domain AP is
`0.8489`. Global target/off-domain AP is `0.8486/0.8475`, whereas Local is
`0.8292/0.7425`: Local is substantially more domain-dependent. Pooled-600 closes part of
the gap but remains `-0.0087 AP` below the target-domain diagonal. U0 therefore requires
target-domain real calibration for its strongest claim.

## 8. OAS covariance candidate

OAS changes only Local D2 covariance estimation and uses no tuned shrinkage strength. It
changes Macro AUC/AP from `0.874072/0.872299` to `0.874123/0.872342`: AP delta
`+0.000043`, 95% CI `[-0.000089,+0.000174]`. It has 13/20 generator non-declines, but
does not meet the required `+0.003 AP`; five-seed AP standard deviation increases from
`0.002210` to `0.002245`. OAS is rejected, stable S0 remains U0, and covariance search is
closed.

## 9. Robustness

The predeclared generator-balanced evaluation subset contains 1,600 videos and uses all 600
locked calibration reals. All 60,780 condition/window rows completed without failure and R0
reproduces locked raw scores exactly. Whitening, U0 structure, and weights remain fixed. B is
condition-matched CDF recalibration, not covariance or whitening refitting.

| Condition | A: original CDF -> perturbed AP | B: matched CDF AP | C: perturbed CDF -> clean AP |
|---|---:|---:|---:|
| Original | 0.8808 | 0.8808 | 0.8808 |
| H.264 CRF 23 | 0.8784 (-0.0024) | 0.8781 | 0.8794 |
| H.264 CRF 35 | 0.8611 (-0.0196) | 0.8584 | 0.8795 |
| Half-resolution restore | 0.8710 (-0.0097) | 0.8706 | 0.8797 |
| Drop 10% / 25% | 0.8783 / 0.8757 | 0.8780 / 0.8751 | 0.8803 / 0.8785 |
| Repeat 10% / 25% | 0.8730 / 0.8582 | 0.8724 / 0.8600 | 0.8782 / 0.8627 |
| Synthetic scene cut | 0.9377 (+0.0569) | 0.9325 | 0.8753 |
| 4 FPS diagnostic | 0.8723 (-0.0085) | 0.8684 | 0.8755 |

Scenario-A CRF23 and 10% frame-drop AP deltas are `-0.0024` and `-0.0024`, with 95%
paired cluster-bootstrap intervals `[-0.0071,+0.0032]` and `[-0.0057,+0.0006]`.
These are limited observed changes, not equivalence claims. CRF35, resize, drop25,
repeat25, and 4 FPS have fully negative AP intervals; repeat25 is
`-0.0226 [-0.0367,-0.0099]`. Matched CDFs do not recover these losses, so the shift is
not a marginal calibration offset alone. In Scenario A, repeat-25 raises mean real/fake scores by
`+0.1405/+0.1364`, lowers rank correlation to `0.9365/0.7632`, and trades fewer real false
alarms for more fake misses. The worst fixed-threshold Macro balanced accuracy is `0.6328`
under perturbed-repeat25 calibration applied to clean input, with real-FP/fake-FN
`0.7067/0.0278`. Scene-cut AP improves because separation increases on the balanced subset;
the real-only injection result shows this is not a monotonic per-video anomaly response.

## 10. Synthetic anomaly localization

The locked analysis uses 300 real videos. The fixed central 25% patch region receives local
freeze, repetition, brightness flicker, affine jitter, or temporal shift; full-frame freeze
and a synthetic scene cut are controls. Target K3 windows and exact temporal/patch masks were
fixed before scoring. Positive score drops mean that an injection makes the video look less
real.

| Injection | K3 Global drop | K3 Local drop | K3 final drop | Patch-time AUPRC |
|---|---:|---:|---:|---:|
| Local freeze (4 frames) | +0.0021 | +0.0027 | +0.0023 | 0.1558 |
| Local repeat | +0.0031 | -0.0007 | +0.0016 | 0.0670 |
| Local flicker | +0.0089 | -0.0153 | -0.0008 | 0.1032 |
| Local affine jitter | -0.0029 | -0.0106 | -0.0060 | 0.1030 |
| Local temporal shift | -0.0045 | -0.0023 | -0.0036 | 0.1245 |
| Full-frame freeze | -0.0055 | -0.4084 | -0.1667 | 0.3424 |
| Synthetic scene cut | -0.0182 | -0.0387 | -0.0264 | -- |

The result is mixed and mostly negative as a mechanism test. Local D2 weakly responds in the
expected direction to local freeze, but not consistently to the other edits. Full-frame
freeze sharply *increases* Local realness because it suppresses second-order token motion;
scene cuts also increase realness on average. Localization is above random only weakly for
some freeze/shift cases and is not a semantic artifact detector. K1 misses the injected native
frames entirely for 165/300 videos, confirming the coverage problem, but K3 merely exposes
more edits and does not guarantee the correct score direction. Maximum R0 score reproduction
error is zero. These findings limit the interpretation of Local D2 to calibrated distributional
evidence on natural videos rather than monotonic response to arbitrary synthetic corruption.

## 11. Locked external validation

The GenVidBench confirmation set contains 199 disjoint VRIPT real calibration videos and
900 evaluation videos: 300 real, 300 ModelScope, and 300 Pika. External Local parameters
use calibration reals only; no alpha, beta, region, layer, K, or aggregation search occurs.

| Method | Pairwise AUC | Real-positive AP | Fake-positive AP |
|---|---:|---:|---:|
| Original STALL | 0.7921 | 0.8043 | 0.7663 |
| Clean K1 | 0.8270 | 0.8369 | 0.8021 |
| **Locked U0 K3** | **0.8410** | **0.8495** | **0.8203** |

U0 minus STALL is `+0.0488 AUC / +0.0460 AP`, with 95% CIs
`[+0.0310,+0.0672] / [+0.0297,+0.0634]`. U0 minus clean K1 is
`+0.0139/+0.0126`; the AP CI `[+0.0017,+0.0247]` is positive, while the AUC CI
`[-0.0007,+0.0294]` crosses zero. ModelScope AUC is effectively tied with K1 but AP rises;
Pika improves clearly. Effective-K is `{1:303, 2:2, 3:595}`. Text2Video-Zero remains
excluded because native 4 FPS cannot provide 16 distinct frames under the strict protocol.

## 12. Current complexity decision

K3 uses 787,690 unique evaluation frames versus 342,736 for K1 (`2.30x`). K5 and all-window
would use `3.21x/4.02x` without AP improvement. The admitted complexity is therefore one
Local D2 score from the same DINO forward plus three-window coverage. Residual, multiscale,
intermediate-layer, Joint, and covariance candidates do not enter the release unless their
predeclared gate is met.

The manuscript main line must contain Original STALL, historical clean single-window, locked
U0, the unified Global/Local/K3 ablation, N=25/50/100/200 calibration stability, and the
locked confirmation experiments. Historical tuned and leakage-affected results belong only
in protocol audit/supplementary material. Failed structure searches belong in one compact
supplementary table.

## 13. Answers to the final questions

1. **Stable U0:** ComGenVid `0.8968/0.9064`, VideoFeedback `0.8597/0.8689`,
   GenVideo `0.8657/0.8416`, and Macro-3 `0.8741/0.8723`.
2. **AP orientation:** the headline is real-positive AP with high-is-real `S`. Fake-positive
   AP uses fake label 1 and `1-S` and is reported separately.
3. **Source of gain:** both branches matter. Local adds `+0.0237 AP` to K3 Global; K3 adds
   `+0.0087 AP` to unified K1. Local is the larger incremental gain, while K3 fixes coverage.
4. **200 real videos:** it is the smallest tested size meeting the declared 95% criterion.
   Five-seed N=200 Macro AP is `0.8680 +/- 0.0022`; calibration split uncertainty remains.
5. **Target-domain calibration:** yes for the strongest claim. Target-domain AP is `0.8723`,
   mean single-source off-domain AP `0.8489`; Local is the domain-sensitive branch.
6. **Robustness:** mild CRF23/drop10 are stable within `0.003 AP`; CRF35, resize, repeat25,
   and 4 FPS lose about `0.020`, `0.010`, `0.023`, and `0.0085 AP`. CDF matching is insufficient.
7. **Localization:** not reliably. Local freeze has weak positive response, but several edits
   have the wrong sign and full-frame freeze raises Local realness sharply.
8. **External transfer:** yes with new-domain real calibration. On GenVidBench U0 improves AP
   by `+0.0460` over STALL and `+0.0126` over clean K1, with positive paired AP intervals.
9. **OAS:** no. Its `+0.000043 AP` interval crosses zero and seed variance slightly increases.
10. **Complexity:** K3 uses `2.30x` K1 frames for a bootstrap-positive `+0.0087 AP`; this is
    justified for the measured coverage gain, but not for latency-critical or severe-shift
    deployment. The release is self-contained at about 57 MB, excluding backbone and videos.
11. **Excluded history:** leakage-affected `0.8737/0.8750` is invalid. Historical
    `0.8694/0.8697` is leakage-free but fake-label selected and remains supplementary only.
12. **Paper allocation:** the main line uses Original STALL, historical clean K1, locked U0,
    unified A0--A10, calibration size, target/cross calibration, external validation, and the
    locked robustness/localization boundaries. Failed structure searches and historical
    tuned/leakage audits remain in compact supplementary tables.

## 14. Runtime and storage

- The clean-from-empty K3 traversal took approximately 52 minutes on two RTX 5090 GPUs;
  locked K1 calibration references took under one minute.
- The 300-video injection scoring stage took about 60 minutes. The full 600-calibration plus
  1,600-evaluation robustness run used eight resumable shards on two RTX 5090 GPUs and took
  approximately 2 hours 58 minutes wall time; its analysis stage is CPU-only and short.
- External parameter fitting and scoring ran concurrently with other jobs; the observed
  end-to-end wall interval was about 83 minutes and is a contention-affected measurement.
- `release/u0/` is approximately 57 MB and includes the three locked Local parameter files,
  manifests, frame indices, hashes, and final scores. It excludes the DINO checkpoint, source
  videos, and feature caches.
- The retained local development outputs for clean-universal, reproduction, calibration,
  cross/OAS, external, injection, and robustness total about 5.6 GB. These caches and raw
  score tables are intentionally excluded from Git.
