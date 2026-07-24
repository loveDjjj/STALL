# Alpha-STALLED local residual and multiscale final report

Date: 2026-07-24

## Executive decision

The frozen method remains K=3 MW2:

```text
G_k = 0.5 * global_spatial_k + 0.5 * global_T1_k
L_k = 0.1 * patch_spatial_k + 0.9 * same_grid_patch_D2_k
G(V), L(V) = effective-K-matched real recalibration of the K-window means
S(V) = 0.6 * G(V) + 0.4 * L(V)
```

Spatial-mean residual D2 (R1) is rejected. It lowers Macro-3 AUC/AP from
`0.8694/0.8697` to `0.8626/0.8609`; the AP delta is `-0.00878`, with 95% paired
bootstrap CI `[-0.01034,-0.00729]`. Only 4/20 generators do not decline in AP.

The predeclared gates therefore stop the remaining search:

- R3 raw/residual score fusion is not run because R1 is not admitted;
- coordinate-median R2 is not rerun because the earlier leakage-free L2 experiment already
  rejected it;
- Stage 3 is not run because old `region=2` is already raw-token 2x2 pooling before D2;
- intermediate-layer D2 and beta pruning are not run because no new temporal candidate
  passes an earlier gate.

No failed component is stacked into the final method. The final detector still has only a
Global branch and a Local branch.

## Protocol and implementation

- Evaluation is the fixed 21,421-video intersection: 4,298 ComGenVid, 3,500
  VideoFeedback, and 13,623 GenVideo videos.
- Each dataset uses its fixed 200-real calibration set. Evaluation real videos and generated
  videos do not enter whitening, CDFs, weights, or thresholds.
- R1 is fitted from the same K1 calibration caches as R0. The operation order is frozen
  region pooling, D2, spatial-mean subtraction, feature-wise L2 normalization, independent
  real-only whitening, Gaussian likelihood, frozen aggregation, and an independent real
  window CDF.
- R1 then uses the exact K=3 windows, beta `0.1`, MW2 branch aggregation, effective-K
  video recalibration, and alpha `0.6`.
- The common-motion diagnostics use unpooled `[T-2,196,1024]` acceleration. They are
  diagnostics, not the scored tensor for ComGenVid/GenVideo, whose frozen region sizes are
  3/2 respectively.
- Every residual window key and frame-index list is matched one-to-one to the frozen K=3
  score file. There are 58,496 windows including calibration and 21,421 evaluation videos,
  with zero duplicate keys, missing rows, decoding failures, or non-finite outputs.
- R0 is read in frozen protocol order and exactly reproduces the published K=3 metrics.

## Stage 0 audit decisions

The detailed code audit is in `reports/local_residual_multiscale_audit.md`. Its decisive
findings are:

1. D2 is exactly `P[t+2]-2P[t+1]+P[t]`, followed by L2 normalization.
2. Patch tensors exclude CLS and four DINO storage/register tokens; the 196 patches retain
   row-major 14x14 order.
3. Old region 1/2/3 performs non-overlapping raw-token averaging before D2. Proposed coarse
   14x14-to-7x7 D2 is therefore the existing region-2 experiment, not a new method.
4. Coordinate-median and CLS-global residuals already failed. Spatial-mean residual was the
   only new eligible residual.
5. DINOv3 intermediate layers are technically accessible in one traversal; layer 23 matched
   the final cached token with maximum error 0.0. The experiment remains gated off.

## R1 results

| Dataset | R0 K=3 MW2 AUC/AP | R1 mean-residual AUC/AP | Delta AP |
|---|---:|---:|---:|
| ComGenVid | 0.8857/0.8996 | 0.8836/0.8954 | -0.0042 |
| VideoFeedback | 0.8623/0.8684 | 0.8495/0.8537 | -0.0147 |
| GenVideo | 0.8601/0.8410 | 0.8547/0.8336 | -0.0074 |
| **Macro-3** | **0.8694/0.8697** | **0.8626/0.8609** | **-0.0088** |

Paired bootstrap is negative on every dataset. VideoFeedback AP has CI
`[-0.01664,-0.01257]`, GenVideo `[-0.01070,-0.00440]`, and ComGenVid
`[-0.00646,-0.00203]`. The Macro AUC CI is also fully negative:
`[-0.00792,-0.00564]`.

The only AP non-declines are GenVideo ModelScope `+0.0070`, VideoFeedback Fast-SVD
`+0.0015`, GenVideo WildScrape `+0.0008`, and VideoFeedback Pika `+0.00005`. Largest
losses are Text2Video-Zero `-0.0481`, VideoFeedback ModelScope `-0.0253`, LaVie-base
`-0.0211`, SoRA-Clip `-0.0170`, Crafter `-0.0169`, and LVDM `-0.0159`.

Admission result:

| Criterion | Required | Observed | Pass |
|---|---:|---:|---|
| Macro AP delta | >= +0.005 | -0.00878 | no |
| worst dataset AP delta | >= -0.005 | -0.01472 | no |
| generator non-decline | >= 12/20 | 4/20 | no |
| Macro AP CI lower bound | > 0 | -0.01034 | no |
| improve ZeroScope/Pika/SoRA/Crafter | >= 2 | 1 | no |
| high-motion real FP count | must not increase | 1365 -> 1260 | yes |

## Common-motion diagnosis

Mean unpooled common-mode energy ratio `rho_mean`:

| Dataset | Real mean | Fake mean | Interpretation |
|---|---:|---:|---|
| ComGenVid | 0.0897 | 0.0595 | real is higher |
| VideoFeedback | 0.0780 | 0.1212 | fake is higher |
| GenVideo | 0.1164 | 0.0924 | real is higher |

The relation is not dataset-invariant. It cannot support a universal rule that high common
motion is a real-video nuisance to remove.

Using the declared diagnostic operating point `S=0.5`, false-positive real videos have
`rho_mean` 0.0883 vs 0.0921 for correctly classified real videos on ComGenVid, 0.0803 vs
0.0737 on VideoFeedback, and 0.1212 vs 0.1113 on GenVideo. Thus false positives show a
modest ratio increase on two datasets but not ComGenVid. Their absolute motion magnitude is
lower than correctly classified real videos on all three datasets, so current false positives
do not generally concentrate in high absolute camera motion.

False-negative generated videos have higher absolute motion than correctly classified fakes
on all datasets: 5.57 vs 3.73 (ComGenVid), 8.30 vs 4.76 (VideoFeedback), and 6.08 vs 3.58
(GenVideo). Their `rho_mean` is also higher, most clearly on VideoFeedback (0.1475 vs
0.1207). However, R1 raises their realness more than it raises correctly detected fakes; for
VideoFeedback the mean R1-R0 change is +0.0429 on false negatives vs +0.0197 otherwise.
Common-mode subtraction therefore worsens the relevant ranking instead of exposing a weak
local residual.

For the four K=3 negative-transfer generators:

| Generator | rho mean | Dataset fake mean | R1-R0 AP |
|---|---:|---:|---:|
| ZeroScope-576w | 0.1953 | 0.1212 | -0.0068 |
| Pika | 0.0836 | 0.1212 | +0.00005 |
| SoRA-Clip | 0.0870 | 0.1212 | -0.0170 |
| Crafter | 0.0868 | 0.0924 | -0.0169 |

Only ZeroScope shows a clear high-common-motion shift. The four failures do not share one
common-motion bias, so residual subtraction is not a general correction for the K=3
negative transfers.

## Cost and retained assets

The dual RTX 5090 extraction took about 41.6 minutes wall time from the first to last
checkpoint: ComGenVid 12.7 minutes, VideoFeedback 4.7 minutes, and GenVideo 25.9 minutes
(dataset phases overlap across GPUs). It uses the same DINO forward as the diagnostic patch
extraction; R1 and `rho` add statistics but no additional backbone inference.

Local outputs before duplicate-shard cleanup occupy about 80 MB; the three independent R1
parameter files are about 8.1 MB each. Required local full tables remain under
`results/local_residual/`; version control keeps only the lightweight metrics, bootstrap,
common-mode summary, and admission decision. Summaries can be rebuilt from the retained
canonical window table with `tools/analyze_local_residual.py --residual-scores
results/local_residual/per_window_scores.csv`; DINO extraction does not need to be repeated.

No scene-cut grouping is reported for R1. The method failed before the final-combination
suite was triggered, and the workspace has no ground-truth scene-cut labels; the older
embedding-distance proxy is not relabeled as annotated scene-cut evidence.

## Answers to the final questions

1. **Does residual D2 reduce common-camera-motion false positives?** At a fixed 0.5
   diagnostic threshold it reduces high-motion real false positives by 105, but the effect is
   inconsistent across datasets and overall AUC/AP decreases significantly.
2. **Does coarse token capture large anomalies missed by fine token?** No new claim is made.
   Raw-token coarse D2 was already the old region-2 experiment; its preference is
   dataset-specific, so the duplicate Stage 3 was not rerun.
3. **Are intermediate layers better for local temporal detection?** Unknown and deliberately
   not tested because no Stage 2/3 candidate passed the gate.
4. **Is patch spatial still needed?** Yes in the frozen method. The earlier clean P0 ablation
   improved all 20 generators over temporal-only D2; no admitted new temporal model triggered
   beta pruning.
5. **Where does the gain come from?** R1 has no overall gain. Small positives occur on
   ModelScope (GenVideo), Fast-SVD, WildScrape, and Pika; the other 16 generators decline.
6. **Does it improve ZeroScope, Pika, SoRA-Clip, and Crafter?** Only Pika is statistically
   negligible positive; the other three decline.
7. **Does the final method still have only two branches?** Yes, Global and Local. No
   volatility, residual, scale, or layer branch is added.
8. **Final improvement relative to STALL, clean single-window, and K=3 MW2?** The retained
   K=3 MW2 is +0.0306/+0.0269 Macro AUC/AP over original STALL and +0.0123/+0.0096 over
   clean single-window Alpha-STALLED. It is unchanged relative to itself. R1 would be
   -0.0068/-0.0088 relative to K=3 MW2 and is rejected.
9. **Is the complexity worthwhile?** K=3 MW2 remains worthwhile. Residual D2 is not;
   multiscale and intermediate-layer complexity is not admitted.
10. **Are conclusions stable over five calibration seeds?** No new candidate reached the
    predeclared final-combination gate, so the five-seed final rerun was correctly not
    triggered. R1 failure is consistent across all three datasets and has a fully negative
    1,000-bootstrap Macro CI, but it should not be mislabeled as a five-calibration-seed
    result.

## Final retained method

Retain K=3 MW2 exactly as frozen in `configs/alpha_stalled_multi_window.yaml`. Reject R1,
reuse the historical rejection of R2, and stop residual, new scale, intermediate-layer, and
beta exploration under the declared gates. The valid final Macro-3 AUC/AP remains
`0.8694/0.8697`.
