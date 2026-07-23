# Multi-order baseline analysis

## Protocol

- Evaluation uses each dataset's frozen holdout index and the strict indexed 2 s / 8 FPS window (16 frames). No 1 s fallback is allowed.
- Every B0-B8 row is the one-to-one key intersection of global CLS-cache scores and the frozen release patch score CSV.
- B0-B2 are recomputed with `precomputed/stall_params_vatex_dino_v3.npz`; B3 is an empirical real-CDF percentile fitted on a disjoint 200-real dataset calibration index.
- B4 is the untuned average `0.5 * B2 + 0.5 * B3`. B7 uses the frozen local weights 0.1 spatial / 0.9 D2; B8 uses 0.6 global / 0.4 patch.
- Motion-conditioned D3 uses 5 calibration quantile bins. Its Stage-2 candidate columns are included in `per_video_scores.csv` but are not relabeled as B baselines.
- AUC/AP are higher-is-real and macro-averaged over pairwise generator comparisons. Each generator uses the same deterministic real sample for every configuration.
- Paired bootstrap resamples real and fake rows within each generator, preserves score pairing, and reports the generator-macro delta.

## Important limitation

The current workspace has no VideoFeedback patch-token cache. B5-B7 therefore reuse the frozen release per-video components, whose patch calibration used all available real videos (4080 for VideoFeedback, 1698 for ComGenVid, 9984 for GenVideo). These branches are retrospectively evaluated on holdout IDs but are not calibration-disjoint. B0-B4 and all D3 variants are reproducible from current caches; a leakage-free B5-B8 rerun requires rebuilding the VideoFeedback patch cache and refitting all three patch models on the same 200-real calibration protocol.

## Dataset macro metrics

| ID | Configuration | ComGenVid AUC/AP | VideoFeedback AUC/AP | GenVideo AUC/AP | Macro-3 AUC/AP |
|---|---|---:|---:|---:|---:|
| B0 | Global spatial | 0.8364/0.8648 | 0.8006/0.8250 | 0.7958/0.8064 | 0.8109/0.8321 |
| B1 | Global first-order temporal | 0.8234/0.8048 | 0.8492/0.8536 | 0.7655/0.7558 | 0.8127/0.8047 |
| B2 | Original STALL | 0.8550/0.8606 | 0.8528/0.8640 | 0.8085/0.8036 | 0.8388/0.8428 |
| B3 | Raw D3 percentile | 0.7421/0.7152 | 0.5245/0.5637 | 0.7218/0.7480 | 0.6628/0.6756 |
| B4 | STALL + raw D3 | 0.8279/0.8260 | 0.6630/0.7153 | 0.7950/0.8161 | 0.7620/0.7858 |
| B5 | Patch spatial | 0.8286/0.8451 | 0.7374/0.7455 | 0.7651/0.7694 | 0.7770/0.7866 |
| B6 | Patch same-grid D2 | 0.9281/0.9298 | 0.8060/0.8153 | 0.8157/0.8049 | 0.8499/0.8500 |
| B7 | Current patch | 0.9288/0.9315 | 0.8079/0.8206 | 0.8178/0.8103 | 0.8515/0.8541 |
| B8 | Current global + patch | 0.9199/0.9231 | 0.8561/0.8687 | 0.8452/0.8332 | 0.8737/0.8750 |

## Positive generator counts

- `add_global_t1` (B2-B0): 14/20 generators improve in AP.
- `add_global_spatial` (B2-B1): 17/20 generators improve in AP.
- `add_raw_d3` (B4-B2): 6/20 generators improve in AP.
- `add_local_d2` (B7-B5): 15/20 generators improve in AP.
- `add_local_spatial` (B7-B6): 13/20 generators improve in AP.
- `add_patch` (B8-B2): 17/20 generators improve in AP.
- `add_global` (B8-B7): 16/20 generators improve in AP.

## GenVideo: raw D3 contribution (B4-B2)

- `Crafter`: delta AUC +0.0416, delta AP +0.0595
- `Gen2`: delta AUC +0.0526, delta AP +0.0517
- `Sora`: delta AUC +0.0207, delta AP +0.0375
- `Show_1`: delta AUC -0.0021, delta AP +0.0349
- `WildScrape`: delta AUC -0.0154, delta AP +0.0186
- `ModelScope`: delta AUC -0.0563, delta AP -0.0138
- `MorphStudio`: delta AUC -0.0710, delta AP -0.0418
- `Lavie`: delta AUC -0.0782, delta AP -0.0469

## VideoFeedback: raw D3 failures and reversals (B4-B2)

- `Pika`: delta AUC +0.0704, delta AP +0.0573
- `Fast-SVD`: delta AUC -0.0279, delta AP -0.0188
- `VideoCrafter2`: delta AUC -0.1031, delta AP -0.0912
- `SoRA-Clip`: delta AUC -0.1491, delta AP -0.1379
- `ModelScope`: delta AUC -0.1818, delta AP -0.1483
- `AnimateDiff`: delta AUC -0.1756, delta AP -0.1657
- `LaVie-base`: delta AUC -0.2081, delta AP -0.1979
- `LVDM`: delta AUC -0.2893, delta AP -0.2156
- `ZeroScope-576w`: delta AUC -0.3474, delta AP -0.2431
- `Text2Video-Zero`: delta AUC -0.4853, delta AP -0.3266

## Patch component diagnosis

- `VEO3`: delta AUC +0.1069, delta AP +0.0999
- `Sora`: delta AUC +0.0937, delta AP +0.0730

The B7-B5 list isolates the local D2 contribution; `generator_metrics.csv` also supports B7-B6 to isolate local spatial contribution on every dataset.

## ComGenVid patch-only versus global+patch

B7 patch-only reaches AUC/AP 0.9288/0.9315; B8 reaches 0.9199/0.9231. The deltas are -0.0090 AUC and -0.0084 AP. Per-generator evidence is in the B8-B7 comparison.

## Bootstrap summary

| Dataset | Comparison | Metric | Delta | 95% CI |
|---|---|---|---:|---:|
| comgenvid | B2-B0 | AUC | +0.0186 | [+0.0114, +0.0259] |
| comgenvid | B2-B0 | AP | -0.0041 | [-0.0122, +0.0035] |
| comgenvid | B2-B1 | AUC | +0.0317 | [+0.0259, +0.0380] |
| comgenvid | B2-B1 | AP | +0.0558 | [+0.0476, +0.0641] |
| comgenvid | B4-B2 | AUC | -0.0271 | [-0.0391, -0.0147] |
| comgenvid | B4-B2 | AP | -0.0346 | [-0.0470, -0.0230] |
| comgenvid | B7-B5 | AUC | +0.1003 | [+0.0900, +0.1109] |
| comgenvid | B7-B5 | AP | +0.0864 | [+0.0758, +0.0969] |
| comgenvid | B7-B6 | AUC | +0.0008 | [-0.0006, +0.0021] |
| comgenvid | B7-B6 | AP | +0.0017 | [+0.0005, +0.0028] |
| comgenvid | B8-B2 | AUC | +0.0648 | [+0.0580, +0.0718] |
| comgenvid | B8-B2 | AP | +0.0624 | [+0.0550, +0.0694] |
| comgenvid | B8-B7 | AUC | -0.0090 | [-0.0133, -0.0048] |
| comgenvid | B8-B7 | AP | -0.0084 | [-0.0128, -0.0038] |
| videofeedback | B2-B0 | AUC | +0.0521 | [+0.0458, +0.0588] |
| videofeedback | B2-B0 | AP | +0.0390 | [+0.0331, +0.0447] |
| videofeedback | B2-B1 | AUC | +0.0035 | [-0.0014, +0.0085] |
| videofeedback | B2-B1 | AP | +0.0105 | [+0.0052, +0.0154] |
| videofeedback | B4-B2 | AUC | -0.1897 | [-0.2024, -0.1776] |
| videofeedback | B4-B2 | AP | -0.1488 | [-0.1586, -0.1376] |
| videofeedback | B7-B5 | AUC | +0.0705 | [+0.0598, +0.0804] |
| videofeedback | B7-B5 | AP | +0.0751 | [+0.0617, +0.0877] |
| videofeedback | B7-B6 | AUC | +0.0019 | [+0.0006, +0.0033] |
| videofeedback | B7-B6 | AP | +0.0053 | [+0.0033, +0.0073] |
| videofeedback | B8-B2 | AUC | +0.0034 | [-0.0040, +0.0103] |
| videofeedback | B8-B2 | AP | +0.0047 | [-0.0011, +0.0107] |
| videofeedback | B8-B7 | AUC | +0.0482 | [+0.0433, +0.0530] |
| videofeedback | B8-B7 | AP | +0.0482 | [+0.0421, +0.0541] |
| genvideo | B2-B0 | AUC | +0.0128 | [+0.0058, +0.0196] |
| genvideo | B2-B0 | AP | -0.0028 | [-0.0095, +0.0043] |
| genvideo | B2-B1 | AUC | +0.0430 | [+0.0342, +0.0525] |
| genvideo | B2-B1 | AP | +0.0478 | [+0.0400, +0.0550] |
| genvideo | B4-B2 | AUC | -0.0135 | [-0.0252, -0.0010] |
| genvideo | B4-B2 | AP | +0.0125 | [+0.0005, +0.0244] |
| genvideo | B7-B5 | AUC | +0.0527 | [+0.0435, +0.0620] |
| genvideo | B7-B5 | AP | +0.0409 | [+0.0295, +0.0538] |
| genvideo | B7-B6 | AUC | +0.0021 | [+0.0008, +0.0036] |
| genvideo | B7-B6 | AP | +0.0054 | [+0.0035, +0.0076] |
| genvideo | B8-B2 | AUC | +0.0367 | [+0.0282, +0.0458] |
| genvideo | B8-B2 | AP | +0.0296 | [+0.0228, +0.0362] |
| genvideo | B8-B7 | AUC | +0.0274 | [+0.0223, +0.0325] |
| genvideo | B8-B7 | AP | +0.0229 | [+0.0149, +0.0308] |

## Missing rows

- `comgenvid`: 0 strict-window/cache rows skipped.
- `videofeedback`: 0 strict-window/cache rows skipped.
- `genvideo`: 0 strict-window/cache rows skipped.
