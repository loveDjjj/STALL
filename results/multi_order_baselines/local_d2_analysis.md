# Local D2 residual analysis

## Protocol

Every local model is refitted on the same dataset-specific, disjoint 200-real calibration index and evaluated on the exact Stage-1 strict 2 s intersection. L0-L2 preserve each dataset's frozen region pooling and aggregation. L3 changes only aggregation to a length-2 temporal-run bottom-20% mean.

- L0: normalized same-grid patch D2.
- L1: normalized `(patch D2 - global CLS D2)`.
- L2: normalized `(patch D2 - spatial median patch D2)`.
- L3: L0 with persistent temporal aggregation.
- P0: leakage-free `0.1 local spatial + 0.9 L0`.

## Dataset macro metrics

| ID | Configuration | ComGenVid AUC/AP | VideoFeedback AUC/AP | GenVideo AUC/AP | Macro-3 AUC/AP |
|---|---|---:|---:|---:|---:|
| Ls | Local spatial | 0.7273/0.7473 | 0.6487/0.6444 | 0.7204/0.7095 | 0.6988/0.7004 |
| L0 | Same-grid D2 | 0.8386/0.8278 | 0.7807/0.7897 | 0.8101/0.7946 | 0.8098/0.8040 |
| L1 | Global-residual D2 | 0.5290/0.5288 | 0.5080/0.5077 | 0.5193/0.5193 | 0.5188/0.5186 |
| L2 | Spatial-median residual D2 | 0.7807/0.7846 | 0.7193/0.7376 | 0.7881/0.7743 | 0.7627/0.7655 |
| L3 | Persistent same-grid D2 | 0.8379/0.8271 | 0.7506/0.7663 | 0.7745/0.7520 | 0.7877/0.7818 |
| P0 | Leakage-free spatial + D2 | 0.8427/0.8554 | 0.7820/0.7926 | 0.8131/0.8006 | 0.8126/0.8162 |

## Cross-generator wins versus L0

- `L1`: AP improves on 0/20 generators; AUC improves on 0/20.
- `L2`: AP improves on 2/20 generators; AUC improves on 1/20.
- `L3`: AP improves on 1/20 generators; AUC improves on 0/20.
- `P0`: AP improves on 20/20 generators; AUC improves on 19/20.

## Paired bootstrap

| Dataset | Variant | Metric | Delta | 95% CI |
|---|---|---|---:|---:|
| comgenvid | L1-L0 | AUC | -0.3096 | [-0.3220, -0.2973] |
| comgenvid | L1-L0 | AP | -0.2990 | [-0.3109, -0.2868] |
| comgenvid | L2-L0 | AUC | -0.0579 | [-0.0681, -0.0475] |
| comgenvid | L2-L0 | AP | -0.0432 | [-0.0521, -0.0339] |
| comgenvid | L3-L0 | AUC | -0.0007 | [-0.0042, +0.0027] |
| comgenvid | L3-L0 | AP | -0.0007 | [-0.0037, +0.0022] |
| comgenvid | P0-L0 | AUC | +0.0041 | [-0.0021, +0.0098] |
| comgenvid | P0-L0 | AP | +0.0276 | [+0.0235, +0.0317] |
| genvideo | L1-L0 | AUC | -0.2908 | [-0.3028, -0.2777] |
| genvideo | L1-L0 | AP | -0.2753 | [-0.2881, -0.2633] |
| genvideo | L2-L0 | AUC | -0.0220 | [-0.0281, -0.0168] |
| genvideo | L2-L0 | AP | -0.0203 | [-0.0257, -0.0150] |
| genvideo | L3-L0 | AUC | -0.0356 | [-0.0401, -0.0308] |
| genvideo | L3-L0 | AP | -0.0426 | [-0.0469, -0.0370] |
| genvideo | P0-L0 | AUC | +0.0030 | [+0.0019, +0.0042] |
| genvideo | P0-L0 | AP | +0.0060 | [+0.0048, +0.0072] |
| videofeedback | L1-L0 | AUC | -0.2727 | [-0.2846, -0.2621] |
| videofeedback | L1-L0 | AP | -0.2820 | [-0.2943, -0.2720] |
| videofeedback | L2-L0 | AUC | -0.0615 | [-0.0655, -0.0574] |
| videofeedback | L2-L0 | AP | -0.0521 | [-0.0559, -0.0481] |
| videofeedback | L3-L0 | AUC | -0.0302 | [-0.0353, -0.0249] |
| videofeedback | L3-L0 | AP | -0.0234 | [-0.0286, -0.0180] |
| videofeedback | P0-L0 | AUC | +0.0013 | [+0.0007, +0.0020] |
| videofeedback | P0-L0 | AP | +0.0029 | [+0.0023, +0.0035] |
