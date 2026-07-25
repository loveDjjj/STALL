# U0 core ablation

All new rows use locked region1+mean components and real-only calibration. A0 is the exact historical Original STALL comparator; A1-A6/A9 use the unified K1 current window, while A7/A8/A10 use locked K3 effective-K recalibration.

| ID | Configuration | ComGenVid | VideoFeedback | GenVideo | Macro-3 |
|---|---|---:|---:|---:|---:|
| A0 | Original STALL | 0.8508/0.8560 | 0.8449/0.8596 | 0.8206/0.8129 | 0.8388/0.8428 |
| A1 | K1 GlobalSpatial only | 0.8351/0.8610 | 0.7855/0.8148 | 0.8044/0.8072 | 0.8083/0.8277 |
| A2 | K1 GlobalT1 only | 0.8163/0.7944 | 0.8454/0.8494 | 0.7828/0.7698 | 0.8149/0.8045 |
| A3 | K1 calibrated Global | 0.8508/0.8537 | 0.8445/0.8569 | 0.8202/0.8099 | 0.8385/0.8402 |
| A4 | K1 PatchSpatial only | 0.7158/0.7242 | 0.6640/0.6585 | 0.7193/0.7093 | 0.6997/0.6973 |
| A5 | K1 PatchD2 only | 0.8651/0.8641 | 0.7962/0.8026 | 0.8123/0.7975 | 0.8245/0.8214 |
| A6 | K1 calibrated Local | 0.8505/0.8506 | 0.7920/0.7949 | 0.8149/0.7976 | 0.8191/0.8144 |
| A7 | K3 calibrated Global only | 0.8627/0.8694 | 0.8442/0.8582 | 0.8386/0.8181 | 0.8485/0.8486 |
| A8 | K3 calibrated Local only | 0.8914/0.8912 | 0.7828/0.7849 | 0.8328/0.8114 | 0.8357/0.8292 |
| A9 | Unified Global+Local K1 | 0.8786/0.8878 | 0.8654/0.8734 | 0.8456/0.8297 | 0.8632/0.8636 |
| A10 | Locked U0 Global+Local K3 | 0.8968/0.9064 | 0.8597/0.8689 | 0.8657/0.8416 | 0.8741/0.8723 |

## Fixed weight sensitivity

| Parameter | ComGenVid | VideoFeedback | GenVideo | Macro-3 |
|---|---:|---:|---:|---:|
| alpha_0.00 | 0.8914/0.8912 | 0.7828/0.7849 | 0.8328/0.8114 | 0.8357/0.8292 |
| alpha_0.20 | 0.9078/0.9138 | 0.8385/0.8381 | 0.8584/0.8331 | 0.8683/0.8617 |
| alpha_0.40 | 0.9051/0.9143 | 0.8541/0.8592 | 0.8667/0.8421 | 0.8753/0.8719 |
| alpha_0.60 | 0.8968/0.9064 | 0.8597/0.8689 | 0.8657/0.8416 | 0.8741/0.8723 |
| alpha_0.80 | 0.8832/0.8920 | 0.8548/0.8670 | 0.8576/0.8344 | 0.8652/0.8645 |
| alpha_1.00 | 0.8627/0.8694 | 0.8442/0.8582 | 0.8386/0.8181 | 0.8485/0.8486 |
| beta_0.00 | 0.8975/0.9069 | 0.8596/0.8684 | 0.8640/0.8405 | 0.8737/0.8719 |
| beta_0.05 | 0.8982/0.9073 | 0.8596/0.8685 | 0.8655/0.8415 | 0.8744/0.8725 |
| beta_0.10 | 0.8968/0.9064 | 0.8597/0.8689 | 0.8657/0.8416 | 0.8741/0.8723 |
| beta_0.20 | 0.8943/0.9048 | 0.8588/0.8686 | 0.8659/0.8421 | 0.8730/0.8718 |
| beta_0.50 | 0.8873/0.8996 | 0.8560/0.8677 | 0.8643/0.8429 | 0.8692/0.8701 |
| beta_1.00 | 0.8728/0.8865 | 0.8498/0.8640 | 0.8431/0.8279 | 0.8552/0.8595 |

Locked U0 remains alpha 0.6 and beta 0.1 regardless of the observed curves.

## Historical multi-window supplements

These rows used the historical dataset-specific region/aggregation configuration, not locked U0, and are included only to document completed prohibited reruns.

| Historical configuration | Macro-3 AUC/AP | Decision |
|---|---:|---|
| clean single-window | 0.8570/0.8600 | historical main ablation |
| K3 MW2 | 0.8694/0.8697 | historical tuned comparator |
| K5 MW2 | 0.8663/0.8672 | reject extra cost |
| all-window MW2 | 0.8712/0.8695 | AP tied, duration/K confounding |
| K3 Local bottom-2 | 0.8680/0.8684 | reject |
| K3 Local hybrid | 0.8687/0.8691 | reject |

## Paired cluster bootstrap

Resampling is paired by score and clustered by video ID; windows are never sampling units.

| Comparison | Metric | Delta | 95% CI |
|---|---|---:|---:|
| add_global_to_k3_local | auc | +0.0384 | [+0.0292, +0.0479] |
| add_global_to_k3_local | fake_positive_ap | +0.0483 | [+0.0345, +0.0619] |
| add_global_to_k3_local | real_positive_ap | +0.0431 | [+0.0350, +0.0513] |
| add_k3_coverage | auc | +0.0109 | [+0.0076, +0.0145] |
| add_k3_coverage | fake_positive_ap | +0.0127 | [+0.0069, +0.0193] |
| add_k3_coverage | real_positive_ap | +0.0087 | [+0.0057, +0.0120] |
| add_local_to_k3_global | auc | +0.0256 | [+0.0212, +0.0303] |
| add_local_to_k3_global | fake_positive_ap | +0.0311 | [+0.0248, +0.0370] |
| add_local_to_k3_global | real_positive_ap | +0.0237 | [+0.0191, +0.0285] |
| add_patch_spatial_to_k1_d2 | auc | -0.0054 | [-0.0082, -0.0027] |
| add_patch_spatial_to_k1_d2 | fake_positive_ap | -0.0134 | [-0.0207, -0.0063] |
| add_patch_spatial_to_k1_d2 | real_positive_ap | -0.0070 | [-0.0088, -0.0054] |
| u0_vs_original_stall | auc | +0.0353 | [+0.0300, +0.0411] |
| u0_vs_original_stall | fake_positive_ap | +0.0373 | [+0.0294, +0.0446] |
| u0_vs_original_stall | real_positive_ap | +0.0295 | [+0.0241, +0.0358] |
