# Locked U0 robustness

The generator-balanced subset and every deterministic perturbation were locked before any perturbation metric was computed. U0 features, whitening, alpha, beta, K, and sampling remain fixed.

A uses original calibration with perturbed test input; B applies the same perturbation to the real calibration references and test input; C uses perturbed calibration references on original test input.

## A_original_calibration_to_perturbed_test

| Condition | Global Macro AUC/AP | Local Macro AUC/AP | Final Macro AUC/AP | Final AP delta |
|---|---:|---:|---:|---:|
| R0_original | 0.8532/0.8501 | 0.8472/0.8410 | 0.8859/0.8808 | +0.0000 |
| R1_h264_crf23 | 0.8551/0.8503 | 0.8440/0.8385 | 0.8827/0.8784 | -0.0024 |
| R2_h264_crf35 | 0.8301/0.8255 | 0.8442/0.8384 | 0.8659/0.8611 | -0.0196 |
| R3_resize_half_restore | 0.8425/0.8352 | 0.8430/0.8387 | 0.8792/0.8710 | -0.0097 |
| R4_drop10 | 0.8471/0.8446 | 0.8466/0.8402 | 0.8829/0.8783 | -0.0024 |
| R5_drop25 | 0.8441/0.8405 | 0.8527/0.8460 | 0.8814/0.8757 | -0.0051 |
| R6_repeat10 | 0.8521/0.8482 | 0.8250/0.8131 | 0.8814/0.8730 | -0.0077 |
| R7_repeat25 | 0.8482/0.8456 | 0.7643/0.7661 | 0.8601/0.8582 | -0.0226 |
| R8_scene_cut | 0.9221/0.9159 | 0.9082/0.9053 | 0.9436/0.9377 | +0.0569 |
| R9_4fps | 0.8356/0.8351 | 0.8519/0.8500 | 0.8771/0.8723 | -0.0085 |

## B_matched_perturbed_calibration_and_test

| Condition | Global Macro AUC/AP | Local Macro AUC/AP | Final Macro AUC/AP | Final AP delta |
|---|---:|---:|---:|---:|
| R0_original | 0.8532/0.8501 | 0.8472/0.8410 | 0.8859/0.8808 | +0.0000 |
| R1_h264_crf23 | 0.8546/0.8504 | 0.8430/0.8390 | 0.8830/0.8781 | -0.0026 |
| R2_h264_crf35 | 0.8283/0.8236 | 0.8409/0.8369 | 0.8652/0.8584 | -0.0223 |
| R3_resize_half_restore | 0.8410/0.8346 | 0.8398/0.8358 | 0.8777/0.8706 | -0.0101 |
| R4_drop10 | 0.8485/0.8453 | 0.8446/0.8399 | 0.8829/0.8780 | -0.0028 |
| R5_drop25 | 0.8443/0.8412 | 0.8474/0.8417 | 0.8804/0.8751 | -0.0056 |
| R6_repeat10 | 0.8519/0.8479 | 0.8201/0.8105 | 0.8773/0.8724 | -0.0084 |
| R7_repeat25 | 0.8464/0.8444 | 0.7531/0.7525 | 0.8605/0.8600 | -0.0207 |
| R8_scene_cut | 0.9222/0.9149 | 0.8826/0.8789 | 0.9375/0.9325 | +0.0518 |
| R9_4fps | 0.8326/0.8323 | 0.8492/0.8466 | 0.8702/0.8684 | -0.0123 |

## C_perturbed_calibration_to_original_test

| Condition | Global Macro AUC/AP | Local Macro AUC/AP | Final Macro AUC/AP | Final AP delta |
|---|---:|---:|---:|---:|
| R0_original | 0.8532/0.8501 | 0.8472/0.8410 | 0.8859/0.8808 | +0.0000 |
| R1_h264_crf23 | 0.8528/0.8496 | 0.8466/0.8410 | 0.8844/0.8794 | -0.0013 |
| R2_h264_crf35 | 0.8535/0.8500 | 0.8475/0.8432 | 0.8855/0.8795 | -0.0013 |
| R3_resize_half_restore | 0.8532/0.8495 | 0.8422/0.8367 | 0.8843/0.8797 | -0.0010 |
| R4_drop10 | 0.8526/0.8496 | 0.8467/0.8402 | 0.8854/0.8803 | -0.0005 |
| R5_drop25 | 0.8530/0.8503 | 0.8453/0.8393 | 0.8839/0.8785 | -0.0023 |
| R6_repeat10 | 0.8530/0.8498 | 0.8357/0.8320 | 0.8810/0.8782 | -0.0026 |
| R7_repeat25 | 0.8527/0.8494 | 0.7702/0.7681 | 0.8621/0.8627 | -0.0181 |
| R8_scene_cut | 0.8526/0.8486 | 0.8218/0.8157 | 0.8800/0.8753 | -0.0055 |
| R9_4fps | 0.8528/0.8497 | 0.8413/0.8353 | 0.8792/0.8755 | -0.0052 |

## Scenario A score and fixed-threshold stability

Score shifts and Spearman correlations are averaged equally across the three datasets within each real/fake subset. The threshold is frozen at 0.5.

| Condition | Balanced acc. | Real FP | Fake FN | Real mean shift | Fake mean shift | Real/Fake Spearman |
|---|---:|---:|---:|---:|---:|---:|
| R0_original | 0.6953 | 0.5600 | 0.0493 | +0.0000 | +0.0000 | 1.0000/1.0000 |
| R1_h264_crf23 | 0.7099 | 0.5333 | 0.0468 | +0.0014 | +0.0001 | 0.9861/0.9650 |
| R2_h264_crf35 | 0.7185 | 0.5000 | 0.0630 | +0.0241 | +0.0264 | 0.9714/0.8877 |
| R3_resize_half_restore | 0.6746 | 0.6000 | 0.0508 | -0.0125 | -0.0020 | 0.9807/0.9728 |
| R4_drop10 | 0.7057 | 0.5400 | 0.0487 | +0.0087 | +0.0065 | 0.9959/0.9895 |
| R5_drop25 | 0.7301 | 0.4867 | 0.0532 | +0.0202 | +0.0133 | 0.9920/0.9822 |
| R6_repeat10 | 0.7197 | 0.5000 | 0.0607 | +0.0325 | +0.0256 | 0.9768/0.9290 |
| R7_repeat25 | 0.7657 | 0.3533 | 0.1153 | +0.1405 | +0.1364 | 0.9365/0.7632 |
| R8_scene_cut | 0.7343 | 0.5000 | 0.0313 | +0.0483 | -0.0056 | 0.6621/0.6206 |
| R9_4fps | 0.7494 | 0.4400 | 0.0612 | +0.0626 | +0.0393 | 0.9771/0.9481 |

## Integrity and fixed threshold

- R0 raw-score maximum reproduction error: `0`.
- Worst Macro balanced accuracy at threshold 0.5 is `0.6328` for `C_perturbed_calibration_to_original_test__R7_repeat25__S`; real FP/fake FN rates are `0.7067/0.0278`.
- Per-dataset/generator metrics, score shifts, correlations, and threshold errors are stored as CSV.
- Mild CRF 23 and 10% frame drop change Scenario-A final AP by less than 0.003; CRF 35 and 25% frame repetition lower it by about 0.020 and 0.023.
- Condition-matched CDF recalibration does not recover the severe shifts, so their loss is not explained by a one-dimensional calibration offset alone.
- Scene-cut AP improves on this balanced detection subset, but the real-only injection study shows that this must not be interpreted as a monotonic per-video anomaly response.
