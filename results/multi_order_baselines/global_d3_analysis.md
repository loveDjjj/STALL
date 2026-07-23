# Global D3 calibration analysis

## Protocol

All variants use the Stage-1 strict 2 s / 8 FPS video intersection. G0 is original STALL. G1-G4 use the fixed structured weights `0.5 spatial + 0.25 first-order temporal + 0.25 second-order temporal`; no weight is selected on a target test set.

- G1: one-sided real-CDF percentile of raw D3 volatility.
- G2: two-sided real-CDF realness.
- G3: two-sided realness conditioned on five real-calibration motion bins.
- G4: G3 after native timestamp normalization of velocity and acceleration.

## Dataset macro metrics

| ID | Configuration | ComGenVid AUC/AP | VideoFeedback AUC/AP | GenVideo AUC/AP | Macro-3 AUC/AP |
|---|---|---:|---:|---:|---:|
| G0 | Original STALL | 0.8550/0.8606 | 0.8528/0.8640 | 0.8085/0.8036 | 0.8388/0.8428 |
| G1 | STALL + raw D3 | 0.8671/0.8720 | 0.7537/0.7974 | 0.8190/0.8277 | 0.8132/0.8324 |
| G2 | STALL + two-sided D3 | 0.8486/0.8506 | 0.7838/0.8160 | 0.7425/0.7569 | 0.7916/0.8078 |
| G3 | STALL + motion-conditioned D3 | 0.8271/0.8397 | 0.7829/0.8166 | 0.7300/0.7583 | 0.7800/0.8049 |
| G4 | STALL + time-normalized conditional D3 | 0.8238/0.8383 | 0.7829/0.8166 | 0.7487/0.7710 | 0.7852/0.8086 |

## Cross-generator wins

- `G1` vs G0: AP improves on 10/20 generators; AUC improves on 9/20.
- `G2` vs G0: AP improves on 3/20 generators; AUC improves on 4/20.
- `G3` vs G0: AP improves on 1/20 generators; AUC improves on 1/20.
- `G4` vs G0: AP improves on 0/20 generators; AUC improves on 0/20.

## GenVideo AP retention

| Variant | Delta AP vs G0 | Retention relative to raw G1 |
|---|---:|---:|
| G1 | +0.0241 | 100.0% |
| G2 | -0.0467 | -193.7% |
| G3 | -0.0453 | -187.9% |
| G4 | -0.0326 | -135.4% |

## Paired bootstrap

| Dataset | Variant | Metric | Delta | 95% CI |
|---|---|---|---:|---:|
| comgenvid | G1-G0 | AUC | +0.0120 | [+0.0038, +0.0202] |
| comgenvid | G1-G0 | AP | +0.0114 | [+0.0042, +0.0187] |
| comgenvid | G2-G0 | AUC | -0.0064 | [-0.0149, +0.0023] |
| comgenvid | G2-G0 | AP | -0.0101 | [-0.0176, -0.0023] |
| comgenvid | G3-G0 | AUC | -0.0279 | [-0.0372, -0.0177] |
| comgenvid | G3-G0 | AP | -0.0209 | [-0.0293, -0.0130] |
| comgenvid | G4-G0 | AUC | -0.0312 | [-0.0409, -0.0215] |
| comgenvid | G4-G0 | AP | -0.0223 | [-0.0305, -0.0144] |
| videofeedback | G1-G0 | AUC | -0.0990 | [-0.1091, -0.0892] |
| videofeedback | G1-G0 | AP | -0.0666 | [-0.0737, -0.0596] |
| videofeedback | G2-G0 | AUC | -0.0690 | [-0.0787, -0.0596] |
| videofeedback | G2-G0 | AP | -0.0481 | [-0.0550, -0.0414] |
| videofeedback | G3-G0 | AUC | -0.0698 | [-0.0787, -0.0607] |
| videofeedback | G3-G0 | AP | -0.0475 | [-0.0543, -0.0403] |
| videofeedback | G4-G0 | AUC | -0.0698 | [-0.0801, -0.0603] |
| videofeedback | G4-G0 | AP | -0.0475 | [-0.0548, -0.0397] |
| genvideo | G1-G0 | AUC | +0.0104 | [+0.0013, +0.0190] |
| genvideo | G1-G0 | AP | +0.0241 | [+0.0167, +0.0304] |
| genvideo | G2-G0 | AUC | -0.0661 | [-0.0768, -0.0555] |
| genvideo | G2-G0 | AP | -0.0467 | [-0.0547, -0.0393] |
| genvideo | G3-G0 | AUC | -0.0786 | [-0.0901, -0.0677] |
| genvideo | G3-G0 | AP | -0.0453 | [-0.0534, -0.0382] |
| genvideo | G4-G0 | AUC | -0.0598 | [-0.0708, -0.0503] |
| genvideo | G4-G0 | AP | -0.0326 | [-0.0411, -0.0252] |

## Decision

Raw G1 retains a GenVideo AP gain of +0.0241, but transfers negatively to VideoFeedback (-0.0666 AP) and lowers Macro-3 AP by -0.0104. The calibrated G2-G4 variants do not preserve the raw GenVideo gain and do not resolve VideoFeedback negative transfer. None satisfies the admission criteria, so global D3 remains an ablation rather than a default branch.
