# Real-only calibration-size selection

Calibration size is selected before generated-candidate scoring. For each dataset, the final nested block of real videos is held out from every candidate fit and CDF. Patch-spatial and D2 validation percentiles are compared with a uniform real-reference distribution using quantile MSE, weighted 0.1/0.9 to match the frozen Local branch.

The minimum-risk candidate is identified first. The one-standard-error rule then chooses the smallest candidate whose point risk is no greater than the minimum risk plus its bootstrap standard error.

| Dataset | Candidate risks | Held-out real | Minimum | Selected |
|---|---|---:|---:|---:|
| comgenvid | N=200: 0.085623, N=400: 0.037188, N=600: 0.015167 | 200 | 600 | **600** |
| videofeedback | N=200: 0.096285, N=300: 0.051260, N=400: 0.031334 | 100 | 400 | **400** |
| genvideo | N=200: 0.084512, N=500: 0.022431, N=1000: 0.005967, N=1500: 0.002541 | 500 | 1500 | **1500** |

## Integrity

- Generated videos used for selection: `0`.
- Selection status: `frozen_before_generated_candidate_scoring`.
- Real-only bootstrap iterations: `1000`.
- Calibration, real validation, and final evaluation identities are disjoint by role.
- ComGenVid selection uses the current DINO extractor rather than its stale compact cache; N=200 and N=800 reproduce the formal raw scorer to numerical precision.

## Frozen decision

The selected dataset-specific upper limits are ComGenVid N=600, VideoFeedback N=400, and GenVideo N=1,500. These values must not be revised after inspecting generated-video metrics.
