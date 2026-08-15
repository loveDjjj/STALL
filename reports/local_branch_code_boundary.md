# Local branch code boundary

## Decision

Locked Alpha-STALLED Local scoring is defined only by
`src/alpha_stalled/local_branch.py` and its numerical dependency
`src/alpha_stalled/whitening.py`. The formal surface is deliberately narrow:

```text
Patch Spatial: raw patch tokens -> fixed Gaussian -> mean likelihood
Patch Temporal: same-grid D2 -> feature L2 normalization -> fixed Gaussian -> mean likelihood
Local: 0.1 * calibrated Patch Spatial + 0.9 * calibrated Patch D2
```

`temporal_order=1` is exposed solely for the registered controlled D1 ablation.
No formal Local API accepts generator labels, target scores, region size, matching mode,
multi-lag settings, residual mode, or derivative orders above two.

## Canonical symbols

| Symbol | Role |
|---|---|
| `local_d1_features` | Controlled D1 ablation feature |
| `local_d2_features` | Locked same-grid D2 feature |
| `score_local_raw` | Region1/mean float64 raw Local scoring |
| `fuse_local_components` | Fixed real-calibrated `0.1/0.9` fusion |
| `LocalRawScores` | Explicit spatial/temporal raw score result |

`alpha_stalled.calibration` re-exports `LOCAL_SPATIAL_WEIGHT` and
`fuse_local_components` for compatibility, but does not define a second formula.

## Experimental compatibility boundary

`src/patch_matching.py` remains available to reproduce development experiments. It contains:

- hard/soft local matching and matching diagnostics;
- raw-token region pooling;
- multi-lag combinations;
- spatial/global common-mode residuals;
- third- and fourth-order finite differences.

These variants are not imported by locked U0, its K1 control, the duration-aware extension,
or the robustness scorer. Current direct consumers are generic parameter fitting and explicit
audit/rejected-experiment tools: `create_patch_params.py`, `fit_intermediate_layer_params.py`,
`fit_clean_universal_layer17.py`, `fit_u0_cross_and_oas_params.py`,
`audit_patch_likelihood_assumptions.py`, and `audit_videofeedback_whitening.py`.

`src/patch_math.py` similarly remains a generic aggregation compatibility helper.
Its percentile function now delegates to the canonical right-inclusive ECDF. Locked U0 uses
region1/mean and therefore does not use its bottom-k implementation.

## Verification

`tests/test_local_branch.py` proves:

- D1/D2 are exactly the shared numerical primitives;
- `score_local_raw` is bit-identical to the direct float64 mean-likelihood computation;
- the Local fusion function is shared with calibration;
- the formal public surface excludes matching, residual, pooling, and D3/D4 controls;
- locked U0, K1, and duration-aware K1 import the canonical raw scorer;
- the legacy percentile helper uses the canonical tie policy.

The full release verifier must continue to report Macro AUC/AP
`0.8740724396350226/0.8722992273320395` after any Local refactor.
