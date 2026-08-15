# Global branch code boundary

## Decision

Locked Alpha-STALLED Global scoring is defined by
`src/alpha_stalled/global_branch.py` and the shared float64 numerical primitives in
`src/alpha_stalled/whitening.py`:

```text
Global Spatial: frame CLS -> fixed Gaussian -> max likelihood
Global T1: lag-1 CLS difference -> feature L2 normalization
           -> exact-zero mask -> fixed Gaussian -> min likelihood
Global: 0.5 * calibrated Global Spatial + 0.5 * calibrated Global T1
```

Exact-zero temporal differences are set to positive infinity before min aggregation,
matching original STALL behavior without allowing a zero vector to become an anomaly.

## Canonical symbols

| Symbol | Role |
|---|---|
| `global_t1_features` | Normalized lag-1 features and exact-zero mask |
| `score_global_raw` | Float64 max/min raw Global scoring |
| `fuse_global_components` | Fixed real-calibrated `0.5/0.5` fusion |
| `GlobalRawScores` | Explicit spatial/T1 raw score result |

`alpha_stalled.calibration` re-exports `GLOBAL_SPATIAL_WEIGHT` and
`fuse_global_components`, but does not define a second Global formula.

## Original STALL compatibility boundary

`src/stall.py` remains the original detector and DINOv3 global-feature extraction API.
Its numpy likelihood implementation is required for original STALL compatibility and historical
entry points. Locked U0, K1, duration-aware, and robustness raw scoring instead use the shared
float64 `score_global_raw` path so outer batching cannot alter a window score.

## Rejected and diagnostic variants

Global D3, two-sided D3, motion-conditioned D3, time-normalized D3, volatility, and
higher-order combinations remain in explicit experiment tools such as
`tools/build_multi_order_baselines.py`, `tools/eval_global_second_order_volatility.py`, and
the D3 protocol/evaluation utilities. They are absent from `global_branch.__all__` and are not
imported by the locked scorers. Their registered outcomes remain rejected or diagnostic; this
refactor does not promote them into the method.

## Verification

`tests/test_global_branch.py` proves:

- T1 is exactly the shared normalized lag-1 primitive;
- the zero-difference mask is retained and an all-zero T1 window scores `+inf`;
- `score_global_raw` is bit-identical to direct float64 max/min likelihood scoring;
- calibration and the Global module share one fusion function;
- locked U0, K1, duration-aware K1, and robustness import the canonical scorer;
- D3 and volatility controls are absent from the formal public surface.

The full release verifier must remain at Macro AUC/AP
`0.8740724396350226/0.8722992273320395` after any Global refactor.
