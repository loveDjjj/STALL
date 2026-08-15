# Locked U0 protocol and evaluation code boundary

## Decision

The locked U0 scorer, analyzer, and release verifier are CLI entry points, not shared Python
libraries. Their reusable protocol and metric logic now lives under `src/alpha_stalled/`:

| Shared module | Owned behavior |
|---|---|
| `alpha_stalled/u0_protocol.py` | locked manifest identity, raw shard loading, 600 K1 references, frame-index verification, effective-K reference selection |
| `alpha_stalled/u0_scoring.py` | strict deduplicated decode, per-video DINO extraction boundary, Global/Local raw window scoring |
| `alpha_stalled/u0_analysis.py` | four-component window calibration, branch fusion, video means, effective-K video CDFs |
| `alpha_stalled/metrics.py` | balanced-real sampling, per-generator pair construction, real-positive AUC/AP, dataset/Macro-3 tables, paired bootstrap |
| `alpha_stalled/calibration.py` | sorted right-inclusive empirical CDF and locked window/video fusion |

The formal dependency direction is now:

```text
score_u0_locked_windows -> alpha_stalled.u0_protocol + u0_scoring
analyze_u0_locked       -> alpha_stalled.u0_protocol + u0_analysis + metrics
verify_u0_locked_release -> alpha_stalled.u0_protocol + metrics + release_io
```

None of those three formal CLIs imports another file from `tools/`. This is enforced with an
AST import-boundary test in `tests/test_u0_protocol.py`.

## Compatibility boundary

`tools/analyze_u0_locked.py` continues to expose `load_raw_windows`,
`load_calibration_references`, `verify_calibration_reference_windows`, and
`selected_calibration_means` as the same imported function objects. Likewise,
`tools/score_u0_locked_windows.py` exposes the shared `load_release_rows` object. Historical
callers also retain shared `decode_row`, `score_batch`, branch scorers, and required-video
resolution objects. `tools/analyze_u0_locked.py` similarly reexports the two `u0_analysis`
functions. Historical callers therefore retain their import surface while each implementation
has one owner.

Other scoring and analysis tools no longer import either formal CLI as a library. Required source
path resolution comes from `alpha_stalled.release_io`; K1 reference scoring imports decode/score
from `alpha_stalled.u0_scoring`. The repository-wide no-consumer rule is checked by AST.

`tools/build_multi_order_baselines.py` retains only thin wrappers that supply historical B0-B8
default names to shared `metric_tables` and `paired_bootstrap`. D3-specific cache loading,
conditional calibration, and variant definitions remain in that historical tool. General
analysis scripts import metrics directly from `alpha_stalled.metrics`; only D3-specific tools
still depend on the multi-order implementation.

Macro-3 cluster bootstrap and the extended binary metric/bootstrap helpers are also owned by
`alpha_stalled.metrics`; historical multi-window and metric-audit CLIs only reexport the same
function objects. Locked K1 raw-component calibration references and K3 effective-K candidate
calibration now live in `alpha_stalled.u0_analysis`, so the core and second-order ablations no
longer import scorer or analyzer CLIs as libraries.

## Metric semantics

The shared metric path preserves the locked protocol:

- real video is the positive class for AUC and AP;
- every generated source is compared against a deterministic real sample of equal size;
- real sources receive equal sampling quotas;
- generator metrics are averaged within a dataset;
- Macro-3 is the equal mean of the three dataset metrics;
- paired bootstrap resamples real and generated rows within each fixed generator pair.

The generic sorted `empirical_cdf` wrapper is owned by `alpha_stalled.calibration` and delegates
to the single right-inclusive implementation in `alpha_stalled.whitening`.

## Verification

`tests/test_metrics.py` covers pair construction, class direction, dataset/Macro-3 reduction,
and deterministic paired bootstrap. `tests/test_u0_pipeline.py` locks deduplicated frame mapping,
per-video extraction calls, raw score row construction, two-level calibration, and effective-K
reference use. `tests/test_u0_protocol.py` covers shared-object compatibility,
manifest identity binding, shard completeness, finite raw scores, calibration membership, locked
K1 indices, and the no-tool-to-tool-import rule.

The full locked release verifier still passes 61 checks with Macro AUC/AP
`0.8740724396350226/0.8722992273320395`. No released score or parameter was rewritten by this
code-boundary migration.

`tools/verify_u0_pipeline_reconstruction.py` provides the stronger optional local audit when the
retained raw evidence directories are present. It recalibrates all 58,496 windows, reconstructs
21,421 videos, and requires exact column order, values, and dtypes against
`release/u0/final_video_scores.csv`. It is read-only so the historical 61-check validation JSON
and run-manifest hashes remain unchanged.
