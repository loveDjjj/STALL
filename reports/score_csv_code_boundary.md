# Score CSV code boundary

The lightweight historical score-table workflow has one reusable implementation:

| Layer | Responsibility |
|---|---|
| `src/alpha_stalled/score_csv.py` | keyed CSV validation, strict Global/Local fusion, score direction, pairwise-balanced metric tables |
| `tools/eval_alpha_stalled.py` | single fixed-alpha fusion CLI and output files |
| `tools/eval_score_csv.py` | one-column evaluation CLI and output files |
| `tools/fuse_scores.py` | historical alpha-sweep CLI, summaries, and optional fused CSV files |

The three commands import their numerical and identity operations from the shared module. Tests
assert that the fixed-fusion and one-column CLI exports are the same function objects as the
shared API. New protocol code must import `alpha_stalled.score_csv`, not a file under `tools/`.

This module preserves the historical three-column identity
`(subset, source_model, filename)` and pairwise-balanced metric semantics. It is not the locked
U0 window/video identity protocol; locked U0 continues to use `alpha_stalled.u0_protocol` and
`alpha_stalled.metrics.metric_tables`.
