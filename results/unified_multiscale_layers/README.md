# Unified multiscale and intermediate-layer results

This directory tracks only the compact evidence for the frozen K=3 MW2 study.
The full per-window/per-video scores, raw combined shards, calibration token
caches, and fitted parameter arrays remain local because they occupy about
8.6 GB.

- `multiscale_*`: MS0-MS4 dataset, generator, motion, false-negative, and
  1,000-iteration paired-bootstrap results.
- `layer_*`: H0-H5 versions of the same analyses.
- `focus_generator_metrics.csv`: ZeroScope-576w, Pika, SoRA-Clip, Crafter,
  and Text2Video-Zero comparisons.
- `admission_summary.csv`: all predeclared admission gates and final decisions.

No candidate is admitted. See
`reports/unified_multiscale_intermediate_layers.md` for the full protocol,
leakage audit, numerical-stability audit, costs, and interpretation.
