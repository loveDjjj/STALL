# Alpha-STALLED Exploration Log

This document records explored directions that are not part of the current
main-paper protocol. It is the single historical index for decisions that
should not be repeated without a new protocol and a new hypothesis.

## Current Main Line

The only current authority is `u0_locked_v1`, defined by
`configs/alpha_stalled_u0_locked.yaml` and `release/u0/`.

The main paper keeps five experiment families:

| ID | Family | Current evidence |
|---|---|---|
| M01 | Unified benchmark comparison | `release/u0/`, `results/research_summary/` |
| M02 | Evidence composition and local temporal dynamics | `reports/u0_core_ablation.md`, `reports/second_order_and_independent_calibration.md` |
| M03 | Local modeling versus K=1/K=3 coverage | U0 K1/K3 controlled comparison |
| M04 | Real-only calibration reliability | `reports/u0_calibration_size_and_seed.md` |
| M05 | Frozen external generalization | `release/u0_external_genvidbench/` |

## Rejected or Superseded Directions

| Direction | Status | Decision |
|---|---|---|
| Historical leakage-affected calibration | `invalid_leakage` | Never use for model selection or performance claims. |
| Dataset-specific region/aggregation | `historical` | Fake-informed tuning; not a universal main protocol. |
| Pre-release temporal-unified U0 | `superseded` | PatchSpatial still inherited historical dataset-specific aggregation. |
| Legacy K=1 paper-score pipeline | `historical` | Keep only for archaeology, not current U0 reproduction. |
| K=5 and all-window aggregation | `rejected` | Higher cost without an AP gain over K=3. |
| Bottom-k and hybrid Local aggregation | `rejected` | Lower-tail aggregation amplified transfer failures. |
| Joint Typicality J2/J3 | `rejected` | Below fixed linear Global/Local fusion. |
| Spatial-mean residual D2 | `rejected` | Negative paired delta and unstable generator behavior. |
| Fine plus coarse intermediate layers | `rejected` | Did not exceed final-layer fine-only Local evidence. |
| Layer-17/layer-23 fusion and cross-layer min | `rejected` | Confidence intervals crossed zero or decreased Macro AP. |
| Motion gates and temporal hard/soft gates | `rejected` | No robust improvement over the locked same-grid D2 branch. |
| OAS covariance | `rejected` | AP gain was negligible and its confidence interval crossed zero. |
| D3/D4 and multi-lag derivatives | `rejected` | Did not exceed same-grid second-order dynamics. |
| Injection localization | `diagnostic_only` | Does not establish semantic artifact localization. |
| Pixel/compression robustness | `diagnostic_only` | Deployment boundary evidence, not a main-paper method experiment. |
| Duration-aware 23-source protocol | `coverage_extension` | Separate coverage protocol; it cannot replace strict-20 U0. |
| Source-aware routing, fallback, persistence and selector gates | `diagnostic_only` | Test/source-aware behavior is outside the locked inference path. |

## Historical Code Families

The following families are exploratory or compatibility-only and are not
required by M01-M05:

- D3/D4, multi-order, intermediate-layer and clean-universal experiments.
- Duration-aware/full-coverage protocols.
- Joint Typicality, residual, OAS, injection and robustness pipelines.
- PatchField, frequency, motion-matching and adaptive-reliability probes.
- Routing, fallback, selector and source-aware fusion experiments.
- Journal experiment launchers and legacy paper-score wrappers.

When these families are removed, retain only their result, reason for rejection,
and replacement decision in this document. Their raw outputs, sweep parameters,
launchers and dedicated reports are intentionally not part of the main checkout.

## Data Boundaries

Keep the three development datasets:

```text
datasets/comgenvid/
datasets/videofeedback/
datasets/genvideo/
```

Keep only the data directories referenced by the locked external GenVidBench
release:

```text
datasets/genvidbench_pair1_ms_vript_calib/
datasets/genvidbench_pair1_ms_vript_eval/
datasets/genvidbench_pair1_pika_vript_eval/
```

The demo dataset, T2VZ 4-FPS stress data, calibration-size copies, selected
calibration pool, raw GenVidBench archives and temporary extraction files are
not main-paper data.

## Cleanup Rules

1. Never delete `release/u0/` or `release/u0_external_genvidbench/`.
2. Never delete the current VATEX Global parameter
   `precomputed/stall_params_vatex_dino_v3.npz` without creating a new release
   and updating its hash manifest.
3. Delete or keep large caches only on the server; preserve inventory metadata.
4. New main-paper runs must use one of M01-M05 and a unique run manifest.
5. A new direction requires a new protocol ID before any result can become
   evidence.
