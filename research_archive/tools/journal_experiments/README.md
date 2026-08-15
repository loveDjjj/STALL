# Archived journal experiment utilities

This directory contains one-time pre-release journal diagnostics whose results are already
materialized under `results/`. They are not authorities for `u0_locked_v1` and must not be used
for new method selection.

The original commands remain available as compatibility wrappers:

| Compatibility command | Archived implementation |
|---|---|
| `tools/analyze_cross_dataset_frozen_hyperparams.py` | `research_archive/tools/journal_experiments/analyze_cross_dataset_frozen_hyperparams.py` |
| `tools/analyze_journal_experiments.py` | `research_archive/tools/journal_experiments/analyze_journal_experiments.py` |
| `tools/audit_failure_cases.py` | `research_archive/tools/journal_experiments/audit_failure_cases.py` |
| `tools/benchmark_video_stage_runtime.py` | `research_archive/tools/journal_experiments/benchmark_video_stage_runtime.py` |
| `tools/bootstrap_macro_average_delta.py` | `research_archive/tools/journal_experiments/bootstrap_macro_average_delta.py` |
| `tools/summarize_duration_window_representative.py` | `research_archive/tools/journal_experiments/summarize_duration_window_representative.py` |
| `tools/analyze_duration_window_feasibility.py` | `research_archive/tools/journal_experiments/analyze_duration_window_feasibility.py` |
| `tools/audit_reference_experiment_alignment.py` | `research_archive/tools/journal_experiments/audit_reference_experiment_alignment.py` |
| `tools/benchmark_csv_stage_runtime.py` | `research_archive/tools/journal_experiments/benchmark_csv_stage_runtime.py` |
| `tools/summarize_journal_experiments.py` | `research_archive/tools/journal_experiments/summarize_journal_experiments.py` |
| `tools/inspect_dinov3_tokens.py` | `research_archive/tools/journal_experiments/inspect_dinov3_tokens.py` |
| `tools/summarize_metrics_average_rows.py` | `research_archive/tools/journal_experiments/summarize_metrics_average_rows.py` |
| `tools/audit_patch_likelihood_assumptions.py` | `research_archive/tools/journal_experiments/audit_patch_likelihood_assumptions.py` |

`configs/tool_dependencies.yaml` records each implementation's pre-move source hash, current
archive hash, historical result evidence, and wrapper target. Both direct archived commands and
the old compatibility commands resolve repository paths independently of the working directory.

The patch-likelihood audit uses the superseded ComGenVid region-3 configuration. Its table and
figure are not included by the current U0 manuscript and must not be cited as locked-U0 evidence.
