# Tool dependency inventory

This report is generated from `configs/tool_dependencies.yaml` and the current AST of
top-level Python files in `tools/`. It is a governance snapshot, not an execution graph.

## Summary

- Tool modules: 126
- Registered `tools -> tools` edges: 8
- Source modules with internal imports: 7
- Imported target modules: 5
- Edges importing private symbols: 2
- Retained tool families: 4
- Dependency cycles: 0

Protected formal, governance, and migrated historical entrypoints listed under
`isolated_entrypoints` have no internal tool imports. Any new edge, removed edge, symbol change,
or cycle fails verification.

## Tool lifecycle

| Lifecycle | Tools | Meaning |
|---|---:|---|
| `compatibility` | 16 | Legacy generic command surfaces retained for existing callers but not used as current protocol authorities. |
| `formal_release` | 10 | Canonical locked-U0 release, public inference, and exact reproduction entrypoints. |
| `governance` | 16 | Deterministic repository catalogs, manifests, provenance capture, and consistency verifiers. |
| `historical_frozen` | 38 | Superseded, rejected, or protocol-specific experiment entrypoints retained only for exact historical reproduction. |
| `paper_evidence` | 45 | Controlled ablations, extensions, robustness studies, and paper asset builders supporting the current manuscript. |
| `research_utility` | 1 | Exploratory audits, benchmarks, token inspection, and generic summaries that are not authoritative claim sources. |

Every top-level Python tool belongs to exactly one lifecycle. Formal release and governance tools
must not import another CLI, and all remaining internal edges must stay within one lifecycle.

### `compatibility`

- New work: Do not add new method logic; route reusable behavior through src/alpha_stalled and preserve command compatibility.
- Archive: Remove only after repository-wide reference checks and a documented replacement command.
- Tools: `analyze_cross_dataset_frozen_hyperparams`, `analyze_duration_window_feasibility`, `analyze_journal_experiments`, `audit_failure_cases`, `audit_patch_likelihood_assumptions`, `audit_reference_experiment_alignment`, `benchmark_csv_stage_runtime`, `benchmark_video_stage_runtime`, `bootstrap_macro_average_delta`, `eval_score_csv`, `fuse_scores`, `inspect_dinov3_tokens`, `summarize_duration_window_representative`, `summarize_journal_experiments`, `summarize_metrics_average_rows`, `verify_alpha_stalled_release`

### `formal_release`

- New work: Bug fixes require locked-result regression evidence; behavior changes require a new protocol and release ID.
- Archive: Never archive while u0_locked_v1 is the current release; preserve public command paths.
- Tools: `analyze_u0_locked`, `build_u0_release_manifests`, `eval_alpha_stalled`, `finalize_u0_locked_calibration_reference`, `finalize_u0_locked_shard`, `score_u0_locked_calibration_reference`, `score_u0_locked_k1_cache`, `score_u0_locked_windows`, `verify_u0_locked_release`, `verify_u0_pipeline_reconstruction`

### `governance`

- New work: May evolve only with schema tests, deterministic generated artifacts, and documentation checks.
- Archive: Keep active while the governed asset exists; governance entrypoints must remain independent of other CLIs.
- Tools: `build_cache_inventory`, `build_data_catalog`, `build_run_manifest`, `build_tool_dependency_inventory`, `capture_experiment_run`, `verify_cache_inventory`, `verify_config_registry`, `verify_data_catalog`, `verify_environment_lock`, `verify_experiment_registry`, `verify_feature_cache_contract`, `verify_parameter_assets`, `verify_project_documentation`, `verify_run_capture`, `verify_run_manifest`, `verify_tool_dependencies`

### `historical_frozen`

- New work: Do not use for new parameter searches or main claims; changes are limited to compatibility and verified refactors.
- Archive: Move only as a complete retained family or with explicit evidence and compatibility-path preservation.
- Tools: `analyze_clean_universal`, `analyze_d3_window_cache_variants`, `analyze_global_d3_variants`, `analyze_global_local_fusion`, `analyze_intermediate_layers`, `analyze_joint_typicality`, `analyze_local_residual`, `analyze_multi_window_feasibility`, `analyze_multi_window_scores`, `analyze_unified_multiscale`, `audit_d3_exact_readiness`, `audit_d3_protocol`, `audit_videofeedback_whitening`, `build_multi_order_baselines`, `compare_d3_smoke_with_existing_scores`, `compare_multi_window_candidates`, `eval_d3_exact_from_frames`, `eval_genvideo_d3_comparable_protocol`, `eval_global_second_order_volatility`, `evaluate_d3_pixel_robustness`, `evaluate_d3_robustness`, `extract_d3_frames_from_runlist`, `fit_clean_universal_layer17`, `fit_clean_universal_params`, `fit_intermediate_layer_params`, `prepare_genvideo_d3_exact_protocol`, `run_d3_exact_genvideo_batch`, `run_local_d2_residuals`, `score_clean_universal_windows`, `score_intermediate_layer_windows`, `score_k1_calibration_from_cache`, `score_local_residual_windows`, `score_multi_window`, `score_multi_window_incremental`, `score_unified_multiscale_windows`, `summarize_d3_exact_genvideo_metrics`, `summarize_temporal_derivative_order`, `summarize_unified_multiscale_layers`

### `paper_evidence`

- New work: New runs require frozen factors, an experiment ID, leakage declarations, and registration before manuscript use.
- Archive: Retain through review and artifact release; archive only with result manifests and stable compatibility commands.
- Tools: `analyze_duration_aware_23source`, `analyze_duration_aware_k1_factorial`, `analyze_full_coverage_protocol`, `analyze_independent_real_complement`, `analyze_second_order_independent_calibration`, `analyze_u0_calibration_sensitivity`, `analyze_u0_core_ablation`, `analyze_u0_cross_dataset_calibration`, `analyze_u0_external_genvidbench`, `analyze_u0_injections`, `analyze_u0_oas_candidate`, `analyze_u0_robustness`, `assemble_duration_aware_k1_reuse`, `audit_runtime_storage_costs`, `audit_u0_metric_protocol`, `audit_u0_numerical_stability`, `bootstrap_u0_core_ablation`, `bootstrap_u0_oas_candidate`, `bootstrap_u0_robustness`, `build_duration_aware_23source_protocol`, `build_duration_aware_k1_protocol`, `build_independent_remaining_real_manifest`, `build_manuscript_rich_evidence`, `build_u0_calibration_reserve`, `build_u0_external_genvidbench_manifests`, `build_u0_robustness_manifests`, `build_u0_robustness_plan`, `collect_partial_k1_checkpoints`, `create_keyframe_case_explanations`, `create_patch_case_visualizations`, `fit_duration_aware_local_params`, `fit_u0_calibration_sensitivity`, `fit_u0_cross_and_oas_params`, `fit_u0_external_genvidbench_params`, `fit_u0_local_d1_params`, `score_duration_aware_23source`, `score_duration_aware_original_k1`, `score_duration_aware_real_curve`, `score_duration_aware_size_candidates`, `score_u0_calibration_candidates`, `score_u0_external_genvidbench_k1`, `score_u0_injections`, `score_u0_local_d1_windows`, `score_u0_robustness`, `select_duration_aware_calibration_size`

### `research_utility`

- New work: Results remain diagnostic until promoted through a registered protocol and evidence review.
- Archive: May be archived independently after confirming that no active script, report command, or cache workflow references it.
- Tools: `prefill_patch_cache`

## Maintenance decisions

- `compatibility_wrapper`: 14
- `retain_referenced`: 3

Reference counts scan non-result repository text while excluding this policy, its generated
artifacts, and each tool's own source file. Archive candidates must have zero references and
content-addressed evidence; compatibility wrappers preserve old command paths after physical
archival; referenced candidates must name a replacement before moving.

| Tool | Lifecycle | Decision | Reference files | Archive target |
|---|---|---|---:|---|
| `analyze_cross_dataset_frozen_hyperparams` | `compatibility` | `compatibility_wrapper` | 2 | `research_archive/tools/journal_experiments/analyze_cross_dataset_frozen_hyperparams.py` |
| `analyze_duration_window_feasibility` | `compatibility` | `compatibility_wrapper` | 3 | `research_archive/tools/journal_experiments/analyze_duration_window_feasibility.py` |
| `analyze_journal_experiments` | `compatibility` | `compatibility_wrapper` | 2 | `research_archive/tools/journal_experiments/analyze_journal_experiments.py` |
| `audit_failure_cases` | `compatibility` | `compatibility_wrapper` | 2 | `research_archive/tools/journal_experiments/audit_failure_cases.py` |
| `audit_patch_likelihood_assumptions` | `compatibility` | `compatibility_wrapper` | 4 | `research_archive/tools/journal_experiments/audit_patch_likelihood_assumptions.py` |
| `audit_reference_experiment_alignment` | `compatibility` | `compatibility_wrapper` | 3 | `research_archive/tools/journal_experiments/audit_reference_experiment_alignment.py` |
| `benchmark_csv_stage_runtime` | `compatibility` | `compatibility_wrapper` | 3 | `research_archive/tools/journal_experiments/benchmark_csv_stage_runtime.py` |
| `benchmark_video_stage_runtime` | `compatibility` | `compatibility_wrapper` | 2 | `research_archive/tools/journal_experiments/benchmark_video_stage_runtime.py` |
| `bootstrap_macro_average_delta` | `compatibility` | `compatibility_wrapper` | 2 | `research_archive/tools/journal_experiments/bootstrap_macro_average_delta.py` |
| `eval_score_csv` | `compatibility` | `retain_referenced` | 9 | `-` |
| `fuse_scores` | `compatibility` | `retain_referenced` | 13 | `-` |
| `inspect_dinov3_tokens` | `compatibility` | `compatibility_wrapper` | 6 | `research_archive/tools/journal_experiments/inspect_dinov3_tokens.py` |
| `prefill_patch_cache` | `research_utility` | `retain_referenced` | 16 | `-` |
| `summarize_duration_window_representative` | `compatibility` | `compatibility_wrapper` | 2 | `research_archive/tools/journal_experiments/summarize_duration_window_representative.py` |
| `summarize_journal_experiments` | `compatibility` | `compatibility_wrapper` | 2 | `research_archive/tools/journal_experiments/summarize_journal_experiments.py` |
| `summarize_metrics_average_rows` | `compatibility` | `compatibility_wrapper` | 5 | `research_archive/tools/journal_experiments/summarize_metrics_average_rows.py` |
| `verify_alpha_stalled_release` | `compatibility` | `compatibility_wrapper` | 10 | `research_archive/tools/pre_release_assets/verify_alpha_stalled_release.py` |

### Archived wrapper integrity

The original implementation hash is preserved from immediately before the move. Wrapper and
archived implementation hashes are recomputed from the current files on every inventory build.

| Tool | Original implementation SHA-256 | Wrapper SHA-256 | Archived implementation SHA-256 |
|---|---|---|---|
| `analyze_cross_dataset_frozen_hyperparams` | `97d0064a53fdd702501f90348b2ea5f2aecc78a97a28446f2ec0b129cc5858d1` | `61f58df96e55c60357a0ef52bf3fa1931590a045970e0940e0013b517f8d1e1c` | `306044f1735fa5bc9f89cfa20a0f38a5268eccbc9ae868b017cb3c065406eb49` |
| `analyze_duration_window_feasibility` | `b134e1ae69ddfea84bc11a2d01695bad90dfa6f9175f07471764f565be9c6693` | `3ea916d8c195f8b39cb0081afb7c10aaf3cf24ac91b81a41e9792ab8503359ff` | `3d2ec262d636bc342138e6417ece4ccee801c0d7ec384d82bbfe22fb34e2b4dd` |
| `analyze_journal_experiments` | `7594a16da983924974a5b2fef90d1d8461923d7b0076337f80274a3eea9eca2a` | `843aa55a40bec335f84d3908c111d25dd38a5cf2be6a8d051aaf60dd1706cf35` | `d29a3c58bcb950aa253292920de85bd88a5e169123bc409c8b743b90d899655f` |
| `audit_failure_cases` | `0feceb8eb0524729e228900d8f59c21caaad409b27fcddf98f7c07ccb2e14e09` | `9b87dfd113b01d0fb8535b772344382c9c137f7479b56c1b012e7d9d7c972424` | `6e679915ac5f54dda13cfd22e62dfd989b9264f548a6581f31e374a41d182e78` |
| `audit_patch_likelihood_assumptions` | `f6d1eae85a6c45ab6bf75665afb573dce4547a9e22ef392d4618fe00e2a22958` | `235a989fe6f1ddfa6c989d801cb010f78eb8ab6b2581d5438bb25dd19e21c8ea` | `8ba2439f6dca629a17bda8a7968322f521fce9da705c8e0d2e49e95bd85298d1` |
| `audit_reference_experiment_alignment` | `e59604846309b249f0f9c49aa4f5c114bc023c1c20e08568e91f892dc764bbaf` | `8457f6d0e630d6860b59fc4566885aad7068bd37137188f295bc1d9a555f4ff9` | `2e9b71706e9e1fde97892b57ed16d203c907ce6776abbebdb6e0ef55c4f79ee1` |
| `benchmark_csv_stage_runtime` | `3395b357a500827f418c6d249425c87a1d60f62bd2e5b737774e2f606fe522bf` | `bd180ca7d7140419fdcecd8e706cd6ec4934fe25bef17e992303317583e70123` | `b3e2a217eb30a472905dabdef01d4f9687e40c7dd1d049263f429648a68c5f6c` |
| `benchmark_video_stage_runtime` | `48e844730488cd59074ec7cf824c3bd02c20b24c3aa11a94669789720ee1cce5` | `ed376cd0336aedc8b2e3b81830043041adf40454469f4eda21209ab9a977be81` | `b50db6373b90776ca3fd999c4b0231b52b2a1725d04fd744137fe40cb394a4e8` |
| `bootstrap_macro_average_delta` | `99091cb46daa21827e288ccda56d0458eeca9963a4d18635f8e9559ca661935f` | `3ae54fedb61c62e000ae24e9c2561f473c465a11ecf568ff927bec0569139777` | `9e8813e3ffb1773334e64f129321f72a6a0dd3feee3cd27f3e65112d1ceea5bb` |
| `inspect_dinov3_tokens` | `96e870cbf79e7eb7d346a5945435bac19ad848928777022e795f61df93f02909` | `b5938825edba45de68942b4889b72961bee16b0fef8558b88805cf71e357c6b4` | `c88ec18f2d0968b6a6598d1d3a3ef751abc5fe40a20d306665b651a21ed667d5` |
| `summarize_duration_window_representative` | `3b986751ff13a73d0be17983e3158d5991024a1d13b611b7148794457a40c71d` | `d58449f04a7abb548ebb0c376928c4969d2c7fa91c11eba8fc54be587c99cfaf` | `9ac7c647d5bc72864a8c986f8dd2b95ce17c56da60600df1483bda909794bc5c` |
| `summarize_journal_experiments` | `3242b7a63c6efe60d4b47750d121fd2b97b77764ffc603438e62c795918ba7a9` | `422421698c30391cf8c491c4f429a23ec52b1aea6b148de8043c13cd6115ef60` | `5b99e2e5a79967a08489198e34438afeccff445066eb28accdf326da3ee872aa` |
| `summarize_metrics_average_rows` | `1d955dc8dee4b37712e77ce1b7727bcba10d50cbc15d3acc3bbe4531df6811e8` | `ba6032bfc33b3e3638b815effe3ff526193cc0351354496f9d403e027820f843` | `1d955dc8dee4b37712e77ce1b7727bcba10d50cbc15d3acc3bbe4531df6811e8` |
| `verify_alpha_stalled_release` | `3f03f58905ec76399592f17790232d71b46079d0b22b14cd759abcc76398f965` | `e888eed314ce3d6ba5aaadd34f9f143cc5b7c1572ccfa2694826c6a131858b85` | `2c6995b95af32ef7649a53039e5363a0c1e0e9c532bfa18bf5571f65db8e740a` |

## Categories

- `historical_d3`: 3
- `historical_experiment_family`: 3
- `research_visualization`: 1
- `u0_extension`: 1

## Migration status

- `retain_historical`: 8

`migrate_to_shared` marks reusable behavior that should move into `src/alpha_stalled/`.
`retain_historical` marks tightly coupled historical evidence that should remain stable until its
whole experiment family is archived.

## Retained family status

- `frozen_historical`: 2
- `frozen_supplement`: 2

Every retained family owns all tools at the endpoints of its registered edges. Evidence files are
content-addressed in the JSON inventory. Physical moves are forbidden until the recorded archive
gate is met.

## Retained families

### `d3_operator_controls`

- Status: `frozen_historical`
- Tools: `analyze_global_d3_variants`, `build_multi_order_baselines`, `evaluate_d3_pixel_robustness`, `evaluate_d3_robustness`
- Archive target: `research_archive/tools/d3_operator_controls`
- Reason: These scripts reproduce the rejected DINOv3 operator-controlled D3 family and share its exact historical cache and calibration semantics.
- Archive gate: Move only with a compatibility wrapper, the registered result files, and a command manifest that reproduces all three dataset-metric tables.
- Evidence:
  - `docs/EXPLORATION_LOG.md` (4733 bytes, sha256 `8c1ccb3215731b30eaa7491413888ef4baabc8cf9336a64063898215bb28a166`)

### `keyframe_case_visualization`

- Status: `frozen_supplement`
- Tools: `create_keyframe_case_explanations`, `create_patch_case_visualizations`
- Archive target: `research_archive/tools/keyframe_case_visualization`
- Reason: The explanation generator is a thin second stage of the paper's frozen qualitative patch-visualization workflow.
- Archive gate: Move only after the paper figure and table have a standalone source-data manifest and their current paths remain available through wrappers.
- Evidence:
  - `docs/EXPLORATION_LOG.md` (4733 bytes, sha256 `8c1ccb3215731b30eaa7491413888ef4baabc8cf9336a64063898215bb28a166`)

### `multiscale_intermediate_layers`

- Status: `frozen_historical`
- Tools: `fit_clean_universal_layer17`, `fit_intermediate_layer_params`, `score_intermediate_layer_windows`, `score_unified_multiscale_windows`
- Archive target: `research_archive/tools/multiscale_intermediate_layers`
- Reason: These fitters and scorers form the rejected multiscale and DINO intermediate-layer experiment family with shared frozen registries.
- Archive gate: Move only after parameter files, raw score shards, checkpoint identity, and the four registered result files are captured by an archival run manifest.
- Evidence:
  - `docs/EXPLORATION_LOG.md` (4733 bytes, sha256 `8c1ccb3215731b30eaa7491413888ef4baabc8cf9336a64063898215bb28a166`)

### `u0_robustness_injection`

- Status: `frozen_supplement`
- Tools: `score_u0_injections`, `score_u0_robustness`
- Archive target: `research_archive/tools/u0_robustness_injection`
- Reason: Synthetic injection scoring intentionally reuses the frozen robustness perturbation and raw U0 scoring implementation.
- Archive gate: Keep in place while robustness and localization are paper supplements; move only with wrapper commands and manifests for both result directories.
- Evidence:
  - `docs/EXPLORATION_LOG.md` (4733 bytes, sha256 `8c1ccb3215731b30eaa7491413888ef4baabc8cf9336a64063898215bb28a166`)

## Registered edges

| Source | Target | Imported symbols | Category | Migration | Family |
|---|---|---|---|---|---|
| `analyze_global_d3_variants` | `build_multi_order_baselines` | `GLOBAL_D3_CONFIG_NAMES` | `historical_d3` | `retain_historical` | `d3_operator_controls` |
| `create_keyframe_case_explanations` | `create_patch_case_visualizations` | `FastPatchScorer`, `compute_anomaly_fields` | `research_visualization` | `retain_historical` | `keyframe_case_visualization` |
| `evaluate_d3_pixel_robustness` | `build_multi_order_baselines` | `KEY_COLUMNS`, `_load_index`, `d3_statistics`, `dataset_specs`, `empirical_cdf`, `two_sided_realness` | `historical_d3` | `retain_historical` | `d3_operator_controls` |
| `evaluate_d3_robustness` | `build_multi_order_baselines` | `KEY_COLUMNS`, `_load_index`, `d3_statistics`, `dataset_specs`, `empirical_cdf`, `load_strict_window`, `two_sided_realness` | `historical_d3` | `retain_historical` | `d3_operator_controls` |
| `fit_clean_universal_layer17` | `fit_intermediate_layer_params` | `cache_name`, `checkpoint_hash` | `historical_experiment_family` | `retain_historical` | `multiscale_intermediate_layers` |
| `score_intermediate_layer_windows` | `fit_intermediate_layer_params` | `LAYERS`, `SPECS` | `historical_experiment_family` | `retain_historical` | `multiscale_intermediate_layers` |
| `score_intermediate_layer_windows` | `score_unified_multiscale_windows` | `PARAMS` | `historical_experiment_family` | `retain_historical` | `multiscale_intermediate_layers` |
| `score_u0_injections` | `score_u0_robustness` | `donor_window_id`, `embed_sequence`, `score_condition` | `u0_extension` | `retain_historical` | `u0_robustness_injection` |
