#!/usr/bin/env python3
"""Build a reproducible runbook for the sample-level fallback candidate."""

from __future__ import annotations

import argparse
import json
import shlex
from pathlib import Path
from typing import Any


DATASET_BEST_DATASETS = [
    {
        "dataset": "comgenvid_full",
        "global_csv": "results/comgenvid_results.csv",
        "raw_patch_csv": "results/patch_region3_bottomk_sweep/comgenvid_patch_region3_second_order_bottomk0p50_spat10_temp90.csv",
        "persistence_csv": "results/patch_calibrated_persistence/comgenvid_persistence_full.csv",
        "persistence_score_col": "anom_mass_thr0p2_real_pct_real",
        "base_alpha": 0.20,
        "final_scores_csv": "results/patch_calibrated_persistence/comgenvid_sample_fallback_final_scores.csv",
        "metrics_csv": "results/patch_calibrated_persistence/comgenvid_sample_fallback_metrics.csv",
        "per_source_csv": "results/patch_calibrated_persistence/sample_fallback_comgenvid_per_source.csv",
    },
    {
        "dataset": "genvideo_balanced300",
        "global_csv": "results/genvideo_results.csv",
        "raw_patch_csv": "results/patch_cross_dataset_ablation/genvideo_patch_same_grid_second_order_region2_mean_spat10_temp90.csv",
        "persistence_csv": "results/patch_calibrated_persistence/genvideo_persistence_balanced300.csv",
        "persistence_score_col": "temp_pct_mean_real_pct_real",
        "base_alpha": 0.60,
        "final_scores_csv": "results/patch_calibrated_persistence/genvideo_sample_fallback_final_scores.csv",
        "metrics_csv": "results/patch_calibrated_persistence/genvideo_sample_fallback_metrics.csv",
        "per_source_csv": "results/patch_calibrated_persistence/sample_fallback_genvideo_per_source.csv",
    },
    {
        "dataset": "videofeedback_full",
        "global_csv": "results/videofeedback_results.csv",
        "raw_patch_csv": "results/patch_cross_dataset_ablation/videofeedback_patch_same_grid_second_order_region1_mean_spat10_temp90.csv",
        "persistence_csv": "results/patch_calibrated_persistence/videofeedback_persistence_full.csv",
        "persistence_score_col": "max_frame_mass_thr0p2_real_pct_real",
        "base_alpha": 0.60,
        "final_scores_csv": "results/patch_calibrated_persistence/videofeedback_sample_fallback_final_scores.csv",
        "metrics_csv": "results/patch_calibrated_persistence/videofeedback_sample_fallback_metrics.csv",
        "per_source_csv": "results/patch_calibrated_persistence/sample_fallback_videofeedback_per_source.csv",
    },
]


DEPLOYABLE_ALPHA060_DATASETS = [
    {
        "dataset": "comgenvid_full_alpha060",
        "global_csv": "results/comgenvid_results.csv",
        "raw_patch_csv": "results/patch_region3_bottomk_sweep/comgenvid_patch_region3_second_order_bottomk0p50_spat10_temp90.csv",
        "persistence_csv": "results/patch_calibrated_persistence/comgenvid_persistence_full.csv",
        "persistence_score_col": "anom_mass_thr0p2_real_pct_real",
        "base_alpha": 0.60,
        "final_scores_csv": "results/patch_calibrated_persistence/comgenvid_alpha060_sample_fallback_final_scores.csv",
        "metrics_csv": "results/patch_calibrated_persistence/comgenvid_alpha060_sample_fallback_metrics.csv",
        "per_source_csv": "results/patch_calibrated_persistence/sample_fallback_comgenvid_alpha060_full_per_source.csv",
    },
    {
        "dataset": "genvideo_full_alpha060",
        "global_csv": "results/genvideo_results.csv",
        "raw_patch_csv": "results/patch_cross_dataset_ablation/genvideo_patch_same_grid_second_order_region2_mean_spat10_temp90.csv",
        "persistence_csv": "results/patch_calibrated_persistence/genvideo_persistence_full.csv",
        "persistence_score_col": "temp_pct_mean_real_pct_real",
        "base_alpha": 0.60,
        "final_scores_csv": "results/patch_calibrated_persistence/genvideo_full_alpha060_sample_fallback_final_scores.csv",
        "metrics_csv": "results/patch_calibrated_persistence/genvideo_full_alpha060_sample_fallback_metrics.csv",
        "per_source_csv": "results/patch_calibrated_persistence/sample_fallback_genvideo_full_alpha060_per_source.csv",
    },
    {
        "dataset": "videofeedback_full",
        "global_csv": "results/videofeedback_results.csv",
        "raw_patch_csv": "results/patch_cross_dataset_ablation/videofeedback_patch_same_grid_second_order_region1_mean_spat10_temp90.csv",
        "persistence_csv": "results/patch_calibrated_persistence/videofeedback_persistence_full.csv",
        "persistence_score_col": "max_frame_mass_thr0p2_real_pct_real",
        "base_alpha": 0.60,
        "final_scores_csv": "results/patch_calibrated_persistence/videofeedback_sample_fallback_final_scores.csv",
        "metrics_csv": "results/patch_calibrated_persistence/videofeedback_sample_fallback_metrics.csv",
        "per_source_csv": "results/patch_calibrated_persistence/sample_fallback_videofeedback_per_source.csv",
    },
]


DEPLOYABLE_ALPHA060_RAWBASE_SPLITMINUS_DATASETS = [
    {
        "dataset": "comgenvid_full_alpha060_rawbase_splitminus",
        "global_csv": "results/comgenvid_results.csv",
        "raw_patch_csv": "results/patch_region3_bottomk_sweep/comgenvid_patch_region3_second_order_bottomk0p50_spat10_temp90.csv",
        "persistence_csv": "results/patch_calibrated_persistence/comgenvid_persistence_full.csv",
        "persistence_score_col": "anom_mass_thr0p2_real_pct_real",
        "base_alpha": 0.60,
        "base_space": "raw",
        "persistence_scale": 1.0,
        "selector_feature": "split_minus_universal",
        "selector_direction": "lt",
        "selector_threshold": 0.0,
        "real_policy": "split",
        "final_scores_csv": (
            "results/patch_calibrated_persistence/"
            "comgenvid_alpha060_rawbase_splitminus_sample_fallback_final_scores.csv"
        ),
        "metrics_csv": (
            "results/patch_calibrated_persistence/"
            "comgenvid_alpha060_rawbase_splitminus_sample_fallback_metrics.csv"
        ),
        "per_source_csv": (
            "results/patch_calibrated_persistence/"
            "sample_fallback_comgenvid_alpha060_rawbase_per_source.csv"
        ),
    },
    {
        "dataset": "genvideo_full_alpha060_rawbase_splitminus",
        "global_csv": "results/genvideo_results.csv",
        "raw_patch_csv": "results/patch_cross_dataset_ablation/genvideo_patch_same_grid_second_order_region2_mean_spat10_temp90.csv",
        "persistence_csv": "results/patch_calibrated_persistence/genvideo_persistence_full.csv",
        "persistence_score_col": "temp_pct_mean_real_pct_real",
        "base_alpha": 0.60,
        "base_space": "raw",
        "persistence_scale": 1.0,
        "selector_feature": "split_minus_universal",
        "selector_direction": "lt",
        "selector_threshold": 0.0,
        "real_policy": "split",
        "final_scores_csv": (
            "results/patch_calibrated_persistence/"
            "genvideo_full_alpha060_rawbase_splitminus_sample_fallback_final_scores.csv"
        ),
        "metrics_csv": (
            "results/patch_calibrated_persistence/"
            "genvideo_full_alpha060_rawbase_splitminus_sample_fallback_metrics.csv"
        ),
        "per_source_csv": (
            "results/patch_calibrated_persistence/"
            "sample_fallback_genvideo_full_alpha060_rawbase_per_source.csv"
        ),
    },
    {
        "dataset": "videofeedback_full_alpha060_rawbase_splitminus",
        "global_csv": "results/videofeedback_results.csv",
        "raw_patch_csv": "results/patch_cross_dataset_ablation/videofeedback_patch_same_grid_second_order_region1_mean_spat10_temp90.csv",
        "persistence_csv": "results/patch_calibrated_persistence/videofeedback_persistence_full.csv",
        "persistence_score_col": "max_frame_mass_thr0p2_real_pct_real",
        "base_alpha": 0.60,
        "base_space": "raw",
        "persistence_scale": 1.0,
        "selector_feature": "split_minus_universal",
        "selector_direction": "lt",
        "selector_threshold": 0.0,
        "real_policy": "split",
        "final_scores_csv": (
            "results/patch_calibrated_persistence/"
            "videofeedback_alpha060_rawbase_splitminus_sample_fallback_final_scores.csv"
        ),
        "metrics_csv": (
            "results/patch_calibrated_persistence/"
            "videofeedback_alpha060_rawbase_splitminus_sample_fallback_metrics.csv"
        ),
        "per_source_csv": (
            "results/patch_calibrated_persistence/"
            "sample_fallback_videofeedback_alpha060_rawbase_per_source.csv"
        ),
    },
]


LEGACY_RANK_CONFIG = {
    "rule": "persistence_minus_base/0.0/ge/rule",
    "objective": "robust_universal_then_base",
    "base_space": "rank",
    "global_score_col": "final_score",
    "raw_patch_score_col": "final_score",
    "universal_weight": 0.15,
    "real_weight": 0.25,
    "fake_weight": 0.25,
    "source_gate_threshold": 0.45,
    "disagreement_threshold": 0.20,
    "confidence_threshold": 0.30,
    "fallback_threshold": 0.0,
}

CURRENT_PROMOTION_CONFIG = {
    "rule": "split_minus_universal/0.0/lt/split",
    "objective": "mean_universal_then_base",
    "base_alpha": 0.60,
    "base_space": "raw",
    "persistence_scale": 1.0,
    "selector_feature": "split_minus_universal",
    "selector_direction": "lt",
    "selector_threshold": 0.0,
    "real_policy": "split",
    "global_score_col": "final_score",
    "raw_patch_score_col": "final_score",
    "universal_weight": 0.15,
    "real_weight": 0.25,
    "fake_weight": 0.25,
    "source_gate_threshold": 0.45,
    "disagreement_threshold": 0.20,
    "confidence_threshold": 0.30,
}


PROMOTION_THRESHOLDS = {
    "min_delta_vs_base_auc": 0.0,
    "min_delta_vs_base_ap": 0.0,
    "min_delta_vs_universal_auc": 0.0,
    "min_delta_vs_universal_ap": 0.0,
    "min_mean_delta_vs_universal_auc": 0.001,
    "min_mean_delta_vs_universal_ap": 0.001,
    "min_datasets": 3,
    "min_sources": 20,
    "max_negative_source_auc": 0,
    "max_negative_source_ap": 0,
    "min_source_delta_auc": 0.0,
    "min_source_delta_ap": 0.0,
}


def _cmd(parts: list[Any]) -> str:
    return " ".join(shlex.quote(str(p)) for p in parts)


def _apply_command(dataset: dict[str, Any]) -> str:
    parts: list[Any] = [
        "python3",
        "tools/apply_sample_fallback_rule.py",
        "--dataset",
        dataset["dataset"],
        "--global-csv",
        dataset["global_csv"],
        "--raw-patch-csv",
        dataset["raw_patch_csv"],
        "--persistence-csv",
        dataset["persistence_csv"],
        "--persistence-score-col",
        dataset["persistence_score_col"],
        "--base-alpha",
        dataset["base_alpha"],
    ]
    if "base_space" in dataset:
        parts.extend(["--base-space", dataset["base_space"]])
    if "persistence_scale" in dataset:
        parts.extend(["--persistence-scale", dataset["persistence_scale"]])
    if "selector_feature" in dataset:
        parts.extend(
            [
                "--selector-feature",
                dataset["selector_feature"],
                "--selector-direction",
                dataset["selector_direction"],
                "--selector-threshold",
                dataset["selector_threshold"],
                "--real-policy",
                dataset["real_policy"],
            ]
        )
    parts.extend(["--output-csv", dataset["final_scores_csv"], "--metrics-csv", dataset["metrics_csv"]])
    return _cmd(parts)


def _validation_command(datasets: list[dict[str, Any]], output_csv: str, summary_csv: str) -> str:
    return _cmd(
        [
            "python3",
            "tools/validate_sample_fallback_rules.py",
            "--per-source-csv",
            *[d["per_source_csv"] for d in datasets],
            "--output-csv",
            output_csv,
            "--summary-csv",
            summary_csv,
        ]
    )


def _promotion_command(
    validation_summary_csv: str,
    final_score_summary_csv: str,
    output_csv: str,
    checks_csv: str,
    objective: str = "robust_universal_then_base",
    source_audit_summary_csv: str | None = None,
) -> str:
    parts: list[Any] = [
        "python3",
        "tools/sample_fallback_promotion_gate.py",
        "--validation-summary-csv",
        validation_summary_csv,
        "--final-score-summary-csv",
        final_score_summary_csv,
    ]
    if source_audit_summary_csv is not None:
        parts.extend(["--source-audit-summary-csv", source_audit_summary_csv])
    parts.extend(
        [
            "--objective",
            objective,
            "--output-csv",
            output_csv,
            "--checks-csv",
            checks_csv,
        ]
    )
    return _cmd(parts)


def _protocol(
    name: str,
    description: str,
    datasets: list[dict[str, Any]],
    validation_csv: str,
    validation_summary_csv: str,
    final_score_summary_csv: str,
    promotion_decision_csv: str,
    promotion_checks_csv: str,
    promotion_objective: str,
    source_audit_summary_csv: str | None = None,
) -> dict[str, Any]:
    return {
        "description": description,
        "datasets": datasets,
        "commands": {
            "apply_fixed_scorer": [_apply_command(d) for d in datasets],
            "validate_rule_selection": _validation_command(datasets, validation_csv, validation_summary_csv),
            "promotion_gate": _promotion_command(
                validation_summary_csv,
                final_score_summary_csv,
                promotion_decision_csv,
                promotion_checks_csv,
                objective=promotion_objective,
                source_audit_summary_csv=source_audit_summary_csv,
            ),
        },
        "outputs": {
            "validation_csv": validation_csv,
            "validation_summary_csv": validation_summary_csv,
            "final_score_summary_csv": final_score_summary_csv,
            "promotion_decision_csv": promotion_decision_csv,
            "promotion_checks_csv": promotion_checks_csv,
        },
        "promotion_objective": promotion_objective,
        "source_audit_summary_csv": source_audit_summary_csv,
    }


def _manifest() -> dict[str, Any]:
    return {
        "name": "sample_level_fallback_promotion_candidate",
        "status": "internal_gate_passed_needs_fresh_external_validation",
        "current_promotion_config": CURRENT_PROMOTION_CONFIG,
        "legacy_rank_config": LEGACY_RANK_CONFIG,
        "promotion_thresholds": PROMOTION_THRESHOLDS,
        "protocols": {
            "dataset_best": _protocol(
                name="dataset_best",
                description=(
                    "Research protocol with ComGenVid base_alpha=0.20 and other datasets at 0.60; "
                    "best for absolute current benchmark score, not the strict default-replacement audit."
                ),
                datasets=DATASET_BEST_DATASETS,
                validation_csv="results/patch_calibrated_persistence/sample_fallback_rule_validation.csv",
                validation_summary_csv="results/patch_calibrated_persistence/sample_fallback_rule_validation_summary.csv",
                final_score_summary_csv="results/patch_calibrated_persistence/sample_fallback_final_score_summary.csv",
                promotion_decision_csv="results/patch_calibrated_persistence/sample_fallback_promotion_decision.csv",
                promotion_checks_csv="results/patch_calibrated_persistence/sample_fallback_promotion_checks.csv",
                promotion_objective="robust_universal_then_base",
            ),
            "deployable_default_alpha060": _protocol(
                name="deployable_default_alpha060",
                description=(
                    "Default-replacement audit protocol; every dataset uses base_alpha=0.60."
                ),
                datasets=DEPLOYABLE_ALPHA060_DATASETS,
                validation_csv="results/patch_calibrated_persistence/sample_fallback_rule_validation_full_scope_alpha060.csv",
                validation_summary_csv=(
                    "results/patch_calibrated_persistence/sample_fallback_rule_validation_full_scope_alpha060_summary.csv"
                ),
                final_score_summary_csv=(
                    "results/patch_calibrated_persistence/sample_fallback_full_scope_alpha060_final_score_summary.csv"
                ),
                promotion_decision_csv=(
                    "results/patch_calibrated_persistence/sample_fallback_full_scope_alpha060_promotion_decision.csv"
                ),
                promotion_checks_csv=(
                    "results/patch_calibrated_persistence/sample_fallback_full_scope_alpha060_promotion_checks.csv"
                ),
                promotion_objective="mean_universal_then_base",
            ),
            "deployable_default_alpha060_rawbase_splitminus": _protocol(
                name="deployable_default_alpha060_rawbase_splitminus",
                description=(
                    "Default-replacement audit protocol with raw alpha=0.60 base and validated "
                    "split_minus_universal<0 selector."
                ),
                datasets=DEPLOYABLE_ALPHA060_RAWBASE_SPLITMINUS_DATASETS,
                validation_csv=(
                    "results/patch_calibrated_persistence/"
                    "sample_fallback_rule_validation_full_scope_alpha060_rawbase.csv"
                ),
                validation_summary_csv=(
                    "results/patch_calibrated_persistence/"
                    "sample_fallback_rule_validation_full_scope_alpha060_rawbase_summary.csv"
                ),
                final_score_summary_csv=(
                    "results/patch_calibrated_persistence/"
                    "sample_fallback_full_scope_alpha060_rawbase_splitminus_final_score_summary.csv"
                ),
                promotion_decision_csv=(
                    "results/patch_calibrated_persistence/"
                    "sample_fallback_full_scope_alpha060_rawbase_splitminus_strict_promotion_decision.csv"
                ),
                promotion_checks_csv=(
                    "results/patch_calibrated_persistence/"
                    "sample_fallback_full_scope_alpha060_rawbase_splitminus_strict_promotion_checks.csv"
                ),
                promotion_objective="mean_universal_then_base",
                source_audit_summary_csv=(
                    "results/patch_calibrated_persistence/"
                    "rawbase_splitminus_vs_default_source_audit_summary.csv"
                ),
            ),
        },
        "future_dataset_template": {
            "dataset": "<new_dataset_name>",
            "global_csv": "<path/to/global_scores.csv>",
            "raw_patch_csv": "<path/to/best_patch_scores.csv>",
            "persistence_csv": "<path/to/persistence_scores.csv>",
            "persistence_score_col": "<pre_registered_persistence_column>",
            "base_alpha": 0.60,
            "base_space": "raw",
            "persistence_scale": 1.0,
            "selector_feature": "split_minus_universal",
            "selector_direction": "lt",
            "selector_threshold": 0.0,
            "real_policy": "split",
            "final_scores_csv": "results/patch_calibrated_persistence/<new_dataset>_sample_fallback_final_scores.csv",
            "metrics_csv": "results/patch_calibrated_persistence/<new_dataset>_sample_fallback_metrics.csv",
            "per_source_csv": "results/patch_calibrated_persistence/sample_fallback_<new_dataset>_per_source.csv",
        },
    }


def _runbook_text(manifest: dict[str, Any]) -> str:
    protocol_sections = []
    for name, protocol in manifest["protocols"].items():
        protocol_sections.extend(
            [
                f"## Protocol: {name}",
                "",
                protocol["description"],
                "",
                "Datasets:",
                "",
                "```json",
                json.dumps(protocol["datasets"], indent=2, sort_keys=True),
                "```",
                "",
                "Scorer commands:",
                "",
                "```bash",
                *protocol["commands"]["apply_fixed_scorer"],
                "```",
                "",
                "Rule-selection validation command:",
                "",
                "```bash",
                protocol["commands"]["validate_rule_selection"],
                "```",
                "",
                "Promotion gate command:",
                "",
                "```bash",
                protocol["commands"]["promotion_gate"],
                "```",
                "",
            ]
        )
    lines = [
        "# Sample-Level Fallback Pre-Registered Runbook",
        "",
        "Status: internal gate passed; promotion still requires a fresh external benchmark or pre-registered future run.",
        "",
        "Current promotion rule:",
        "",
        "```text",
        "base = raw alpha=0.60 fusion",
        "use split score when split_score - universal_score < 0",
        "otherwise use universal score",
        "```",
        "",
        "Current locked parameters:",
        "",
        "```json",
        json.dumps(manifest["current_promotion_config"], indent=2, sort_keys=True),
        "```",
        "",
        "Legacy rank-space parameters:",
        "",
        "```json",
        json.dumps(manifest["legacy_rank_config"], indent=2, sort_keys=True),
        "```",
        "",
        "Promotion thresholds:",
        "",
        "```json",
        json.dumps(manifest["promotion_thresholds"], indent=2, sort_keys=True),
        "```",
        "",
        "Protocol distinction:",
        "",
        "```text",
        "dataset_best: highest current benchmark score; ComGenVid uses alpha=0.20.",
        "deployable_default_alpha060: rank-base default-replacement audit; all datasets use alpha=0.60.",
        "deployable_default_alpha060_rawbase_splitminus: current internal-gate-passed default-replacement candidate.",
        "```",
        "",
        "Diagnostic variants:",
        "",
        "```text",
        "persistence_scale=0.75: passes the strict gate but is not the default candidate;",
        "it improves the thinnest AP margin slightly while lowering mean AUC/AP.",
        "Keep persistence_scale=1.0 unless a future pre-registered validation",
        "explicitly prioritizes min-AP margin over mean AUC/AP.",
        "```",
        "",
        "Frozen manifest verification:",
        "",
        "```bash",
        "python3 tools/verify_frozen_sample_fallback_manifest.py "
        "--manifest-json results/patch_calibrated_persistence/sample_fallback_preregistered_manifest.json "
        "--output-csv results/patch_calibrated_persistence/sample_fallback_frozen_manifest_verification.csv",
        "```",
        "",
        "Fresh validation scaffold:",
        "",
        "The generated scaffold runs input validation, frozen scoring, source audit, and a single-dataset fresh gate.",
        "",
        "```bash",
        "python3 tools/build_fresh_validation_scaffold.py "
        "--manifest-json results/patch_calibrated_persistence/sample_fallback_preregistered_manifest.json "
        "--dataset <fresh_dataset_name> "
        "--global-csv <path/to/fresh_global_scores.csv> "
        "--raw-patch-csv <path/to/fresh_best_patch_scores.csv> "
        "--persistence-csv <path/to/fresh_persistence_scores.csv> "
        "--persistence-score-col <pre_registered_persistence_column> "
        "--output-json results/patch_calibrated_persistence/<fresh_dataset_name>_fresh_validation_scaffold.json "
        "--output-md results/patch_calibrated_persistence/<fresh_dataset_name>_fresh_validation_scaffold.md",
        "```",
        "",
        "Fresh scaffold verification:",
        "",
        "Run this before executing the scaffold commands, so a stale or edited scaffold fails fast.",
        "",
        "```bash",
        "python3 tools/verify_fresh_validation_scaffold.py "
        "--scaffold-json results/patch_calibrated_persistence/<fresh_dataset_name>_fresh_validation_scaffold.json "
        "--output-csv results/patch_calibrated_persistence/<fresh_dataset_name>_fresh_validation_scaffold_verification.csv",
        "```",
        "",
        *protocol_sections,
        "",
        "Future dataset template:",
        "",
        "```json",
        json.dumps(manifest["future_dataset_template"], indent=2, sort_keys=True),
        "```",
        "",
        "Promotion rule: do not change thresholds, objective, or scorer parameters for the future run.",
    ]
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest-json", type=Path, required=True)
    parser.add_argument("--runbook-md", type=Path, required=True)
    args = parser.parse_args()

    manifest = _manifest()
    args.manifest_json.parent.mkdir(parents=True, exist_ok=True)
    args.runbook_md.parent.mkdir(parents=True, exist_ok=True)
    args.manifest_json.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    args.runbook_md.write_text(_runbook_text(manifest), encoding="utf-8")
    print(f"Saved manifest -> {args.manifest_json}")
    print(f"Saved runbook -> {args.runbook_md}")


if __name__ == "__main__":
    main()
