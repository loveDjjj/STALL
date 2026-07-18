#!/usr/bin/env python3
"""Build a pre-registered scaffold for a fresh sample-fallback validation run."""

from __future__ import annotations

import argparse
import json
import shlex
from pathlib import Path
from typing import Any


FROZEN_PROTOCOL = "deployable_default_alpha060_rawbase_splitminus"


def _cmd(parts: list[Any]) -> str:
    return " ".join(shlex.quote(str(p)) for p in parts)


def _dataset_entry(args: argparse.Namespace, manifest: dict[str, Any]) -> dict[str, Any]:
    template = manifest["future_dataset_template"].copy()
    dataset = args.dataset
    template.update(
        {
            "dataset": dataset,
            "global_csv": str(args.global_csv),
            "raw_patch_csv": str(args.raw_patch_csv),
            "persistence_csv": str(args.persistence_csv),
            "persistence_score_col": args.persistence_score_col,
            "final_scores_csv": f"results/patch_calibrated_persistence/{dataset}_sample_fallback_final_scores.csv",
            "metrics_csv": f"results/patch_calibrated_persistence/{dataset}_sample_fallback_metrics.csv",
            "per_source_csv": f"results/patch_calibrated_persistence/sample_fallback_{dataset}_per_source.csv",
        }
    )
    return template


def _apply_command(dataset: dict[str, Any]) -> str:
    return _cmd(
        [
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
            "--base-space",
            dataset["base_space"],
            "--persistence-scale",
            dataset["persistence_scale"],
            "--selector-feature",
            dataset["selector_feature"],
            "--selector-direction",
            dataset["selector_direction"],
            "--selector-threshold",
            dataset["selector_threshold"],
            "--real-policy",
            dataset["real_policy"],
            "--output-csv",
            dataset["final_scores_csv"],
            "--metrics-csv",
            dataset["metrics_csv"],
        ]
    )


def _input_validation_command(dataset: dict[str, Any]) -> str:
    return _cmd(
        [
            "python3",
            "tools/validate_fresh_score_inputs.py",
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
            "--output-summary-csv",
            f"results/patch_calibrated_persistence/{dataset['dataset']}_fresh_input_validation_summary.csv",
            "--output-source-summary-csv",
            f"results/patch_calibrated_persistence/{dataset['dataset']}_fresh_input_validation_sources.csv",
        ]
    )


def _source_audit_command(dataset: dict[str, Any]) -> str:
    return _cmd(
        [
            "python3",
            "tools/audit_fallback_vs_default_by_source.py",
            "--dataset",
            dataset["dataset"],
            "--global-csv",
            dataset["global_csv"],
            "--raw-patch-csv",
            dataset["raw_patch_csv"],
            "--fallback-csv",
            dataset["final_scores_csv"],
            "--output-csv",
            f"results/patch_calibrated_persistence/{dataset['dataset']}_vs_default_by_source.csv",
            "--summary-csv",
            f"results/patch_calibrated_persistence/{dataset['dataset']}_vs_default_by_source_summary.csv",
        ]
    )


def _fresh_gate_command(dataset: dict[str, Any]) -> str:
    return _cmd(
        [
            "python3",
            "tools/fresh_validation_gate.py",
            "--dataset",
            dataset["dataset"],
            "--metrics-csv",
            dataset["metrics_csv"],
            "--source-audit-summary-csv",
            f"results/patch_calibrated_persistence/{dataset['dataset']}_vs_default_by_source_summary.csv",
            "--output-csv",
            f"results/patch_calibrated_persistence/{dataset['dataset']}_fresh_validation_decision.csv",
            "--checks-csv",
            f"results/patch_calibrated_persistence/{dataset['dataset']}_fresh_validation_checks.csv",
        ]
    )


def _scaffold(args: argparse.Namespace) -> dict[str, Any]:
    manifest = json.loads(args.manifest_json.read_text(encoding="utf-8"))
    frozen = manifest["current_promotion_config"]
    dataset = _dataset_entry(args, manifest)
    return {
        "name": "fresh_sample_fallback_validation",
        "status": "pre_registered_not_run",
        "source_manifest": str(args.manifest_json),
        "frozen_protocol": FROZEN_PROTOCOL,
        "frozen_config": frozen,
        "dataset": dataset,
        "commands": {
            "validate_fresh_inputs": _input_validation_command(dataset),
            "apply_frozen_scorer": _apply_command(dataset),
            "source_audit_vs_alpha060_default": _source_audit_command(dataset),
            "fresh_validation_gate": _fresh_gate_command(dataset),
        },
        "acceptance_criteria": {
            "required": [
                "Use the frozen scorer command without changing parameters.",
                "Use raw alpha=0.60 global+patch fusion as the default comparator.",
                "Report dataset Average AUC/AP from the metrics CSV.",
                "Report source-level deltas against alpha=0.60 default.",
                "No source_model may have negative AUC or AP delta unless the run is explicitly marked failed.",
            ],
            "promotion_note": (
                "A single fresh dataset is supporting evidence, not automatic promotion. "
                "Promotion should compare the fresh result with the frozen internal gate and document any source-level regression."
            ),
        },
        "manual_prerequisites": [
            "global_csv exists with subset/source_model/filename/final_score columns.",
            "raw_patch_csv exists with subset/source_model/filename/final_score columns.",
            "persistence_csv exists with subset/source_model/filename and the pre-registered persistence score column.",
            "The fake source_model labels are stable enough for source-level non-regression reporting.",
        ],
    }


def _markdown(scaffold: dict[str, Any]) -> str:
    return "\n".join(
        [
            "# Fresh Sample-Fallback Validation Scaffold",
            "",
            f"Status: {scaffold['status']}",
            "",
            "Frozen config:",
            "",
            "```json",
            json.dumps(scaffold["frozen_config"], indent=2, sort_keys=True),
            "```",
            "",
            "Dataset entry:",
            "",
            "```json",
            json.dumps(scaffold["dataset"], indent=2, sort_keys=True),
            "```",
            "",
            "Prerequisites:",
            "",
            *[f"- {item}" for item in scaffold["manual_prerequisites"]],
            "",
            "Commands:",
            "",
            "```bash",
            scaffold["commands"]["validate_fresh_inputs"],
            scaffold["commands"]["apply_frozen_scorer"],
            scaffold["commands"]["source_audit_vs_alpha060_default"],
            scaffold["commands"]["fresh_validation_gate"],
            "```",
            "",
            "Acceptance criteria:",
            "",
            *[f"- {item}" for item in scaffold["acceptance_criteria"]["required"]],
            "",
            f"Promotion note: {scaffold['acceptance_criteria']['promotion_note']}",
            "",
        ]
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest-json", type=Path, required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--global-csv", required=True)
    parser.add_argument("--raw-patch-csv", required=True)
    parser.add_argument("--persistence-csv", required=True)
    parser.add_argument("--persistence-score-col", required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-md", type=Path, required=True)
    args = parser.parse_args()

    scaffold = _scaffold(args)
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_md.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(scaffold, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    args.output_md.write_text(_markdown(scaffold), encoding="utf-8")
    print(f"Saved scaffold JSON -> {args.output_json}")
    print(f"Saved scaffold Markdown -> {args.output_md}")
    print(scaffold["commands"]["apply_frozen_scorer"])


if __name__ == "__main__":
    main()
