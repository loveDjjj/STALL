#!/usr/bin/env python3
"""Build a pre-run intake checklist for fresh sample-fallback validation."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


REQUIRED_KEY = "subset/source_model/filename"


CHECKLIST_ROWS = [
    {
        "phase": "preflight",
        "item": "current pipeline verifier passes before preparing a fresh run",
        "required": True,
        "evidence": "results/patch_calibrated_persistence/current_patch_pipeline_verification_summary.csv",
        "failure_action": "stop and fix stale or failing frozen artifacts",
    },
    {
        "phase": "preflight",
        "item": "dataset name is stable and does not reuse a previous result prefix",
        "required": True,
        "evidence": "<fresh_dataset_name>",
        "failure_action": "rename before building scaffold artifacts",
    },
    {
        "phase": "inputs",
        "item": "global_csv exists with subset/source_model/filename/final_score columns",
        "required": True,
        "evidence": "<path/to/fresh_global_scores.csv>",
        "failure_action": "do not run scorer until global input is regenerated",
    },
    {
        "phase": "inputs",
        "item": "raw_patch_csv exists with subset/source_model/filename/final_score columns",
        "required": True,
        "evidence": "<path/to/fresh_best_patch_scores.csv>",
        "failure_action": "do not run scorer until raw patch input is regenerated",
    },
    {
        "phase": "inputs",
        "item": "persistence_csv exists with subset/source_model/filename and the pre-registered persistence score column",
        "required": True,
        "evidence": "<path/to/fresh_persistence_scores.csv>",
        "failure_action": "do not substitute a new persistence feature ad hoc",
    },
    {
        "phase": "inputs",
        "item": "no duplicated subset/source_model/filename keys in any input",
        "required": True,
        "evidence": "tools/validate_fresh_score_inputs.py output",
        "failure_action": "deduplicate upstream and rerun input validation",
    },
    {
        "phase": "inputs",
        "item": "global, raw patch, and persistence inputs merge one-to-one",
        "required": True,
        "evidence": "all_inputs_aligned=True",
        "failure_action": "repair missing rows before scoring",
    },
    {
        "phase": "inputs",
        "item": "fresh dataset has real rows, fake rows, and source_model labels for fake rows",
        "required": True,
        "evidence": "n_real>0, n_fake>0, n_fake_sources>0",
        "failure_action": "fresh run is invalid for AUC/AP source audit",
    },
    {
        "phase": "scaffold",
        "item": "fresh scaffold is generated from the frozen manifest",
        "required": True,
        "evidence": "tools/build_fresh_validation_scaffold.py",
        "failure_action": "rebuild scaffold from manifest rather than editing commands manually",
    },
    {
        "phase": "scaffold",
        "item": "fresh scaffold verifier passes before execution",
        "required": True,
        "evidence": "tools/verify_fresh_validation_scaffold.py",
        "failure_action": "stop and fix scaffold drift",
    },
    {
        "phase": "scaffold",
        "item": "fresh scaffold runner is used to execute the frozen command sequence",
        "required": True,
        "evidence": "tools/run_fresh_validation_scaffold.py",
        "failure_action": "do not manually reorder or skip scaffold commands",
    },
    {
        "phase": "scoring",
        "item": "apply frozen scorer without changing base_alpha/base_space/persistence_scale/selector/real_policy",
        "required": True,
        "evidence": "tools/apply_sample_fallback_rule.py command in scaffold",
        "failure_action": "mark run exploratory, not fresh validation",
    },
    {
        "phase": "audit",
        "item": "audit against raw alpha=0.60 global+patch default by source_model",
        "required": True,
        "evidence": "tools/audit_fallback_vs_default_by_source.py",
        "failure_action": "fresh result cannot support default replacement",
    },
    {
        "phase": "decision",
        "item": "No source_model may have negative AUC or AP delta unless the run is explicitly marked failed",
        "required": True,
        "evidence": "tools/fresh_validation_gate.py decision/checks CSV",
        "failure_action": "do not promote; diagnose source-level regression",
    },
    {
        "phase": "decision",
        "item": "single fresh dataset is supporting evidence, not automatic promotion",
        "required": True,
        "evidence": "final report section for the fresh run",
        "failure_action": "require broader evidence before external default promotion",
    },
]


def _frozen_lines(config: dict[str, Any]) -> list[str]:
    keys = [
        "base_alpha",
        "base_space",
        "persistence_scale",
        "selector_feature",
        "selector_direction",
        "selector_threshold",
        "real_policy",
        "objective",
        "rule",
    ]
    return [f"- {key} = {config[key]}" for key in keys]


def _markdown(manifest: dict[str, Any]) -> str:
    config = manifest["current_promotion_config"]
    template = manifest["future_dataset_template"]
    rows = ["# Fresh Validation Intake Checklist", ""]
    rows.extend(
        [
            "Status: pre-run template for true fresh dataset validation.",
            "",
            "Purpose: freeze the intake checks before any new data result is scored.",
            "",
            "Frozen scorer identity:",
            "",
            *_frozen_lines(config),
            "",
            f"Required row key: {REQUIRED_KEY}",
            "",
            "Fresh dataset placeholders:",
            "",
            "```json",
            json.dumps(template, indent=2, sort_keys=True),
            "```",
            "",
            "Command order:",
            "",
            "```bash",
            "python3 tools/verify_current_patch_pipeline.py --output-csv results/patch_calibrated_persistence/current_patch_pipeline_verification_summary.csv",
            "python3 tools/build_fresh_validation_scaffold.py --manifest-json results/patch_calibrated_persistence/sample_fallback_preregistered_manifest.json --dataset <fresh_dataset_name> --global-csv <path/to/fresh_global_scores.csv> --raw-patch-csv <path/to/fresh_best_patch_scores.csv> --persistence-csv <path/to/fresh_persistence_scores.csv> --persistence-score-col <pre_registered_persistence_column> --output-json results/patch_calibrated_persistence/<fresh_dataset_name>_fresh_validation_scaffold.json --output-md results/patch_calibrated_persistence/<fresh_dataset_name>_fresh_validation_scaffold.md",
            "python3 tools/verify_fresh_validation_scaffold.py --scaffold-json results/patch_calibrated_persistence/<fresh_dataset_name>_fresh_validation_scaffold.json --output-csv results/patch_calibrated_persistence/<fresh_dataset_name>_fresh_validation_scaffold_verification.csv",
            "python3 tools/run_fresh_validation_scaffold.py --scaffold-json results/patch_calibrated_persistence/<fresh_dataset_name>_fresh_validation_scaffold.json --output-summary-csv results/patch_calibrated_persistence/<fresh_dataset_name>_fresh_validation_runner_summary.csv --output-steps-csv results/patch_calibrated_persistence/<fresh_dataset_name>_fresh_validation_runner_steps.csv",
            "```",
            "",
            "Runner executes these scaffold steps in order: validate_fresh_inputs, apply_frozen_scorer, source_audit_vs_alpha060_default, fresh_validation_gate.",
            "",
            "Checklist:",
            "",
            "| phase | required item | evidence | failure action |",
            "|---|---|---|---|",
        ]
    )
    for row in CHECKLIST_ROWS:
        rows.append(
            f"| {row['phase']} | {row['item']} | {row['evidence']} | {row['failure_action']} |"
        )
    rows.extend(
        [
            "",
            "Acceptance boundary:",
            "",
            "- No source_model may have negative AUC or AP delta unless the run is explicitly marked failed.",
            "- A single fresh dataset is supporting evidence, not automatic promotion.",
            "- Any changed scorer parameter makes the run exploratory rather than frozen fresh validation.",
            "",
        ]
    )
    return "\n".join(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest-json", type=Path, required=True)
    parser.add_argument("--output-md", type=Path, required=True)
    parser.add_argument("--output-csv", type=Path, required=True)
    args = parser.parse_args()

    manifest = json.loads(args.manifest_json.read_text(encoding="utf-8"))
    args.output_md.parent.mkdir(parents=True, exist_ok=True)
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    args.output_md.write_text(_markdown(manifest), encoding="utf-8")
    with args.output_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["phase", "item", "required", "evidence", "failure_action"])
        writer.writeheader()
        writer.writerows(CHECKLIST_ROWS)
    print(f"Saved checklist Markdown -> {args.output_md}")
    print(f"Saved checklist CSV -> {args.output_csv}")


if __name__ == "__main__":
    main()
