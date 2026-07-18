#!/usr/bin/env python3
"""Verify a generated fresh validation scaffold before running it."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


REQUIRED_CONFIG = {
    "base_alpha": 0.6,
    "base_space": "raw",
    "persistence_scale": 1.0,
    "selector_feature": "split_minus_universal",
    "selector_direction": "lt",
    "selector_threshold": 0.0,
    "real_policy": "split",
}

REQUIRED_COMMANDS = [
    "validate_fresh_inputs",
    "apply_frozen_scorer",
    "source_audit_vs_alpha060_default",
    "fresh_validation_gate",
]

COMMAND_MARKERS = {
    "validate_fresh_inputs": ["tools/validate_fresh_score_inputs.py", "--persistence-score-col"],
    "apply_frozen_scorer": [
        "tools/apply_sample_fallback_rule.py",
        "--base-space raw",
        "--persistence-scale 1.0",
        "--selector-feature split_minus_universal",
        "--real-policy split",
    ],
    "source_audit_vs_alpha060_default": ["tools/audit_fallback_vs_default_by_source.py", "--fallback-csv"],
    "fresh_validation_gate": ["tools/fresh_validation_gate.py", "--source-audit-summary-csv"],
}


def _add(rows: list[dict[str, object]], check: str, passed: bool, detail: str) -> None:
    rows.append({"check": check, "passed": bool(passed), "detail": detail})


def run(args: argparse.Namespace) -> pd.DataFrame:
    scaffold = json.loads(args.scaffold_json.read_text(encoding="utf-8"))
    rows: list[dict[str, object]] = []

    _add(rows, "name", scaffold.get("name") == "fresh_sample_fallback_validation", str(scaffold.get("name")))
    _add(rows, "status", scaffold.get("status") == "pre_registered_not_run", str(scaffold.get("status")))
    _add(
        rows,
        "frozen_protocol",
        scaffold.get("frozen_protocol") == "deployable_default_alpha060_rawbase_splitminus",
        str(scaffold.get("frozen_protocol")),
    )

    frozen = scaffold.get("frozen_config", {})
    dataset = scaffold.get("dataset", {})
    for key, expected in REQUIRED_CONFIG.items():
        _add(rows, f"frozen_config:{key}", frozen.get(key) == expected, f"actual={frozen.get(key)!r}, expected={expected!r}")
        _add(rows, f"dataset:{key}", dataset.get(key) == expected, f"actual={dataset.get(key)!r}, expected={expected!r}")

    dataset_name = str(dataset.get("dataset", ""))
    _add(rows, "dataset_name_present", bool(dataset_name), dataset_name)
    for path_key in ["global_csv", "raw_patch_csv", "persistence_csv", "persistence_score_col"]:
        value = str(dataset.get(path_key, ""))
        _add(rows, f"dataset_input:{path_key}", bool(value), value)
    for path_key in ["final_scores_csv", "metrics_csv", "per_source_csv"]:
        value = str(dataset.get(path_key, ""))
        _add(rows, f"dataset_output:{path_key}", dataset_name in value, value)

    commands = scaffold.get("commands", {})
    for command_name in REQUIRED_COMMANDS:
        command = str(commands.get(command_name, ""))
        _add(rows, f"command_exists:{command_name}", bool(command), command)
        for marker in COMMAND_MARKERS[command_name]:
            _add(rows, f"command_marker:{command_name}:{marker}", marker in command, command)

    criteria = scaffold.get("acceptance_criteria", {}).get("required", [])
    joined = "\n".join(str(x) for x in criteria)
    _add(rows, "acceptance:has_required_items", len(criteria) >= 5, str(criteria))
    _add(rows, "acceptance:frozen_scorer", "frozen scorer" in joined, joined)
    _add(rows, "acceptance:alpha060_default", "alpha=0.60" in joined, joined)
    _add(rows, "acceptance:source_delta", "source-level" in joined or "source_model" in joined, joined)

    prereqs = scaffold.get("manual_prerequisites", [])
    prereq_text = "\n".join(str(x) for x in prereqs)
    _add(rows, "prereq:key_columns", "subset/source_model/filename" in prereq_text, prereq_text)
    _add(rows, "prereq:persistence_column", "persistence score column" in prereq_text, prereq_text)

    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scaffold-json", type=Path, required=True)
    parser.add_argument("--output-csv", type=Path, required=True)
    args = parser.parse_args()

    checks = run(args)
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    checks.to_csv(args.output_csv, index=False)
    n_passed = int(checks["passed"].sum())
    print(f"PASS {n_passed}/{len(checks)}" if n_passed == len(checks) else f"FAIL {n_passed}/{len(checks)}")
    print(checks.to_string(index=False))
    if n_passed != len(checks):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
