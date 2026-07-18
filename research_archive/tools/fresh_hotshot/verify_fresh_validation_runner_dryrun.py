#!/usr/bin/env python3
"""Verify a dry-run output from run_fresh_validation_scaffold.py."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


EXPECTED_STEPS = [
    "validate_fresh_inputs",
    "apply_frozen_scorer",
    "source_audit_vs_alpha060_default",
    "fresh_validation_gate",
]


def _row(check: str, passed: bool, detail: str) -> dict[str, object]:
    return {"check": check, "passed": bool(passed), "detail": detail}


def run(args: argparse.Namespace) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    summary_exists = args.summary_csv.exists()
    steps_exists = args.steps_csv.exists()
    rows.append(_row("summary_exists", summary_exists, str(args.summary_csv)))
    rows.append(_row("steps_exists", steps_exists, str(args.steps_csv)))
    if not summary_exists or not steps_exists:
        return pd.DataFrame(rows)

    summary = pd.read_csv(args.summary_csv)
    steps = pd.read_csv(args.steps_csv)
    rows.append(_row("single_summary_row", len(summary) == 1, str(len(summary))))
    rows.append(_row("decision_dry_run_pass", str(summary.iloc[0].get("decision")) == "DRY_RUN_PASS", str(summary.iloc[0].get("decision"))))
    rows.append(_row("dry_run_true", bool(summary.iloc[0].get("dry_run")), str(summary.iloc[0].get("dry_run"))))
    rows.append(_row("placeholder_detected", bool(summary.iloc[0].get("contains_placeholders")), str(summary.iloc[0].get("contains_placeholders"))))
    actual_steps = steps["step"].astype(str).tolist() if "step" in steps.columns else []
    rows.append(_row("step_order", actual_steps == EXPECTED_STEPS, str(actual_steps)))
    rows.append(_row("all_steps_dry_run", bool((steps["status"].astype(str) == "DRY_RUN").all()), str(steps.get("status", []))))
    rows.append(_row("all_returncodes_zero", bool((steps["returncode"].fillna(-1).astype(int) == 0).all()), str(steps.get("returncode", []))))
    required_commands = [
        "tools/validate_fresh_score_inputs.py",
        "tools/apply_sample_fallback_rule.py",
        "tools/audit_fallback_vs_default_by_source.py",
        "tools/fresh_validation_gate.py",
    ]
    commands_text = "\n".join(steps["command"].astype(str).tolist()) if "command" in steps.columns else ""
    for command_marker in required_commands:
        rows.append(_row(f"command_marker:{command_marker}", command_marker in commands_text, command_marker))
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary-csv", type=Path, required=True)
    parser.add_argument("--steps-csv", type=Path, required=True)
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
