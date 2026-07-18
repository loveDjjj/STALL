#!/usr/bin/env python3
"""Verify the fresh validation intake checklist template."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


REQUIRED_MARKERS = [
    "base_alpha = 0.6",
    "base_space = raw",
    "persistence_scale = 1.0",
    "selector_feature = split_minus_universal",
    "selector_direction = lt",
    "selector_threshold = 0.0",
    "real_policy = split",
    "subset/source_model/filename",
    "tools/verify_current_patch_pipeline.py",
    "tools/build_fresh_validation_scaffold.py",
    "tools/verify_fresh_validation_scaffold.py",
    "tools/run_fresh_validation_scaffold.py",
    "tools/validate_fresh_score_inputs.py",
    "tools/apply_sample_fallback_rule.py",
    "tools/audit_fallback_vs_default_by_source.py",
    "tools/fresh_validation_gate.py",
    "No source_model may have negative AUC or AP delta",
    "single fresh dataset is supporting evidence, not automatic promotion",
    "validate_fresh_inputs, apply_frozen_scorer, source_audit_vs_alpha060_default, fresh_validation_gate",
]

REQUIRED_PHASES = ["preflight", "inputs", "scaffold", "scoring", "audit", "decision"]


def _row(check: str, passed: bool, detail: str) -> dict[str, object]:
    return {"check": check, "passed": bool(passed), "detail": detail}


def run(args: argparse.Namespace) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    md_exists = args.checklist_md.exists()
    csv_exists = args.checklist_csv.exists()
    rows.append(_row("markdown_exists", md_exists, str(args.checklist_md)))
    rows.append(_row("csv_exists", csv_exists, str(args.checklist_csv)))
    if not md_exists or not csv_exists:
        return pd.DataFrame(rows)

    text = args.checklist_md.read_text(encoding="utf-8")
    rows.append(_row("markdown_nonempty", len(text.strip()) > 0, f"{len(text)} chars"))
    for marker in REQUIRED_MARKERS:
        rows.append(_row(f"marker:{marker}", marker in text, marker))

    df = pd.read_csv(args.checklist_csv)
    required_columns = ["phase", "item", "required", "evidence", "failure_action"]
    missing_columns = [col for col in required_columns if col not in df.columns]
    rows.append(_row("csv_required_columns", not missing_columns, str(missing_columns)))
    if missing_columns:
        return pd.DataFrame(rows)

    rows.append(_row("csv_nonempty", len(df) > 0, f"{len(df)} rows"))
    required_bool = df["required"].astype(str).str.lower().isin(["true", "1", "yes"])
    rows.append(_row("all_rows_required", bool(required_bool.all()), f"{int(required_bool.sum())}/{len(df)}"))
    for phase in REQUIRED_PHASES:
        rows.append(_row(f"phase:{phase}", phase in set(df["phase"].astype(str)), phase))
    rows.append(
        _row(
            "failure_actions_present",
            bool(df["failure_action"].astype(str).str.len().gt(0).all()),
            "all rows have failure_action",
        )
    )
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checklist-md", type=Path, required=True)
    parser.add_argument("--checklist-csv", type=Path, required=True)
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
