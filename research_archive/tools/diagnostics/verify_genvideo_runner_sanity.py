#!/usr/bin/env python3
"""Verify the concrete GenVideo runner sanity execution."""

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


def _exists(rows: list[dict[str, object]], label: str, path: Path) -> bool:
    ok = path.exists()
    rows.append(_row(f"exists:{label}", ok, str(path)))
    return ok


def run(args: argparse.Namespace) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    required = {
        "runner_summary": args.runner_summary_csv,
        "runner_steps": args.runner_steps_csv,
        "input_summary": args.input_summary_csv,
        "metrics": args.metrics_csv,
        "source_audit_summary": args.source_audit_summary_csv,
        "gate_decision": args.gate_decision_csv,
        "gate_checks": args.gate_checks_csv,
    }
    if not all(_exists(rows, label, path) for label, path in required.items()):
        return pd.DataFrame(rows)

    runner_summary = pd.read_csv(args.runner_summary_csv)
    runner_steps = pd.read_csv(args.runner_steps_csv)
    input_summary = pd.read_csv(args.input_summary_csv)
    metrics = pd.read_csv(args.metrics_csv)
    source_audit = pd.read_csv(args.source_audit_summary_csv)
    gate_decision = pd.read_csv(args.gate_decision_csv)
    gate_checks = pd.read_csv(args.gate_checks_csv)

    rows.append(_row("runner_decision_pass", str(runner_summary.iloc[0]["decision"]) == "PASS", str(runner_summary.iloc[0]["decision"])))
    rows.append(_row("runner_not_dry_run", not bool(runner_summary.iloc[0]["dry_run"]), str(runner_summary.iloc[0]["dry_run"])))
    rows.append(_row("runner_no_placeholders", not bool(runner_summary.iloc[0]["contains_placeholders"]), str(runner_summary.iloc[0]["contains_placeholders"])))
    rows.append(_row("runner_steps_4", int(runner_summary.iloc[0]["n_steps"]) == 4, str(runner_summary.iloc[0]["n_steps"])))
    rows.append(_row("runner_all_steps_passed", int(runner_summary.iloc[0]["n_passed_or_dry_run"]) == 4, str(runner_summary.iloc[0]["n_passed_or_dry_run"])))
    rows.append(_row("runner_step_order", runner_steps["step"].astype(str).tolist() == EXPECTED_STEPS, str(runner_steps["step"].astype(str).tolist())))
    rows.append(_row("runner_step_status_pass", bool((runner_steps["status"].astype(str) == "PASS").all()), str(runner_steps["status"].astype(str).tolist())))
    rows.append(_row("input_all_aligned", bool(input_summary.iloc[0]["all_inputs_aligned"]), str(input_summary.iloc[0]["all_inputs_aligned"])))
    rows.append(_row("input_has_real", int(input_summary.iloc[0]["n_real"]) > 0, str(input_summary.iloc[0]["n_real"])))
    rows.append(_row("input_has_fake", int(input_summary.iloc[0]["n_fake"]) > 0, str(input_summary.iloc[0]["n_fake"])))
    rows.append(_row("input_fake_sources_8", int(input_summary.iloc[0]["n_fake_sources"]) == 8, str(input_summary.iloc[0]["n_fake_sources"])))
    avg = metrics[metrics["source_model"] == "Average"]
    rows.append(_row("metrics_has_average", len(avg) == 1, str(len(avg))))
    if len(avg) == 1:
        rows.append(_row("metrics_avg_auc_positive", float(avg.iloc[0]["auc"]) > 0, str(avg.iloc[0]["auc"])))
        rows.append(_row("metrics_avg_ap_positive", float(avg.iloc[0]["ap"]) > 0, str(avg.iloc[0]["ap"])))
    rows.append(_row("source_audit_sources_8", int(source_audit.iloc[0]["n_sources"]) == 8, str(source_audit.iloc[0]["n_sources"])))
    rows.append(_row("source_audit_no_negative_auc", int(source_audit.iloc[0]["n_negative_auc"]) == 0, str(source_audit.iloc[0]["n_negative_auc"])))
    rows.append(_row("source_audit_no_negative_ap", int(source_audit.iloc[0]["n_negative_ap"]) == 0, str(source_audit.iloc[0]["n_negative_ap"])))
    rows.append(_row("source_audit_min_delta_auc_positive", float(source_audit.iloc[0]["min_delta_auc"]) > 0, str(source_audit.iloc[0]["min_delta_auc"])))
    rows.append(_row("source_audit_min_delta_ap_positive", float(source_audit.iloc[0]["min_delta_ap"]) > 0, str(source_audit.iloc[0]["min_delta_ap"])))
    rows.append(_row("gate_decision_pass", str(gate_decision.iloc[0]["decision"]) == "PASS", str(gate_decision.iloc[0]["decision"])))
    rows.append(_row("gate_checks_10", int(gate_decision.iloc[0]["n_checks"]) == 10, str(gate_decision.iloc[0]["n_checks"])))
    rows.append(_row("gate_all_checks_passed", int(gate_decision.iloc[0]["n_passed"]) == 10, str(gate_decision.iloc[0]["n_passed"])))
    rows.append(_row("gate_checks_all_true", bool(gate_checks["passed"].astype(bool).all()), f"{int(gate_checks['passed'].astype(bool).sum())}/{len(gate_checks)}"))
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runner-summary-csv", type=Path, required=True)
    parser.add_argument("--runner-steps-csv", type=Path, required=True)
    parser.add_argument("--input-summary-csv", type=Path, required=True)
    parser.add_argument("--metrics-csv", type=Path, required=True)
    parser.add_argument("--source-audit-summary-csv", type=Path, required=True)
    parser.add_argument("--gate-decision-csv", type=Path, required=True)
    parser.add_argument("--gate-checks-csv", type=Path, required=True)
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
