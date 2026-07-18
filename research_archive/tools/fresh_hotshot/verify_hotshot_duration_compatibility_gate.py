#!/usr/bin/env python3
"""Verify Hotshot duration compatibility gate."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def _row(check: str, passed: bool, detail: str) -> dict[str, object]:
    return {"check": check, "passed": bool(passed), "detail": detail}


def run(args: argparse.Namespace) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    exists = args.gate_csv.exists()
    rows.append(_row("gate_exists", exists, str(args.gate_csv)))
    if not exists:
        return pd.DataFrame(rows)
    gate = pd.read_csv(args.gate_csv)
    rows.append(_row("single_gate_row", len(gate) == 1, str(len(gate))))
    row = gate.iloc[0]
    rows.append(_row("decision_duration_mismatch", str(row["decision"]) == "DURATION_MISMATCH_LOCAL_VARIANT_ONLY", str(row["decision"])))
    rows.append(_row("protocol_not_compatible", not bool(row["protocol_compatible"]), str(row["protocol_compatible"])))
    rows.append(_row("runbook_duration_1", int(row["runbook_duration"]) == 1, str(row["runbook_duration"])))
    rows.append(_row("params_duration_2", int(row["params_duration"]) == 2, str(row["params_duration"])))
    rows.append(_row("target_rows_3251", int(row["target_rows"]) == 3251, str(row["target_rows"])))
    rows.append(_row("target_all_one_second", bool(row["target_all_one_second"]), str(row["target_all_one_second"])))
    rows.append(_row("allowed_next_local_only", str(row["allowed_next"]) == "local_variant_only", str(row["allowed_next"])))
    rows.append(_row("required_label_has_caveat", "duration=1 caveat" in str(row["required_label"]), str(row["required_label"])))
    rows.append(_row("params_mode_second_order", str(row["params_patch_temp_mode"]) == "same_grid_second_order", str(row["params_patch_temp_mode"])))
    rows.append(_row("params_region1", int(row["params_patch_region_size"]) == 1, str(row["params_patch_region_size"])))
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gate-csv", type=Path, required=True)
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
