#!/usr/bin/env python3
"""Verify the patch candidate decision board against expected branch states."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


EXPECTED_STATUS = {
    "frozen_rawbase_splitminus_candidate": "PROMOTE_INTERNAL_DEFAULT",
    "primitive_action_headroom": "REJECT_MORE_SOURCE_LEVEL_THREE_WAY_ROUTING",
    "posthoc_hard_tail_score_cap": "REJECT",
    "residual_tail_feature_signal": "DIAGNOSTIC_SIGNAL_CONFIRMED",
    "residual_tail_feature_gate": "HOLD_DIAGNOSTIC_ONLY",
    "fresh_validation_pipeline": "READY_NOT_RUN",
    "hotshot_local_variant_pipeline": "COMPLETED_LOCAL_VARIANT_WITH_CAVEAT",
}

EXPECTED_MARKERS = {
    "frozen_rawbase_splitminus_candidate": [
        "strict_gate=PASS",
        "min delta AUC/AP=+0.006018/+0.000101",
        "+Hotshot local 31-source negatives=0/0",
    ],
    "primitive_action_headroom": ["sources_with_primitive_headroom=0"],
    "posthoc_hard_tail_score_cap": ["best deployable mean delta AUC/AP=+0.000000/+0.000000"],
    "residual_tail_feature_signal": ["AUC=0.989276"],
    "residual_tail_feature_gate": ["negative heldout sources", "-0.001516/-0.006000"],
    "fresh_validation_pipeline": [
        "PASS 46/46",
        "WAITING_FOR_TRUE_FRESH_DATASET",
        "fresh_like_ready_count=0",
        "protocol-compatible fresh dataset",
    ],
    "hotshot_local_variant_pipeline": [
        "local delta AUC/AP=+0.045099/+0.031405",
        "Hotshot-XL AUC/AP=0.832344/0.871717",
        "DURATION_MISMATCH_LOCAL_VARIANT_ONLY",
        "completed local stress-test evidence only",
    ],
}


def _add(rows: list[dict[str, object]], check: str, passed: bool, detail: str) -> None:
    rows.append({"check": check, "passed": bool(passed), "detail": detail})


def _evidence_paths(value: str) -> list[Path]:
    paths = []
    for part in value.split(";"):
        item = part.strip()
        if item:
            paths.append(Path(item))
    return paths


def run(args: argparse.Namespace) -> pd.DataFrame:
    board = pd.read_csv(args.board_csv)
    rows: list[dict[str, object]] = []
    branches = set(board["branch"].astype(str))
    _add(rows, "branch_count", len(board) == len(EXPECTED_STATUS), f"actual={len(board)}, expected={len(EXPECTED_STATUS)}")
    for branch, expected_status in EXPECTED_STATUS.items():
        exists = branch in branches
        _add(rows, f"branch_exists:{branch}", exists, branch)
        if not exists:
            continue
        item = board[board["branch"] == branch].iloc[0]
        status = str(item["status"])
        _add(rows, f"status:{branch}", status == expected_status, f"actual={status}, expected={expected_status}")
        text = "\n".join(str(item.get(col, "")) for col in ["primary_result", "risk_result", "next_action"])
        for marker in EXPECTED_MARKERS[branch]:
            _add(rows, f"marker:{branch}:{marker}", marker in text, text)
        for path in _evidence_paths(str(item["evidence"])):
            _add(rows, f"evidence_exists:{branch}:{path}", path.exists(), str(path))

    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--board-csv", type=Path, required=True)
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
