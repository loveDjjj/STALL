#!/usr/bin/env python3
"""Summarize the remaining gap to promotion-grade fresh validation."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def _read(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    return pd.read_csv(path)


def _bool_series(df: pd.DataFrame, column: str) -> pd.Series:
    return df[column].astype(str).str.lower().isin({"true", "1", "yes"})


def run(args: argparse.Namespace) -> pd.DataFrame:
    inventory = _read(args.inventory_csv)
    pipeline = _read(args.pipeline_summary_csv)
    board = _read(args.decision_board_csv)

    ready = inventory[_bool_series(inventory, "scaffold_ready")].copy()
    fresh_like = ready[ready["fresh_status"].astype(str) == "LOCAL_READY_NEEDS_PROVENANCE"]
    hotshot = inventory[inventory["candidate"].astype(str) == "videofeedback_hotshot"]
    fresh_branch = board[board["branch"].astype(str) == "fresh_validation_pipeline"]

    pipeline_passed = bool((pipeline["passed"].astype(str).str.lower() == "true").all())
    hotshot_status = str(hotshot.iloc[0]["fresh_status"]) if len(hotshot) == 1 else "MISSING"
    hotshot_scaffold_ready = bool(_bool_series(hotshot, "scaffold_ready").iloc[0]) if len(hotshot) == 1 else False
    fresh_branch_status = str(fresh_branch.iloc[0]["status"]) if len(fresh_branch) == 1 else "MISSING"

    promotion_ready = (
        pipeline_passed
        and fresh_branch_status == "READY_NOT_RUN"
        and len(fresh_like) > 0
    )
    if promotion_ready:
        status = "READY_FOR_FRESH_RUN_NEEDS_PROVENANCE_REVIEW"
        blocking_reason = "at least one scaffold-ready candidate is not pre-classified as internal/local variant"
        next_action = "review provenance, build scaffold, and run frozen fresh validation"
    else:
        status = "WAITING_FOR_TRUE_FRESH_DATASET"
        blocking_reason = (
            "no scaffold-ready candidate has fresh_status=LOCAL_READY_NEEDS_PROVENANCE; "
            "existing scaffold-ready candidates are internal datasets or local variants"
        )
        next_action = "provide or generate a protocol-compatible fresh dataset with global, raw-patch, and persistence scores"

    row = {
        "status": status,
        "pipeline_passed": pipeline_passed,
        "fresh_branch_status": fresh_branch_status,
        "candidate_count": int(len(inventory)),
        "scaffold_ready_count": int(len(ready)),
        "fresh_like_ready_count": int(len(fresh_like)),
        "internal_or_variant_ready_count": int(
            ready["fresh_status"].astype(str).isin(
                {"NOT_FRESH_INTERNAL_OR_VARIANT", "NOT_FRESH_LOCAL_VARIANT_WITH_DURATION_CAVEAT"}
            ).sum()
        ),
        "hotshot_status": hotshot_status,
        "hotshot_scaffold_ready": hotshot_scaffold_ready,
        "hotshot_duration_caveat": "DURATION_CAVEAT" in hotshot_status,
        "blocking_reason": blocking_reason,
        "next_action": next_action,
    }
    return pd.DataFrame([row])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--inventory-csv", type=Path, default=Path("results/patch_calibrated_persistence/fresh_candidate_inventory.csv"))
    parser.add_argument("--pipeline-summary-csv", type=Path, default=Path("results/patch_calibrated_persistence/current_patch_pipeline_verification_summary.csv"))
    parser.add_argument("--decision-board-csv", type=Path, default=Path("results/patch_calibrated_persistence/patch_candidate_decision_board.csv"))
    parser.add_argument("--output-csv", type=Path, required=True)
    args = parser.parse_args()

    out = run(args)
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.output_csv, index=False)
    print(out.to_string(index=False))
    print(f"Saved readiness gap -> {args.output_csv}")


if __name__ == "__main__":
    main()
