#!/usr/bin/env python3
"""Verify the current frozen patch pipeline audit artifacts."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


DEFAULT_CHECKS = [
    (
        "frozen_manifest",
        Path("results/patch_calibrated_persistence/sample_fallback_frozen_manifest_verification.csv"),
        68,
    ),
    (
        "fresh_scaffold",
        Path("results/patch_calibrated_persistence/fresh_validation_scaffold_template_verification.csv"),
        46,
    ),
    (
        "fresh_intake_checklist",
        Path("results/patch_calibrated_persistence/fresh_validation_intake_checklist_template_verification.csv"),
        32,
    ),
    (
        "fresh_runner_dryrun",
        Path("results/patch_calibrated_persistence/fresh_validation_runner_dryrun_verification.csv"),
        13,
    ),
    (
        "genvideo_runner_sanity",
        Path("results/patch_calibrated_persistence/genvideo_runner_sanity_verification.csv"),
        30,
    ),
    (
        "fresh_candidate_inventory",
        Path("results/patch_calibrated_persistence/fresh_candidate_inventory_verification.csv"),
        23,
    ),
    (
        "fresh_promotion_readiness_gap",
        Path("results/patch_calibrated_persistence/fresh_promotion_readiness_gap_verification.csv"),
        15,
    ),
    (
        "local_fresh_candidate_discovery",
        Path("results/patch_calibrated_persistence/local_fresh_candidate_discovery_verification.csv"),
        10,
    ),
    (
        "demo_fresh_prep_gap",
        Path("results/patch_calibrated_persistence/demo_fresh_prep_gap_verification.csv"),
        21,
    ),
    (
        "demo_fresh_score_runbook",
        Path("results/patch_calibrated_persistence/demo_fresh_score_runbook_verification.csv"),
        30,
    ),
    (
        "demo_duration2_filtered_index",
        Path("results/patch_calibrated_persistence/demo_dataset_duration2_filtered_verification.csv"),
        21,
    ),
    (
        "demo_duration2_filtered_score_runbook",
        Path("results/patch_calibrated_persistence/demo_dataset_duration2_filtered_score_runbook_verification.csv"),
        22,
    ),
    (
        "demo_duration2_filtered_result",
        Path("results/patch_calibrated_persistence/demo_dataset_duration2_filtered_result_verification.csv"),
        29,
    ),
    (
        "hotshot_scaffold_gap",
        Path("results/patch_calibrated_persistence/hotshot_scaffold_gap_verification.csv"),
        14,
    ),
    (
        "hotshot_completion_runbook",
        Path("results/patch_calibrated_persistence/hotshot_completion_runbook_verification.csv"),
        27,
    ),
    (
        "hotshot_cache_prefill_plan",
        Path("results/patch_calibrated_persistence/hotshot_cache_prefill_plan_verification.csv"),
        10,
    ),
    (
        "hotshot_duration_compatibility",
        Path("results/patch_calibrated_persistence/hotshot_duration_compatibility_verification.csv"),
        12,
    ),
    (
        "hotshot_merge_preparation",
        Path("results/patch_calibrated_persistence/hotshot_merge_preparation_verification.csv"),
        9,
    ),
    (
        "hotshot_plus_source_audit",
        Path("results/patch_calibrated_persistence/hotshot_plus_source_audit_verification.csv"),
        85,
    ),
    (
        "decision_board",
        Path("results/patch_calibrated_persistence/patch_candidate_decision_board_verification.csv"),
        46,
    ),
]


def _verify_one(name: str, path: Path, expected_checks: int) -> dict[str, object]:
    if not path.exists():
        return {
            "artifact": name,
            "path": str(path),
            "passed": False,
            "n_checks": 0,
            "n_passed": 0,
            "expected_checks": expected_checks,
            "detail": "missing",
        }
    df = pd.read_csv(path)
    if "passed" not in df.columns:
        return {
            "artifact": name,
            "path": str(path),
            "passed": False,
            "n_checks": int(len(df)),
            "n_passed": 0,
            "expected_checks": expected_checks,
            "detail": "missing passed column",
        }
    n_checks = int(len(df))
    n_passed = int(df["passed"].astype(bool).sum())
    ok = n_checks == expected_checks and n_passed == n_checks
    return {
        "artifact": name,
        "path": str(path),
        "passed": ok,
        "n_checks": n_checks,
        "n_passed": n_passed,
        "expected_checks": expected_checks,
        "detail": "ok" if ok else f"expected {expected_checks} all-pass checks",
    }


def run(args: argparse.Namespace) -> pd.DataFrame:
    rows = []
    for name, rel_path, expected in DEFAULT_CHECKS:
        rows.append(_verify_one(name, args.root / rel_path, expected))
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--output-csv", type=Path, required=True)
    args = parser.parse_args()

    summary = run(args)
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(args.output_csv, index=False)
    n_passed = int(summary["passed"].sum())
    print(f"PASS {n_passed}/{len(summary)}" if n_passed == len(summary) else f"FAIL {n_passed}/{len(summary)}")
    print(summary.to_string(index=False))
    if n_passed != len(summary):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
