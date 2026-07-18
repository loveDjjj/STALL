#!/usr/bin/env python3
"""Verify executed results for the duration=2 filtered demo branch."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def _row(check: str, passed: bool, detail: str) -> dict[str, object]:
    return {"check": check, "passed": bool(passed), "detail": detail}


def run(args: argparse.Namespace) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    paths = {
        "global_scores": args.global_scores_csv,
        "raw_patch_scores": args.raw_patch_scores_csv,
        "persistence_scores": args.persistence_scores_csv,
        "patch_cache_summary": args.patch_cache_summary_csv,
        "scaffold_verification": args.scaffold_verification_csv,
        "runner_summary": args.runner_summary_csv,
        "metrics": args.metrics_csv,
        "source_audit_summary": args.source_audit_summary_csv,
        "decision": args.decision_csv,
        "checks": args.checks_csv,
    }
    for name, path in paths.items():
        rows.append(_row(f"{name}_exists", path.exists(), str(path)))
    if not all(path.exists() for path in paths.values()):
        return pd.DataFrame(rows)

    global_scores = pd.read_csv(args.global_scores_csv)
    raw_patch = pd.read_csv(args.raw_patch_scores_csv)
    persistence = pd.read_csv(args.persistence_scores_csv)
    patch_cache = pd.read_csv(args.patch_cache_summary_csv)
    scaffold_verification = pd.read_csv(args.scaffold_verification_csv)
    runner = pd.read_csv(args.runner_summary_csv)
    metrics = pd.read_csv(args.metrics_csv)
    source_audit = pd.read_csv(args.source_audit_summary_csv)
    decision = pd.read_csv(args.decision_csv)
    checks = pd.read_csv(args.checks_csv)

    rows.extend(
        [
            _row("global_rows_34", len(global_scores) == 34, f"rows={len(global_scores)}"),
            _row("raw_patch_rows_34", len(raw_patch) == 34, f"rows={len(raw_patch)}"),
            _row("persistence_rows_34", len(persistence) == 34, f"rows={len(persistence)}"),
            _row("patch_cache_no_misses", int(patch_cache.iloc[0]["misses_before"]) == 0, patch_cache.to_string(index=False)),
            _row("patch_cache_files_34", args.patch_cache_file_count == 34, str(args.patch_cache_file_count)),
            _row("scaffold_verification_pass_46", len(scaffold_verification) == 46 and bool(scaffold_verification["passed"].astype(bool).all()), f"rows={len(scaffold_verification)}"),
            _row("runner_pass", str(runner.iloc[0]["decision"]) == "PASS", runner.to_string(index=False)),
            _row("runner_four_steps", int(runner.iloc[0]["n_steps"]) == 4 and int(runner.iloc[0]["n_passed_or_dry_run"]) == 4, runner.to_string(index=False)),
            _row("metrics_rows_8", len(metrics) == 8, f"rows={len(metrics)}"),
            _row("metrics_average_auc_1", float(metrics.loc[metrics["source_model"] == "Average", "auc"].iloc[0]) == 1.0, metrics.to_string(index=False)),
            _row("metrics_average_ap_1", float(metrics.loc[metrics["source_model"] == "Average", "ap"].iloc[0]) == 1.0, metrics.to_string(index=False)),
            _row("source_audit_7_sources", int(source_audit.iloc[0]["n_sources"]) == 7, source_audit.to_string(index=False)),
            _row("source_audit_no_negative_auc", int(source_audit.iloc[0]["n_negative_auc"]) == 0, source_audit.to_string(index=False)),
            _row("source_audit_no_negative_ap", int(source_audit.iloc[0]["n_negative_ap"]) == 0, source_audit.to_string(index=False)),
            _row("source_audit_zero_mean_delta_auc", float(source_audit.iloc[0]["mean_delta_auc"]) == 0.0, source_audit.to_string(index=False)),
            _row("source_audit_zero_mean_delta_ap", float(source_audit.iloc[0]["mean_delta_ap"]) == 0.0, source_audit.to_string(index=False)),
            _row("decision_pass", str(decision.iloc[0]["decision"]) == "PASS", decision.to_string(index=False)),
            _row("decision_10_checks", int(decision.iloc[0]["n_checks"]) == 10 and int(decision.iloc[0]["n_passed"]) == 10, decision.to_string(index=False)),
            _row("checks_all_passed", bool(checks["passed"].astype(bool).all()), checks.to_string(index=False)),
        ]
    )
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--global-scores-csv", type=Path, default=Path("results/patch_calibrated_persistence/demo_dataset_duration2_filtered_global_scores.csv"))
    parser.add_argument("--raw-patch-scores-csv", type=Path, default=Path("results/patch_calibrated_persistence/demo_dataset_duration2_filtered_raw_patch_scores.csv"))
    parser.add_argument("--persistence-scores-csv", type=Path, default=Path("results/patch_calibrated_persistence/demo_dataset_duration2_filtered_persistence_scores.csv"))
    parser.add_argument("--patch-cache-summary-csv", type=Path, default=Path("results/patch_calibrated_persistence/demo_dataset_duration2_filtered_patch_cache_prefill_summary.csv"))
    parser.add_argument("--scaffold-verification-csv", type=Path, default=Path("results/patch_calibrated_persistence/demo_dataset_duration2_filtered_fresh_validation_scaffold_verification.csv"))
    parser.add_argument("--runner-summary-csv", type=Path, default=Path("results/patch_calibrated_persistence/demo_dataset_duration2_filtered_fresh_validation_runner_summary.csv"))
    parser.add_argument("--metrics-csv", type=Path, default=Path("results/patch_calibrated_persistence/demo_dataset_duration2_filtered_sample_fallback_metrics.csv"))
    parser.add_argument("--source-audit-summary-csv", type=Path, default=Path("results/patch_calibrated_persistence/demo_dataset_duration2_filtered_vs_default_by_source_summary.csv"))
    parser.add_argument("--decision-csv", type=Path, default=Path("results/patch_calibrated_persistence/demo_dataset_duration2_filtered_fresh_validation_decision.csv"))
    parser.add_argument("--checks-csv", type=Path, default=Path("results/patch_calibrated_persistence/demo_dataset_duration2_filtered_fresh_validation_checks.csv"))
    parser.add_argument("--patch-cache-file-count", type=int, required=True)
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
