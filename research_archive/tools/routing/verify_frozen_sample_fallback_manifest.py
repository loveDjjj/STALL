#!/usr/bin/env python3
"""Verify the frozen sample-fallback manifest and strict promotion artifacts."""

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
    "objective": "mean_universal_then_base",
    "rule": "split_minus_universal/0.0/lt/split",
}

REQUIRED_DATASETS = {
    "comgenvid_full_alpha060_rawbase_splitminus",
    "genvideo_full_alpha060_rawbase_splitminus",
    "videofeedback_full_alpha060_rawbase_splitminus",
}


def _add(rows: list[dict[str, object]], check: str, passed: bool, detail: str) -> None:
    rows.append({"check": check, "passed": bool(passed), "detail": detail})


def _exists(path: Path) -> bool:
    return path.exists() and path.stat().st_size > 0


def run(args: argparse.Namespace) -> pd.DataFrame:
    root = args.root
    manifest = json.loads(args.manifest_json.read_text(encoding="utf-8"))
    rows: list[dict[str, object]] = []

    config = manifest.get("current_promotion_config", {})
    for key, expected in REQUIRED_CONFIG.items():
        actual = config.get(key)
        _add(rows, f"current_config:{key}", actual == expected, f"actual={actual!r}, expected={expected!r}")

    protocol = manifest.get("protocols", {}).get(args.protocol)
    _add(rows, "protocol_exists", protocol is not None, args.protocol)
    if protocol is None:
        return pd.DataFrame(rows)

    datasets = protocol.get("datasets", [])
    dataset_names = {d.get("dataset") for d in datasets}
    _add(
        rows,
        "protocol_datasets",
        dataset_names == REQUIRED_DATASETS,
        f"actual={sorted(dataset_names)}, expected={sorted(REQUIRED_DATASETS)}",
    )
    for dataset in datasets:
        name = str(dataset.get("dataset"))
        for key, expected in REQUIRED_CONFIG.items():
            if key in {"objective", "rule"}:
                continue
            actual = dataset.get(key)
            _add(rows, f"dataset_config:{name}:{key}", actual == expected, f"actual={actual!r}, expected={expected!r}")
        for path_key in ["global_csv", "raw_patch_csv", "persistence_csv", "final_scores_csv", "metrics_csv", "per_source_csv"]:
            path = root / str(dataset.get(path_key))
            _add(rows, f"dataset_path:{name}:{path_key}", _exists(path), str(path))

    commands = protocol.get("commands", {})
    scorer_commands = commands.get("apply_fixed_scorer", [])
    for i, command in enumerate(scorer_commands):
        has_scale = "--persistence-scale 1.0" in command
        has_raw = "--base-space raw" in command
        has_selector = "--selector-feature split_minus_universal" in command
        _add(rows, f"scorer_command:{i}:persistence_scale", has_scale, command)
        _add(rows, f"scorer_command:{i}:base_space", has_raw, command)
        _add(rows, f"scorer_command:{i}:selector", has_selector, command)

    outputs = protocol.get("outputs", {})
    for key in ["validation_csv", "validation_summary_csv", "final_score_summary_csv", "promotion_decision_csv", "promotion_checks_csv"]:
        path = root / str(outputs.get(key))
        _add(rows, f"output_exists:{key}", _exists(path), str(path))

    source_audit_path = root / str(protocol.get("source_audit_summary_csv"))
    _add(rows, "output_exists:source_audit_summary_csv", _exists(source_audit_path), str(source_audit_path))

    decision_path = root / str(outputs.get("promotion_decision_csv"))
    if _exists(decision_path):
        decision = pd.read_csv(decision_path)
        passed = (
            len(decision) == 1
            and decision.iloc[0].get("decision") == "PASS"
            and int(decision.iloc[0].get("n_checks")) == args.expected_checks
            and int(decision.iloc[0].get("n_passed")) == args.expected_checks
        )
        _add(rows, "promotion_decision", passed, decision.to_dict(orient="records").__repr__())

    final_summary_path = root / str(outputs.get("final_score_summary_csv"))
    if _exists(final_summary_path):
        summary = pd.read_csv(final_summary_path)
        mean = summary[summary["dataset"] == "Mean"]
        passed = (
            len(summary[summary["dataset"] != "Mean"]) == args.expected_datasets
            and len(mean) == 1
            and int(mean.iloc[0]["n_sources"]) == args.expected_sources
            and float(mean.iloc[0]["avg_auc"]) >= args.min_mean_auc
            and float(mean.iloc[0]["avg_ap"]) >= args.min_mean_ap
        )
        _add(rows, "final_score_summary", passed, summary.to_dict(orient="records").__repr__())

    if _exists(source_audit_path):
        audit = pd.read_csv(source_audit_path)
        all_rows = audit[audit["dataset"] == "ALL"]
        passed = (
            len(all_rows) == 1
            and int(all_rows.iloc[0]["n_sources"]) == args.expected_sources
            and int(all_rows.iloc[0]["n_negative_auc"]) == 0
            and int(all_rows.iloc[0]["n_negative_ap"]) == 0
            and float(all_rows.iloc[0]["min_delta_auc"]) >= 0.0
            and float(all_rows.iloc[0]["min_delta_ap"]) >= 0.0
        )
        _add(rows, "source_audit_summary", passed, all_rows.to_dict(orient="records").__repr__())

    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest-json", type=Path, required=True)
    parser.add_argument("--output-csv", type=Path, required=True)
    parser.add_argument("--protocol", default="deployable_default_alpha060_rawbase_splitminus")
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--expected-checks", type=int, default=19)
    parser.add_argument("--expected-datasets", type=int, default=3)
    parser.add_argument("--expected-sources", type=int, default=20)
    parser.add_argument("--min-mean-auc", type=float, default=0.89)
    parser.add_argument("--min-mean-ap", type=float, default=0.947)
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
