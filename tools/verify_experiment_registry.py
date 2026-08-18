#!/usr/bin/env python3
"""Validate the experiment registry and its locked-U0 authoritative row."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Mapping

import yaml


ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from alpha_stalled.experiment_registry import (
    EVIDENCE_REQUIRED_STATUSES,
    read_registry,
    validate_registry,
)


DEFAULT_REGISTRY = ROOT / "reports/u0_experiment_registry.csv"
DEFAULT_CONFIG = ROOT / "configs/alpha_stalled_u0_locked.yaml"
LOCKED_EXPERIMENT_ID = "alpha_stalled_u0_locked"
CSV_METRIC_EVIDENCE = {
    "original_stall_strict20_k1": (
        "results/research_summary/experiment_metrics.csv",
        {"dataset": "Macro-3", "method": "original_stall_strict20_k1"},
        "auc",
        "ap",
    ),
    "u0_unified_k1": (
        "results/research_summary/experiment_metrics.csv",
        {"dataset": "Macro-3", "method": "unified_k1_region1_mean"},
        "auc",
        "ap",
    ),
    "u0_k3_global_only": (
        "results/research_summary/experiment_metrics.csv",
        {"dataset": "Macro-3", "method": "k3_global_only"},
        "auc",
        "ap",
    ),
    "u0_k3_local_only": (
        "results/research_summary/experiment_metrics.csv",
        {"dataset": "Macro-3", "method": "k3_local_only"},
        "auc",
        "ap",
    ),
    "u0_local_d1_only": (
        "results/research_summary/experiment_metrics.csv",
        {"dataset": "Macro-3", "method": "local_d1_only"},
        "auc",
        "ap",
    ),
    "u0_local_d2_only": (
        "results/research_summary/experiment_metrics.csv",
        {"dataset": "Macro-3", "method": "local_d2_only"},
        "auc",
        "ap",
    ),
    "u0_full_local_d1": (
        "results/research_summary/experiment_metrics.csv",
        {"dataset": "Macro-3", "method": "full_local_d1"},
        "auc",
        "ap",
    ),
    "u0_external_genvidbench": (
        "results/research_summary/experiment_metrics.csv",
        {"dataset": "GenVidBench", "method": "locked_u0_k3"},
        "auc",
        "ap",
    ),
}


def _read_csv_metric(
    path: Path,
    selector: Mapping[str, str],
    auc_column: str,
    ap_column: str,
) -> tuple[float, float]:
    with path.open(newline="", encoding="utf-8") as handle:
        matches = [
            row
            for row in csv.DictReader(handle)
            if all(row.get(key) == value for key, value in selector.items())
        ]
    if len(matches) != 1:
        raise AssertionError(
            f"metric evidence selector matched {len(matches)} rows: {path} {selector}"
        )
    return float(matches[0][auc_column]), float(matches[0][ap_column])


def verify(
    registry_path: Path = DEFAULT_REGISTRY,
    config_path: Path = DEFAULT_CONFIG,
) -> dict[str, object]:
    rows, columns = read_registry(registry_path)
    summary = validate_registry(rows, columns, repository_root=ROOT)
    rows_by_id = {row["experiment_id"]: row for row in rows}
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))

    locked = rows_by_id[LOCKED_EXPERIMENT_ID]
    expected_protocol = str(config["release"]["protocol_version"])
    expected_auc = float(config["release"]["stable_macro_auc"])
    expected_ap = float(config["release"]["stable_macro_real_positive_ap"])
    expected_count = int(config["evaluation"]["expected_evaluation_counts"]["total"])

    if summary.main_experiment_id != LOCKED_EXPERIMENT_ID:
        raise AssertionError(
            f"locked registry row is {summary.main_experiment_id}, "
            f"expected {LOCKED_EXPERIMENT_ID}"
        )
    checks = {
        "protocol_id": locked["protocol_id"] == expected_protocol,
        "evaluation_videos": int(locked["evaluation_videos"]) == expected_count,
        "macro_auc": float(locked["macro_auc"]) == expected_auc,
        "macro_ap": float(locked["macro_ap"]) == expected_ap,
        "status": locked["status"] == "main_method_locked",
    }
    failed = [name for name, passed in checks.items() if not passed]
    if failed:
        raise AssertionError(f"locked U0 registry mismatch: {failed}")

    claim_ids = {
        row_id
        for row_id, row in rows_by_id.items()
        if row["status"] in EVIDENCE_REQUIRED_STATUSES
    }
    evidence_ids = {LOCKED_EXPERIMENT_ID, *CSV_METRIC_EVIDENCE}
    if claim_ids != evidence_ids:
        raise AssertionError(
            "claim-level evidence map mismatch: "
            f"missing={sorted(claim_ids - evidence_ids)} "
            f"extra={sorted(evidence_ids - claim_ids)}"
        )

    evidence_checks = 1
    for experiment_id, evidence in CSV_METRIC_EVIDENCE.items():
        relative_path, selector, auc_column, ap_column = evidence
        auc, ap = _read_csv_metric(
            ROOT / relative_path, selector, auc_column, ap_column
        )
        registry_row = rows_by_id[experiment_id]
        if auc != float(registry_row["macro_auc"]):
            raise AssertionError(f"{experiment_id}: registry/evidence Macro AUC mismatch")
        if ap != float(registry_row["macro_ap"]):
            raise AssertionError(f"{experiment_id}: registry/evidence Macro AP mismatch")
        if registry_row["result_path"] != relative_path:
            raise AssertionError(f"{experiment_id}: result_path is not its metric evidence")
        evidence_checks += 1

    return {
        "passed": True,
        "row_count": summary.row_count,
        "main_experiment_id": summary.main_experiment_id,
        "status_counts": summary.status_counts,
        "locked_checks": checks,
        "claim_evidence_checks": evidence_checks,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--registry", type=Path, default=DEFAULT_REGISTRY)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--json", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    payload = verify(args.registry, args.config)
    if args.json:
        print(json.dumps(payload, indent=2, ensure_ascii=False))
    else:
        print(
            "experiment registry verification passed: "
            f"rows={payload['row_count']} main={payload['main_experiment_id']} "
            f"claim_evidence={payload['claim_evidence_checks']}"
        )


if __name__ == "__main__":
    main()
