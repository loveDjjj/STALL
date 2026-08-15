#!/usr/bin/env python3
"""Validate a run manifest against artifacts, registry, and locked U0 config."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from alpha_stalled.experiment_registry import read_registry
from alpha_stalled.run_manifest import validate_run_manifest


DEFAULT_MANIFEST = (
    ROOT
    / "results/research_summary/run_manifests/alpha_stalled_u0_locked.json"
)
DEFAULT_REGISTRY = ROOT / "reports/u0_experiment_registry.csv"
LOCKED_CONFIG = ROOT / "configs/alpha_stalled_u0_locked.yaml"


def repository_path(path: Path) -> Path:
    return path if path.is_absolute() else ROOT / path


def verify(
    manifest_path: Path = DEFAULT_MANIFEST,
    registry_path: Path = DEFAULT_REGISTRY,
    *,
    verify_hashes: bool = True,
) -> dict[str, object]:
    manifest_path = repository_path(manifest_path)
    registry_path = repository_path(registry_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    summary = validate_run_manifest(
        manifest, repository_root=ROOT, verify_hashes=verify_hashes
    )
    registry_rows, _ = read_registry(registry_path)
    rows_by_id = {row["experiment_id"]: row for row in registry_rows}
    if summary.experiment_id not in rows_by_id:
        raise AssertionError(f"run manifest is not registered: {summary.experiment_id}")
    row = rows_by_id[summary.experiment_id]
    identity = manifest["identity"]
    metrics = manifest["metrics"]
    data = manifest["data"]
    selection = manifest["selection"]
    registry_checks = {
        "protocol_id": identity["protocol_id"] == row["protocol_id"],
        "parent_experiment_id": identity["parent_experiment_id"]
        == row["parent_experiment_id"],
        "status": identity["status"] == row["status"],
        "macro_auc": float(metrics["macro_auc"]) == float(row["macro_auc"]),
        "macro_ap": float(metrics["macro_ap"]) == float(row["macro_ap"]),
        "evaluation_video_count": int(data["evaluation_video_count"])
        == int(row["evaluation_videos"]),
        "uses_generated_for_selection": bool(
            selection["uses_generated_for_selection"]
        )
        == (row["uses_fake_for_selection"].lower() == "true"),
        "score_direction": data["score_direction"] == row["score_direction"],
    }
    failed = [name for name, passed in registry_checks.items() if not passed]
    if failed:
        raise AssertionError(f"run manifest/registry mismatch: {failed}")

    locked_checks: dict[str, bool] = {}
    if summary.experiment_id == "alpha_stalled_u0_locked":
        config = yaml.safe_load(LOCKED_CONFIG.read_text(encoding="utf-8"))
        locked_checks = {
            "protocol_id": identity["protocol_id"]
            == config["release"]["protocol_version"],
            "macro_auc": float(metrics["macro_auc"])
            == float(config["release"]["stable_macro_auc"]),
            "macro_ap": float(metrics["macro_ap"])
            == float(config["release"]["stable_macro_real_positive_ap"]),
            "calibration_real_count": int(data["calibration_real_count"])
            == int(config["evaluation"]["expected_calibration_real_counts"]["total"]),
            "evaluation_video_count": int(data["evaluation_video_count"])
            == int(config["evaluation"]["expected_evaluation_counts"]["total"]),
        }
        failed = [name for name, passed in locked_checks.items() if not passed]
        if failed:
            raise AssertionError(f"run manifest/locked config mismatch: {failed}")

    return {
        "passed": True,
        "experiment_id": summary.experiment_id,
        "protocol_id": summary.protocol_id,
        "provenance_mode": summary.provenance_mode,
        "artifact_count": summary.artifact_count,
        "verified_hash_count": summary.verified_hash_count,
        "registry_checks": registry_checks,
        "locked_checks": locked_checks,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--registry", type=Path, default=DEFAULT_REGISTRY)
    parser.add_argument("--skip-hashes", action="store_true")
    parser.add_argument("--json", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    payload = verify(
        args.manifest,
        args.registry,
        verify_hashes=not args.skip_hashes,
    )
    if args.json:
        print(json.dumps(payload, indent=2, ensure_ascii=False))
    else:
        print(
            "run manifest verification passed: "
            f"experiment={payload['experiment_id']} "
            f"artifacts={payload['artifact_count']} "
            f"hashes={payload['verified_hash_count']}"
        )


if __name__ == "__main__":
    main()
