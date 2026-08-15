#!/usr/bin/env python3
"""Verify frozen parameter identities, NPZ schemas, and local sweep boundaries."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from alpha_stalled.parameter_assets import (
    read_parameter_assets,
    render_parameter_asset_report,
    validate_parameter_assets,
)


DEFAULT_CATALOG = ROOT / "configs/parameter_assets.yaml"
DEFAULT_REPORT = ROOT / "reports/parameter_asset_inventory.md"


def repository_path(path: Path) -> Path:
    return path if path.is_absolute() else ROOT / path


def verify(catalog_path: Path, report_path: Path, *, write_report: bool) -> dict[str, object]:
    payload = read_parameter_assets(repository_path(catalog_path))
    summary = validate_parameter_assets(payload, ROOT)
    report = render_parameter_asset_report(payload, summary)
    resolved_report = repository_path(report_path)
    if write_report:
        resolved_report.write_text(report, encoding="utf-8")
    elif not resolved_report.is_file() or resolved_report.read_text(encoding="utf-8") != report:
        raise ValueError(
            f"parameter asset report is stale: run {Path(__file__).name} --write-report"
        )
    return {
        "passed": True,
        "governed_asset_count": summary.governed_asset_count,
        "current_release_count": summary.current_release_count,
        "external_confirmation_count": summary.external_confirmation_count,
        "historical_frozen_count": summary.historical_frozen_count,
        "local_unregistered_count": summary.local_unregistered_count,
        "local_unregistered_bytes": summary.local_unregistered_bytes,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--catalog", type=Path, default=DEFAULT_CATALOG)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--write-report", action="store_true")
    parser.add_argument("--json", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = verify(args.catalog, args.report, write_report=args.write_report)
    if args.json:
        print(json.dumps(result, indent=2, ensure_ascii=False, sort_keys=True))
    else:
        print(
            "parameter asset verification passed: "
            f"governed={result['governed_asset_count']} "
            f"current={result['current_release_count']} "
            f"external={result['external_confirmation_count']} "
            f"historical={result['historical_frozen_count']} "
            f"local_ignored={result['local_unregistered_count']}"
        )


if __name__ == "__main__":
    main()
