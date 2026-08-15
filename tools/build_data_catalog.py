#!/usr/bin/env python3
"""Build or deterministically check the canonical dataset catalog."""

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

from alpha_stalled.data_catalog import build_data_catalog, render_data_catalog
from alpha_stalled.release_io import write_json


DEFAULT_SPEC = ROOT / "configs/data_catalog.yaml"
DEFAULT_OUTPUT = ROOT / "results/research_summary/data_catalog.json"
DEFAULT_REPORT = ROOT / "reports/data_catalog.md"


def repository_path(path: Path) -> Path:
    return path if path.is_absolute() else ROOT / path


def run(args: argparse.Namespace) -> dict[str, object]:
    spec_path = repository_path(args.spec)
    output_path = repository_path(args.output)
    report_path = repository_path(args.report)
    spec = yaml.safe_load(spec_path.read_text(encoding="utf-8"))
    if not isinstance(spec, dict):
        raise ValueError("data catalog specification must be a YAML object")
    catalog = build_data_catalog(spec, ROOT)
    report = render_data_catalog(catalog)
    if args.check:
        if json.loads(output_path.read_text(encoding="utf-8")) != catalog:
            raise AssertionError(f"data catalog JSON is stale: {output_path}")
        if report_path.read_text(encoding="utf-8") != report:
            raise AssertionError(f"data catalog report is stale: {report_path}")
        action = "checked"
    else:
        existing = [path for path in (output_path, report_path) if path.exists()]
        if existing and not args.overwrite:
            shown = ", ".join(str(path) for path in existing)
            raise FileExistsError(
                f"refusing to overwrite data catalog without --overwrite: {shown}"
            )
        write_json(output_path, catalog)
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(report, encoding="utf-8")
        action = "wrote"
    summary = catalog["summary"]
    return {
        "action": action,
        "snapshot_id": catalog["snapshot_id"],
        "dataset_count": summary["dataset_count"],
        "canonical_video_count": summary["canonical_video_count"],
        "release_video_count": summary["release_video_count"],
        "missing_file_count": summary["source_file_missing_count"],
        "output": str(output_path.relative_to(ROOT)),
        "report": str(report_path.relative_to(ROOT)),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", type=Path, default=DEFAULT_SPEC)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if args.check and args.overwrite:
        parser.error("--check and --overwrite are mutually exclusive")
    return args


def main() -> None:
    print(json.dumps(run(parse_args()), ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
