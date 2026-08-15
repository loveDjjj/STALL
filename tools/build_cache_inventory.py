#!/usr/bin/env python3
"""Build or deterministically check the repository cache inventory."""

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

from alpha_stalled.cache_inventory import build_cache_inventory, render_cache_inventory
from alpha_stalled.release_io import write_json


DEFAULT_SPEC = ROOT / "configs/cache_inventory.yaml"
DEFAULT_OUTPUT = ROOT / "results/research_summary/cache_inventory.json"
DEFAULT_REPORT = ROOT / "reports/cache_inventory.md"


def repository_path(path: Path) -> Path:
    return path if path.is_absolute() else ROOT / path


def run(args: argparse.Namespace) -> dict[str, object]:
    spec_path = repository_path(args.spec)
    output_path = repository_path(args.output)
    report_path = repository_path(args.report)
    spec = yaml.safe_load(spec_path.read_text(encoding="utf-8"))
    if not isinstance(spec, dict):
        raise ValueError("cache inventory specification must be a YAML object")
    inventory = build_cache_inventory(spec, ROOT)
    report = render_cache_inventory(inventory)

    if args.check:
        existing = json.loads(output_path.read_text(encoding="utf-8"))
        if existing != inventory:
            raise AssertionError(f"cache inventory JSON is stale: {output_path}")
        if report_path.read_text(encoding="utf-8") != report:
            raise AssertionError(f"cache inventory report is stale: {report_path}")
        action = "checked"
    else:
        existing_paths = [path for path in (output_path, report_path) if path.exists()]
        if existing_paths and not args.overwrite:
            shown = ", ".join(str(path) for path in existing_paths)
            raise FileExistsError(
                f"refusing to overwrite cache inventory without --overwrite: {shown}"
            )
        write_json(output_path, inventory)
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(report, encoding="utf-8")
        action = "wrote"

    summary = inventory["summary"]
    return {
        "action": action,
        "snapshot_id": inventory["snapshot_id"],
        "group_count": summary["group_count"],
        "file_count": summary["file_count"],
        "logical_bytes": summary["logical_bytes"],
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
    payload = run(parse_args())
    print(json.dumps(payload, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
