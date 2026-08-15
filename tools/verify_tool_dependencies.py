#!/usr/bin/env python3
"""Verify tools import policy and the checked-in dependency artifacts."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from alpha_stalled.tool_dependencies import (
    dependency_inventory,
    read_tool_dependency_policy,
    render_dependency_report,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--policy", type=Path, default=ROOT / "configs/tool_dependencies.yaml"
    )
    parser.add_argument(
        "--inventory",
        type=Path,
        default=ROOT / "results/research_summary/tool_dependency_inventory.json",
    )
    parser.add_argument(
        "--report", type=Path, default=ROOT / "reports/tool_dependency_inventory.md"
    )
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    payload = dependency_inventory(ROOT / "tools", read_tool_dependency_policy(args.policy))
    expected_json = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    expected_report = render_dependency_report(payload)
    if not args.inventory.is_file() or args.inventory.read_text(encoding="utf-8") != expected_json:
        raise SystemExit(f"tool dependency inventory is missing or stale: {args.inventory}")
    if not args.report.is_file() or args.report.read_text(encoding="utf-8") != expected_report:
        raise SystemExit(f"tool dependency report is missing or stale: {args.report}")
    if args.json:
        print(json.dumps(payload, indent=2, sort_keys=True))
    else:
        print(
            "tool dependency verification passed: "
            f"modules={payload['tool_module_count']} edges={payload['edge_count']} "
            f"cycles=0 isolated={len(payload['isolated_entrypoints'])}"
        )


if __name__ == "__main__":
    main()
