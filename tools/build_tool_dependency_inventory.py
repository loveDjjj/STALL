#!/usr/bin/env python3
"""Build or check the deterministic tools dependency inventory and report."""

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
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()

    payload = dependency_inventory(ROOT / "tools", read_tool_dependency_policy(args.policy))
    inventory_text = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    report_text = render_dependency_report(payload)
    if args.check:
        failures = []
        for path, expected in (
            (args.inventory, inventory_text),
            (args.report, report_text),
        ):
            if not path.is_file() or path.read_text(encoding="utf-8") != expected:
                failures.append(str(path))
        if failures:
            raise SystemExit(f"tool dependency artifacts are stale: {failures}")
    else:
        args.inventory.parent.mkdir(parents=True, exist_ok=True)
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.inventory.write_text(inventory_text, encoding="utf-8")
        args.report.write_text(report_text, encoding="utf-8")
    print(
        "tool dependency inventory "
        f"{'check' if args.check else 'build'} passed: "
        f"modules={payload['tool_module_count']} edges={payload['edge_count']}"
    )


if __name__ == "__main__":
    main()
