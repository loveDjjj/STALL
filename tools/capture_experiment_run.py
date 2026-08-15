#!/usr/bin/env python3
"""Run one experiment command while capturing exact, immutable provenance."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from alpha_stalled.run_capture import execute_captured_run


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__,
        epilog=(
            "Place '--' before the wrapped argv, for example: "
            "-- python tools/my_experiment.py --output-dir results/runs/example"
        ),
    )
    parser.add_argument("--experiment-id", required=True)
    parser.add_argument("--protocol-id", required=True)
    parser.add_argument("--environment")
    parser.add_argument("--require-clean", action="store_true")
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    if args.command and args.command[0] == "--":
        args.command = args.command[1:]
    if not args.command:
        parser.error("a wrapped command is required after '--'")
    return args


def main() -> None:
    args = parse_args()
    summary = execute_captured_run(
        repository_root=ROOT,
        experiment_id=args.experiment_id,
        protocol_id=args.protocol_id,
        command=args.command,
        environment=args.environment,
        require_clean=args.require_clean,
    )
    payload = {
        "experiment_id": summary.experiment_id,
        "protocol_id": summary.protocol_id,
        "state": summary.state,
        "exit_code": summary.exit_code,
        "capture_path": str(summary.capture_path.relative_to(ROOT)),
    }
    print(json.dumps(payload, ensure_ascii=False, sort_keys=True))
    raise SystemExit(summary.exit_code or 0)


if __name__ == "__main__":
    main()
