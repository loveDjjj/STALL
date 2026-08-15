#!/usr/bin/env python3
"""Validate one captured experiment execution record."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from alpha_stalled.run_capture import read_run_capture, validate_run_capture


def verify(
    path: Path,
    *,
    require_completed: bool = True,
    repository_root: Path = ROOT,
) -> dict[str, object]:
    repository_root = repository_root.resolve()
    path = path if path.is_absolute() else repository_root / path
    capture = read_run_capture(path)
    summary = validate_run_capture(
        capture,
        repository_root=repository_root,
        require_completed=require_completed,
    )
    if path.resolve() != summary.capture_path.resolve():
        raise ValueError("capture is not at results/runs/<experiment_id>/run_capture.json")
    return {
        "passed": True,
        "experiment_id": summary.experiment_id,
        "protocol_id": summary.protocol_id,
        "state": summary.state,
        "exit_code": summary.exit_code,
        "capture_path": str(path.relative_to(repository_root)),
        "completed_required": require_completed,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--capture", type=Path, required=True)
    parser.add_argument(
        "--allow-running",
        action="store_true",
        help="Inspect a running record; this is not completion evidence.",
    )
    parser.add_argument("--json", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    payload = verify(args.capture, require_completed=not args.allow_running)
    if args.json:
        print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))
    else:
        print(
            "run capture verification passed: "
            f"experiment={payload['experiment_id']} state={payload['state']} "
            f"exit_code={payload['exit_code']}"
        )


if __name__ == "__main__":
    main()
