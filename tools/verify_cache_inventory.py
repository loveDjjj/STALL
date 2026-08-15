#!/usr/bin/env python3
"""Validate cache lifecycle declarations and optionally rescan the cache tree."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from alpha_stalled.cache_inventory import validate_cache_inventory


DEFAULT_INVENTORY = ROOT / "results/research_summary/cache_inventory.json"


def repository_path(path: Path) -> Path:
    return path if path.is_absolute() else ROOT / path


def verify(path: Path = DEFAULT_INVENTORY, *, verify_layout: bool = True) -> dict[str, object]:
    path = repository_path(path)
    inventory = json.loads(path.read_text(encoding="utf-8"))
    summary = validate_cache_inventory(
        inventory,
        repository_root=ROOT,
        verify_layout=verify_layout,
    )
    return {
        "passed": True,
        "snapshot_id": summary.snapshot_id,
        "group_count": summary.group_count,
        "file_count": summary.file_count,
        "logical_bytes": summary.logical_bytes,
        "safe_delete_candidate_bytes": summary.safe_delete_candidate_bytes,
        "layout_verified": verify_layout,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inventory", type=Path, default=DEFAULT_INVENTORY)
    parser.add_argument(
        "--skip-layout",
        action="store_true",
        help="Validate declarations only; this is not evidence that the snapshot is current.",
    )
    parser.add_argument("--json", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    payload = verify(args.inventory, verify_layout=not args.skip_layout)
    if args.json:
        print(json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True))
    else:
        print(
            "cache inventory verification passed: "
            f"snapshot={payload['snapshot_id']} groups={payload['group_count']} "
            f"files={payload['file_count']} layout_verified={payload['layout_verified']}"
        )


if __name__ == "__main__":
    main()
