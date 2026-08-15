#!/usr/bin/env python3
"""Validate the canonical dataset catalog against indexes and release manifests."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from alpha_stalled.data_catalog import validate_data_catalog


DEFAULT_CATALOG = ROOT / "results/research_summary/data_catalog.json"


def repository_path(path: Path) -> Path:
    return path if path.is_absolute() else ROOT / path


def verify(path: Path = DEFAULT_CATALOG, *, verify_inputs: bool = True) -> dict[str, object]:
    path = repository_path(path)
    catalog = json.loads(path.read_text(encoding="utf-8"))
    summary = validate_data_catalog(
        catalog,
        repository_root=ROOT,
        verify_inputs=verify_inputs,
    )
    return {
        "passed": True,
        "snapshot_id": summary.snapshot_id,
        "dataset_count": summary.dataset_count,
        "canonical_video_count": summary.canonical_video_count,
        "release_video_count": summary.release_video_count,
        "missing_file_count": summary.missing_file_count,
        "inputs_verified": verify_inputs,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--catalog", type=Path, default=DEFAULT_CATALOG)
    parser.add_argument(
        "--skip-inputs",
        action="store_true",
        help="Validate catalog structure only; canonical index/manifests are not rescanned.",
    )
    parser.add_argument("--json", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    payload = verify(args.catalog, verify_inputs=not args.skip_inputs)
    if args.json:
        print(json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True))
    else:
        print(
            "data catalog verification passed: "
            f"snapshot={payload['snapshot_id']} datasets={payload['dataset_count']} "
            f"canonical={payload['canonical_video_count']} "
            f"release={payload['release_video_count']} "
            f"missing={payload['missing_file_count']} "
            f"inputs_verified={payload['inputs_verified']}"
        )


if __name__ == "__main__":
    main()
