#!/usr/bin/env python3
"""Build or deterministically check a hashed experiment run manifest."""

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

from alpha_stalled.release_io import write_json
from alpha_stalled.run_manifest import build_run_manifest


def repository_path(path: Path) -> Path:
    return path if path.is_absolute() else ROOT / path


def run(args: argparse.Namespace) -> dict[str, object]:
    spec_path = repository_path(args.spec)
    output_path = repository_path(args.output)
    spec = yaml.safe_load(spec_path.read_text(encoding="utf-8"))
    if not isinstance(spec, dict):
        raise ValueError("run manifest specification must be a YAML object")
    manifest = build_run_manifest(spec, ROOT)

    if args.check:
        existing = json.loads(output_path.read_text(encoding="utf-8"))
        if existing != manifest:
            raise AssertionError(f"run manifest is stale: {output_path}")
        action = "checked"
    else:
        if output_path.exists() and not args.overwrite:
            raise FileExistsError(
                f"refusing to overwrite run manifest without --overwrite: {output_path}"
            )
        write_json(output_path, manifest)
        action = "wrote"
    return {
        "action": action,
        "output": str(output_path.relative_to(ROOT)),
        "experiment_id": manifest["identity"]["experiment_id"],
        "artifact_count": sum(len(items) for items in manifest["artifacts"].values()),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
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
