#!/usr/bin/env python3
"""Validate configuration identity, lifecycle, and complete registry coverage."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from alpha_stalled.config_registry import read_config_registry, validate_config_registry


DEFAULT_REGISTRY = ROOT / "configs/config_registry.yaml"


def repository_path(path: Path) -> Path:
    return path if path.is_absolute() else ROOT / path


def verify(path: Path = DEFAULT_REGISTRY) -> dict[str, object]:
    registry = read_config_registry(repository_path(path))
    summary = validate_config_registry(registry, ROOT)
    return {
        "passed": True,
        "asset_count": summary.asset_count,
        "protocol_config_count": summary.protocol_config_count,
        "historical_config_count": summary.historical_config_count,
        "current_path": summary.current_path,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--registry", type=Path, default=DEFAULT_REGISTRY)
    parser.add_argument("--json", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    payload = verify(args.registry)
    if args.json:
        print(json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True))
    else:
        print(
            "config registry verification passed: "
            f"assets={payload['asset_count']} "
            f"protocols={payload['protocol_config_count']} "
            f"historical={payload['historical_config_count']} "
            f"current={payload['current_path']}"
        )


if __name__ == "__main__":
    main()
