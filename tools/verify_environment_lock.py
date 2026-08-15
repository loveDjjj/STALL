#!/usr/bin/env python3
"""Verify the exact environment lock and optionally compare the active environment."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from alpha_stalled.environment_lock import (
    read_environment_spec,
    validate_environment_files,
    validate_installed_environment,
)


def _current_conda_records() -> list[dict[str, object]]:
    result = subprocess.run(
        ["conda", "list", "--json"],
        check=True,
        capture_output=True,
        text=True,
    )
    listed = json.loads(result.stdout)
    if not isinstance(listed, list):
        raise ValueError("conda list did not return a package list")
    records = [record for record in listed if record.get("channel") == "pypi"]
    metadata_root = Path(sys.prefix) / "conda-meta"
    if not metadata_root.is_dir():
        raise ValueError(f"active Python is not inside a Conda environment: {sys.prefix}")
    for path in sorted(metadata_root.glob("*.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        records.append(
            {
                "name": payload["name"],
                "version": payload["version"],
                "build_string": payload["build"],
                "channel": "conda-meta",
            }
        )
    return records


def _verify_pip_health() -> None:
    subprocess.run(
        [sys.executable, "-m", "pip", "check"],
        check=True,
        capture_output=True,
        text=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--environment", type=Path, default=ROOT / "environment.yml")
    parser.add_argument("--lock", type=Path, default=ROOT / "environment.lock.yml")
    parser.add_argument(
        "--check-current",
        action="store_true",
        help="compare the active Conda environment with every locked package",
    )
    args = parser.parse_args()

    summary = validate_environment_files(ROOT, args.environment, args.lock)
    message = (
        "environment lock verification passed: "
        f"direct={summary.direct_conda_count}+{summary.direct_pip_count} "
        f"locked={summary.locked_conda_count}+{summary.locked_pip_count} "
        f"imports={summary.external_import_count}"
    )
    if args.check_current:
        lock = read_environment_spec(args.lock)
        installed = validate_installed_environment(lock, _current_conda_records())
        _verify_pip_health()
        message += (
            f" current={installed.matched_conda_count}+{installed.matched_pip_count}"
            f" overlaid_conda={installed.overlaid_conda_count}"
            f" extra_pip={installed.extra_pip_count} pip_check=passed"
        )
    print(message)


if __name__ == "__main__":
    main()
