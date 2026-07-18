#!/usr/bin/env python3
"""Run or dry-run a fresh validation scaffold command sequence."""

from __future__ import annotations

import argparse
import json
import shlex
import subprocess
import time
from pathlib import Path
from typing import Any

import pandas as pd


COMMAND_ORDER = [
    "validate_fresh_inputs",
    "apply_frozen_scorer",
    "source_audit_vs_alpha060_default",
    "fresh_validation_gate",
]


def _has_placeholder(value: str) -> bool:
    return "<" in value or ">" in value


def _run_command(step: str, command: str, dry_run: bool) -> dict[str, object]:
    row: dict[str, object] = {
        "step": step,
        "command": command,
        "dry_run": dry_run,
        "status": "DRY_RUN" if dry_run else "PENDING",
        "returncode": 0 if dry_run else None,
        "duration_sec": 0.0,
        "stdout_tail": "",
        "stderr_tail": "",
    }
    if dry_run:
        return row

    started = time.time()
    proc = subprocess.run(
        shlex.split(command),
        check=False,
        capture_output=True,
        text=True,
    )
    row["duration_sec"] = round(time.time() - started, 3)
    row["returncode"] = int(proc.returncode)
    row["status"] = "PASS" if proc.returncode == 0 else "FAIL"
    row["stdout_tail"] = proc.stdout[-2000:]
    row["stderr_tail"] = proc.stderr[-2000:]
    return row


def _load_scaffold(path: Path) -> dict[str, Any]:
    scaffold = json.loads(path.read_text(encoding="utf-8"))
    if "commands" not in scaffold:
        raise ValueError(f"{path} missing commands")
    missing = [step for step in COMMAND_ORDER if step not in scaffold["commands"]]
    if missing:
        raise ValueError(f"{path} missing command steps: {missing}")
    return scaffold


def run(args: argparse.Namespace) -> tuple[pd.DataFrame, pd.DataFrame]:
    scaffold = _load_scaffold(args.scaffold_json)
    rows: list[dict[str, object]] = []
    commands = scaffold["commands"]
    contains_placeholders = any(_has_placeholder(str(commands[step])) for step in COMMAND_ORDER)
    if contains_placeholders and not args.dry_run:
        raise ValueError("Scaffold commands contain placeholders; use --dry-run or build a concrete fresh scaffold")

    failed = False
    for step in COMMAND_ORDER:
        command = str(commands[step])
        if failed:
            rows.append(
                {
                    "step": step,
                    "command": command,
                    "dry_run": args.dry_run,
                    "status": "SKIPPED_AFTER_FAILURE",
                    "returncode": None,
                    "duration_sec": 0.0,
                    "stdout_tail": "",
                    "stderr_tail": "",
                }
            )
            continue
        row = _run_command(step, command, args.dry_run)
        rows.append(row)
        failed = row["status"] == "FAIL"

    steps = pd.DataFrame(rows)
    passed = bool(steps["status"].isin(["PASS", "DRY_RUN"]).all())
    summary = pd.DataFrame(
        [
            {
                "scaffold_json": str(args.scaffold_json),
                "dataset": scaffold.get("dataset", {}).get("dataset", ""),
                "dry_run": args.dry_run,
                "contains_placeholders": contains_placeholders,
                "decision": "DRY_RUN_PASS" if args.dry_run and passed else ("PASS" if passed else "FAIL"),
                "n_steps": len(steps),
                "n_passed_or_dry_run": int(steps["status"].isin(["PASS", "DRY_RUN"]).sum()),
                "failed_step": "" if passed else str(steps.loc[steps["status"] == "FAIL", "step"].iloc[0]),
            }
        ]
    )
    return summary, steps


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scaffold-json", type=Path, required=True)
    parser.add_argument("--output-summary-csv", type=Path, required=True)
    parser.add_argument("--output-steps-csv", type=Path, required=True)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    summary, steps = run(args)
    args.output_summary_csv.parent.mkdir(parents=True, exist_ok=True)
    args.output_steps_csv.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(args.output_summary_csv, index=False)
    steps.to_csv(args.output_steps_csv, index=False)
    print(summary.to_string(index=False))
    print(steps[["step", "status", "returncode", "duration_sec"]].to_string(index=False))
    if summary.iloc[0]["decision"] == "FAIL":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
