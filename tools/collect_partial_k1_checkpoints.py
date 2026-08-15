#!/usr/bin/env python3
"""Collect interrupted K1 checkpoint rows and emit a strict remaining manifest."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]


def run(args: argparse.Namespace) -> None:
    paths = sorted(
        path
        for directory in args.checkpoint_dir
        for path in directory.glob("shard_*/part_*.csv")
    )
    if not paths:
        raise FileNotFoundError(f"no checkpoint parts under {args.checkpoint_dir}")
    completed = pd.concat(
        [pd.read_csv(path, float_precision="round_trip") for path in paths],
        ignore_index=True,
    )
    if completed["k1_task_id"].duplicated().any():
        raise ValueError("duplicate K1 task IDs across checkpoint parts")
    source = pd.read_csv(args.input, float_precision="round_trip")
    source = source[source["dataset"].eq(args.dataset)].copy()
    completed_ids = set(completed["k1_task_id"].astype(str))
    source_ids = set(source["k1_task_id"].astype(str))
    if not completed_ids <= source_ids:
        raise ValueError("checkpoint contains task IDs outside the source manifest")
    remaining = source[~source["k1_task_id"].astype(str).isin(completed_ids)].copy()
    args.completed_output.parent.mkdir(parents=True, exist_ok=True)
    completed.sort_values("k1_task_id").to_csv(args.completed_output, index=False)
    remaining.sort_values("k1_task_id").to_csv(args.remaining_output, index=False)
    print(
        f"dataset={args.dataset} source={len(source)} completed={len(completed)} "
        f"remaining={len(remaining)}"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--checkpoint-dir", type=Path, required=True, nargs="+")
    parser.add_argument("--completed-output", type=Path, required=True)
    parser.add_argument("--remaining-output", type=Path, required=True)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
