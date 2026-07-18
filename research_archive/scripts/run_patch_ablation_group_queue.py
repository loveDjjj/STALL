#!/usr/bin/env python3
"""Run cross-dataset patch ablation groups with one cache pass per group."""

from __future__ import annotations

import json
import subprocess
import time
from pathlib import Path


ROOT = Path("/data/OneDay/STALL_project")
STATUS_PATH = ROOT / "STALL/results/patch_cross_dataset_ablation/group_status.jsonl"
LOG_DIR = ROOT / "STALL/logs/patch_cross_dataset_ablation_groups"

DATASETS = ["videofeedback", "genvideo"]
MODES = ["same_grid_lag1", "same_grid_second_order"]
REGIONS = [1, 2, 3]
AGGS = ["bottomk0p2", "bottomk0p5", "mean"]


def result_path(dataset: str, mode: str, region: int, agg: str) -> Path:
    return (
        ROOT
        / "STALL/results/patch_cross_dataset_ablation"
        / f"{dataset}_patch_{mode}_region{region}_{agg}_spat10_temp90.csv"
    )


def append_status(record: dict) -> None:
    STATUS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with STATUS_PATH.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def group_complete(dataset: str, mode: str, region: int) -> bool:
    return all(
        (p := result_path(dataset, mode, region, agg)).exists() and p.stat().st_size > 0
        for agg in AGGS
    )


def main() -> int:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    groups = [
        (dataset, mode, region)
        for dataset in DATASETS
        for mode in MODES
        for region in REGIONS
    ]

    for idx, (dataset, mode, region) in enumerate(groups, start=1):
        group_id = f"{dataset}_{mode}_region{region}"
        log_path = LOG_DIR / f"{group_id}.log"

        if group_complete(dataset, mode, region):
            print(f"[{idx}/{len(groups)}] skip complete {group_id}", flush=True)
            append_status({"time": time.time(), "event": "skip_complete", "group": group_id})
            continue

        print(f"[{idx}/{len(groups)}] start {group_id}", flush=True)
        append_status({"time": time.time(), "event": "start", "group": group_id})
        cmd = [
            "bash",
            "STALL/scripts/run_one_patch_ablation_group.sh",
            dataset,
            mode,
            str(region),
        ]
        with log_path.open("ab") as log:
            proc = subprocess.run(
                cmd,
                cwd=ROOT,
                stdout=log,
                stderr=subprocess.STDOUT,
                check=False,
            )

        ok = proc.returncode == 0 and group_complete(dataset, mode, region)
        append_status(
            {
                "time": time.time(),
                "event": "finish" if ok else "fail",
                "group": group_id,
                "returncode": proc.returncode,
                "log": str(log_path),
            }
        )
        if not ok:
            print(f"[{idx}/{len(groups)}] failed {group_id}; see {log_path}", flush=True)
            return proc.returncode or 1
        print(f"[{idx}/{len(groups)}] done {group_id}", flush=True)

    print("all groups complete", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
