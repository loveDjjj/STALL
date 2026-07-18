#!/usr/bin/env python3
"""Run patch ablation jobs one at a time with resumable status files."""

from __future__ import annotations

import json
import subprocess
import time
from pathlib import Path


ROOT = Path("/data/OneDay/STALL_project")
STATUS_PATH = ROOT / "STALL/results/patch_cross_dataset_ablation/status.jsonl"
LOG_DIR = ROOT / "STALL/logs/patch_cross_dataset_ablation"

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


def params_path(dataset: str, mode: str, region: int, agg: str) -> Path:
    return (
        ROOT
        / "STALL/precomputed"
        / f"patch_params_{dataset}_real_{mode}_region{region}_{agg}_v2.npz"
    )


def append_status(record: dict) -> None:
    STATUS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with STATUS_PATH.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def main() -> int:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    jobs = [
        (dataset, mode, region, agg)
        for dataset in DATASETS
        for mode in MODES
        for region in REGIONS
        for agg in AGGS
    ]

    for idx, (dataset, mode, region, agg) in enumerate(jobs, start=1):
        params = params_path(dataset, mode, region, agg)
        result = result_path(dataset, mode, region, agg)
        job_id = f"{dataset}_{mode}_region{region}_{agg}"
        log_path = LOG_DIR / f"{job_id}.log"

        if result.exists() and result.stat().st_size > 0:
            print(f"[{idx}/{len(jobs)}] skip complete {job_id}", flush=True)
            append_status(
                {
                    "time": time.time(),
                    "event": "skip_complete",
                    "job": job_id,
                    "params": str(params),
                    "result": str(result),
                }
            )
            continue

        print(f"[{idx}/{len(jobs)}] start {job_id}", flush=True)
        append_status(
            {
                "time": time.time(),
                "event": "start",
                "job": job_id,
                "params_exists": params.exists(),
                "result_exists": result.exists(),
            }
        )

        cmd = [
            "bash",
            "STALL/scripts/run_one_patch_ablation_job.sh",
            dataset,
            mode,
            str(region),
            agg,
        ]
        with log_path.open("ab") as log:
            proc = subprocess.run(
                cmd,
                cwd=ROOT,
                stdout=log,
                stderr=subprocess.STDOUT,
                check=False,
            )

        ok = proc.returncode == 0 and result.exists() and result.stat().st_size > 0
        append_status(
            {
                "time": time.time(),
                "event": "finish" if ok else "fail",
                "job": job_id,
                "returncode": proc.returncode,
                "params_exists": params.exists(),
                "result_exists": result.exists(),
                "log": str(log_path),
            }
        )
        if not ok:
            print(f"[{idx}/{len(jobs)}] failed {job_id}; see {log_path}", flush=True)
            return proc.returncode or 1
        print(f"[{idx}/{len(jobs)}] done {job_id}", flush=True)

    print("all jobs complete", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
