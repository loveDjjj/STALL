#!/usr/bin/env python3
"""Score the disjoint K=1 calibration videos from existing compact patch caches."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from eval_patch_fast import FastPatchScorer, iter_cache_jobs, load_cache_batch
from alpha_stalled.legacy_window_scoring import LOCAL_PARAMS
from alpha_stalled.legacy_local_d2_protocol import dataset_specs
from stall import STALL


def score_dataset(
    spec,
    global_scorer: STALL,
    device: str,
    batch_size: int,
) -> pd.DataFrame:
    local_params = LOCAL_PARAMS[spec.name]
    if not local_params.exists():
        raise FileNotFoundError(local_params)
    local_scorer = FastPatchScorer(str(local_params), device=device)
    local_scorer.validate("same_grid_second_order")
    jobs = list(iter_cache_jobs(str(spec.calib_index), str(spec.patch_cache), 2, True, None))
    rows: list[dict] = []
    for start in range(0, len(jobs), batch_size):
        batch_jobs = jobs[start : start + batch_size]
        patch_batch, global_batch, grid_size = load_cache_batch(batch_jobs, include_global=True)
        if grid_size != local_scorer.patch_grid_size:
            raise ValueError(f"{spec.name}: cache grid {grid_size} != params {local_scorer.patch_grid_size}")
        global_scores = global_scorer._scores_from_embs(global_batch.numpy())
        local_scores = local_scorer.score_batch(
            patch_batch,
            patch_temp_mode="same_grid_second_order",
            patch_spat_weight=0.1,
            patch_temp_weight=0.9,
            aggregation=local_scorer.aggregation_config.get("mode", "bottomk_mean"),
            bottomk_ratio=local_scorer.params_bottomk_ratio,
            temporal_run_length=local_scorer.params_temporal_run_length,
            patch_region_size=local_scorer.params_patch_region_size,
            global_batch=global_batch,
        )
        for index, job in enumerate(batch_jobs):
            g = float(global_scores["final_score"][index])
            local = float(local_scores["patch_final_score"][index])
            rows.append(
                {
                    "dataset": spec.name,
                    "protocol_split": "calibration",
                    "subset": job["subset"],
                    "source_model": job["source_model"],
                    "filename": job["filename"],
                    "global_spatial": float(global_scores["spat_percentile"][index]),
                    "global_t1": float(global_scores["temp_percentile"][index]),
                    "G_k": g,
                    "patch_spatial": float(local_scores["patch_spat_percentile"][index]),
                    "patch_d2": float(local_scores["patch_temp_percentile"][index]),
                    "L_k": local,
                    "S_k": 0.6 * g + 0.4 * local,
                }
            )
        print(f"[{spec.name}] {min(start + batch_size, len(jobs))}/{len(jobs)}", flush=True)
    if local_scorer._executor is not None:
        local_scorer._executor.shutdown(wait=True)
    frame = pd.DataFrame(rows)
    if len(frame) != 200:
        raise ValueError(f"{spec.name}: expected 200 calibration rows, found {len(frame)}")
    return frame


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=REPO_ROOT)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument(
        "--global-params",
        type=Path,
        default=REPO_ROOT / "precomputed/stall_params_vatex_dino_v3.npz",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=REPO_ROOT / "results/multi_window_joint_typicality/K1_calibration_window_scores.csv",
    )
    args = parser.parse_args()
    global_scorer = STALL(
        args.device,
        np.load(args.global_params, allow_pickle=True),
        load_dino=False,
    )
    frames = [
        score_dataset(spec, global_scorer, args.device, args.batch_size)
        for spec in dataset_specs(args.root)
    ]
    output = pd.concat(frames, ignore_index=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(args.output, index=False)
    print(f"saved {len(output)} rows -> {args.output}")


if __name__ == "__main__":
    main()
