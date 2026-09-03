#!/usr/bin/env python3
"""一次扫描严格 Patch cache，完成 Stage 2 T1-T5 轨迹几何矩阵。"""

from __future__ import annotations

import argparse
import contextlib
import json
import shlex
import sys
import traceback
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from artifacts import create_run_directory, write_csv, write_progress, write_run_manifest
from config import apply_overrides, load_config
from evaluation.tables import build_metric_tables, build_pairwise_metric_table
from pipeline import run_local_candidate_matrix_from_cache
from runner import _Tee


MATRICES = {
    "trajectory": {
        "run_name": "stage2_trajectory_matrix",
        "candidates": {
            "t1_curvature": "curvature",
            "t2_speed_ratio": "speed_ratio",
            "t3_path_chord": "path_chord",
            "t4_d2_curvature": "d2_curvature",
            "t5_geometry": "geometry",
        },
    },
    "conditional": {
        "run_name": "stage3_conditional_matrix",
        "candidates": {
            "d1_conditioned_d2": {
                "dynamics": "finite_difference", "conditional": True,
            },
            "d3_conditioned_geometry": {
                "dynamics": "geometry", "conditional": True,
            },
        },
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--matrix", choices=sorted(MATRICES), default="trajectory")
    parser.add_argument("--config", type=Path, default=ROOT / "configs/benchmark.yaml")
    parser.add_argument("--set", dest="overrides", action="append", default=[])
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    matrix = MATRICES[args.matrix]
    run_name = str(matrix["run_name"])
    candidates = dict(matrix["candidates"])
    config_path = args.config if args.config.is_absolute() else ROOT / args.config
    fixed = [
        "runtime.reuse_global_run=alpha_stall_full_d2_k3_no_spatial_refit",
        "runtime.device=cuda:1",
        "runtime.devices=[cuda:1]",
        "method.local.correspondence.type=same_grid",
        "method.local.correspondence.confidence=none",
    ]
    config = apply_overrides(load_config(config_path), [*args.overrides, *fixed])
    # 候选矩阵属于可复现配置的一部分，必须进入 resolved config 与 config hash。
    config["experiment"] = {
        "matrix": args.matrix,
        "candidates": candidates,
    }
    if args.dry_run:
        print(f"dry-run：{args.matrix} 矩阵将一次扫描缓存运行 {', '.join(candidates)}")
        print(f"设备：{config['runtime'].get('devices') or [config['runtime']['device']]}")
        print(f"复用 Global：{config['runtime']['reuse_global_run']}")
        return

    output_dir = create_run_directory(ROOT, run_name, overwrite=args.overwrite)
    command = [sys.executable, *sys.argv]
    (output_dir / "command.txt").write_text(
        "命令：" + " ".join(shlex.quote(item) for item in command) + "\n",
        encoding="utf-8",
    )
    write_run_manifest(
        output_dir, ROOT, run_name, config, status="running",
        pipeline_metadata={"matrix": args.matrix, "candidates": candidates},
    )
    progress = {"status": "running", "run_name": run_name, "phase": "initializing"}
    write_progress(output_dir, progress)

    def report(event: dict) -> None:
        progress.update(event)
        progress["updated_at"] = datetime.now(timezone.utc).isoformat()
        write_progress(output_dir, progress)
        if event.get("message"):
            print(event["message"], flush=True)

    logs = output_dir / "logs"
    logs.mkdir(exist_ok=True)
    with (logs / "run.log").open("a", encoding="utf-8", buffering=1) as log, \
         contextlib.redirect_stdout(_Tee(sys.stdout, log)), \
         contextlib.redirect_stderr(_Tee(sys.stderr, log)):
        try:
            windows, videos, metadata = run_local_candidate_matrix_from_cache(
                ROOT, config, candidates, report=report, artifact_dir=output_dir
            )
            dataset_tables, generator_tables, pairwise_tables = [], [], []
            for variant, frame in videos.groupby("variant", sort=False):
                clean = frame.drop(columns="variant")
                dataset, generator = build_metric_tables(clean, str(variant))
                pairwise = build_pairwise_metric_table(
                    clean, str(variant), int(config["metrics"]["pairwise_seed"])
                )
                for table in (dataset, generator, pairwise):
                    table.insert(0, "variant", variant)
                dataset_tables.append(dataset)
                generator_tables.append(generator)
                pairwise_tables.append(pairwise)
            write_csv(output_dir, "window_scores.csv", windows)
            write_csv(output_dir, "video_scores.csv", videos)
            write_csv(output_dir, "dataset_metrics.csv", pd.concat(dataset_tables, ignore_index=True))
            write_csv(output_dir, "generator_metrics.csv", pd.concat(generator_tables, ignore_index=True))
            write_csv(output_dir, "pairwise_metrics.csv", pd.concat(pairwise_tables, ignore_index=True))
            progress.update({"status": "completed", "phase": "completed"})
            write_progress(output_dir, progress)
            write_run_manifest(
                output_dir, ROOT, run_name, config, status="completed",
                pipeline_metadata=metadata,
            )
            print(f"[完成] {args.matrix} 矩阵已写入 {output_dir}", flush=True)
        except BaseException as error:
            failure = {
                "exception_type": type(error).__name__,
                "message": str(error),
                "occurred_at_utc": datetime.now(timezone.utc).isoformat(),
                "traceback": traceback.format_exc(),
            }
            (output_dir / "failure.json").write_text(
                json.dumps(failure, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )
            progress.update({"status": "interrupted", "phase": "interrupted"})
            write_progress(output_dir, progress)
            write_run_manifest(
                output_dir, ROOT, run_name, config, status="interrupted",
                pipeline_metadata={"matrix": args.matrix, "candidates": candidates},
            )
            print(f"[中断] {type(error).__name__}: {error}", flush=True)
            raise


if __name__ == "__main__":
    main()
