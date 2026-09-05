#!/usr/bin/env python3
"""汇总冻结 ViF-Bench 确认集的分支、权重与真实阈值结果。"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from config import load_config
from cross_domain import evaluate_cross_domain_cell
from temporal_selection.evaluation import (
    build_matched_selector_pairwise_table,
    paired_selector_bootstrap,
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run-dir",
        type=Path,
        default=ROOT / "results/runs/caes_confirmation_vifbench",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "results/analysis/vifbench_confirmation",
    )
    parser.add_argument(
        "--random-run-dir",
        type=Path,
        default=ROOT / "results/runs/caes_confirmation_vifbench_random",
    )
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    progress = json.loads((args.run_dir / "progress.json").read_text(encoding="utf-8"))
    if progress.get("status") != "completed":
        raise ValueError("ViF-Bench确认run尚未完成")
    if args.output_dir.exists() and args.overwrite:
        shutil.rmtree(args.output_dir)
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError(f"分析目录已存在：{args.output_dir}")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    config = load_config(args.run_dir / "resolved_config.yaml")
    windows = pd.read_csv(args.run_dir / "window_scores.csv", float_precision="round_trip")
    videos = pd.read_csv(args.run_dir / "video_scores.csv", float_precision="round_trip")
    random_progress = json.loads(
        (args.random_run_dir / "progress.json").read_text(encoding="utf-8")
    )
    if random_progress.get("status") != "completed":
        raise ValueError("ViF-Bench Random run尚未完成")
    random_windows = pd.read_csv(
        args.random_run_dir / "window_scores.csv", float_precision="round_trip"
    )
    random_videos = pd.read_csv(
        args.random_run_dir / "video_scores.csv", float_precision="round_trip"
    )
    main_uniform = videos[videos["selector"].eq("uniform")].sort_values("video_id")
    random_uniform = random_videos[
        random_videos["selector"].eq("uniform")
    ].sort_values("video_id")
    if not main_uniform["video_id"].reset_index(drop=True).equals(
        random_uniform["video_id"].reset_index(drop=True)
    ):
        raise ValueError("Random run与主确认run的Uniform视频身份不同")
    for column in ("global_score", "local_score", "final_score"):
        difference = (
            main_uniform[column].to_numpy()
            - random_uniform[column].to_numpy()
        )
        if float(abs(difference).max()) != 0.0:
            raise ValueError(f"Random run的Uniform {column}发生漂移")
    combined_videos = pd.concat([
        videos,
        random_videos[random_videos["selector"].eq("random")],
    ], ignore_index=True)
    selector_windows = {
        "uniform": windows[windows["selector"].eq("uniform")],
        "feature_change": windows[windows["selector"].eq("feature_change")],
        "random": random_windows[random_windows["selector"].eq("random")],
    }
    summaries, operating = [], []
    for selector, selected_source in selector_windows.items():
        selected = selected_source.copy()
        summary, points, _ = evaluate_cross_domain_cell(
            selected[selected["split"].eq("calibration")],
            selected[selected["split"].eq("evaluation")],
            config,
            calibration_bank=f"vifbench_{selector}",
            evaluation_domain="vifbench",
        )
        summary.insert(0, "selector", selector)
        points.insert(0, "selector", selector)
        summaries.append(summary)
        operating.append(points)

    anchor = videos[videos["selector"].eq("uniform")].copy()
    branches = [anchor]
    for selector in ("uniform", "feature_change"):
        selected = videos[videos["selector"].eq(selector)].copy()
        for component in ("global_score", "local_score", "final_score"):
            current = selected.copy()
            current["selector"] = f"{selector}:{component}"
            current["final_score"] = current[component]
            branches.append(current)
    branch_metrics = build_matched_selector_pairwise_table(
        pd.concat(branches, ignore_index=True), baseline="uniform", seed=42
    )
    branch_metrics = branch_metrics[
        branch_metrics["scope"].eq("generator_pairwise_dataset_macro")
    ]

    sensitivity = [anchor]
    for selector in ("uniform", "feature_change"):
        selected = videos[videos["selector"].eq(selector)].copy()
        for global_weight in (0.0, 0.25, 0.4, 0.5, 0.6, 2.0 / 3.0, 0.75, 1.0):
            current = selected.copy()
            current["selector"] = f"{selector}:g{global_weight:.4f}"
            current["final_score"] = (
                global_weight * current["global_score"]
                + (1.0 - global_weight) * current["local_score"]
            )
            sensitivity.append(current)
    weight_metrics = build_matched_selector_pairwise_table(
        pd.concat(sensitivity, ignore_index=True), baseline="uniform", seed=42
    )
    weight_metrics = weight_metrics[
        weight_metrics["scope"].eq("generator_pairwise_dataset_macro")
    ]
    selector_pairwise = build_matched_selector_pairwise_table(
        combined_videos, baseline="uniform", seed=42
    )
    selector_bootstrap = paired_selector_bootstrap(
        combined_videos, baseline="uniform", seed=42, iterations=1000
    )
    feature_vs_random = paired_selector_bootstrap(
        combined_videos[
            combined_videos["selector"].isin(["random", "feature_change"])
        ],
        baseline="random",
        seed=42,
        iterations=1000,
    )

    outputs = {
        "summary_metrics.csv": pd.concat(summaries, ignore_index=True),
        "real_only_operating_points.csv": pd.concat(operating, ignore_index=True),
        "branch_metrics.csv": branch_metrics,
        "weight_sensitivity_diagnostic.csv": weight_metrics,
        "selector_pairwise_metrics.csv": selector_pairwise,
        "selector_paired_bootstrap.csv": pd.concat(
            [selector_bootstrap, feature_vs_random], ignore_index=True
        ),
    }
    for name, frame in outputs.items():
        frame.to_csv(args.output_dir / name, index=False)
    manifest = {
        "status": "completed",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_run": str(args.run_dir),
        "source_video_scores_sha256": _sha256(args.run_dir / "video_scores.csv"),
        "random_video_scores_sha256": _sha256(
            args.random_run_dir / "video_scores.csv"
        ),
        "uniform_max_abs_difference": 0.0,
        "artifacts": {
            name: _sha256(args.output_dir / name) for name in outputs
        },
        "weight_sensitivity_role": "post-confirmation diagnostic only",
    }
    (args.output_dir / "analysis_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(outputs["summary_metrics.csv"].to_string(index=False), flush=True)
    print(outputs["real_only_operating_points.csv"].to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
