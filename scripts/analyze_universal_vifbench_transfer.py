#!/usr/bin/env python3
"""严格配对比较 ViF target-real 与不含ViF的Universal-4校准。"""

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
for path in (ROOT / "src", ROOT / "scripts"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from config import load_config
from cross_domain import _calibrate_windows_and_videos
from evaluation.tables import build_pairwise_metric_table
from finalize_cross_domain_calibration import _read_split
from run_cross_domain_calibration import ALL_DOMAINS
from temporal_selection.evaluation import paired_selector_bootstrap


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--cross-domain-dir",
        type=Path,
        default=ROOT / "results/runs/cross_domain_feature_k3_equal_fusion",
    )
    parser.add_argument(
        "--transfer-dir",
        type=Path,
        default=ROOT / "results/runs/universal4_to_vifbench_feature_k3",
    )
    parser.add_argument(
        "--target-run-dir",
        type=Path,
        default=ROOT / "results/runs/caes_confirmation_vifbench",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "results/analysis/universal4_to_vifbench",
    )
    parser.add_argument("--iterations", type=int, default=1000)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if args.output_dir.exists() and args.overwrite:
        shutil.rmtree(args.output_dir)
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError(f"Universal-4→ViF分析已存在：{args.output_dir}")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    calibration = pd.concat([
        _read_split(args.cross_domain_dir, domain, "calibration")
        for domain in ALL_DOMAINS
    ], ignore_index=True)
    calibration["patch_temporal_raw"] = calibration[
        "patch_temporal_raw__universal4"
    ]
    evaluation = _read_split(args.transfer_dir, "vifbench", "evaluation")
    evaluation["patch_temporal_raw"] = evaluation[
        "patch_temporal_raw__universal4"
    ]
    config = load_config(ROOT / "configs/benchmark.yaml")
    config["method"]["fusion"]["global_weight"] = 0.5
    config["method"]["fusion"]["local_weight"] = 0.5
    _, universal = _calibrate_windows_and_videos(
        calibration.assign(dataset="vifbench"), evaluation, config
    )
    universal.insert(0, "selector", "universal4")

    target = pd.read_csv(
        args.target_run_dir / "video_scores.csv", float_precision="round_trip"
    )
    target = target[target["selector"].eq("feature_change")].copy()
    target["selector"] = "target_real"
    scores = pd.concat([target, universal], ignore_index=True, sort=False)
    pairwise = []
    for selector, frame in scores.groupby("selector", sort=False):
        current = build_pairwise_metric_table(frame, selector, 42)
        current.insert(0, "selector", selector)
        pairwise.append(current)
    pairwise = pd.concat(pairwise, ignore_index=True)
    bootstrap = paired_selector_bootstrap(
        scores,
        baseline="target_real",
        seed=42,
        iterations=args.iterations,
    )
    outputs = {
        "pairwise_metrics.csv": pairwise,
        "paired_bootstrap.csv": bootstrap,
        "video_scores.csv": scores,
    }
    for name, frame in outputs.items():
        frame.to_csv(args.output_dir / name, index=False)
    manifest = {
        "status": "completed",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "universal_source_domains": list(ALL_DOMAINS),
        "vifbench_real_used_by_universal": 0,
        "artifacts": {
            name: _sha256(args.output_dir / name) for name in outputs
        },
    }
    (args.output_dir / "analysis_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(pairwise.to_string(index=False), flush=True)
    print(bootstrap.to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
