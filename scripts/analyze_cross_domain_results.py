#!/usr/bin/env python3
"""生成跨真实域矩阵的论文配对指标、bootstrap与对角回归审计。"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT / "src", ROOT / "scripts"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from config import load_config
from cross_domain import _calibrate_windows_and_videos
from evaluation.tables import build_pairwise_metric_table
from finalize_cross_domain_calibration import _read_split
from run_cross_domain_calibration import ALL_DOMAINS, UNIVERSAL_BANKS
from temporal_selection.evaluation import paired_selector_bootstrap


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run-dir",
        type=Path,
        default=ROOT / "results/runs/cross_domain_feature_k3_equal_fusion",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "results/analysis/cross_domain_feature_k3_equal_fusion",
    )
    parser.add_argument("--bootstrap-iterations", type=int, default=1000)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if args.output_dir.exists() and args.overwrite:
        shutil.rmtree(args.output_dir)
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError(f"跨域分析目录已存在：{args.output_dir}")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    config = load_config(ROOT / "configs/benchmark.yaml")
    config["method"]["fusion"]["global_weight"] = 0.5
    config["method"]["fusion"]["local_weight"] = 0.5
    raw = {
        (domain, split): _read_split(args.run_dir, domain, split)
        for domain in ALL_DOMAINS
        for split in ("calibration", "evaluation")
    }
    banks = [*ALL_DOMAINS, *UNIVERSAL_BANKS]
    video_frames = []
    pairwise_frames = []
    for bank in banks:
        source_domains = (
            UNIVERSAL_BANKS[bank] if bank in UNIVERSAL_BANKS else (bank,)
        )
        calibration = pd.concat(
            [raw[(domain, "calibration")] for domain in source_domains],
            ignore_index=True,
        )
        calibration["patch_temporal_raw"] = calibration[
            f"patch_temporal_raw__{bank}"
        ]
        for target in ALL_DOMAINS:
            evaluation = raw[(target, "evaluation")].copy()
            evaluation["patch_temporal_raw"] = evaluation[
                f"patch_temporal_raw__{bank}"
            ]
            _, videos = _calibrate_windows_and_videos(
                calibration.assign(dataset=target), evaluation, config
            )
            videos.insert(0, "calibration_bank", bank)
            videos.insert(1, "evaluation_domain", target)
            video_frames.append(videos)
            pairwise = build_pairwise_metric_table(
                videos, run_name=bank, pairwise_seed=42
            )
            pairwise.insert(0, "calibration_bank", bank)
            pairwise.insert(1, "evaluation_domain", target)
            pairwise_frames.append(pairwise)
    all_videos = pd.concat(video_frames, ignore_index=True)
    pairwise = pd.concat(pairwise_frames, ignore_index=True)
    pairwise = pairwise[
        pairwise["scope"].eq("generator_pairwise_dataset_macro")
    ].reset_index(drop=True)

    bootstrap_frames = []
    for target in ALL_DOMAINS:
        selected = all_videos[
            all_videos["evaluation_domain"].eq(target)
            & all_videos["calibration_bank"].isin(
                [target, "universal3", "universal4"]
            )
        ].copy()
        selected["selector"] = selected["calibration_bank"]
        bootstrap = paired_selector_bootstrap(
            selected,
            baseline=target,
            seed=42,
            iterations=args.bootstrap_iterations,
        )
        bootstrap["evaluation_domain"] = target
        bootstrap_frames.append(bootstrap)

    source_scores = {
        domain: (
            ROOT / "results/runs/caes_external_genvidbench_feature_change/video_scores.csv"
            if domain == "genvidbench"
            else ROOT / "results/runs/caes_stage_fs/video_scores.csv"
        )
        for domain in ALL_DOMAINS
    }
    audit_rows = []
    for domain in ALL_DOMAINS:
        expected = pd.read_csv(source_scores[domain], float_precision="round_trip")
        expected = expected[
            expected["selector"].eq("feature_change")
            & expected["dataset"].eq(domain)
        ][["video_id", "global_score", "local_score"]]
        expected["expected_equal_final"] = (
            0.5 * expected["global_score"] + 0.5 * expected["local_score"]
        )
        actual = all_videos[
            all_videos["calibration_bank"].eq(domain)
            & all_videos["evaluation_domain"].eq(domain)
        ][["video_id", "final_score"]]
        merged = expected.merge(actual, on="video_id", validate="one_to_one")
        if len(merged) != len(expected) or len(merged) != len(actual):
            raise ValueError(f"{domain}对角视频身份审计不完整")
        difference = np.abs(
            merged["expected_equal_final"].to_numpy()
            - merged["final_score"].to_numpy()
        )
        audit_rows.append({
            "domain": domain,
            "videos": len(merged),
            "max_abs_final_difference": float(difference.max()),
            "mean_abs_final_difference": float(difference.mean()),
            "pearson": float(merged["expected_equal_final"].corr(merged["final_score"])),
            "spearman": float(
                merged["expected_equal_final"].corr(
                    merged["final_score"], method="spearman"
                )
            ),
        })

    outputs = {
        "pairwise_matrix.csv": pairwise,
        "universal_vs_target_bootstrap.csv": pd.concat(
            bootstrap_frames, ignore_index=True
        ),
        "diagonal_video_audit.csv": pd.DataFrame(audit_rows),
        "video_scores.csv.gz": all_videos,
    }
    for name, frame in outputs.items():
        frame.to_csv(
            args.output_dir / name,
            index=False,
            compression="gzip" if name.endswith(".gz") else None,
        )
    manifest = {
        "status": "completed",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_finalized_manifest_sha256": _sha256(
            args.run_dir / "finalized_manifest.json"
        ),
        "bootstrap_iterations": args.bootstrap_iterations,
        "artifacts": {
            name: _sha256(args.output_dir / name) for name in outputs
        },
    }
    (args.output_dir / "analysis_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(pairwise.to_string(index=False), flush=True)
    print(pd.DataFrame(audit_rows).to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
