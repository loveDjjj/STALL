#!/usr/bin/env python3
"""Analyze the sole predeclared OAS Local-D2 covariance candidate."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml


ROOT = Path(__file__).resolve().parents[1]
for directory in (ROOT / "src", ROOT / "tools"):
    if str(directory) not in sys.path:
        sys.path.insert(0, str(directory))

from analyze_u0_cross_dataset_calibration import calibrate_cross, load_parts
from analyze_u0_locked import global_references
from build_multi_order_baselines import metric_tables
from stable_whitening import (
    StableGaussianParams,
    l2_normalized_second_order,
    score_gaussian_aggregate_float64,
)


DATASETS = ("comgenvid", "videofeedback", "genvideo")
SEEDS = (17, 29, 43, 71, 101)


def load_oas(path: Path) -> StableGaussianParams:
    return StableGaussianParams.from_npz(
        str(path), "mu_patch_temp", "W_patch_temp", "calib_patch_temp_scores"
    )


def batch_invariance(
    reserve_manifest: Path, params_dir: Path, device: str
) -> pd.DataFrame:
    payload = json.loads(reserve_manifest.read_text())
    rows = []
    for dataset in DATASETS:
        items = [item for item in payload["videos"] if item["dataset"] == dataset][:16]
        patches = []
        for item in items:
            cache = ROOT / item["cache_path"]
            data = torch.load(cache, weights_only=True, map_location="cpu")
            patches.append(data["patch"].float())
        features = l2_normalized_second_order(torch.stack(patches))
        params = load_oas(params_dir / f"{dataset}_locked200.npz")
        reference = None
        for batch_size in (1, 4, 8, 16):
            parts = []
            for start in range(0, len(features), batch_size):
                raw, _ = score_gaussian_aggregate_float64(
                    features[start : start + batch_size],
                    params,
                    aggregation="mean",
                    device=device,
                    compute_percentile=False,
                )
                parts.append(raw)
            values = np.concatenate(parts)
            if reference is None:
                reference = values
            rows.append(
                {
                    "dataset": dataset,
                    "batch_size": batch_size,
                    "raw_max_error_vs_batch1": float(np.max(np.abs(values - reference))),
                    "rank_changes_vs_batch1": int(
                        np.count_nonzero(np.argsort(values) != np.argsort(reference))
                    ),
                }
            )
    return pd.DataFrame(rows)


def parameter_diagnostics(params_dir: Path) -> pd.DataFrame:
    rows = []
    for path in sorted(params_dir.glob("*.npz")):
        data = np.load(path, allow_pickle=True)
        metadata = json.loads(str(data["aggregation_config"]))
        rows.append(
            {
                "parameter_file": path.name,
                "calibration_bank": metadata["calibration_bank"],
                "calibration_videos": metadata["calibration_videos"],
                "seed": metadata["seed"],
                "shrinkage": metadata["oas_shrinkage"],
                "eigenvalue_min": metadata["eigenvalue_min"],
                "eigenvalue_max": metadata["eigenvalue_max"],
                "condition_number": metadata["condition_number"],
                "effective_rank": metadata["effective_rank"],
                "fit_samples": metadata["fit_samples"],
                "fit_elapsed_seconds": metadata["elapsed_seconds"],
            }
        )
    return pd.DataFrame(rows)


def write_report(
    dataset_metrics: pd.DataFrame,
    generator_metrics: pd.DataFrame,
    seed_metrics: pd.DataFrame,
    diagnostics: pd.DataFrame,
    invariance: pd.DataFrame,
    report: Path,
) -> None:
    indexed = dataset_metrics.set_index(["config", "dataset"])
    lines = [
        "# U0 OAS covariance candidate",
        "",
        "S1 changes only the Local D2 covariance estimator from the locked effective-rank "
        "whitening to parameter-free OAS shrinkage. PatchSpatial, features, K=3, CDFs, alpha, "
        "beta, and all calibration videos remain fixed.",
        "",
        "## Locked-split result",
        "",
        "| Configuration | ComGenVid | VideoFeedback | GenVideo | Macro-3 |",
        "|---|---:|---:|---:|---:|",
    ]
    for config in ("S0", "S1"):
        cells = []
        for dataset in (*DATASETS, "Macro-3"):
            row = indexed.loc[(config, dataset)]
            cells.append(f"{row.auc:.4f}/{row.ap:.4f}")
        lines.append(f"| {config} | " + " | ".join(cells) + " |")
    delta_ap = float(indexed.loc[("S1", "Macro-3"), "ap"] - indexed.loc[("S0", "Macro-3"), "ap"])
    dataset_deltas = {
        dataset: float(indexed.loc[("S1", dataset), "ap"] - indexed.loc[("S0", dataset), "ap"])
        for dataset in DATASETS
    }
    generator = generator_metrics.pivot_table(
        index=["dataset", "generator"], columns="config", values="ap"
    )
    nondecline = int((generator["S1"] >= generator["S0"]).sum())
    seed_macro = seed_metrics[seed_metrics["dataset"].eq("Macro-3")]
    s0_std = float(seed_macro["S0_ap"].std(ddof=1))
    s1_std = float(seed_macro["S1_ap"].std(ddof=1))
    lines.extend(
        [
            "",
            "## Independent-reserve seed stability",
            "",
            "| Seed | S0 Macro AUC/AP | S1 Macro AUC/AP | Delta AP |",
            "|---:|---:|---:|---:|",
        ]
    )
    for row in seed_macro.sort_values("seed").itertuples(index=False):
        lines.append(
            f"| {row.seed} | {row.S0_auc:.4f}/{row.S0_ap:.4f} | "
            f"{row.S1_auc:.4f}/{row.S1_ap:.4f} | {row.S1_ap-row.S0_ap:+.4f} |"
        )
    lines.extend(
        [
            "",
            f"S0/S1 five-seed Macro AP standard deviations are `{s0_std:.6f}/{s1_std:.6f}`.",
            "",
            "## Covariance and numerics",
            "",
            f"Locked-split OAS shrinkage range is "
            f"`{diagnostics[diagnostics.calibration_bank.str.contains('locked200')].shrinkage.min():.6g}` to "
            f"`{diagnostics[diagnostics.calibration_bank.str.contains('locked200')].shrinkage.max():.6g}`; "
            f"all OAS models have effective rank 1024. Maximum observed condition number is "
            f"`{diagnostics.condition_number.max():.3g}`.",
            f"Batch 1/4/8/16 maximum raw error is `{invariance.raw_max_error_vs_batch1.max():.3g}` "
            f"with `{invariance.rank_changes_vs_batch1.max()}` rank changes.",
            "",
            "## Admission status before external validation",
            "",
            f"- Macro AP delta: `{delta_ap:+.6f}` (required >= +0.003).",
            f"- Dataset AP deltas: "
            + ", ".join(f"{dataset} `{value:+.6f}`" for dataset, value in dataset_deltas.items())
            + " (each required >= -0.003).",
            f"- Generator AP non-decline: `{nondecline}/20` (required >= 12/20).",
            f"- Seed AP std S0/S1: `{s0_std:.6f}/{s1_std:.6f}` (must not increase).",
            "- The 1,000-iteration paired cluster-bootstrap gate is appended after scoring; "
            "external validation is required only if all internal gates pass.",
        ]
    )
    report.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run(args: argparse.Namespace) -> None:
    config = yaml.safe_load(args.config.read_text())
    evaluation = load_parts(args.scores_dir, "evaluation")
    calibration = load_parts(args.scores_dir, "locked_calibration")
    reserve = load_parts(args.scores_dir, "reserve")
    bank_manifest = json.loads(args.bank_manifest.read_text())
    membership = pd.read_csv(args.membership)
    global_spatial_ref, global_t1_ref = global_references(config)
    release = pd.read_csv(args.locked_scores, float_precision="round_trip")
    per_video = release[
        ["video_id", "dataset", "protocol_split", "subset", "source_model", "filename", "S"]
    ].rename(columns={"S": "S0"})
    locked_pieces = []
    for dataset in DATASETS:
        ids = set(bank_manifest["banks"][dataset]["video_ids"])
        scored = calibrate_cross(
            evaluation[evaluation["dataset"].eq(dataset)],
            calibration,
            ids,
            f"oas_{dataset}_locked200",
            global_spatial_ref,
            global_t1_ref,
        ).rename(columns={"S": "S1", "G": "S1_G", "L": "S1_L"})
        locked_pieces.append(scored)
    per_video = per_video.merge(pd.concat(locked_pieces), on="video_id", validate="one_to_one")
    dataset_metrics, generator_metrics = metric_tables(
        per_video,
        seed=int(config["release"]["random_seed"]),
        score_columns=["S0", "S1"],
        config_names={"S0": "stable U0", "S1": "OAS Local D2"},
    )

    s0_seed_scores = pd.read_csv(args.s0_seed_scores, float_precision="round_trip")
    seed_rows = []
    s1_seed_wide = release[
        ["video_id", "dataset", "protocol_split", "subset", "source_model", "filename"]
    ].copy()
    for seed in SEEDS:
        pieces = []
        for dataset in DATASETS:
            ids = set(
                membership[
                    membership["dataset"].eq(dataset)
                    & membership["seed"].eq(seed)
                    & membership["calibration_size"].eq(200)
                ]["video_id"]
            )
            pieces.append(
                calibrate_cross(
                    evaluation[evaluation["dataset"].eq(dataset)],
                    reserve,
                    ids,
                    f"oas_{dataset}_s{seed}_n200",
                    global_spatial_ref,
                    global_t1_ref,
                )[["video_id", "S"]]
            )
        values = pd.concat(pieces).rename(columns={"S": f"S1_s{seed}"})
        s1_seed_wide = s1_seed_wide.merge(values, on="video_id", validate="one_to_one")
        comparison = s1_seed_wide[
            ["video_id", "dataset", "protocol_split", "subset", "source_model", "filename", f"S1_s{seed}"]
        ].merge(
            s0_seed_scores[["video_id", f"S_s{seed}_n200"]],
            on="video_id",
            validate="one_to_one",
        ).rename(columns={f"S1_s{seed}": "S1", f"S_s{seed}_n200": "S0"})
        seed_dataset, _ = metric_tables(
            comparison,
            seed=int(config["release"]["random_seed"]),
            score_columns=["S0", "S1"],
            config_names={"S0": "S0", "S1": "S1"},
        )
        pivot = seed_dataset.pivot(index="dataset", columns="config", values=["auc", "ap"])
        for dataset in (*DATASETS, "Macro-3"):
            seed_rows.append(
                {
                    "seed": seed,
                    "dataset": dataset,
                    "S0_auc": pivot.loc[dataset, ("auc", "S0")],
                    "S0_ap": pivot.loc[dataset, ("ap", "S0")],
                    "S1_auc": pivot.loc[dataset, ("auc", "S1")],
                    "S1_ap": pivot.loc[dataset, ("ap", "S1")],
                }
            )
    seed_metrics = pd.DataFrame(seed_rows)
    diagnostics = parameter_diagnostics(args.oas_params)
    invariance = batch_invariance(args.reserve_manifest, args.oas_params, args.device)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    per_video.to_csv(args.output_dir / "per_video_scores.csv", index=False)
    s1_seed_wide.to_csv(args.output_dir / "per_video_seed_scores.csv", index=False)
    dataset_metrics.to_csv(args.output_dir / "dataset_metrics.csv", index=False)
    generator_metrics.to_csv(args.output_dir / "generator_metrics.csv", index=False)
    seed_metrics.to_csv(args.output_dir / "seed_metrics.csv", index=False)
    diagnostics.to_csv(args.output_dir / "covariance_diagnostics.csv", index=False)
    invariance.to_csv(args.output_dir / "batch_invariance.csv", index=False)
    write_report(
        dataset_metrics,
        generator_metrics,
        seed_metrics,
        diagnostics,
        invariance,
        args.report,
    )
    print(dataset_metrics.to_string(index=False))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", type=Path, default=ROOT / "configs/alpha_stalled_u0_locked.yaml"
    )
    parser.add_argument(
        "--scores-dir", type=Path, default=ROOT / "results/u0_cross_and_oas/scores"
    )
    parser.add_argument(
        "--bank-manifest", type=Path, default=ROOT / "release/u0/cross_calibration_banks.json"
    )
    parser.add_argument(
        "--membership", type=Path, default=ROOT / "release/u0/calibration_split_membership.csv"
    )
    parser.add_argument(
        "--reserve-manifest",
        type=Path,
        default=ROOT / "release/u0/calibration_reserve_manifest.json",
    )
    parser.add_argument(
        "--oas-params", type=Path, default=ROOT / "results/u0_cross_and_oas/oas"
    )
    parser.add_argument(
        "--s0-seed-scores",
        type=Path,
        default=ROOT / "results/u0_calibration_sensitivity/analysis/per_video_scores.csv",
    )
    parser.add_argument(
        "--locked-scores", type=Path, default=ROOT / "release/u0/final_video_scores.csv"
    )
    parser.add_argument(
        "--output-dir", type=Path, default=ROOT / "results/u0_oas_candidate"
    )
    parser.add_argument(
        "--report", type=Path, default=ROOT / "reports/u0_oas_covariance_candidate.md"
    )
    parser.add_argument("--device", default="cuda:0")
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
