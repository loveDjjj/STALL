#!/usr/bin/env python3
"""Evaluate H0-H5 calibrated DINO layer temporal scores under frozen K=3 MW2."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
TOOLS_DIR = REPO_ROOT / "tools"
if str(TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(TOOLS_DIR))

from analyze_multi_window_scores import macro_cluster_bootstrap
from analyze_unified_multiscale import (
    KEY_COLUMNS,
    WINDOW_KEYS,
    target_k_reference,
)
from build_multi_order_baselines import empirical_cdf, metric_tables, paired_bootstrap


CONFIG_NAMES = {
    "H0": "Final layer 23",
    "H1": "Mid layer 11",
    "H2": "Late layer 17",
    "H3": "Equal calibrated layer 11+23",
    "H4": "Equal calibrated layer 17+23",
    "H5": "Equal calibrated layer 11+17+23",
}


def read_layer_scores(root: Path, num_shards: int) -> pd.DataFrame:
    frames = []
    for dataset in ("comgenvid", "videofeedback", "genvideo"):
        for shard in range(num_shards):
            path = root / f"{dataset}_layer_shard{shard}.csv"
            if not path.exists():
                path = root / f"{dataset}_combined_shard{shard}.csv"
            if not path.exists():
                raise FileNotFoundError(path)
            frames.append(pd.read_csv(path, float_precision="round_trip"))
    scores = pd.concat(frames, ignore_index=True)
    if scores.duplicated(WINDOW_KEYS).any():
        raise ValueError("duplicate layer window keys")
    return scores


def merge_layer_windows(baseline: pd.DataFrame, layer: pd.DataFrame) -> pd.DataFrame:
    columns = [
        *WINDOW_KEYS,
        "frame_indices",
        "layer11_temporal",
        "layer17_temporal",
        "layer23_temporal",
    ]
    merged = baseline.merge(
        layer[columns],
        on=WINDOW_KEYS,
        how="inner",
        validate="one_to_one",
        suffixes=("", "_layer"),
    )
    if len(merged) != len(baseline) or len(merged) != len(layer):
        raise ValueError("layer join changed frozen window count")
    if not (merged["frame_indices"] == merged["frame_indices_layer"]).all():
        raise ValueError("layer frame indices differ from frozen K=3 windows")
    merged = merged.drop(columns="frame_indices_layer")
    error = float((merged["layer23_temporal"] - merged["patch_d2"]).abs().max())
    if error > 1e-7:
        raise ValueError(f"layer 23 score differs from frozen final score: {error}")
    temporal = {
        "H0": merged["patch_d2"],
        "H1": merged["layer11_temporal"],
        "H2": merged["layer17_temporal"],
        "H3": 0.5 * (merged["layer11_temporal"] + merged["patch_d2"]),
        "H4": 0.5 * (merged["layer17_temporal"] + merged["patch_d2"]),
        "H5": (
            merged["layer11_temporal"]
            + merged["layer17_temporal"]
            + merged["patch_d2"]
        )
        / 3.0,
    }
    for config, values in temporal.items():
        merged[f"{config}_temporal"] = values
        merged[f"{config}_L_k"] = (
            merged["L_k"]
            if config == "H0"
            else 0.1 * merged["patch_spatial"] + 0.9 * values
        )
    return merged


def aggregate_and_calibrate(windows: pd.DataFrame) -> pd.DataFrame:
    aggregations = {
        "duration_seconds": ("duration_seconds", "first"),
        "effective_k": ("effective_k", "first"),
        "windows": ("window_id", "size"),
        "G_mean_raw": ("G_k", "mean"),
    }
    for config in CONFIG_NAMES:
        aggregations[f"{config}_L_raw"] = (f"{config}_L_k", "mean")
    per_video = (
        windows.groupby(KEY_COLUMNS, sort=False, observed=True)
        .agg(**aggregations)
        .reset_index()
    )
    if not (per_video["effective_k"] == per_video["windows"]).all():
        raise ValueError("layer effective_k does not match window count")
    frames = []
    for dataset, dataset_video in per_video.groupby("dataset", sort=False):
        evaluation = dataset_video[dataset_video["protocol_split"] == "evaluation"]
        dataset_windows = windows[windows["dataset"] == dataset]
        groups = []
        for effective_k, target in evaluation.groupby("effective_k", sort=True):
            target = target.copy()
            global_reference = target_k_reference(
                dataset_windows, int(effective_k), "H0_L_k"
            )
            target["G"] = empirical_cdf(
                target["G_mean_raw"].to_numpy(),
                global_reference["G_mean_raw"].to_numpy(),
            )
            for config in CONFIG_NAMES:
                reference = target_k_reference(
                    dataset_windows, int(effective_k), f"{config}_L_k"
                )
                target[f"{config}_L"] = empirical_cdf(
                    target[f"{config}_L_raw"].to_numpy(),
                    reference["L_raw"].to_numpy(),
                )
                target[config] = 0.6 * target["G"] + 0.4 * target[f"{config}_L"]
            groups.append(target)
        frames.append(pd.concat(groups, ignore_index=True))
    return pd.concat(frames, ignore_index=True)


def attach_frozen(evaluation: pd.DataFrame, path: Path) -> pd.DataFrame:
    frozen = pd.read_csv(path, float_precision="round_trip")[
        [*KEY_COLUMNS, "G_mean", "MW2"]
    ].copy()
    frozen["_order"] = np.arange(len(frozen), dtype=np.int64)
    frozen = frozen.rename(columns={"G_mean": "frozen_G", "MW2": "frozen_H0"})
    merged = evaluation.merge(frozen, on=KEY_COLUMNS, how="inner", validate="one_to_one")
    if len(merged) != len(frozen):
        raise ValueError("frozen evaluation join changed row count")
    if float((merged["G"] - merged["frozen_G"]).abs().max()) > 1e-12:
        raise ValueError("layer study changed global calibration")
    if float((merged["H0"] - merged["frozen_H0"]).abs().max()) > 1e-12:
        raise ValueError("H0 does not reproduce frozen K=3 MW2")
    return merged.sort_values("_order").drop(columns="_order").reset_index(drop=True)


def generator_deltas(metrics: pd.DataFrame) -> pd.DataFrame:
    pivot = metrics.pivot(
        index=["dataset", "generator"], columns="config", values=["auc", "ap"]
    )
    rows = []
    for dataset, generator in pivot.index:
        for config in CONFIG_NAMES:
            rows.append(
                {
                    "dataset": dataset,
                    "generator": generator,
                    "config": config,
                    "delta_auc_vs_H0": float(
                        pivot.loc[(dataset, generator), ("auc", config)]
                        - pivot.loc[(dataset, generator), ("auc", "H0")]
                    ),
                    "delta_ap_vs_H0": float(
                        pivot.loc[(dataset, generator), ("ap", config)]
                        - pivot.loc[(dataset, generator), ("ap", "H0")]
                    ),
                }
            )
    return pd.DataFrame(rows)


def motion_failure_tables(evaluation: pd.DataFrame, diagnostics_path: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    diagnostics = pd.read_csv(diagnostics_path)[
        [*KEY_COLUMNS, "motion_group", "fixed_error_type"]
    ]
    merged = evaluation.merge(diagnostics, on=KEY_COLUMNS, how="left", validate="one_to_one")
    motion_rows = []
    for group in ("low", "mid", "high"):
        subset = pd.concat(
            [
                merged[merged["subset"] == "real"],
                merged[
                    (merged["subset"] == "annotated")
                    & (merged["motion_group"] == group)
                ],
            ],
            ignore_index=True,
        )
        metrics, _ = metric_tables(
            subset,
            42,
            score_columns=tuple(CONFIG_NAMES),
            config_names=CONFIG_NAMES,
        )
        metrics.insert(0, "fake_motion_group", group)
        motion_rows.append(metrics)
    failures = merged[
        (merged["subset"] == "annotated")
        & (merged["fixed_error_type"] == "false_negative_at_0.5")
    ]
    failure_rows = pd.DataFrame(
        [
            {
                "config": config,
                "n_baseline_false_negatives": len(failures),
                "mean_score": float(failures[config].mean()),
                "still_false_negative_at_0.5": int((failures[config] >= 0.5).sum()),
            }
            for config in CONFIG_NAMES
        ]
    )
    return pd.concat(motion_rows, ignore_index=True), failure_rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--layer-score-root", type=Path, required=True)
    parser.add_argument("--num-shards", type=int, default=2)
    parser.add_argument("--output-dir", type=Path, default=REPO_ROOT / "results/unified_multiscale_layers")
    parser.add_argument("--bootstrap-iterations", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--baseline-windows", type=Path, default=REPO_ROOT / "results/multi_window_joint_typicality/K3_uniform_window_scores.csv")
    parser.add_argument("--frozen-evaluation", type=Path, default=REPO_ROOT / "results/multi_window_joint_typicality/K3_uniform_evaluation_scores.csv")
    parser.add_argument("--motion-diagnostics", type=Path, default=REPO_ROOT / "results/local_residual/per_video_scores.csv")
    args = parser.parse_args()

    baseline = pd.read_csv(args.baseline_windows, float_precision="round_trip")
    layers = read_layer_scores(args.layer_score_root, args.num_shards)
    windows = merge_layer_windows(baseline, layers)
    evaluation = attach_frozen(
        aggregate_and_calibrate(windows), args.frozen_evaluation
    )
    dataset_metrics, generator_metrics = metric_tables(
        evaluation,
        args.seed,
        score_columns=tuple(CONFIG_NAMES),
        config_names=CONFIG_NAMES,
    )
    comparisons = tuple(
        (config, "H0", f"{config}-H0") for config in CONFIG_NAMES if config != "H0"
    )
    bootstrap = pd.concat(
        [
            paired_bootstrap(
                evaluation, args.seed, args.bootstrap_iterations, comparisons
            ),
            macro_cluster_bootstrap(
                evaluation,
                [config for config in CONFIG_NAMES if config != "H0"],
                args.seed,
                args.bootstrap_iterations,
                base_config="H0",
            ),
        ],
        ignore_index=True,
    )
    deltas = generator_deltas(generator_metrics)
    motion, failures = motion_failure_tables(evaluation, args.motion_diagnostics)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    windows.to_csv(args.output_dir / "layer_per_window_scores.csv", index=False)
    evaluation.to_csv(args.output_dir / "layer_per_video_scores.csv", index=False)
    dataset_metrics.to_csv(args.output_dir / "layer_dataset_metrics.csv", index=False)
    generator_metrics.to_csv(args.output_dir / "layer_generator_metrics.csv", index=False)
    deltas.to_csv(args.output_dir / "layer_generator_deltas.csv", index=False)
    bootstrap.to_csv(args.output_dir / "layer_bootstrap_deltas.csv", index=False)
    motion.to_csv(args.output_dir / "layer_motion_metrics.csv", index=False)
    failures.to_csv(args.output_dir / "layer_baseline_false_negatives.csv", index=False)
    print(dataset_metrics.to_string(index=False))


if __name__ == "__main__":
    main()
