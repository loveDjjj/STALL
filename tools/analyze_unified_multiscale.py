#!/usr/bin/env python3
"""Evaluate unified region-scale temporal fusion under frozen K=3 MW2."""

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
from build_multi_order_baselines import empirical_cdf, metric_tables, paired_bootstrap


KEY_COLUMNS = ["dataset", "protocol_split", "subset", "source_model", "filename"]
WINDOW_KEYS = [*KEY_COLUMNS, "window_id"]
CONFIG_NAMES = {
    "MS0": "Dataset-specific region reference",
    "MS1": "Unified region 1",
    "MS2": "Unified region 2",
    "MS3": "Unified equal region 1+2",
    "MS4": "Unified equal region 1+2+3",
}


def read_region_scores(root: Path, num_shards: int) -> pd.DataFrame:
    frames = []
    for dataset in ("comgenvid", "videofeedback", "genvideo"):
        for shard in range(num_shards):
            path = root / f"{dataset}_region_shard{shard}.csv"
            if not path.exists():
                path = root / f"{dataset}_combined_shard{shard}.csv"
            if not path.exists():
                raise FileNotFoundError(path)
            frames.append(pd.read_csv(path, float_precision="round_trip"))
    scores = pd.concat(frames, ignore_index=True)
    if scores.duplicated(WINDOW_KEYS).any():
        raise ValueError("duplicate multiscale window keys")
    return scores


def merge_window_scores(baseline: pd.DataFrame, region: pd.DataFrame) -> pd.DataFrame:
    columns = [
        *WINDOW_KEYS,
        "frame_indices",
        "region1_temporal",
        "region2_temporal",
        "region3_temporal",
    ]
    merged = baseline.merge(
        region[columns],
        on=WINDOW_KEYS,
        how="inner",
        validate="one_to_one",
        suffixes=("", "_region"),
    )
    if len(merged) != len(baseline) or len(merged) != len(region):
        raise ValueError(
            f"window join changed rows: baseline={len(baseline)} region={len(region)} "
            f"merged={len(merged)}"
        )
    if not (merged["frame_indices"] == merged["frame_indices_region"]).all():
        raise ValueError("multiscale frame indices differ from frozen K=3 windows")
    merged = merged.drop(columns="frame_indices_region")
    temporal = {
        "MS1": merged["region1_temporal"],
        "MS2": merged["region2_temporal"],
        "MS3": 0.5 * (merged["region1_temporal"] + merged["region2_temporal"]),
        "MS4": (
            merged["region1_temporal"]
            + merged["region2_temporal"]
            + merged["region3_temporal"]
        )
        / 3.0,
    }
    merged["MS0_L_k"] = merged["L_k"]
    for config, values in temporal.items():
        merged[f"{config}_temporal"] = values
        merged[f"{config}_L_k"] = 0.1 * merged["patch_spatial"] + 0.9 * values
    return merged


def aggregate_videos(window_scores: pd.DataFrame) -> pd.DataFrame:
    aggregations = {
        "duration_seconds": ("duration_seconds", "first"),
        "effective_k": ("effective_k", "first"),
        "windows": ("window_id", "size"),
        "G_mean_raw": ("G_k", "mean"),
    }
    for config in CONFIG_NAMES:
        aggregations[f"{config}_L_raw"] = (f"{config}_L_k", "mean")
    result = (
        window_scores.groupby(KEY_COLUMNS, sort=False, observed=True)
        .agg(**aggregations)
        .reset_index()
    )
    if not (result["effective_k"] == result["windows"]).all():
        raise ValueError("effective_k does not match window count")
    return result


def target_k_reference(
    window_scores: pd.DataFrame, target_k: int, local_column: str
) -> pd.DataFrame:
    calibration = window_scores[
        (window_scores["protocol_split"] == "calibration")
        & (window_scores["subset"] == "real")
    ]
    rows = []
    for key, frame in calibration.groupby(KEY_COLUMNS, sort=False, observed=True):
        ordered = frame.sort_values("window_id")
        if len(ordered) < target_k:
            continue
        positions = (
            np.array([(len(ordered) - 1) // 2], dtype=int)
            if target_k == 1
            else np.rint(np.linspace(0, len(ordered) - 1, target_k)).astype(int)
        )
        selected = ordered.iloc[np.unique(positions)]
        if len(selected) != target_k:
            continue
        rows.append(
            {
                **dict(zip(KEY_COLUMNS, key)),
                "G_mean_raw": float(selected["G_k"].mean()),
                "L_raw": float(selected[local_column].mean()),
            }
        )
    reference = pd.DataFrame(rows)
    if reference.empty:
        raise ValueError(f"no calibration reference for effective_k={target_k}")
    return reference


def calibrate_configs(per_video: pd.DataFrame, windows: pd.DataFrame) -> pd.DataFrame:
    frames = []
    for dataset, dataset_video in per_video.groupby("dataset", sort=False):
        evaluation = dataset_video[dataset_video["protocol_split"] == "evaluation"].copy()
        dataset_windows = windows[windows["dataset"] == dataset]
        groups = []
        for effective_k, target in evaluation.groupby("effective_k", sort=True):
            target = target.copy()
            global_reference = target_k_reference(
                dataset_windows, int(effective_k), "MS0_L_k"
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


def attach_frozen_order(evaluation: pd.DataFrame, frozen_path: Path) -> pd.DataFrame:
    frozen = pd.read_csv(frozen_path, float_precision="round_trip")[
        [*KEY_COLUMNS, "G_mean", "MW2"]
    ].copy()
    frozen["_protocol_order"] = np.arange(len(frozen), dtype=np.int64)
    frozen = frozen.rename(columns={"G_mean": "frozen_G", "MW2": "frozen_MS0"})
    merged = evaluation.merge(frozen, on=KEY_COLUMNS, how="inner", validate="one_to_one")
    if len(merged) != len(frozen):
        raise ValueError("frozen K=3 join changed row count")
    if float((merged["G"] - merged["frozen_G"]).abs().max()) > 1e-12:
        raise ValueError("unified study changed frozen global calibration")
    if float((merged["MS0"] - merged["frozen_MS0"]).abs().max()) > 1e-12:
        raise ValueError("MS0 does not exactly reproduce frozen K=3 MW2")
    return (
        merged.sort_values("_protocol_order")
        .drop(columns="_protocol_order")
        .reset_index(drop=True)
    )


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
                    "delta_auc_vs_MS0": float(
                        pivot.loc[(dataset, generator), ("auc", config)]
                        - pivot.loc[(dataset, generator), ("auc", "MS0")]
                    ),
                    "delta_ap_vs_MS0": float(
                        pivot.loc[(dataset, generator), ("ap", config)]
                        - pivot.loc[(dataset, generator), ("ap", "MS0")]
                    ),
                }
            )
    return pd.DataFrame(rows)


def motion_and_failure_tables(evaluation: pd.DataFrame, diagnostics_path: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    diagnostics = pd.read_csv(diagnostics_path)[
        [*KEY_COLUMNS, "motion_group", "fixed_error_type"]
    ]
    merged = evaluation.merge(diagnostics, on=KEY_COLUMNS, how="left", validate="one_to_one")
    if merged[["motion_group", "fixed_error_type"]].isna().any().any():
        raise ValueError("missing motion diagnostics")
    motion_rows = []
    for motion_group in ("low", "mid", "high"):
        fake = merged[
            (merged["subset"] == "annotated")
            & (merged["motion_group"] == motion_group)
        ]
        real = merged[merged["subset"] == "real"]
        subset = pd.concat([real, fake], ignore_index=True)
        datasets, _ = metric_tables(
            subset,
            42,
            score_columns=tuple(CONFIG_NAMES),
            config_names=CONFIG_NAMES,
        )
        datasets.insert(0, "fake_motion_group", motion_group)
        motion_rows.append(datasets)
    false_negative = merged[
        (merged["subset"] == "annotated")
        & (merged["fixed_error_type"] == "false_negative_at_0.5")
    ].copy()
    failure_rows = []
    for config in CONFIG_NAMES:
        failure_rows.append(
            {
                "config": config,
                "n_baseline_false_negatives": len(false_negative),
                "mean_score": float(false_negative[config].mean()),
                "still_false_negative_at_0.5": int((false_negative[config] >= 0.5).sum()),
            }
        )
    return pd.concat(motion_rows, ignore_index=True), pd.DataFrame(failure_rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--region-score-root", type=Path, required=True)
    parser.add_argument("--num-shards", type=int, default=2)
    parser.add_argument("--output-dir", type=Path, default=REPO_ROOT / "results/unified_multiscale_layers")
    parser.add_argument("--bootstrap-iterations", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--baseline-windows", type=Path, default=REPO_ROOT / "results/multi_window_joint_typicality/K3_uniform_window_scores.csv")
    parser.add_argument("--frozen-evaluation", type=Path, default=REPO_ROOT / "results/multi_window_joint_typicality/K3_uniform_evaluation_scores.csv")
    parser.add_argument("--motion-diagnostics", type=Path, default=REPO_ROOT / "results/local_residual/per_video_scores.csv")
    args = parser.parse_args()

    baseline = pd.read_csv(args.baseline_windows, float_precision="round_trip")
    region = read_region_scores(args.region_score_root, args.num_shards)
    windows = merge_window_scores(baseline, region)
    per_video = aggregate_videos(windows)
    evaluation = attach_frozen_order(
        calibrate_configs(per_video, windows), args.frozen_evaluation
    )
    dataset_metrics, generator_metrics = metric_tables(
        evaluation,
        args.seed,
        score_columns=tuple(CONFIG_NAMES),
        config_names=CONFIG_NAMES,
    )
    comparisons = [(config, "MS0", f"{config}-MS0") for config in CONFIG_NAMES if config != "MS0"]
    dataset_bootstrap = paired_bootstrap(
        evaluation,
        args.seed,
        args.bootstrap_iterations,
        comparisons,
    )
    macro_bootstrap = macro_cluster_bootstrap(
        evaluation,
        [config for config in CONFIG_NAMES if config != "MS0"],
        args.seed,
        args.bootstrap_iterations,
        base_config="MS0",
    )
    bootstrap = pd.concat([dataset_bootstrap, macro_bootstrap], ignore_index=True)
    deltas = generator_deltas(generator_metrics)
    motion, failures = motion_and_failure_tables(evaluation, args.motion_diagnostics)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    windows.to_csv(args.output_dir / "multiscale_per_window_scores.csv", index=False)
    evaluation.to_csv(args.output_dir / "multiscale_per_video_scores.csv", index=False)
    dataset_metrics.to_csv(args.output_dir / "multiscale_dataset_metrics.csv", index=False)
    generator_metrics.to_csv(args.output_dir / "multiscale_generator_metrics.csv", index=False)
    deltas.to_csv(args.output_dir / "multiscale_generator_deltas.csv", index=False)
    bootstrap.to_csv(args.output_dir / "multiscale_bootstrap_deltas.csv", index=False)
    motion.to_csv(args.output_dir / "multiscale_motion_metrics.csv", index=False)
    failures.to_csv(args.output_dir / "multiscale_baseline_false_negatives.csv", index=False)
    print(dataset_metrics.to_string(index=False))


if __name__ == "__main__":
    main()
