#!/usr/bin/env python3
"""Aggregate and evaluate frozen K=3 raw D2 versus spatial-mean residual D2."""

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
    "R0": "Frozen K=3 MW2 raw D2",
    "R1": "K=3 MW2 spatial-mean residual D2",
}
ADMISSION_FOCUS = {"ZeroScope-576w", "Pika", "SoRA-Clip", "Crafter"}


def read_residual_shards(root: Path, num_shards: int) -> pd.DataFrame:
    frames = []
    for dataset in ("comgenvid", "videofeedback", "genvideo"):
        for shard in range(num_shards):
            path = root / f"{dataset}_R1_shard{shard}.csv"
            if not path.exists():
                raise FileNotFoundError(path)
            frames.append(pd.read_csv(path, float_precision="round_trip"))
    scores = pd.concat(frames, ignore_index=True)
    if scores.duplicated(WINDOW_KEYS).any():
        raise ValueError("duplicate residual window keys")
    return scores


def merge_window_scores(baseline: pd.DataFrame, residual: pd.DataFrame) -> pd.DataFrame:
    residual_columns = [
        *WINDOW_KEYS,
        "frame_indices",
        "residual_raw_likelihood",
        "residual_d2",
        "rho_mean",
        "rho_median",
        "mean_motion_magnitude",
        "mean_common_magnitude",
        "median_common_magnitude",
    ]
    merged = baseline.merge(
        residual[residual_columns],
        on=WINDOW_KEYS,
        how="inner",
        validate="one_to_one",
        suffixes=("", "_residual"),
    )
    if len(merged) != len(baseline) or len(merged) != len(residual):
        raise ValueError(
            f"window join changed row count: baseline={len(baseline)} "
            f"residual={len(residual)} merged={len(merged)}"
        )
    if not (merged["frame_indices"] == merged["frame_indices_residual"]).all():
        raise ValueError("residual frame indices differ from frozen K=3 windows")
    merged = merged.drop(columns="frame_indices_residual")
    merged["R0_temporal"] = merged["patch_d2"]
    merged["R1_temporal"] = merged["residual_d2"]
    merged["R0_L_k"] = merged["L_k"]
    merged["R1_L_k"] = 0.1 * merged["patch_spatial"] + 0.9 * merged["R1_temporal"]
    return merged


def aggregate_videos(window_scores: pd.DataFrame) -> pd.DataFrame:
    grouped = window_scores.groupby(KEY_COLUMNS, sort=False, observed=True)
    per_video = grouped.agg(
        video_path=("video_path", "first"),
        duration_seconds=("duration_seconds", "first"),
        effective_k=("effective_k", "first"),
        windows=("window_id", "size"),
        G_mean_raw=("G_k", "mean"),
        R0_L_mean_raw=("R0_L_k", "mean"),
        R1_L_mean_raw=("R1_L_k", "mean"),
        raw_d2_score=("R0_temporal", "mean"),
        residual_d2_score=("R1_temporal", "mean"),
        local_spatial_score=("patch_spatial", "mean"),
        global_T1_score=("global_t1", "mean"),
        rho_mean=("rho_mean", "mean"),
        rho_mean_max=("rho_mean", "max"),
        rho_median=("rho_median", "mean"),
        mean_motion_magnitude=("mean_motion_magnitude", "mean"),
        mean_common_magnitude=("mean_common_magnitude", "mean"),
        median_common_magnitude=("median_common_magnitude", "mean"),
    ).reset_index()
    if not (per_video["effective_k"] == per_video["windows"]).all():
        raise ValueError("effective_k does not match residual window count")
    return per_video


def target_k_reference(window_scores: pd.DataFrame, target_k: int) -> pd.DataFrame:
    calibration = window_scores[
        (window_scores["protocol_split"] == "calibration")
        & (window_scores["subset"] == "real")
    ]
    rows = []
    for key, frame in calibration.groupby(KEY_COLUMNS, sort=False, observed=True):
        ordered = frame.sort_values("window_id")
        if len(ordered) < target_k:
            continue
        if target_k == 1:
            positions = np.array([(len(ordered) - 1) // 2], dtype=int)
        else:
            positions = np.rint(np.linspace(0, len(ordered) - 1, target_k)).astype(int)
        selected = ordered.iloc[np.unique(positions)]
        if len(selected) != target_k:
            continue
        rows.append(
            {
                **dict(zip(KEY_COLUMNS, key)),
                "G_mean_raw": float(selected["G_k"].mean()),
                "R1_L_mean_raw": float(selected["R1_L_k"].mean()),
            }
        )
    reference = pd.DataFrame(rows)
    if reference.empty:
        raise ValueError(f"no calibration reference supports effective_k={target_k}")
    return reference


def calibrate_r1(
    per_video: pd.DataFrame,
    window_scores: pd.DataFrame,
    frozen_evaluation: pd.DataFrame,
) -> pd.DataFrame:
    frames = []
    for dataset, dataset_videos in per_video.groupby("dataset", sort=False):
        evaluation = dataset_videos[dataset_videos["protocol_split"] == "evaluation"].copy()
        dataset_windows = window_scores[window_scores["dataset"] == dataset]
        groups = []
        for effective_k, target in evaluation.groupby("effective_k", sort=True):
            reference = target_k_reference(dataset_windows, int(effective_k))
            target = target.copy()
            target["calibration_reference_n"] = len(reference)
            target["R1_G"] = empirical_cdf(
                target["G_mean_raw"].to_numpy(), reference["G_mean_raw"].to_numpy()
            )
            target["R1_L"] = empirical_cdf(
                target["R1_L_mean_raw"].to_numpy(), reference["R1_L_mean_raw"].to_numpy()
            )
            target["R1"] = 0.6 * target["R1_G"] + 0.4 * target["R1_L"]
            groups.append(target)
        frames.append(pd.concat(groups, ignore_index=True))
    evaluation = pd.concat(frames, ignore_index=True)
    frozen = frozen_evaluation[[*KEY_COLUMNS, "MW2", "G_mean", "L_mean"]].copy()
    frozen["_protocol_order"] = np.arange(len(frozen), dtype=np.int64)
    frozen = frozen.rename(
        columns={"MW2": "R0", "G_mean": "R0_G", "L_mean": "R0_L"}
    )
    evaluation = evaluation.merge(frozen, on=KEY_COLUMNS, how="inner", validate="one_to_one")
    if len(evaluation) != len(frozen):
        raise ValueError("frozen K=3 evaluation join changed row count")
    max_global_error = float((evaluation["R1_G"] - evaluation["R0_G"]).abs().max())
    if max_global_error > 1e-12:
        raise ValueError(f"R1 changed frozen global calibration: max error={max_global_error}")
    evaluation["final_score"] = evaluation["R0"]
    evaluation["R1_minus_R0"] = evaluation["R1"] - evaluation["R0"]
    return evaluation.sort_values("_protocol_order").drop(columns="_protocol_order").reset_index(drop=True)


def add_bins(per_video: pd.DataFrame) -> pd.DataFrame:
    out = per_video.copy()
    out["duration_group"] = pd.cut(
        out["duration_seconds"],
        bins=[-np.inf, 4, 6, 10, 20, np.inf],
        labels=["2-4s", "4-6s", "6-10s", "10-20s", "20s+"],
        right=False,
    )
    out["effective_k_group"] = out["effective_k"].map(lambda value: str(int(value)))
    real = out[out["subset"] == "real"]
    motion_bins = {}
    for dataset, frame in real.groupby("dataset", sort=False):
        motion_bins[dataset] = np.quantile(frame["mean_motion_magnitude"], [1 / 3, 2 / 3])
    labels = []
    for row in out.itertuples(index=False):
        low, high = motion_bins[row.dataset]
        if row.mean_motion_magnitude <= low:
            labels.append("low")
        elif row.mean_motion_magnitude <= high:
            labels.append("mid")
        else:
            labels.append("high")
    out["motion_group"] = labels
    out["fixed_error_type"] = "correct_at_0.5"
    out.loc[(out["subset"] == "real") & (out["R0"] < 0.5), "fixed_error_type"] = "false_positive_at_0.5"
    out.loc[(out["subset"] == "annotated") & (out["R0"] >= 0.5), "fixed_error_type"] = "false_negative_at_0.5"
    return out


def common_mode_tables(per_video: pd.DataFrame) -> pd.DataFrame:
    rows = []
    group_specs = [
        ("dataset_subset", ["dataset", "subset"]),
        ("generator", ["dataset", "subset", "source_model"]),
        ("error_type", ["dataset", "fixed_error_type"]),
        ("motion_group", ["dataset", "motion_group"]),
        ("effective_k", ["dataset", "effective_k_group"]),
        ("duration", ["dataset", "duration_group"]),
    ]
    measures = [
        "rho_mean",
        "rho_median",
        "mean_motion_magnitude",
        "mean_common_magnitude",
        "raw_d2_score",
        "residual_d2_score",
        "R0",
        "R1",
        "R1_minus_R0",
    ]
    for group_type, columns in group_specs:
        for key, frame in per_video.groupby(columns, observed=True, sort=True):
            key = key if isinstance(key, tuple) else (key,)
            identity = {column: value for column, value in zip(columns, key)}
            for measure in measures:
                values = frame[measure].dropna().to_numpy(dtype=np.float64)
                if not len(values):
                    continue
                rows.append(
                    {
                        "group_type": group_type,
                        **identity,
                        "measure": measure,
                        "n": len(values),
                        "mean": float(np.mean(values)),
                        "median": float(np.median(values)),
                        "q10": float(np.quantile(values, 0.1)),
                        "q90": float(np.quantile(values, 0.9)),
                    }
                )
    return pd.DataFrame(rows)


def failure_cases(per_video: pd.DataFrame, count: int = 100) -> pd.DataFrame:
    frames = []
    for dataset, frame in per_video.groupby("dataset", sort=False):
        real = frame[frame["subset"] == "real"].nsmallest(count, "R0").copy()
        real["failure_case"] = "lowest_R0_real"
        frames.append(real)
        for _, fake in frame[frame["subset"] == "annotated"].groupby("source_model", sort=True):
            selected = fake.nlargest(min(count, len(fake)), "R0").copy()
            selected["failure_case"] = "highest_R0_fake"
            frames.append(selected)
    return pd.concat(frames, ignore_index=True)


def admission_summary(
    dataset_metrics: pd.DataFrame,
    generator_metrics: pd.DataFrame,
    bootstrap: pd.DataFrame,
    per_video: pd.DataFrame,
) -> dict[str, object]:
    dataset = dataset_metrics[dataset_metrics["dataset"] != "Macro-3"].pivot(
        index="dataset", columns="config", values="ap"
    )
    macro = dataset_metrics[dataset_metrics["dataset"] == "Macro-3"].set_index("config")
    generator = generator_metrics.pivot(index=["dataset", "generator"], columns="config", values="ap")
    macro_boot = bootstrap[
        (bootstrap["dataset"] == "Macro-3")
        & (bootstrap["comparison"] == "R1-R0")
        & (bootstrap["metric"] == "ap")
    ].iloc[0]
    focus = generator.reset_index()
    focus = focus[focus["generator"].isin(ADMISSION_FOCUS)]
    high_real = per_video[(per_video["subset"] == "real") & (per_video["motion_group"] == "high")]
    fp_r0 = int((high_real["R0"] < 0.5).sum())
    fp_r1 = int((high_real["R1"] < 0.5).sum())
    checks = {
        "macro_ap_delta": float(macro.loc["R1", "ap"] - macro.loc["R0", "ap"]),
        "worst_dataset_ap_delta": float((dataset["R1"] - dataset["R0"]).min()),
        "generator_non_decline_count": int((generator["R1"] >= generator["R0"]).sum()),
        "generator_total": len(generator),
        "macro_ap_ci95_low": float(macro_boot.ci95_low),
        "macro_ap_ci95_high": float(macro_boot.ci95_high),
        "focus_improvement_count": int((focus["R1"] > focus["R0"]).sum()),
        "high_motion_real_fp_R0": fp_r0,
        "high_motion_real_fp_R1": fp_r1,
    }
    checks["admitted"] = bool(
        checks["macro_ap_delta"] >= 0.005
        and checks["worst_dataset_ap_delta"] >= -0.005
        and checks["generator_non_decline_count"] >= 12
        and checks["macro_ap_ci95_low"] > 0
        and checks["focus_improvement_count"] >= 2
        and fp_r1 <= fp_r0
    )
    return checks


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--num-shards", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--bootstrap-iterations", type=int, default=1000)
    parser.add_argument(
        "--window-scores",
        type=Path,
        default=REPO_ROOT / "results/multi_window_joint_typicality/K3_uniform_window_scores.csv",
    )
    parser.add_argument(
        "--frozen-evaluation",
        type=Path,
        default=REPO_ROOT / "results/multi_window_joint_typicality/K3_uniform_evaluation_scores.csv",
    )
    parser.add_argument(
        "--residual-shards",
        type=Path,
        default=REPO_ROOT / "results/local_residual/window_shards",
    )
    parser.add_argument(
        "--residual-scores",
        type=Path,
        help="Canonical merged residual scores; use this to rebuild summaries after shard cleanup.",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=REPO_ROOT / "results/local_residual"
    )
    args = parser.parse_args()

    baseline = pd.read_csv(args.window_scores, float_precision="round_trip")
    residual = (
        pd.read_csv(args.residual_scores, float_precision="round_trip")
        if args.residual_scores is not None
        else read_residual_shards(args.residual_shards, args.num_shards)
    )
    windows = merge_window_scores(baseline, residual)
    videos = aggregate_videos(windows)
    frozen = pd.read_csv(args.frozen_evaluation, float_precision="round_trip")
    evaluation = calibrate_r1(videos, windows, frozen)
    evaluation = add_bins(evaluation)
    dataset_metrics, generator_metrics = metric_tables(
        evaluation,
        args.seed,
        score_columns=tuple(CONFIG_NAMES),
        config_names=CONFIG_NAMES,
    )
    comparisons = (("R1", "R0", "R1-R0"),)
    dataset_bootstrap = paired_bootstrap(
        evaluation, args.seed, args.bootstrap_iterations, comparisons=comparisons
    )
    macro_bootstrap = macro_cluster_bootstrap(
        evaluation.rename(columns={"R0": "MW0"}),
        ["R1"],
        args.seed,
        args.bootstrap_iterations,
    ).replace({"comparison": {"R1-MW0": "R1-R0"}, "base_config": {"MW0": "R0"}})
    bootstrap = pd.concat([dataset_bootstrap, macro_bootstrap], ignore_index=True)
    admission = admission_summary(dataset_metrics, generator_metrics, bootstrap, evaluation)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    windows.to_csv(args.output_dir / "per_window_scores.csv", index=False)
    evaluation.to_csv(args.output_dir / "per_video_scores.csv", index=False)
    dataset_metrics.to_csv(args.output_dir / "dataset_metrics.csv", index=False)
    generator_metrics.to_csv(args.output_dir / "generator_metrics.csv", index=False)
    bootstrap.to_csv(args.output_dir / "bootstrap_deltas.csv", index=False)
    common_mode_tables(evaluation).to_csv(
        args.output_dir / "common_mode_analysis.csv", index=False
    )
    failure_cases(evaluation).to_csv(args.output_dir / "failure_cases.csv", index=False)
    pd.DataFrame([admission]).to_csv(args.output_dir / "admission_summary.csv", index=False)
    print(pd.DataFrame([admission]).to_string(index=False))


if __name__ == "__main__":
    main()
