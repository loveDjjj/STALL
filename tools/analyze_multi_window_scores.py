#!/usr/bin/env python3
"""Aggregate K=3 window scores, recalibrate per video, and evaluate MW0-MW4."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from alpha_stalled.calibration import empirical_cdf
from alpha_stalled.historical_window_analysis import (
    KEY_COLUMNS,
    bottom2_mean,
    calibration_references,
)
from alpha_stalled.metrics import (
    macro_cluster_bootstrap,
    metric_tables,
    paired_bootstrap,
    pairwise_frames,
)


SCORE_KEYS = ["subset", "source_model", "filename"]
BASELINE_KEYS = ["dataset", *SCORE_KEYS]
CONFIG_NAMES = {
    "MW0": "Frozen single-window baseline",
    "MW0R": "Single-window with video-level recalibration",
    "MW1": "K=3 mean final score",
    "MW2": "K=3 recalibrated branch means",
    "MW3": "K=3 global mean + local bottom-2",
    "MW4": "K=3 global mean + local hybrid",
}


def config_names_for_sampling(sampling: str) -> dict[str, str]:
    label = {
        "K3_uniform": "K=3",
        "K5_uniform": "K=5",
        "all_nonoverlap": "all non-overlap",
    }.get(sampling, sampling)
    return {
        config: name.replace("K=3", label)
        for config, name in CONFIG_NAMES.items()
    }


def aggregate_window_scores(window_scores: pd.DataFrame) -> pd.DataFrame:
    grouped = window_scores.groupby(KEY_COLUMNS, sort=False, observed=True)
    aggregated = grouped.agg(
        duration_seconds=("duration_seconds", "first"),
        effective_k=("effective_k", "first"),
        unique_frame_count=("unique_frame_count", "first"),
        windows=("window_id", "size"),
        G_mean_raw=("G_k", "mean"),
        L_mean_raw=("L_k", "mean"),
        L_bottom2_raw=("L_k", bottom2_mean),
        S_mean_raw=("S_k", "mean"),
        G_std=("G_k", lambda values: float(np.std(values, ddof=0))),
        L_std=("L_k", lambda values: float(np.std(values, ddof=0))),
        S_std=("S_k", lambda values: float(np.std(values, ddof=0))),
        low_local_windows=("L_k", lambda values: int((values < 0.2).sum())),
    ).reset_index()
    if not (aggregated["effective_k"] == aggregated["windows"]).all():
        raise ValueError("window count does not match effective_k")
    aggregated["L_hybrid_raw"] = 0.5 * (
        aggregated["L_mean_raw"] + aggregated["L_bottom2_raw"]
    )
    return aggregated


def add_recalibrated_scores(
    per_video: pd.DataFrame,
    window_scores: pd.DataFrame,
    calibration_mode: str = "effective_k",
) -> pd.DataFrame:
    if calibration_mode not in {"effective_k", "unconditional"}:
        raise ValueError(f"unknown calibration mode: {calibration_mode}")
    frames: list[pd.DataFrame] = []
    for dataset, frame in per_video.groupby("dataset", sort=False):
        calibration = frame[frame["protocol_split"] == "calibration"]
        evaluation = frame[frame["protocol_split"] == "evaluation"].copy()
        if len(calibration) != 200:
            raise ValueError(f"{dataset}: expected 200 calibration videos, found {len(calibration)}")
        evaluation["MW1"] = evaluation["S_mean_raw"]
        recalibrated_groups: list[pd.DataFrame] = []
        dataset_windows = window_scores[window_scores["dataset"] == dataset]
        if calibration_mode == "effective_k":
            target_groups = evaluation.groupby("effective_k", sort=True)
        else:
            target_groups = [("all", evaluation)]
        for effective_k, target in target_groups:
            reference = (
                calibration_references(dataset_windows, int(effective_k))
                if calibration_mode == "effective_k"
                else calibration
            )
            target = target.copy()
            target["calibration_reference_n"] = len(reference)
            target["calibration_mode"] = calibration_mode
            target["G_mean"] = empirical_cdf(
                target["G_mean_raw"].to_numpy(), reference["G_mean_raw"].to_numpy()
            )
            for raw, calibrated in (
                ("L_mean_raw", "L_mean"),
                ("L_bottom2_raw", "L_bottom2"),
                ("L_hybrid_raw", "L_hybrid"),
            ):
                target[calibrated] = empirical_cdf(
                    target[raw].to_numpy(), reference[raw].to_numpy()
                )
            recalibrated_groups.append(target)
        evaluation = pd.concat(recalibrated_groups, ignore_index=True)
        evaluation["MW2"] = 0.6 * evaluation["G_mean"] + 0.4 * evaluation["L_mean"]
        evaluation["MW3"] = 0.6 * evaluation["G_mean"] + 0.4 * evaluation["L_bottom2"]
        evaluation["MW4"] = 0.6 * evaluation["G_mean"] + 0.4 * evaluation["L_hybrid"]
        frames.append(evaluation)
    return pd.concat(frames, ignore_index=True)


def add_frozen_baseline(evaluation: pd.DataFrame, baseline_path: Path) -> pd.DataFrame:
    baseline = pd.read_csv(baseline_path, float_precision="round_trip")[
        BASELINE_KEYS + ["B2", "P0", "final_selected"]
    ].copy()
    baseline["_protocol_order"] = np.arange(len(baseline), dtype=np.int64)
    baseline = baseline.rename(columns={"final_selected": "MW0"})
    merged = evaluation.merge(baseline, on=BASELINE_KEYS, how="inner", validate="one_to_one")
    if len(merged) != len(evaluation):
        raise ValueError(f"baseline join changed evaluation rows: {len(evaluation)} -> {len(merged)}")
    return merged.sort_values("_protocol_order").drop(columns="_protocol_order").reset_index(drop=True)


def add_recalibrated_k1_control(
    evaluation: pd.DataFrame,
    k1_calibration_scores: pd.DataFrame,
) -> pd.DataFrame:
    out = evaluation.copy()
    controls: list[pd.DataFrame] = []
    for dataset, frame in out.groupby("dataset", sort=False):
        calibration = k1_calibration_scores[
            (k1_calibration_scores["dataset"] == dataset)
            & (k1_calibration_scores["protocol_split"] == "calibration")
        ]
        if len(calibration) != 200:
            raise ValueError(f"{dataset}: K1 control requires 200 calibration rows")
        current = frame.copy()
        current["K1_G_recal"] = empirical_cdf(
            current["B2"].to_numpy(), calibration["G_k"].to_numpy()
        )
        current["K1_L_recal"] = empirical_cdf(
            current["P0"].to_numpy(), calibration["L_k"].to_numpy()
        )
        current["MW0R"] = 0.6 * current["K1_G_recal"] + 0.4 * current["K1_L_recal"]
        controls.append(current)
    return pd.concat(controls, ignore_index=True)


def _auc_ap(frame: pd.DataFrame, score: str) -> tuple[float, float]:
    labels = frame["subset"].eq("real").astype(np.uint8).to_numpy()
    values = frame[score].to_numpy(dtype=np.float64)
    return float(roc_auc_score(labels, values)), float(average_precision_score(labels, values))


def generator_deltas(generator_metrics: pd.DataFrame) -> pd.DataFrame:
    pivot = generator_metrics.pivot(index=["dataset", "generator"], columns="config", values=["auc", "ap"])
    rows: list[dict] = []
    for dataset, generator in pivot.index:
        for candidate in ("MW0R", "MW1", "MW2", "MW3", "MW4"):
            if ("ap", candidate) not in pivot.columns:
                continue
            rows.append(
                {
                    "dataset": dataset,
                    "generator": generator,
                    "config": candidate,
                    "delta_auc_vs_MW0": float(pivot.loc[(dataset, generator), ("auc", candidate)] - pivot.loc[(dataset, generator), ("auc", "MW0")]),
                    "delta_ap_vs_MW0": float(pivot.loc[(dataset, generator), ("ap", candidate)] - pivot.loc[(dataset, generator), ("ap", "MW0")]),
                }
            )
    return pd.DataFrame(rows)


def stratified_metric_tables(
    per_video: pd.DataFrame,
    group_column: str,
    seed: int,
    config_names: dict[str, str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    dataset_rows: list[pd.DataFrame] = []
    generator_rows: list[pd.DataFrame] = []
    for group, frame in per_video.groupby(group_column, observed=True, sort=True):
        valid = [
            dataset_frame
            for _, dataset_frame in frame.groupby("dataset", sort=False)
            if dataset_frame["subset"].nunique() == 2
        ]
        if not valid:
            continue
        frame = pd.concat(valid, ignore_index=True)
        try:
            datasets, generators = metric_tables(
                frame,
                seed,
                score_columns=tuple(config_names),
                config_names=config_names,
            )
        except (ValueError, ZeroDivisionError):
            continue
        datasets.insert(0, group_column, group)
        generators.insert(0, group_column, group)
        dataset_rows.append(datasets)
        generator_rows.append(generators)
    return (
        pd.concat(dataset_rows, ignore_index=True) if dataset_rows else pd.DataFrame(),
        pd.concat(generator_rows, ignore_index=True) if generator_rows else pd.DataFrame(),
    )


def read_shards(score_root: Path, sampling: str, num_shards: int) -> pd.DataFrame:
    if score_root.is_file():
        scores = pd.read_csv(score_root, float_precision="round_trip")
    else:
        shard_paths = [
            score_root / f"{dataset}_{sampling}_shard{shard}.csv"
            for dataset in ("comgenvid", "videofeedback", "genvideo")
            for shard in range(num_shards)
        ]
        existing_shards = [path for path in shard_paths if path.exists()]
        if existing_shards and len(existing_shards) != len(shard_paths):
            missing = next(path for path in shard_paths if not path.exists())
            raise FileNotFoundError(f"incomplete score shards; first missing file: {missing}")
        if existing_shards:
            scores = pd.concat(
                [pd.read_csv(path, float_precision="round_trip") for path in shard_paths],
                ignore_index=True,
            )
        else:
            merged_candidates = (
                score_root / f"{sampling}_window_scores.csv",
                score_root.parent / f"{sampling}_window_scores.csv",
            )
            merged = next((path for path in merged_candidates if path.exists()), None)
            if merged is None:
                raise FileNotFoundError(
                    "no complete score shards or merged window-score artifact found; "
                    f"checked {shard_paths[0]} and {merged_candidates}"
                )
            scores = pd.read_csv(merged, float_precision="round_trip")
    if scores.duplicated(KEY_COLUMNS + ["window_id"]).any():
        raise ValueError("duplicate window-score keys across shards")
    return scores


def write_report(
    dataset_metrics: pd.DataFrame,
    generator_delta: pd.DataFrame,
    macro_bootstrap: pd.DataFrame,
    per_video: pd.DataFrame,
    output: Path,
    sampling: str,
) -> None:
    pivot = dataset_metrics.pivot(index="config", columns="dataset", values=["auc", "ap"])
    lines = [
        f"# {sampling} multi-window analysis",
        "",
        "MW0 is the exact frozen single-window baseline. MW0R applies the same video-level branch recalibration to K=1 and isolates recalibration from additional temporal coverage.",
        "MW1 is the literal mean of window-level final scores. MW2-MW4 recalibrate each aggregated branch on the disjoint 200-real multi-window calibration videos before fixed 0.6/0.4 fusion.",
        f"Calibration mode: `{per_video.calibration_mode.iloc[0]}`. Effective-K matching is used for fixed-K experiments; all-non-overlap uses one naturally aggregated score per real video and an unconditional, video-weighted CDF.",
        "",
        "## Dataset metrics",
        "",
        "| config | ComGenVid AUC/AP | VideoFeedback AUC/AP | GenVideo AUC/AP | Macro-3 AUC/AP |",
        "|---|---:|---:|---:|---:|",
    ]
    for config in CONFIG_NAMES:
        cells = []
        for dataset in ("comgenvid", "videofeedback", "genvideo", "Macro-3"):
            cells.append(f"{pivot.loc[config, ('auc', dataset)]:.4f}/{pivot.loc[config, ('ap', dataset)]:.4f}")
        lines.append(f"| {config} | " + " | ".join(cells) + " |")
    lines.extend(["", "## Admission diagnostics", ""])
    macro = dataset_metrics[dataset_metrics["dataset"] == "Macro-3"].set_index("config")
    dataset_only = dataset_metrics[dataset_metrics["dataset"] != "Macro-3"]
    for config in ("MW1", "MW2", "MW3", "MW4"):
        delta_ap = float(macro.loc[config, "ap"] - macro.loc["MW0", "ap"])
        joined = dataset_only[dataset_only["config"].isin(["MW0", config])].pivot(index="dataset", columns="config", values="ap")
        worst = float((joined[config] - joined["MW0"]).min())
        wins = int((generator_delta[generator_delta["config"] == config]["delta_ap_vs_MW0"] > 0).sum())
        ci = macro_bootstrap[(macro_bootstrap["new_config"] == config) & (macro_bootstrap["metric"] == "ap")].iloc[0]
        lines.append(
            f"- {config}: Macro AP {delta_ap:+.4f}, worst dataset {worst:+.4f}, generator wins {wins}/20, bootstrap 95% CI [{ci.ci95_low:+.4f}, {ci.ci95_high:+.4f}]."
        )
    long_subset = per_video[per_video["effective_k"] >= 3]
    lines.extend(
        [
            "",
            "## Coverage and score variation",
            "",
            f"- Evaluation videos: {len(per_video)}; effective_K>=3: {len(long_subset)}.",
            f"- Mean window-score std: G={per_video.G_std.mean():.4f}, L={per_video.L_std.mean():.4f}, S={per_video.S_std.mean():.4f}.",
            f"- Unique decoded frames: {int(per_video.unique_frame_count.sum())}; windows: {int(per_video.windows.sum())}.",
        ]
    )
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--score-root",
        type=Path,
        default=REPO_ROOT / "results/multi_window_joint_typicality/window_scores",
    )
    parser.add_argument(
        "--k1-calibration",
        type=Path,
        default=REPO_ROOT / "results/multi_window_joint_typicality/K1_calibration_window_scores.csv",
    )
    parser.add_argument(
        "--baseline",
        type=Path,
        default=REPO_ROOT / "results/multi_order_baselines/fusion_selected_per_video_scores.csv",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=REPO_ROOT / "results/multi_window_joint_typicality/multi_window_manifest.csv",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPO_ROOT / "results/multi_window_joint_typicality",
    )
    parser.add_argument("--num-shards", type=int, default=2)
    parser.add_argument(
        "--sampling",
        choices=("K3_uniform", "K5_uniform", "all_nonoverlap"),
        default="K3_uniform",
    )
    parser.add_argument("--output-prefix")
    parser.add_argument(
        "--calibration-mode",
        choices=("auto", "effective_k", "unconditional"),
        default="auto",
    )
    parser.add_argument("--bootstrap-iterations", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    prefix = args.output_prefix or {
        "K3_uniform": "K3",
        "K5_uniform": "K5",
        "all_nonoverlap": "all",
    }[args.sampling]
    config_names = config_names_for_sampling(args.sampling)
    calibration_mode = args.calibration_mode
    if calibration_mode == "auto":
        calibration_mode = (
            "unconditional" if args.sampling == "all_nonoverlap" else "effective_k"
        )
    window_scores = read_shards(args.score_root, args.sampling, args.num_shards)
    per_video_all = aggregate_window_scores(window_scores)
    evaluation = add_recalibrated_scores(
        per_video_all, window_scores, calibration_mode=calibration_mode
    )
    evaluation = add_frozen_baseline(evaluation, args.baseline)
    evaluation = add_recalibrated_k1_control(
        evaluation,
        pd.read_csv(args.k1_calibration, float_precision="round_trip"),
    )
    manifest = pd.read_csv(args.manifest)[KEY_COLUMNS + ["duration_bin"]]
    evaluation = evaluation.drop(columns=["duration_seconds"], errors="ignore").merge(
        manifest[manifest["protocol_split"] == "evaluation"],
        on=KEY_COLUMNS,
        how="left",
        validate="one_to_one",
    )

    configs = tuple(CONFIG_NAMES)
    dataset_metrics, generator_metrics = metric_tables(
        evaluation,
        args.seed,
        score_columns=configs,
        config_names=config_names,
    )
    comparisons = tuple((config, "MW0", f"{config}-MW0") for config in configs if config != "MW0")
    dataset_bootstrap = paired_bootstrap(
        evaluation,
        args.seed,
        args.bootstrap_iterations,
        comparisons=comparisons,
    )
    macro_bootstrap = macro_cluster_bootstrap(
        evaluation,
        [config for config in configs if config != "MW0"],
        args.seed,
        args.bootstrap_iterations,
    )
    deltas = generator_deltas(generator_metrics)
    length_metrics, length_generator_metrics = stratified_metric_tables(
        evaluation, "duration_bin", args.seed, config_names
    )
    effective_metrics, effective_generator_metrics = stratified_metric_tables(
        evaluation, "effective_k", args.seed, config_names
    )
    long_video_metrics, long_video_generator_metrics = metric_tables(
        evaluation[evaluation["effective_k"] >= 3],
        args.seed,
        score_columns=configs,
        config_names=config_names,
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    window_scores.to_csv(args.output_dir / f"{args.sampling}_window_scores.csv", index=False)
    per_video_all.to_csv(args.output_dir / f"{args.sampling}_all_video_aggregates.csv", index=False)
    evaluation.to_csv(args.output_dir / f"{args.sampling}_evaluation_scores.csv", index=False)
    dataset_metrics.to_csv(args.output_dir / f"{prefix}_dataset_metrics.csv", index=False)
    generator_metrics.to_csv(args.output_dir / f"{prefix}_generator_metrics.csv", index=False)
    deltas.to_csv(args.output_dir / f"{prefix}_generator_deltas.csv", index=False)
    dataset_bootstrap.to_csv(args.output_dir / f"{prefix}_dataset_bootstrap.csv", index=False)
    macro_bootstrap.to_csv(args.output_dir / f"{prefix}_macro_bootstrap.csv", index=False)
    length_metrics.to_csv(args.output_dir / f"{prefix}_duration_group_metrics.csv", index=False)
    length_generator_metrics.to_csv(
        args.output_dir / f"{prefix}_duration_group_generator_metrics.csv", index=False
    )
    effective_metrics.to_csv(args.output_dir / f"{prefix}_effective_k_metrics.csv", index=False)
    effective_generator_metrics.to_csv(
        args.output_dir / f"{prefix}_effective_k_generator_metrics.csv", index=False
    )
    long_video_metrics.to_csv(args.output_dir / f"{prefix}_long_video_metrics.csv", index=False)
    long_video_generator_metrics.to_csv(
        args.output_dir / f"{prefix}_long_video_generator_metrics.csv", index=False
    )
    write_report(
        dataset_metrics,
        deltas,
        macro_bootstrap,
        evaluation,
        args.output_dir / f"{prefix}_analysis.md",
        args.sampling,
    )
    print(f"window_rows={len(window_scores)} evaluation_videos={len(evaluation)}")
    print(f"saved -> {args.output_dir}")


if __name__ == "__main__":
    main()
