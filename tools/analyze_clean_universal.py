#!/usr/bin/env python3
"""Evaluate clean-universal U0-U5 and real-only C0-C2 on frozen K=3 MW2."""

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
UNIVERSAL_CONFIGS = {
    "U0": (1, "mean"),
    "U1": (2, "mean"),
    "U2": (3, "mean"),
    "U3": (1, "bottom20"),
    "U4": (2, "bottom20"),
    "U5": (3, "bottom20"),
}
CROSS_LAYER_CONFIGS = ("C0", "C1", "C2")


def config_registry() -> pd.DataFrame:
    rows = [
        {
            "config": config,
            "family": "clean_universal",
            "region": region,
            "aggregation": aggregation,
            "layer": 23,
            "selection_role": (
                "pre_release_temporal_unified_main"
                if config == "U0"
                else "temporal_sensitivity_only"
            ),
            "uses_fake_for_fit_or_calibration": False,
            "uses_dataset_specific_fake_selection": True,
            "patch_spatial_inherited_from_historical_scores": True,
            "development_fake_informed": True,
        }
        for config, (region, aggregation) in UNIVERSAL_CONFIGS.items()
    ]
    rows.extend(
        [
            {
                "config": "HistoricalTuned",
                "family": "historical_dataset_specific_tuned",
                "region": "3/1/2",
                "aggregation": "bottom20/mean/mean",
                "layer": 23,
                "selection_role": "target_fake_selected_supplement_only",
                "uses_fake_for_fit_or_calibration": False,
                "uses_dataset_specific_fake_selection": True,
                "patch_spatial_inherited_from_historical_scores": True,
                "development_fake_informed": True,
            },
            {
                "config": "C0",
                "family": "cross_layer",
                "region": 1,
                "aggregation": "mean",
                "layer": 23,
                "selection_role": "U0_identity_control",
                "uses_fake_for_fit_or_calibration": False,
                "uses_dataset_specific_fake_selection": True,
                "patch_spatial_inherited_from_historical_scores": True,
                "development_fake_informed": True,
            },
            {
                "config": "C1",
                "family": "cross_layer",
                "region": 1,
                "aggregation": "mean",
                "layer": "min(17,23)+real_window_CDF",
                "selection_role": "predeclared_candidate",
                "uses_fake_for_fit_or_calibration": False,
                "uses_dataset_specific_fake_selection": True,
                "patch_spatial_inherited_from_historical_scores": True,
                "development_fake_informed": True,
            },
            {
                "config": "C2",
                "family": "cross_layer",
                "region": 1,
                "aggregation": "mean",
                "layer": "real_motion_median_gate(17,23)",
                "selection_role": "optional_predeclared_candidate",
                "uses_fake_for_fit_or_calibration": False,
                "uses_dataset_specific_fake_selection": True,
                "patch_spatial_inherited_from_historical_scores": True,
                "development_fake_informed": True,
            },
        ]
    )
    return pd.DataFrame(rows)


def calibration_audit(windows: pd.DataFrame) -> pd.DataFrame:
    rows = []
    identity = ["dataset", "subset", "source_model", "filename"]
    for dataset_name, frame in windows.groupby("dataset", sort=False):
        calibration = frame[frame["protocol_split"] == "calibration"]
        evaluation = frame[frame["protocol_split"] == "evaluation"]
        calibration_keys = set(map(tuple, calibration[identity].drop_duplicates().to_numpy()))
        evaluation_keys = set(map(tuple, evaluation[identity].drop_duplicates().to_numpy()))
        rows.append(
            {
                "dataset": dataset_name,
                "calibration_real_videos": calibration[
                    calibration["subset"] == "real"
                ][identity].drop_duplicates().shape[0],
                "calibration_fake_videos": calibration[
                    calibration["subset"] != "real"
                ][identity].drop_duplicates().shape[0],
                "calibration_windows": len(calibration),
                "evaluation_videos": evaluation[identity].drop_duplicates().shape[0],
                "evaluation_windows": len(evaluation),
                "calibration_evaluation_video_overlap": len(
                    calibration_keys & evaluation_keys
                ),
            }
        )
    return pd.DataFrame(rows)


def c2_gate_usage(windows: pd.DataFrame) -> pd.DataFrame:
    return (
        windows.groupby(
            ["dataset", "protocol_split", "subset"],
            sort=True,
            observed=True,
        )
        .agg(
            windows=("window_id", "size"),
            c1_windows=("C2_uses_C1", "sum"),
            c1_fraction=("C2_uses_C1", "mean"),
        )
        .reset_index()
    )


def canonicalize_temporal_percentiles(
    clean: pd.DataFrame, params_dir: Path
) -> pd.DataFrame:
    """Rebuild percentiles from raw scores and the final real-only CDF files."""
    out = clean.copy()
    for dataset, index in out.groupby("dataset", sort=False).groups.items():
        positions = np.asarray(list(index), dtype=int)
        for region in (1, 2, 3):
            for aggregation in ("mean", "bottom20"):
                params = np.load(
                    params_dir
                    / f"{dataset}_region{region}_{aggregation}.npz",
                    allow_pickle=True,
                )
                reference = np.sort(params["calib_patch_temp_scores"])
                raw_column = f"region{region}_{aggregation}_raw"
                output_column = f"region{region}_{aggregation}_temporal"
                out.loc[positions, output_column] = (
                    np.searchsorted(
                        reference, out.loc[positions, raw_column].to_numpy(), side="right"
                    )
                    / len(reference)
                )
        params = np.load(
            params_dir / f"{dataset}_layer17_region1_mean.npz", allow_pickle=True
        )
        reference = np.sort(params["calib_patch_temp_scores"])
        out.loc[positions, "layer17_temporal"] = (
            np.searchsorted(
                reference, out.loc[positions, "layer17_raw"].to_numpy(), side="right"
            )
            / len(reference)
        )
    return out


def read_score_shards(root: Path, num_shards: int) -> pd.DataFrame:
    frames = []
    for dataset in ("comgenvid", "videofeedback", "genvideo"):
        for shard in range(num_shards):
            path = root / f"{dataset}_clean_shard{shard}.csv"
            if not path.exists():
                raise FileNotFoundError(path)
            frames.append(pd.read_csv(path, float_precision="round_trip"))
    scores = pd.concat(frames, ignore_index=True)
    if scores.duplicated(WINDOW_KEYS).any():
        raise ValueError("duplicate clean-universal window keys")
    return scores


def merge_windows(
    baseline: pd.DataFrame,
    clean: pd.DataFrame,
    motion: pd.DataFrame,
) -> pd.DataFrame:
    score_columns = [
        *WINDOW_KEYS,
        "frame_indices",
        "layer17_raw",
        "layer17_temporal",
    ]
    for region in (1, 2, 3):
        for aggregation in ("mean", "bottom20"):
            score_columns.extend(
                [
                    f"region{region}_{aggregation}_raw",
                    f"region{region}_{aggregation}_temporal",
                ]
            )
    merged = baseline.merge(
        clean[score_columns],
        on=WINDOW_KEYS,
        how="inner",
        validate="one_to_one",
        suffixes=("", "_clean"),
    )
    if len(merged) != len(baseline) or len(merged) != len(clean):
        raise ValueError("clean-universal join changed frozen window count")
    if not (merged["frame_indices"] == merged["frame_indices_clean"]).all():
        raise ValueError("clean-universal frame indices differ from frozen K=3")
    merged = merged.drop(columns="frame_indices_clean")
    motion_columns = [*WINDOW_KEYS, "mean_motion_magnitude"]
    merged = merged.merge(
        motion[motion_columns], on=WINDOW_KEYS, how="left", validate="one_to_one"
    )
    if merged["mean_motion_magnitude"].isna().any():
        raise ValueError("missing window motion diagnostics")
    # Every first-level temporal score is an empirical percentile of exactly
    # 200 real windows. Restore that discrete grid after CSV serialization so
    # 1e-16 text roundoff cannot change later video-level CDF tie handling.
    temporal_columns = ["layer17_temporal"] + [
        f"region{region}_{aggregation}_temporal"
        for region in (1, 2, 3)
        for aggregation in ("mean", "bottom20")
    ]
    for column in temporal_columns:
        merged[column] = np.rint(merged[column].to_numpy() * 200.0) / 200.0
    merged["_baseline_temporal_200"] = (
        np.rint(merged["patch_d2"].to_numpy() * 200.0) / 200.0
    )
    oracle_identity_columns = {
        "comgenvid": "region3_bottom20_temporal",
        "videofeedback": "region1_mean_temporal",
        "genvideo": "region2_mean_temporal",
    }
    for dataset, column in oracle_identity_columns.items():
        mask = merged["dataset"] == dataset
        merged.loc[mask, column] = merged.loc[mask, "_baseline_temporal_200"]
    for config, (region, aggregation) in UNIVERSAL_CONFIGS.items():
        temporal = f"region{region}_{aggregation}_temporal"
        merged[f"{config}_temporal"] = merged[temporal]
        merged[f"{config}_L_k"] = merged["L_k"] + 0.9 * (
            merged[temporal] - merged["_baseline_temporal_200"]
        )
    return add_cross_layer_temporal_scores(merged)


def add_cross_layer_temporal_scores(windows: pd.DataFrame) -> pd.DataFrame:
    frames = []
    for _, dataset in windows.groupby("dataset", sort=False):
        target = dataset.copy()
        target["C0_temporal"] = target["region1_mean_temporal"]
        target["C1_raw"] = np.minimum(
            target["layer17_temporal"].to_numpy(),
            target["C0_temporal"].to_numpy(),
        )
        calibration = target[
            (target["protocol_split"] == "calibration")
            & (target["subset"] == "real")
        ]
        if len(calibration) < 2:
            raise ValueError("C1/C2 require real calibration windows")
        target["C1_temporal"] = empirical_cdf(
            target["C1_raw"].to_numpy(), calibration["C1_raw"].to_numpy()
        )
        target["motion_real_cdf"] = empirical_cdf(
            target["mean_motion_magnitude"].to_numpy(),
            calibration["mean_motion_magnitude"].to_numpy(),
        )
        target["C2_temporal"] = np.where(
            target["motion_real_cdf"] < 0.5,
            target["C1_temporal"],
            target["C0_temporal"],
        )
        target["C2_uses_C1"] = target["motion_real_cdf"] < 0.5
        for config in CROSS_LAYER_CONFIGS:
            baseline_temporal = target.get(
                "_baseline_temporal_200",
                np.rint(target.get("patch_d2", 0.0) * 200.0) / 200.0,
            )
            baseline_local = target.get(
                "L_k", 0.1 * target["patch_spatial"] + 0.9 * baseline_temporal
            )
            target[f"{config}_L_k"] = baseline_local + 0.9 * (
                target[f"{config}_temporal"] - baseline_temporal
            )
        frames.append(target)
    return pd.concat(frames, ignore_index=True)


def aggregate_videos(windows: pd.DataFrame) -> pd.DataFrame:
    aggregations: dict[str, tuple[str, str]] = {
        "duration_seconds": ("duration_seconds", "first"),
        "effective_k": ("effective_k", "first"),
        "windows": ("window_id", "size"),
        "G_mean_raw": ("G_k", "mean"),
        "mean_motion_magnitude": ("mean_motion_magnitude", "mean"),
    }
    for config in (*UNIVERSAL_CONFIGS, *CROSS_LAYER_CONFIGS):
        aggregations[f"{config}_L_raw"] = (f"{config}_L_k", "mean")
    result = (
        windows.groupby(KEY_COLUMNS, sort=False, observed=True)
        .agg(**aggregations)
        .reset_index()
    )
    if not (result["effective_k"] == result["windows"]).all():
        raise ValueError("effective_k does not match clean-universal window count")
    return result


def target_k_reference(
    windows: pd.DataFrame, target_k: int, local_column: str
) -> pd.DataFrame:
    calibration = windows[
        (windows["protocol_split"] == "calibration")
        & (windows["subset"] == "real")
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
    result = pd.DataFrame(rows)
    if result.empty:
        raise ValueError(f"no real calibration reference for effective_k={target_k}")
    return result


def calibrate_videos(per_video: pd.DataFrame, windows: pd.DataFrame) -> pd.DataFrame:
    configs = (*UNIVERSAL_CONFIGS, *CROSS_LAYER_CONFIGS)
    frames = []
    for dataset_name, dataset_videos in per_video.groupby("dataset", sort=False):
        evaluation = dataset_videos[
            dataset_videos["protocol_split"] == "evaluation"
        ].copy()
        dataset_windows = windows[windows["dataset"] == dataset_name]
        groups = []
        for effective_k, target in evaluation.groupby("effective_k", sort=True):
            target = target.copy()
            global_reference = target_k_reference(
                dataset_windows, int(effective_k), "U0_L_k"
            )
            target["G"] = empirical_cdf(
                target["G_mean_raw"].to_numpy(),
                global_reference["G_mean_raw"].to_numpy(),
            )
            for config in configs:
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


def attach_historical_tuned(
    evaluation: pd.DataFrame, frozen_path: Path
) -> pd.DataFrame:
    frozen = pd.read_csv(frozen_path, float_precision="round_trip")[
        [*KEY_COLUMNS, "G_mean", "MW2"]
    ].copy()
    frozen["_order"] = np.arange(len(frozen), dtype=np.int64)
    frozen = frozen.rename(
        columns={"G_mean": "frozen_G", "MW2": "HistoricalTuned"}
    )
    merged = evaluation.merge(frozen, on=KEY_COLUMNS, how="inner", validate="one_to_one")
    if len(merged) != len(frozen):
        raise ValueError("historical tuned join changed fixed evaluation row count")
    global_error = float((merged["G"] - merged["frozen_G"]).abs().max())
    if global_error > 1e-12:
        raise ValueError(f"clean universal changed frozen Global: {global_error}")
    c0_error = float((merged["C0"] - merged["U0"]).abs().max())
    if c0_error > 1e-12:
        raise ValueError(f"C0 does not reproduce U0: {c0_error}")
    return merged.sort_values("_order").drop(columns="_order").reset_index(drop=True)


def generator_deltas(metrics: pd.DataFrame, base: str) -> pd.DataFrame:
    pivot = metrics.pivot(
        index=["dataset", "generator"], columns="config", values=["auc", "ap"]
    )
    rows = []
    for dataset, generator in pivot.index:
        for config in pivot.columns.levels[1]:
            rows.append(
                {
                    "dataset": dataset,
                    "generator": generator,
                    "config": config,
                    "base_config": base,
                    "delta_auc": float(
                        pivot.loc[(dataset, generator), ("auc", config)]
                        - pivot.loc[(dataset, generator), ("auc", base)]
                    ),
                    "delta_ap": float(
                        pivot.loc[(dataset, generator), ("ap", config)]
                        - pivot.loc[(dataset, generator), ("ap", base)]
                    ),
                }
            )
    return pd.DataFrame(rows)


def real_motion_thresholds(windows: pd.DataFrame) -> pd.DataFrame:
    calibration = (
        windows[
            (windows["protocol_split"] == "calibration")
            & (windows["subset"] == "real")
        ]
        .groupby(KEY_COLUMNS, sort=False, observed=True)["mean_motion_magnitude"]
        .mean()
        .reset_index()
    )
    window_calibration = windows[
        (windows["protocol_split"] == "calibration")
        & (windows["subset"] == "real")
    ]
    rows = []
    for dataset, frame in calibration.groupby("dataset", sort=False):
        window_values = window_calibration[
            window_calibration["dataset"] == dataset
        ]["mean_motion_magnitude"]
        video_quantiles = np.quantile(frame["mean_motion_magnitude"], [1 / 3, 2 / 3])
        rows.append(
            {
                "dataset": dataset,
                "real_calibration_videos": len(frame),
                "real_calibration_windows": len(window_values),
                "window_motion_median_for_C2": float(np.median(window_values)),
                "video_motion_q33_for_reporting": float(video_quantiles[0]),
                "video_motion_q67_for_reporting": float(video_quantiles[1]),
            }
        )
    return pd.DataFrame(rows)


def motion_metrics(evaluation: pd.DataFrame, windows: pd.DataFrame) -> pd.DataFrame:
    threshold_frame = real_motion_thresholds(windows).set_index("dataset")
    thresholds = {
        dataset: (
            row.video_motion_q33_for_reporting,
            row.video_motion_q67_for_reporting,
        )
        for dataset, row in threshold_frame.iterrows()
    }
    merged = evaluation.copy()
    groups = []
    for row in merged.itertuples(index=False):
        low, high = thresholds[row.dataset]
        if row.mean_motion_magnitude <= low:
            groups.append("low")
        elif row.mean_motion_magnitude <= high:
            groups.append("mid")
        else:
            groups.append("high")
    merged["motion_group"] = groups
    rows = []
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
            score_columns=CROSS_LAYER_CONFIGS,
            config_names={name: name for name in CROSS_LAYER_CONFIGS},
        )
        metrics.insert(0, "fake_motion_group", group)
        rows.append(metrics)
    return pd.concat(rows, ignore_index=True)


def admission_summary(
    dataset_metrics: pd.DataFrame,
    generator_metrics: pd.DataFrame,
    bootstrap: pd.DataFrame,
    motion: pd.DataFrame,
) -> pd.DataFrame:
    dataset_ap = dataset_metrics[dataset_metrics["dataset"] != "Macro-3"].pivot(
        index="dataset", columns="config", values="ap"
    )
    macro_ap = dataset_metrics[dataset_metrics["dataset"] == "Macro-3"].set_index(
        "config"
    )["ap"]
    generator_ap = generator_metrics.pivot(
        index=["dataset", "generator"], columns="config", values="ap"
    )
    high = motion[
        (motion["fake_motion_group"] == "high")
        & (motion["dataset"] == "Macro-3")
    ].set_index("config")["ap"]
    rows = []
    for config in ("C1", "C2"):
        macro_delta = float(macro_ap[config] - macro_ap["C0"])
        worst = float((dataset_ap[config] - dataset_ap["C0"]).min())
        non_decline = int((generator_ap[config] >= generator_ap["C0"]).sum())
        high_delta = float(high[config] - high["C0"])
        ci = bootstrap[
            (bootstrap["dataset"] == "Macro-3")
            & (bootstrap["new_config"] == config)
            & (bootstrap["metric"] == "ap")
        ].iloc[0]
        passes = {
            "passes_macro_ap": macro_delta >= 0.003,
            "passes_worst_dataset": worst >= -0.003,
            "passes_generator_count": non_decline >= 12,
            "passes_high_motion": high_delta >= -0.005,
            "passes_bootstrap": float(ci.ci95_low) > 0.0,
        }
        rows.append(
            {
                "config": config,
                "base_config": "C0",
                "delta_macro_ap": macro_delta,
                "worst_dataset_ap_delta": worst,
                "generator_non_decline_count": non_decline,
                "generator_total": len(generator_ap),
                "high_motion_macro_ap_delta": high_delta,
                "macro_ap_ci95_low": float(ci.ci95_low),
                "macro_ap_ci95_high": float(ci.ci95_high),
                **passes,
                "admitted": all(passes.values()),
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--score-root", type=Path, required=True)
    parser.add_argument("--num-shards", type=int, default=2)
    parser.add_argument(
        "--params-dir",
        type=Path,
        default=REPO_ROOT / "results/clean_universal/params",
    )
    parser.add_argument(
        "--baseline-windows",
        type=Path,
        default=REPO_ROOT
        / "results/multi_window_joint_typicality/K3_uniform_window_scores.csv",
    )
    parser.add_argument(
        "--frozen-evaluation",
        type=Path,
        default=REPO_ROOT
        / "results/multi_window_joint_typicality/K3_uniform_evaluation_scores.csv",
    )
    parser.add_argument(
        "--motion-windows",
        type=Path,
        default=REPO_ROOT / "results/local_residual/per_window_scores.csv",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPO_ROOT / "results/clean_universal",
    )
    parser.add_argument("--bootstrap-iterations", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    baseline = pd.read_csv(args.baseline_windows, float_precision="round_trip")
    clean = canonicalize_temporal_percentiles(
        read_score_shards(args.score_root, args.num_shards), args.params_dir
    )
    motion_windows = pd.read_csv(args.motion_windows, float_precision="round_trip")
    windows = merge_windows(baseline, clean, motion_windows)
    evaluation = attach_historical_tuned(
        calibrate_videos(aggregate_videos(windows), windows), args.frozen_evaluation
    )

    universal_names = (*UNIVERSAL_CONFIGS, "HistoricalTuned")
    universal_dataset, universal_generator = metric_tables(
        evaluation,
        args.seed,
        score_columns=universal_names,
        config_names={name: name for name in universal_names},
    )
    cross_dataset, cross_generator = metric_tables(
        evaluation,
        args.seed,
        score_columns=CROSS_LAYER_CONFIGS,
        config_names={name: name for name in CROSS_LAYER_CONFIGS},
    )
    universal_comparisons = tuple(
        (config, "U0", f"{config}-U0")
        for config in (*tuple(UNIVERSAL_CONFIGS)[1:], "HistoricalTuned")
    )
    cross_comparisons = tuple(
        (config, "C0", f"{config}-C0") for config in ("C1", "C2")
    )
    universal_bootstrap = pd.concat(
        [
            paired_bootstrap(
                evaluation,
                args.seed,
                args.bootstrap_iterations,
                universal_comparisons,
            ),
            macro_cluster_bootstrap(
                evaluation,
                [item[0] for item in universal_comparisons],
                args.seed,
                args.bootstrap_iterations,
                base_config="U0",
            ),
        ],
        ignore_index=True,
    )
    cross_bootstrap = pd.concat(
        [
            paired_bootstrap(
                evaluation, args.seed, args.bootstrap_iterations, cross_comparisons
            ),
            macro_cluster_bootstrap(
                evaluation,
                ["C1", "C2"],
                args.seed,
                args.bootstrap_iterations,
                base_config="C0",
            ),
        ],
        ignore_index=True,
    )
    motion = motion_metrics(evaluation, windows)
    admission = admission_summary(
        cross_dataset, cross_generator, cross_bootstrap, motion
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    windows.to_csv(args.output_dir / "per_window_scores.csv", index=False)
    evaluation.to_csv(args.output_dir / "per_video_scores.csv", index=False)
    universal_dataset.to_csv(args.output_dir / "universal_dataset_metrics.csv", index=False)
    universal_generator.to_csv(
        args.output_dir / "universal_generator_metrics.csv", index=False
    )
    generator_deltas(universal_generator, "U0").to_csv(
        args.output_dir / "universal_generator_deltas.csv", index=False
    )
    universal_bootstrap.to_csv(
        args.output_dir / "universal_bootstrap_deltas.csv", index=False
    )
    cross_dataset.to_csv(args.output_dir / "cross_layer_dataset_metrics.csv", index=False)
    cross_generator.to_csv(
        args.output_dir / "cross_layer_generator_metrics.csv", index=False
    )
    generator_deltas(cross_generator, "C0").to_csv(
        args.output_dir / "cross_layer_generator_deltas.csv", index=False
    )
    cross_bootstrap.to_csv(
        args.output_dir / "cross_layer_bootstrap_deltas.csv", index=False
    )
    motion.to_csv(args.output_dir / "cross_layer_motion_metrics.csv", index=False)
    admission.to_csv(args.output_dir / "cross_layer_admission.csv", index=False)
    config_registry().to_csv(args.output_dir / "config_registry.csv", index=False)
    calibration_audit(windows).to_csv(
        args.output_dir / "calibration_audit.csv", index=False
    )
    real_motion_thresholds(windows).to_csv(
        args.output_dir / "real_motion_thresholds.csv", index=False
    )
    c2_gate_usage(windows).to_csv(
        args.output_dir / "c2_gate_usage.csv", index=False
    )
    print("Universal U0-U5 + HistoricalTuned")
    print(universal_dataset.to_string(index=False))
    print("\nCross-layer C0-C2")
    print(cross_dataset.to_string(index=False))
    print("\nAdmission")
    print(admission.to_string(index=False))


if __name__ == "__main__":
    main()
