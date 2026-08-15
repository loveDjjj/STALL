#!/usr/bin/env python3
"""Evaluate real-only cross-fitted joint Global-Local typicality models."""

from __future__ import annotations

import argparse
import hashlib
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import multivariate_normal, norm
from sklearn.covariance import LedoitWolf
from sklearn.model_selection import KFold


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from alpha_stalled.calibration import empirical_cdf
from alpha_stalled.historical_window_analysis import calibration_references
from alpha_stalled.metrics import macro_cluster_bootstrap, metric_tables, paired_bootstrap


KEY_COLUMNS = ["dataset", "subset", "source_model", "filename"]
SEEDS = (13, 29, 42, 73, 101)
LOCAL_RAW_COLUMNS = {
    "MW1": "L_mean_raw",
    "MW2": "L_mean_raw",
    "MW3": "L_bottom2_raw",
    "MW4": "L_hybrid_raw",
}
CONFIG_NAMES = {
    "J0": "Frozen fixed 0.6/0.4 fusion",
    "J0X": "Cross-fitted fixed 0.6/0.4 control",
    "J1": "Symmetric shrinkage Mahalanobis",
    "J2": "One-sided lower-tail Mahalanobis",
    "J2G": "Real-Q95 conflict-gated J2 diagnostic",
    "J3": "Gaussian copula lower tail",
}


def gaussianize(percentiles: np.ndarray, calibration_size: int) -> np.ndarray:
    epsilon = 1.0 / float(calibration_size + 1)
    return norm.ppf(np.clip(percentiles, epsilon, 1.0 - epsilon))


def upper_tail_realness(values: np.ndarray, calibration: np.ndarray) -> np.ndarray:
    reference = np.sort(np.asarray(calibration, dtype=np.float64))
    positions = np.searchsorted(reference, values, side="left")
    return (len(reference) - positions) / float(len(reference))


def quadratic(values: np.ndarray, center: np.ndarray, precision: np.ndarray) -> np.ndarray:
    centered = values - center
    return np.einsum("ni,ij,nj->n", centered, precision, centered)


def lower_tail_quadratic(values: np.ndarray, precision: np.ndarray) -> np.ndarray:
    deficits = np.maximum(0.0, -values)
    return np.einsum("ni,ij,nj->n", deficits, precision, deficits)


@dataclass(frozen=True)
class CrossfitMargins:
    calibration_percentiles: np.ndarray
    evaluation_fold_percentiles: np.ndarray


def clipped_percentile(values: np.ndarray, reference: np.ndarray) -> np.ndarray:
    epsilon = 1.0 / float(len(reference) + 1)
    return np.clip(empirical_cdf(values, reference), epsilon, 1.0 - epsilon)


def crossfit_margins(
    calibration_raw: np.ndarray,
    evaluation_raw: np.ndarray,
    seed: int,
    folds: int = 5,
) -> CrossfitMargins:
    if calibration_raw.shape[1] != 2 or evaluation_raw.shape[1] != 2:
        raise ValueError("joint margins require exactly two columns")
    splitter = KFold(n_splits=folds, shuffle=True, random_state=seed)
    oof = np.empty_like(calibration_raw, dtype=np.float64)
    evaluation = np.empty((folds, len(evaluation_raw), 2), dtype=np.float64)
    for fold_id, (train, heldout) in enumerate(splitter.split(calibration_raw)):
        for branch in range(2):
            reference = calibration_raw[train, branch]
            oof[heldout, branch] = empirical_cdf(calibration_raw[heldout, branch], reference)
            evaluation[fold_id, :, branch] = empirical_cdf(evaluation_raw[:, branch], reference)
    return CrossfitMargins(oof, evaluation)


def crossfit_conditioned_margins(
    calibration: pd.DataFrame,
    evaluation: pd.DataFrame,
    references: pd.DataFrame,
    local_raw_column: str,
    seed: int,
    folds: int = 5,
) -> CrossfitMargins:
    """Cross-fit margins with effective-K matched, video-weighted references."""
    branch_columns = ("G_mean_raw", local_raw_column)
    oof = np.empty((len(calibration), 2), dtype=np.float64)
    evaluation_fold = np.empty((folds, len(evaluation), 2), dtype=np.float64)
    splitter = KFold(n_splits=folds, shuffle=True, random_state=seed)
    calibration_ids = calibration[KEY_COLUMNS].astype(str).agg("|".join, axis=1).to_numpy()
    reference_ids = references[KEY_COLUMNS].astype(str).agg("|".join, axis=1)
    for fold_id, (train, heldout) in enumerate(splitter.split(calibration)):
        train_ids = set(calibration_ids[train])
        fold_reference = references[reference_ids.isin(train_ids)]
        for target_k, heldout_group in calibration.iloc[heldout].groupby("effective_k", sort=True):
            heldout_positions = heldout_group.index.to_numpy()
            reference = fold_reference[fold_reference["target_k"] == int(target_k)]
            if len(reference) < 20:
                raise ValueError(f"insufficient fold reference for effective_k={target_k}: {len(reference)}")
            for branch, column in enumerate(branch_columns):
                oof[heldout_positions, branch] = clipped_percentile(
                    heldout_group[column].to_numpy(dtype=np.float64),
                    reference[column].to_numpy(dtype=np.float64),
                )
        for target_k, eval_group in evaluation.groupby("effective_k", sort=True):
            eval_positions = eval_group.index.to_numpy()
            reference = fold_reference[fold_reference["target_k"] == int(target_k)]
            if len(reference) < 20:
                raise ValueError(f"insufficient fold reference for effective_k={target_k}: {len(reference)}")
            for branch, column in enumerate(branch_columns):
                evaluation_fold[fold_id, eval_positions, branch] = clipped_percentile(
                    eval_group[column].to_numpy(dtype=np.float64),
                    reference[column].to_numpy(dtype=np.float64),
                )
    return CrossfitMargins(oof, evaluation_fold)


@dataclass(frozen=True)
class JointSeedResult:
    evaluation_scores: dict[str, np.ndarray]
    calibration_scores: dict[str, np.ndarray]
    evaluation_margins: np.ndarray
    calibration_margins: np.ndarray
    conflict_threshold: float


def fit_joint_seed(margins: CrossfitMargins) -> JointSeedResult:
    calibration_size = len(margins.calibration_percentiles)
    fold_train_size = calibration_size * 4 // 5
    z_calibration = gaussianize(margins.calibration_percentiles, fold_train_size)
    z_evaluation = gaussianize(margins.evaluation_fold_percentiles, fold_train_size)

    shrinkage = LedoitWolf(store_precision=True, assume_centered=False).fit(z_calibration)
    precision = shrinkage.precision_
    covariance = shrinkage.covariance_
    scale = np.sqrt(np.diag(covariance))
    rho = float(np.clip(covariance[0, 1] / (scale[0] * scale[1]), -0.99, 0.99))

    d1_calibration = quadratic(z_calibration, shrinkage.location_, precision)
    d2_calibration = lower_tail_quadratic(z_calibration, precision)
    copula_calibration = multivariate_normal.cdf(
        z_calibration,
        mean=np.zeros(2),
        cov=np.array([[1.0, rho], [rho, 1.0]]),
    )

    evaluation_scores = {"J0X": [], "J1": [], "J2": [], "J3": []}
    for fold_id, fold_values in enumerate(z_evaluation):
        d1 = quadratic(fold_values, shrinkage.location_, precision)
        d2 = lower_tail_quadratic(fold_values, precision)
        copula = multivariate_normal.cdf(
            fold_values,
            mean=np.zeros(2),
            cov=np.array([[1.0, rho], [rho, 1.0]]),
        )
        evaluation_scores["J0X"].append(
            0.6 * margins.evaluation_fold_percentiles[fold_id, :, 0]
            + 0.4 * margins.evaluation_fold_percentiles[fold_id, :, 1]
        )
        evaluation_scores["J1"].append(upper_tail_realness(d1, d1_calibration))
        evaluation_scores["J2"].append(upper_tail_realness(d2, d2_calibration))
        evaluation_scores["J3"].append(empirical_cdf(copula, copula_calibration))
    averaged = {
        config: np.mean(np.stack(values), axis=0)
        for config, values in evaluation_scores.items()
    }
    calibration_scores = {
        "J0": 0.6 * margins.calibration_percentiles[:, 0] + 0.4 * margins.calibration_percentiles[:, 1],
        "J0X": 0.6 * margins.calibration_percentiles[:, 0] + 0.4 * margins.calibration_percentiles[:, 1],
        "J1": upper_tail_realness(d1_calibration, d1_calibration),
        "J2": upper_tail_realness(d2_calibration, d2_calibration),
        "J3": empirical_cdf(copula_calibration, copula_calibration),
    }
    return JointSeedResult(
        evaluation_scores=averaged,
        calibration_scores=calibration_scores,
        evaluation_margins=margins.evaluation_fold_percentiles.mean(axis=0),
        calibration_margins=margins.calibration_percentiles,
        conflict_threshold=float(
            np.quantile(
                np.abs(margins.calibration_percentiles[:, 0] - margins.calibration_percentiles[:, 1]),
                0.95,
            )
        ),
    )


def evaluate_dataset(
    dataset: str,
    calibration: pd.DataFrame,
    evaluation: pd.DataFrame,
    local_raw_column: str,
    base_config: str,
    seeds: tuple[int, ...],
    references: pd.DataFrame,
    calibration_mode: str = "effective_k",
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    calibration = calibration.sort_values(KEY_COLUMNS[1:]).reset_index(drop=True)
    evaluation = evaluation.reset_index(drop=True)
    seed_eval: list[pd.DataFrame] = []
    seed_calib: list[pd.DataFrame] = []
    for seed in seeds:
        if calibration_mode == "effective_k":
            margins = crossfit_conditioned_margins(
                calibration,
                evaluation,
                references,
                local_raw_column,
                seed,
            )
        elif calibration_mode == "unconditional":
            columns = ["G_mean_raw", local_raw_column]
            margins = crossfit_margins(
                calibration[columns].to_numpy(dtype=np.float64),
                evaluation[columns].to_numpy(dtype=np.float64),
                seed,
            )
        else:
            raise ValueError(f"unknown margin calibration mode: {calibration_mode}")
        result = fit_joint_seed(margins)
        eval_frame = evaluation[KEY_COLUMNS].copy()
        eval_frame["seed"] = seed
        eval_frame["J0"] = evaluation[base_config].to_numpy()
        eval_frame["G_crossfit"] = result.evaluation_margins[:, 0]
        eval_frame["L_crossfit"] = result.evaluation_margins[:, 1]
        eval_frame["conflict_threshold"] = result.conflict_threshold
        for config, values in result.evaluation_scores.items():
            eval_frame[config] = values
        eval_frame["J2G"] = np.where(
            np.abs(eval_frame["G_crossfit"] - eval_frame["L_crossfit"])
            > result.conflict_threshold,
            eval_frame["J2"],
            eval_frame["J0X"],
        )
        seed_eval.append(eval_frame)

        calib_frame = calibration[KEY_COLUMNS].copy()
        calib_frame["seed"] = seed
        calib_frame["G_crossfit"] = result.calibration_margins[:, 0]
        calib_frame["L_crossfit"] = result.calibration_margins[:, 1]
        calib_frame["conflict_threshold"] = result.conflict_threshold
        for config, values in result.calibration_scores.items():
            calib_frame[config] = values
        calib_frame["J2G"] = np.where(
            np.abs(calib_frame["G_crossfit"] - calib_frame["L_crossfit"])
            > result.conflict_threshold,
            calib_frame["J2"],
            calib_frame["J0X"],
        )
        seed_calib.append(calib_frame)

    evaluation_by_seed = pd.concat(seed_eval, ignore_index=True)
    calibration_by_seed = pd.concat(seed_calib, ignore_index=True)
    ensemble = evaluation[KEY_COLUMNS].copy()
    ensemble["J0"] = evaluation[base_config].to_numpy()
    for column in ("G_crossfit", "L_crossfit", "J0X", "J1", "J2", "J2G", "J3"):
        values = np.stack(
            [frame[column].to_numpy() for frame in seed_eval],
            axis=0,
        )
        ensemble[column] = values.mean(axis=0)
    ensemble["conflict_threshold"] = float(
        np.mean([frame["conflict_threshold"].iloc[0] for frame in seed_eval])
    )
    return ensemble, evaluation_by_seed, calibration_by_seed


def add_conflict_labels(
    evaluation: pd.DataFrame,
    calibration_by_seed: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    output: list[pd.DataFrame] = []
    threshold_rows: list[dict] = []
    for dataset, frame in evaluation.groupby("dataset", sort=False):
        current = frame.copy()
        difference = current["G_crossfit"] - current["L_crossfit"]
        calibration = calibration_by_seed[calibration_by_seed["dataset"] == dataset]
        calibration_ensemble = calibration.groupby(KEY_COLUMNS, observed=True).agg(
            G_crossfit=("G_crossfit", "mean"),
            L_crossfit=("L_crossfit", "mean"),
        )
        threshold = float(
            np.quantile(
                np.abs(
                    calibration_ensemble["G_crossfit"]
                    - calibration_ensemble["L_crossfit"]
                ),
                0.95,
            )
        )
        current["conflict_threshold"] = threshold
        current["conflict"] = difference.abs() > threshold
        current["conflict_direction"] = np.where(
            ~current["conflict"],
            "non_conflict",
            np.where(difference > 0, "global_high_local_low", "global_low_local_high"),
        )
        for config in CONFIG_NAMES:
            real_scores = calibration.groupby(KEY_COLUMNS, observed=True)[config].mean().to_numpy()
            score_threshold = float(np.quantile(real_scores, 0.05))
            current[f"{config}_threshold"] = score_threshold
            current[f"{config}_error"] = np.where(
                current["subset"].eq("real"),
                current[config] < score_threshold,
                current[config] >= score_threshold,
            )
            threshold_rows.append(
                {
                    "dataset": dataset,
                    "config": config,
                    "real_only_q05_threshold": score_threshold,
                    "conflict_q95_threshold": threshold,
                }
            )
        output.append(current)
    return pd.concat(output, ignore_index=True), pd.DataFrame(threshold_rows)


def conflict_summary(evaluation: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict] = []
    groups = ["all_conflict", "global_high_local_low", "global_low_local_high"]
    for dataset, dataset_frame in evaluation.groupby("dataset", sort=False):
        for name in groups:
            if name == "all_conflict":
                frame = dataset_frame[dataset_frame["conflict"]]
            else:
                frame = dataset_frame[dataset_frame["conflict_direction"] == name]
            for config in CONFIG_NAMES:
                rows.append(
                    {
                        "dataset": dataset,
                        "conflict_group": name,
                        "config": config,
                        "videos": len(frame),
                        "real_videos": int(frame["subset"].eq("real").sum()),
                        "fake_videos": int(frame["subset"].eq("annotated").sum()),
                        "error_rate": float(frame[f"{config}_error"].mean()) if len(frame) else np.nan,
                    }
                )
    return pd.DataFrame(rows)


def seed_stability_metrics(evaluation_by_seed: pd.DataFrame, seed: int) -> pd.DataFrame:
    rows: list[pd.DataFrame] = []
    for calibration_seed, frame in evaluation_by_seed.groupby("seed", sort=True):
        metrics, _ = metric_tables(
            frame,
            seed,
            score_columns=tuple(CONFIG_NAMES),
            config_names=CONFIG_NAMES,
        )
        metrics.insert(0, "calibration_seed", calibration_seed)
        rows.append(metrics)
    return pd.concat(rows, ignore_index=True)


def write_report(
    base_config: str,
    calibration_mode: str,
    metrics: pd.DataFrame,
    deltas: pd.DataFrame,
    stability: pd.DataFrame,
    conflicts: pd.DataFrame,
    output: Path,
) -> None:
    pivot = metrics.pivot(index="config", columns="dataset", values=["auc", "ap"])
    lines = [
        "# Real-only Joint Global-Local Typicality",
        "",
        f"Frozen multi-window input: `{base_config}`. J2 is the preregistered primary joint candidate; J1 and J3 are ablations.",
        f"Every result uses five-fold real-only `{calibration_mode}` cross-fitting over each dataset's 200 calibration videos and an ensemble of five fixed fold seeds.",
        "J0 is the frozen Stage-1 score. J0X applies the same fixed 0.6/0.4 fusion to cross-fitted margins, isolating the margin-calibration change from joint geometry.",
        "J2G is a post-specified, parameter-free diagnostic: use J2 only beyond the real-OOF Q95 conflict threshold and J0X otherwise.",
        "",
        "## Metrics",
        "",
        "| config | ComGenVid AUC/AP | VideoFeedback AUC/AP | GenVideo AUC/AP | Macro-3 AUC/AP |",
        "|---|---:|---:|---:|---:|",
    ]
    for config in CONFIG_NAMES:
        cells = []
        for dataset in ("comgenvid", "videofeedback", "genvideo", "Macro-3"):
            cells.append(f"{pivot.loc[config, ('auc', dataset)]:.4f}/{pivot.loc[config, ('ap', dataset)]:.4f}")
        lines.append(f"| {config} | " + " | ".join(cells) + " |")
    lines.extend(["", "## J2 admission", ""])
    macro = metrics[metrics["dataset"] == "Macro-3"].set_index("config")
    dataset_rows = metrics[metrics["dataset"] != "Macro-3"].pivot(index="dataset", columns="config", values="ap")
    j2_deltas = deltas[(deltas["new_config"] == "J2") & (deltas["metric"] == "ap")]
    macro_ci = j2_deltas[j2_deltas["dataset"] == "Macro-3"].iloc[0]
    lines.append(f"- Macro AP delta: {macro.loc['J2', 'ap'] - macro.loc['J0', 'ap']:+.4f}.")
    lines.append(f"- Joint-only Macro AP delta vs J0X: {macro.loc['J2', 'ap'] - macro.loc['J0X', 'ap']:+.4f}.")
    lines.append(f"- Worst dataset AP delta: {(dataset_rows['J2'] - dataset_rows['J0']).min():+.4f}.")
    lines.append(f"- Macro paired cluster-bootstrap 95% CI: [{macro_ci.ci95_low:+.4f}, {macro_ci.ci95_high:+.4f}].")
    stability_macro = stability[(stability["dataset"] == "Macro-3") & (stability["config"] == "J2")]
    lines.append(f"- Five-seed Macro AP mean/std: {stability_macro.ap.mean():.4f}/{stability_macro.ap.std(ddof=0):.4f}.")
    conflict_pivot = conflicts[conflicts["conflict_group"] == "all_conflict"].pivot(index="dataset", columns="config", values="error_rate")
    for dataset in conflict_pivot.index:
        lines.append(f"- {dataset} conflict error J0->J2: {conflict_pivot.loc[dataset, 'J0']:.4f}->{conflict_pivot.loc[dataset, 'J2']:.4f}.")
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--aggregates",
        type=Path,
        default=REPO_ROOT / "results/multi_window_joint_typicality/K3_uniform_all_video_aggregates.csv",
    )
    parser.add_argument(
        "--evaluation",
        type=Path,
        default=REPO_ROOT / "results/multi_window_joint_typicality/K3_uniform_evaluation_scores.csv",
    )
    parser.add_argument(
        "--window-scores",
        type=Path,
        default=REPO_ROOT / "results/multi_window_joint_typicality/K3_uniform_window_scores.csv",
    )
    parser.add_argument("--base-config", choices=tuple(LOCAL_RAW_COLUMNS), default="MW4")
    parser.add_argument(
        "--margin-calibration-mode",
        choices=("auto", "effective_k", "unconditional"),
        default="auto",
    )
    parser.add_argument("--seeds", default=",".join(str(seed) for seed in SEEDS))
    parser.add_argument("--folds", type=int, default=5, choices=[5])
    parser.add_argument("--bootstrap-iterations", type=int, default=1000)
    parser.add_argument("--metric-seed", type=int, default=42)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPO_ROOT / "results/multi_window_joint_typicality",
    )
    args = parser.parse_args()
    seeds = tuple(int(value) for value in args.seeds.split(","))
    if len(seeds) != 5 or len(set(seeds)) != 5:
        raise ValueError("exactly five distinct calibration seeds are required")

    aggregates = pd.read_csv(args.aggregates, float_precision="round_trip")
    evaluation_scores = pd.read_csv(args.evaluation, float_precision="round_trip")
    window_scores = pd.read_csv(args.window_scores, float_precision="round_trip")
    evaluation_scores["_protocol_order"] = np.arange(len(evaluation_scores), dtype=np.int64)
    local_raw = LOCAL_RAW_COLUMNS[args.base_config]
    calibration_mode = args.margin_calibration_mode
    if calibration_mode == "auto":
        recorded = set(evaluation_scores.get("calibration_mode", pd.Series(dtype=str)).dropna())
        if len(recorded) != 1:
            raise ValueError(
                "cannot infer one margin calibration mode from evaluation scores"
            )
        calibration_mode = recorded.pop()
    ensemble_frames: list[pd.DataFrame] = []
    seed_frames: list[pd.DataFrame] = []
    calibration_frames: list[pd.DataFrame] = []
    for dataset in ("comgenvid", "videofeedback", "genvideo"):
        calibration = aggregates[
            (aggregates["dataset"] == dataset)
            & (aggregates["protocol_split"] == "calibration")
        ].copy()
        evaluation_raw = aggregates[
            (aggregates["dataset"] == dataset)
            & (aggregates["protocol_split"] == "evaluation")
        ].copy()
        frozen = evaluation_scores[evaluation_scores["dataset"] == dataset][
            KEY_COLUMNS + [args.base_config, "_protocol_order"]
        ]
        evaluation = evaluation_raw.merge(frozen, on=KEY_COLUMNS, how="inner", validate="one_to_one")
        evaluation = evaluation.sort_values("_protocol_order").reset_index(drop=True)
        references = pd.DataFrame()
        if calibration_mode == "effective_k":
            reference_frames: list[pd.DataFrame] = []
            dataset_windows = window_scores[window_scores["dataset"] == dataset]
            target_values = sorted(
                set(calibration["effective_k"].astype(int))
                | set(evaluation["effective_k"].astype(int))
            )
            for target_k in target_values:
                reference_frames.append(calibration_references(dataset_windows, target_k))
            references = pd.concat(reference_frames, ignore_index=True)
        ensemble, by_seed, calibration_by_seed = evaluate_dataset(
            dataset,
            calibration,
            evaluation,
            local_raw,
            args.base_config,
            seeds,
            references,
            calibration_mode,
        )
        ensemble_frames.append(ensemble)
        seed_frames.append(by_seed)
        calibration_frames.append(calibration_by_seed)

    ensemble = pd.concat(ensemble_frames, ignore_index=True)
    by_seed = pd.concat(seed_frames, ignore_index=True)
    calibration_by_seed = pd.concat(calibration_frames, ignore_index=True)
    ensemble, thresholds = add_conflict_labels(ensemble, calibration_by_seed)
    conflicts = conflict_summary(ensemble)
    stability = seed_stability_metrics(by_seed, args.metric_seed)
    metrics, generator_metrics = metric_tables(
        ensemble,
        args.metric_seed,
        score_columns=tuple(CONFIG_NAMES),
        config_names=CONFIG_NAMES,
    )
    comparisons = tuple(
        (config, "J0", f"{config}-J0")
        for config in ("J0X", "J1", "J2", "J2G", "J3")
    )
    dataset_deltas = paired_bootstrap(
        ensemble,
        args.metric_seed,
        args.bootstrap_iterations,
        comparisons=comparisons,
    )
    # macro_cluster_bootstrap expects MW0 as its baseline column.
    # Recompute through a temporary alias to keep the shared implementation exact.
    aliased = ensemble.copy()
    aliased["MW0"] = aliased["J0"]
    macro_deltas = macro_cluster_bootstrap(
        aliased,
        ["J0X", "J1", "J2", "J2G", "J3"],
        args.metric_seed,
        args.bootstrap_iterations,
    )
    macro_deltas["base_config"] = "J0"
    macro_deltas["comparison"] = macro_deltas["new_config"] + "-J0"
    deltas = pd.concat([dataset_deltas, macro_deltas], ignore_index=True)
    focus_generators = generator_metrics[
        generator_metrics["generator"].isin(
            ["SoRA-Clip", "Text2Video-Zero", "VideoCrafter2"]
        )
    ].copy()
    focus_conflicts = ensemble[
        ensemble["source_model"].isin(
            ["SoRA-Clip", "Text2Video-Zero", "VideoCrafter2"]
        )
    ].copy()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    ensemble.to_csv(args.output_dir / "joint_evaluation_scores.csv", index=False)
    by_seed.to_csv(args.output_dir / "joint_seed_evaluation_scores.csv", index=False)
    calibration_by_seed.to_csv(args.output_dir / "joint_oof_calibration_scores.csv", index=False)
    thresholds.to_csv(args.output_dir / "joint_real_only_thresholds.csv", index=False)
    metrics.to_csv(args.output_dir / "joint_dataset_metrics.csv", index=False)
    generator_metrics.to_csv(args.output_dir / "joint_generator_metrics.csv", index=False)
    stability.to_csv(args.output_dir / "joint_seed_stability.csv", index=False)
    conflicts.to_csv(args.output_dir / "joint_conflict_summary.csv", index=False)
    focus_generators.to_csv(args.output_dir / "joint_focus_generator_metrics.csv", index=False)
    focus_conflicts.to_csv(args.output_dir / "joint_focus_conflict_scores.csv", index=False)
    deltas.to_csv(args.output_dir / "joint_bootstrap_deltas.csv", index=False)
    write_report(
        args.base_config,
        calibration_mode,
        metrics,
        deltas,
        stability,
        conflicts,
        args.output_dir / "joint_typicality_analysis.md",
    )
    print(f"evaluation={len(ensemble)} calibration_oof={len(calibration_by_seed)}")
    print(f"saved -> {args.output_dir}")


if __name__ == "__main__":
    main()
