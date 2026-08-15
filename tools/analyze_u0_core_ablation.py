#!/usr/bin/env python3
"""Build locked U0 component, alpha, beta, and K1/K3 ablations."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml


ROOT = Path(__file__).resolve().parents[1]
for directory in (ROOT / "src", ROOT / "tools"):
    if str(directory) not in sys.path:
        sys.path.insert(0, str(directory))

from alpha_stalled.artifacts import read_expected_shards
from alpha_stalled.metrics import metric_tables
from alpha_stalled.u0_protocol import load_calibration_references
from alpha_stalled.u0_analysis import (
    calibrate_k3_candidate,
    calibrate_raw,
    calibration_raw_references,
)
from alpha_stalled.whitening import empirical_cdf_right_inclusive, stable_sorted


KEY_COLUMNS = [
    "video_id",
    "dataset",
    "protocol_split",
    "subset",
    "source_model",
    "filename",
]
ABLATION_NAMES = {
    "A0": "Original STALL",
    "A1": "K1 GlobalSpatial only",
    "A2": "K1 GlobalT1 only",
    "A3": "K1 calibrated Global",
    "A4": "K1 PatchSpatial only",
    "A5": "K1 PatchD2 only",
    "A6": "K1 calibrated Local",
    "A7": "K3 calibrated Global only",
    "A8": "K3 calibrated Local only",
    "A9": "Unified Global+Local K1",
    "A10": "Locked U0 Global+Local K3",
}
ALPHAS = (0.0, 0.2, 0.4, 0.6, 0.8, 1.0)
BETAS = (0.0, 0.05, 0.1, 0.2, 0.5, 1.0)


def load_k1_scores(directory: Path, num_shards: int) -> pd.DataFrame:
    result = read_expected_shards(
        directory, ("comgenvid", "videofeedback", "genvideo"), num_shards
    )
    if len(result) != 21421 or result["video_id"].duplicated().any():
        raise ValueError("K1 score shards must contain 21,421 unique evaluation videos")
    return result


def scored_k1_calibration(
    calibration_raw_dir: Path, num_shards: int, config: dict
) -> pd.DataFrame:
    raw = load_calibration_references(calibration_raw_dir, num_shards)
    frames = []
    for dataset, frame in raw.groupby("dataset", sort=False):
        raw_arrays = {
            column: frame[column].to_numpy(dtype=np.float64)
            for column in (
                "global_spatial_raw",
                "global_t1_raw",
                "patch_spatial_raw",
                "patch_d2_raw",
            )
        }
        references = calibration_raw_references(
            calibration_raw_dir, num_shards, str(dataset), config
        )
        calibrated = calibrate_raw(raw_arrays, references)
        target = frame[KEY_COLUMNS].copy()
        for name, values in calibrated.items():
            target[name] = values
        frames.append(target)
    return pd.concat(frames, ignore_index=True)


def calibrate_k1_components(
    evaluation: pd.DataFrame, calibration: pd.DataFrame
) -> pd.DataFrame:
    mapping = {
        "A1": "global_spatial",
        "A2": "global_t1",
        "A3": "G_k",
        "A4": "patch_spatial",
        "A5": "patch_d2",
        "A6": "L_k",
    }
    frames = []
    for dataset, target in evaluation.groupby("dataset", sort=False):
        target = target.copy()
        reference = calibration[calibration["dataset"] == dataset]
        if len(reference) != 200:
            raise ValueError(f"{dataset}: expected 200 K1 calibration videos")
        for output, source in mapping.items():
            target[output] = empirical_cdf_right_inclusive(
                target[source].to_numpy(), stable_sorted(reference[source].to_numpy())
            )
        target["A9"] = 0.6 * target["A3"] + 0.4 * target["A6"]
        frames.append(target)
    return pd.concat(frames, ignore_index=True)


def add_k3_candidates(evaluation: pd.DataFrame, windows: pd.DataFrame) -> pd.DataFrame:
    output = evaluation.copy()
    for name, column in (("A7", "G_k"), ("A8", "L_k")):
        candidate = calibrate_k3_candidate(windows, column).rename(
            columns={"score": name}
        )
        output = output.merge(candidate, on="video_id", validate="one_to_one")
    output["A10"] = 0.6 * output["A7"] + 0.4 * output["A8"]
    return output


def add_original_stall(evaluation: pd.DataFrame, path: Path) -> pd.DataFrame:
    old = pd.read_csv(path, float_precision="round_trip")
    keys = ["dataset", "subset", "source_model", "filename"]
    old = old[keys + ["B2"]].rename(columns={"B2": "A0"})
    result = evaluation.merge(old, on=keys, how="left", validate="one_to_one")
    if result["A0"].isna().any():
        raise ValueError("failed to align Original STALL with locked evaluation")
    return result


def add_sensitivities(evaluation: pd.DataFrame, windows: pd.DataFrame) -> pd.DataFrame:
    output = evaluation.copy()
    for alpha in ALPHAS:
        output[f"alpha_{alpha:.2f}"] = alpha * output["A7"] + (1.0 - alpha) * output["A8"]
    for beta in BETAS:
        column = f"beta_window_{beta:.2f}"
        windows[column] = beta * windows["patch_spatial"] + (1.0 - beta) * windows["patch_d2"]
        local = calibrate_k3_candidate(windows, column).rename(
            columns={"score": f"L_beta_{beta:.2f}"}
        )
        output = output.merge(local, on="video_id", validate="one_to_one")
        output[f"beta_{beta:.2f}"] = (
            0.6 * output["A7"] + 0.4 * output[f"L_beta_{beta:.2f}"]
        )
    return output


def write_report(dataset_metrics: pd.DataFrame, path: Path) -> None:
    selected = dataset_metrics[dataset_metrics["config"].isin(ABLATION_NAMES)]
    pivot = selected.pivot(index="config", columns="dataset", values=["auc", "ap"])
    lines = [
        "# U0 core ablation",
        "",
        "All new rows use locked region1+mean components and real-only calibration. "
        "A0 is the exact historical Original STALL comparator; A1-A6/A9 use the unified K1 "
        "current window, while A7/A8/A10 use locked K3 effective-K recalibration.",
        "",
        "| ID | Configuration | ComGenVid | VideoFeedback | GenVideo | Macro-3 |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for config in ABLATION_NAMES:
        cells = []
        for dataset in ("comgenvid", "videofeedback", "genvideo", "Macro-3"):
            cells.append(
                f"{pivot.loc[config, ('auc', dataset)]:.4f}/"
                f"{pivot.loc[config, ('ap', dataset)]:.4f}"
            )
        lines.append(f"| {config} | {ABLATION_NAMES[config]} | " + " | ".join(cells) + " |")
    lines.extend(
        [
            "",
            "## Fixed weight sensitivity",
            "",
            "| Parameter | ComGenVid | VideoFeedback | GenVideo | Macro-3 |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    sensitivity = dataset_metrics.set_index(["config", "dataset"])
    for config in [f"alpha_{value:.2f}" for value in ALPHAS] + [
        f"beta_{value:.2f}" for value in BETAS
    ]:
        cells = []
        for dataset in ("comgenvid", "videofeedback", "genvideo", "Macro-3"):
            row = sensitivity.loc[(config, dataset)]
            cells.append(f"{row.auc:.4f}/{row.ap:.4f}")
        lines.append(f"| {config} | " + " | ".join(cells) + " |")
    lines.extend(
        [
            "",
            "Locked U0 remains alpha 0.6 and beta 0.1 regardless of the observed curves.",
            "",
            "## Historical multi-window supplements",
            "",
            "These rows used the historical dataset-specific region/aggregation configuration, "
            "not locked U0, and are included only to document completed prohibited reruns.",
            "",
            "| Historical configuration | Macro-3 AUC/AP | Decision |",
            "|---|---:|---|",
            "| clean single-window | 0.8570/0.8600 | historical main ablation |",
            "| K3 MW2 | 0.8694/0.8697 | historical tuned comparator |",
            "| K5 MW2 | 0.8663/0.8672 | reject extra cost |",
            "| all-window MW2 | 0.8712/0.8695 | AP tied, duration/K confounding |",
            "| K3 Local bottom-2 | 0.8680/0.8684 | reject |",
            "| K3 Local hybrid | 0.8687/0.8691 | reject |",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run(args: argparse.Namespace) -> None:
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    k1 = load_k1_scores(args.k1_dir, args.num_shards)
    calibration = scored_k1_calibration(
        args.calibration_raw_dir, args.num_shards, config
    )
    evaluation = calibrate_k1_components(k1, calibration)
    windows = pd.read_csv(args.k3_windows, float_precision="round_trip")
    evaluation = add_k3_candidates(evaluation, windows)
    evaluation = add_original_stall(evaluation, args.original_stall)
    evaluation = add_sensitivities(evaluation, windows)
    for column in ("alpha_0.60", "beta_0.10"):
        error = float(np.max(np.abs(evaluation[column] - evaluation["A10"])))
        if error > 1e-15:
            raise ValueError(f"locked sensitivity point {column} differs from A10: {error}")

    locked = pd.read_csv(args.locked_scores, float_precision="round_trip")[["video_id", "S"]]
    checked = evaluation[["video_id", "A10"]].merge(
        locked, on="video_id", validate="one_to_one"
    )
    error = float(np.max(np.abs(checked["A10"] - checked["S"])))
    if error > 1e-15:
        raise ValueError(f"reconstructed locked U0 differs from release: {error}")

    # The historical pairwise evaluator samples real rows by position. Preserve
    # the locked release order so identical scores cannot acquire different
    # metrics merely because cache shards were concatenated in another order.
    evaluation = locked[["video_id"]].merge(
        evaluation, on="video_id", validate="one_to_one"
    )

    ablations = list(ABLATION_NAMES)
    alpha_columns = [f"alpha_{value:.2f}" for value in ALPHAS]
    beta_columns = [f"beta_{value:.2f}" for value in BETAS]
    configs = ablations + alpha_columns + beta_columns
    names = {
        **ABLATION_NAMES,
        **{column: column for column in alpha_columns + beta_columns},
    }
    dataset_metrics, generator_metrics = metric_tables(
        evaluation, seed=args.seed, score_columns=configs, config_names=names
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    evaluation.to_csv(args.output_dir / "per_video_ablation_scores.csv", index=False)
    dataset_metrics.to_csv(args.output_dir / "core_ablation_dataset_metrics.csv", index=False)
    generator_metrics.to_csv(args.output_dir / "core_ablation_generator_metrics.csv", index=False)
    write_report(dataset_metrics, args.report)
    print(
        dataset_metrics[
            (dataset_metrics["dataset"] == "Macro-3")
            & dataset_metrics["config"].isin(ablations)
        ].to_string(index=False)
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", type=Path, default=ROOT / "configs/alpha_stalled_u0_locked.yaml"
    )
    parser.add_argument(
        "--k1-dir", type=Path, default=ROOT / "results/u0_core_ablation/k1_raw"
    )
    parser.add_argument(
        "--calibration-raw-dir",
        type=Path,
        default=ROOT / "results/u0_locked_reproduction/calibration_raw",
    )
    parser.add_argument(
        "--k3-windows",
        type=Path,
        default=ROOT / "results/u0_locked_reproduction/per_window_scores.csv",
    )
    parser.add_argument(
        "--original-stall",
        type=Path,
        default=ROOT / "results/multi_order_baselines/per_video_scores.csv",
    )
    parser.add_argument(
        "--locked-scores", type=Path, default=ROOT / "release/u0/final_video_scores.csv"
    )
    parser.add_argument(
        "--output-dir", type=Path, default=ROOT / "results/u0_core_ablation"
    )
    parser.add_argument(
        "--report", type=Path, default=ROOT / "reports/u0_core_ablation.md"
    )
    parser.add_argument("--num-shards", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
