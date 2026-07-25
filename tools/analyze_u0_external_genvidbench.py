#!/usr/bin/env python3
"""Analyze locked U0 on the confirmatory GenVidBench MS/Pika split."""

from __future__ import annotations

import argparse
import glob
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from sklearn.metrics import average_precision_score, roc_auc_score


ROOT = Path(__file__).resolve().parents[1]
for directory in (ROOT / "src", ROOT / "tools"):
    if str(directory) not in sys.path:
        sys.path.insert(0, str(directory))

from analyze_u0_locked import cdf_with_positive_infinity, global_references
from stable_whitening import empirical_cdf_right_inclusive, stable_sorted


DATASET = "genvidbench_pair1"
CONFIGS = {
    "original_stall": "Original STALL",
    "clean_k1": "Clean single-window",
    "locked_u0": "Locked U0 K3",
}
KEY_COLUMNS = [
    "video_id",
    "dataset",
    "protocol_split",
    "subset",
    "source_model",
    "filename",
]


def load_shards(directory: Path, prefix: str, num_shards: int) -> pd.DataFrame:
    paths = [
        directory / f"{prefix}_shard{shard:02d}_of_{num_shards:02d}.csv"
        for shard in range(num_shards)
    ]
    missing = [path for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(missing[0])
    return pd.concat(
        [pd.read_csv(path, float_precision="round_trip") for path in paths],
        ignore_index=True,
    )


def calibrate_windows(
    frame: pd.DataFrame,
    patch_spatial_ref: np.ndarray,
    patch_d2_ref: np.ndarray,
    global_spatial_ref: np.ndarray,
    global_t1_ref: np.ndarray,
) -> pd.DataFrame:
    output = frame.copy()
    output["global_spatial"] = empirical_cdf_right_inclusive(
        output["global_spatial_raw"].to_numpy(), global_spatial_ref
    )
    output["global_t1"] = cdf_with_positive_infinity(
        output["global_t1_raw"].to_numpy(), global_t1_ref
    )
    output["patch_spatial"] = empirical_cdf_right_inclusive(
        output["patch_spatial_raw"].to_numpy(), patch_spatial_ref
    )
    output["patch_d2"] = empirical_cdf_right_inclusive(
        output["patch_d2_raw"].to_numpy(), patch_d2_ref
    )
    output["G_k"] = 0.5 * output["global_spatial"] + 0.5 * output["global_t1"]
    output["L_k"] = 0.1 * output["patch_spatial"] + 0.9 * output["patch_d2"]
    return output


def selected_reference(calibration: pd.DataFrame, target_k: int, column: str) -> np.ndarray:
    values = []
    for _, frame in calibration.groupby("video_id", sort=False):
        ordered = frame.sort_values("window_id")
        if len(ordered) < target_k:
            continue
        positions = (
            np.asarray([(len(ordered) - 1) // 2], dtype=int)
            if target_k == 1
            else np.rint(np.linspace(0, len(ordered) - 1, target_k)).astype(int)
        )
        selected = ordered.iloc[np.unique(positions)]
        if len(selected) == target_k:
            values.append(float(selected[column].mean()))
    if len(values) < 2:
        raise ValueError(f"insufficient external effective-K={target_k} reference")
    return stable_sorted(np.asarray(values, dtype=np.float64))


def build_scores(
    k1_raw: pd.DataFrame, k3_raw: pd.DataFrame, config: dict
) -> tuple[pd.DataFrame, pd.DataFrame, tuple[float, float]]:
    calibration_k1 = k1_raw[k1_raw["protocol_split"].eq("calibration")]
    evaluation_k1 = k1_raw[k1_raw["protocol_split"].eq("evaluation")]
    if len(calibration_k1) != 199 or len(evaluation_k1) != 900:
        raise ValueError("external K1 score counts must be 199 calibration and 900 evaluation")
    patch_spatial_ref = stable_sorted(calibration_k1["patch_spatial_raw"].to_numpy())
    patch_d2_ref = stable_sorted(calibration_k1["patch_d2_raw"].to_numpy())
    global_spatial_ref, global_t1_ref = global_references(config)
    motion_thresholds = tuple(
        float(value)
        for value in calibration_k1["mean_global_motion"]
        .quantile([1 / 3, 2 / 3])
        .to_numpy()
    )
    calibration_k1 = calibrate_windows(
        calibration_k1,
        patch_spatial_ref,
        patch_d2_ref,
        global_spatial_ref,
        global_t1_ref,
    )
    evaluation_k1 = calibrate_windows(
        evaluation_k1,
        patch_spatial_ref,
        patch_d2_ref,
        global_spatial_ref,
        global_t1_ref,
    )
    for branch in ("G", "L"):
        reference = stable_sorted(calibration_k1[f"{branch}_k"].to_numpy())
        evaluation_k1[branch] = empirical_cdf_right_inclusive(
            evaluation_k1[f"{branch}_k"].to_numpy(), reference
        )
    evaluation = evaluation_k1[
        [*KEY_COLUMNS, "video_path", "duration_seconds", "mean_global_motion"]
    ].copy()
    evaluation["original_stall"] = evaluation_k1["G_k"].to_numpy()
    evaluation["clean_k1"] = (
        0.6 * evaluation_k1["G"].to_numpy() + 0.4 * evaluation_k1["L"].to_numpy()
    )

    calibrated_k3 = calibrate_windows(
        k3_raw,
        patch_spatial_ref,
        patch_d2_ref,
        global_spatial_ref,
        global_t1_ref,
    )
    per_video = (
        calibrated_k3.groupby(KEY_COLUMNS, sort=False, observed=True)
        .agg(
            effective_k=("effective_k", "first"),
            window_count=("window_id", "size"),
            G_raw=("G_k", "mean"),
            L_raw=("L_k", "mean"),
        )
        .reset_index()
    )
    if not (per_video["effective_k"] == per_video["window_count"]).all():
        raise ValueError("external K3 effective_k does not match scored window count")
    calibration_k3 = calibrated_k3[calibrated_k3["protocol_split"].eq("calibration")]
    parts = []
    for effective_k, target in per_video[
        per_video["protocol_split"].eq("evaluation")
    ].groupby("effective_k", sort=True):
        target = target.copy()
        for branch in ("G", "L"):
            reference = selected_reference(calibration_k3, int(effective_k), f"{branch}_k")
            target[branch] = empirical_cdf_right_inclusive(
                target[f"{branch}_raw"].to_numpy(), reference
            )
        target["locked_u0"] = 0.6 * target["G"] + 0.4 * target["L"]
        parts.append(target[["video_id", "effective_k", "G", "L", "locked_u0"]])
    u0 = pd.concat(parts, ignore_index=True)
    evaluation = evaluation.merge(u0, on="video_id", validate="one_to_one")
    if len(evaluation) != 900:
        raise ValueError("external evaluation must contain 900 unique videos")
    return evaluation, calibrated_k3, motion_thresholds


def metric_rows(frame: pd.DataFrame, group_name: str, group_value: str) -> list[dict]:
    rows = []
    for generator in ("ms", "pika"):
        pair = frame[
            frame["subset"].eq("real") | frame["source_model"].eq(generator)
        ]
        label = pair["subset"].eq("real").astype(int).to_numpy()
        if len(np.unique(label)) < 2:
            continue
        for config in CONFIGS:
            score = pair[config].to_numpy(dtype=np.float64)
            rows.append(
                {
                    group_name: group_value,
                    "generator": generator,
                    "config": config,
                    "real_count": int(label.sum()),
                    "fake_count": int((1 - label).sum()),
                    "auc": roc_auc_score(label, score),
                    "real_positive_ap": average_precision_score(label, score),
                    "fake_positive_ap": average_precision_score(1 - label, 1.0 - score),
                }
            )
    return rows


def grouped_metrics(
    evaluation: pd.DataFrame, motion_thresholds: tuple[float, float]
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    generator = pd.DataFrame(metric_rows(evaluation, "group", "all"))
    dataset = (
        generator.groupby("config", as_index=False)
        .agg(
            auc=("auc", "mean"),
            real_positive_ap=("real_positive_ap", "mean"),
            fake_positive_ap=("fake_positive_ap", "mean"),
        )
    )
    duration = evaluation.copy()
    duration["duration_group"] = pd.cut(
        duration["duration_seconds"],
        bins=[-np.inf, 2.1, 5.0, np.inf],
        labels=["short_le_2.1s", "medium_2.1_to_5s", "long_gt_5s"],
    ).astype(str)
    duration_rows = []
    for group, frame in duration.groupby("duration_group", sort=True):
        duration_rows.extend(metric_rows(frame, "duration_group", str(group)))
    q1, q2 = motion_thresholds
    motion = evaluation.copy()
    motion["motion_group"] = pd.cut(
        motion["mean_global_motion"],
        bins=[-np.inf, q1, q2, np.inf],
        labels=["low", "mid", "high"],
        include_lowest=True,
    ).astype(str)
    motion_rows = []
    for group, frame in motion.groupby("motion_group", sort=True):
        motion_rows.extend(metric_rows(frame, "motion_group", str(group)))
    return dataset, pd.DataFrame(duration_rows), pd.DataFrame(motion_rows)


def paired_bootstrap(evaluation: pd.DataFrame, iterations: int, seed: int) -> pd.DataFrame:
    rng = np.random.RandomState(seed)
    real = evaluation[evaluation["subset"].eq("real")].reset_index(drop=True)
    fake = {
        generator: evaluation[evaluation["source_model"].eq(generator)].reset_index(drop=True)
        for generator in ("ms", "pika")
    }
    records = []
    for iteration in range(iterations):
        real_indices = rng.randint(0, len(real), len(real))
        sampled_real = real.iloc[real_indices]
        metrics = {config: {"auc": [], "ap": []} for config in CONFIGS}
        for generator, source in fake.items():
            sampled_fake = source.iloc[rng.randint(0, len(source), len(source))]
            pair = pd.concat([sampled_real, sampled_fake], ignore_index=True)
            label = pair["subset"].eq("real").astype(int).to_numpy()
            for config in CONFIGS:
                score = pair[config].to_numpy()
                metrics[config]["auc"].append(roc_auc_score(label, score))
                metrics[config]["ap"].append(average_precision_score(label, score))
        for baseline in ("original_stall", "clean_k1"):
            for metric in ("auc", "ap"):
                records.append(
                    {
                        "iteration": iteration,
                        "comparison": f"locked_u0_minus_{baseline}",
                        "metric": metric,
                        "delta": float(
                            np.mean(metrics["locked_u0"][metric])
                            - np.mean(metrics[baseline][metric])
                        ),
                    }
                )
    samples = pd.DataFrame(records)
    summary = (
        samples.groupby(["comparison", "metric"], as_index=False)["delta"]
        .agg(
            mean="mean",
            lower=lambda values: np.quantile(values, 0.025),
            upper=lambda values: np.quantile(values, 0.975),
        )
    )
    return samples, summary


def write_report(
    metrics: pd.DataFrame,
    generator: pd.DataFrame,
    bootstrap: pd.DataFrame,
    evaluation: pd.DataFrame,
    report: Path,
) -> None:
    indexed = metrics.set_index("config")
    lines = [
        "# Locked U0 external GenVidBench validation",
        "",
        "The method and manifests were locked before scoring. Local whitening and all CDFs use "
        "199 disjoint VRIPT real calibration videos; no generated video enters fitting, "
        "calibration, or parameter selection. Evaluation contains 300 disjoint real, 300 MS, "
        "and 300 Pika videos.",
        "",
        "| Method | Pairwise AUC | Real-positive AP | Fake-positive AP |",
        "|---|---:|---:|---:|",
    ]
    for config, name in CONFIGS.items():
        row = indexed.loc[config]
        lines.append(
            f"| {name} | {row.auc:.4f} | {row.real_positive_ap:.4f} | "
            f"{row.fake_positive_ap:.4f} |"
        )
    lines.extend(["", "## By generator", "", "| Generator | Method | AUC/AP |", "|---|---|---:|"])
    for row in generator.itertuples(index=False):
        lines.append(
            f"| {row.generator} | {CONFIGS[row.config]} | "
            f"{row.auc:.4f}/{row.real_positive_ap:.4f} |"
        )
    lines.extend(["", "## Paired bootstrap", "", "| Comparison | Metric | Delta 95% CI |", "|---|---|---:|"])
    for row in bootstrap.itertuples(index=False):
        lines.append(
            f"| {row.comparison} | {row.metric} | {row.mean:+.4f} "
            f"[{row.lower:+.4f}, {row.upper:+.4f}] |"
        )
    effective = evaluation["effective_k"].value_counts().sort_index().astype(int).to_dict()
    lines.extend(
        [
            "",
            "## Protocol notes",
            "",
            f"- Evaluation effective-K distribution: `{effective}`.",
            "- Text2Video-Zero is excluded: its native 4 FPS videos cannot provide 16 distinct "
            "frames in two seconds under the locked 8 FPS protocol.",
            "- Duration groups, calibration-real motion-tertile groups, per-video scores, and "
            "failure cases are retained as machine-readable CSV files.",
        ]
    )
    report.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run(args: argparse.Namespace) -> None:
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    k1 = load_shards(args.k1_dir, DATASET, args.num_shards)
    k3 = load_shards(args.k3_dir, DATASET, args.num_shards)
    if k1["video_id"].duplicated().any():
        raise ValueError("duplicate external K1 video IDs")
    if k3.duplicated(["video_id", "window_id"]).any():
        raise ValueError("duplicate external K3 window keys")
    evaluation, calibrated_k3, motion_thresholds = build_scores(k1, k3, config)
    metrics, duration, motion = grouped_metrics(evaluation, motion_thresholds)
    generator = pd.DataFrame(metric_rows(evaluation, "group", "all"))
    bootstrap_samples, bootstrap = paired_bootstrap(
        evaluation, args.bootstrap_iterations, args.seed
    )
    fake_failures = evaluation[evaluation["subset"].ne("real")].nlargest(
        50, "locked_u0"
    ).assign(failure_type="fake_false_negative")
    real_failures = evaluation[evaluation["subset"].eq("real")].nsmallest(
        50, "locked_u0"
    ).assign(failure_type="real_false_positive")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    evaluation.to_csv(args.output_dir / "per_video_scores.csv", index=False)
    calibrated_k3.to_csv(args.output_dir / "per_window_scores.csv", index=False)
    metrics.to_csv(args.output_dir / "dataset_metrics.csv", index=False)
    generator.to_csv(args.output_dir / "generator_metrics.csv", index=False)
    duration.to_csv(args.output_dir / "duration_metrics.csv", index=False)
    motion.to_csv(args.output_dir / "motion_metrics.csv", index=False)
    bootstrap_samples.to_csv(args.output_dir / "bootstrap_samples.csv", index=False)
    bootstrap.to_csv(args.output_dir / "bootstrap_deltas.csv", index=False)
    pd.concat([fake_failures, real_failures], ignore_index=True).to_csv(
        args.output_dir / "failure_cases.csv", index=False
    )
    write_report(metrics, generator, bootstrap, evaluation, args.report)
    print(metrics.to_string(index=False))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", type=Path, default=ROOT / "configs/alpha_stalled_u0_locked.yaml"
    )
    parser.add_argument(
        "--k1-dir", type=Path, default=ROOT / "results/u0_external_genvidbench/k1_raw"
    )
    parser.add_argument(
        "--k3-dir", type=Path, default=ROOT / "results/u0_external_genvidbench/raw"
    )
    parser.add_argument(
        "--output-dir", type=Path, default=ROOT / "results/u0_external_genvidbench/analysis"
    )
    parser.add_argument(
        "--report", type=Path, default=ROOT / "reports/u0_locked_external_validation.md"
    )
    parser.add_argument("--num-shards", type=int, default=4)
    parser.add_argument("--bootstrap-iterations", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
