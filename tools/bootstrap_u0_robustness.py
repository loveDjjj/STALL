#!/usr/bin/env python3
"""Paired video-cluster bootstrap for locked U0 robustness deltas."""

from __future__ import annotations

import argparse
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
TOOLS = ROOT / "tools"
for directory in (ROOT / "src", TOOLS):
    if str(directory) not in sys.path:
        sys.path.insert(0, str(directory))

from alpha_stalled.metrics import binary_metrics, pairwise_frames, repeat_by_count, stable_seed
from u0_perturbations import CONDITIONS


SCENARIO = "A_original_calibration_to_perturbed_test"
BASELINE = "R0_original"
METRICS = ("auc", "real_positive_ap")
INDEX_COLUMNS = ("video_id", "dataset", "subset", "source_model")


def wide_scores(path: Path, locked_scores: Path) -> pd.DataFrame:
    scores = pd.read_csv(path, float_precision="round_trip")
    required = {*INDEX_COLUMNS, "scenario", "condition", "S"}
    missing = required - set(scores.columns)
    if missing:
        raise ValueError(f"robustness score file missing columns: {sorted(missing)}")
    scores = scores[scores["scenario"].eq(SCENARIO)]
    if scores.duplicated(["video_id", "condition"]).any():
        raise ValueError("duplicate video/condition rows in Scenario A")
    missing_conditions = set(CONDITIONS) - set(scores["condition"].unique())
    if missing_conditions:
        raise ValueError(f"missing robustness conditions: {sorted(missing_conditions)}")
    release = pd.read_csv(locked_scores, float_precision="round_trip")
    subset_ids = set(scores["video_id"])
    wide = release[release["video_id"].isin(subset_ids)][list(INDEX_COLUMNS)].copy()
    if len(wide) != scores["video_id"].nunique():
        raise ValueError("locked score order does not cover the robustness subset exactly")
    for condition in CONDITIONS:
        values = scores[scores["condition"].eq(condition)][["video_id", "S"]].rename(
            columns={"S": condition}
        )
        wide = wide.merge(values, on="video_id", how="left", validate="one_to_one")
    if wide[list(CONDITIONS)].isna().any().any():
        raise ValueError("incomplete paired robustness scores")
    return wide


def point_deltas(
    pairs_by_dataset: dict[str, dict[str, pd.DataFrame]],
) -> dict[tuple[str, str, str], float]:
    points: dict[tuple[str, str, str], float] = {}
    macro = {condition: {metric: [] for metric in METRICS} for condition in CONDITIONS}
    for dataset, pairs in pairs_by_dataset.items():
        dataset_values = {
            condition: {metric: [] for metric in METRICS} for condition in CONDITIONS
        }
        for pair in pairs.values():
            baseline = binary_metrics(pair, BASELINE)
            for condition in CONDITIONS:
                current = binary_metrics(pair, condition)
                for metric in METRICS:
                    dataset_values[condition][metric].append(
                        current[metric] - baseline[metric]
                    )
        for condition in CONDITIONS:
            for metric in METRICS:
                value = float(np.mean(dataset_values[condition][metric]))
                points[(dataset, condition, metric)] = value
                macro[condition][metric].append(value)
    for condition in CONDITIONS:
        for metric in METRICS:
            points[("Macro-3", condition, metric)] = float(
                np.mean(macro[condition][metric])
            )
    return points


def sample_chunk(
    pairs_by_dataset: dict[str, dict[str, pd.DataFrame]],
    seed: int,
    iterations: list[int],
) -> dict[tuple[str, str, str], list[float]]:
    samples: dict[tuple[str, str, str], list[float]] = {}
    for iteration in iterations:
        macro = {condition: {metric: [] for metric in METRICS} for condition in CONDITIONS}
        for dataset, pairs in pairs_by_dataset.items():
            real_ids = sorted(
                {
                    str(video_id)
                    for pair in pairs.values()
                    for video_id in pair.loc[pair["subset"].eq("real"), "video_id"]
                }
            )
            real_rng = np.random.default_rng(
                stable_seed(seed, dataset, str(iteration), "real")
            )
            real_counts = (
                pd.Series(real_rng.choice(real_ids, size=len(real_ids), replace=True))
                .value_counts()
                .astype(int)
                .to_dict()
            )
            dataset_values = {
                condition: {metric: [] for metric in METRICS}
                for condition in CONDITIONS
            }
            for generator, pair in pairs.items():
                real = repeat_by_count(pair[pair["subset"].eq("real")], real_counts)
                fake = pair[pair["subset"].ne("real")]
                fake_rng = np.random.default_rng(
                    stable_seed(seed, dataset, str(iteration), str(generator), "fake")
                )
                fake = fake.iloc[fake_rng.integers(0, len(fake), size=len(fake))]
                sample = pd.concat([real, fake], ignore_index=True)
                baseline = binary_metrics(sample, BASELINE)
                for condition in CONDITIONS:
                    current = binary_metrics(sample, condition)
                    for metric in METRICS:
                        dataset_values[condition][metric].append(
                            current[metric] - baseline[metric]
                        )
            for condition in CONDITIONS:
                for metric in METRICS:
                    value = float(np.mean(dataset_values[condition][metric]))
                    samples.setdefault((dataset, condition, metric), []).append(value)
                    macro[condition][metric].append(value)
        for condition in CONDITIONS:
            for metric in METRICS:
                samples.setdefault(("Macro-3", condition, metric), []).append(
                    float(np.mean(macro[condition][metric]))
                )
    return samples


def run(args: argparse.Namespace) -> None:
    scores = wide_scores(args.scores, args.locked_scores)
    pairs = {
        str(dataset): pairwise_frames(frame, args.seed)
        for dataset, frame in scores.groupby("dataset", sort=True)
    }
    points = point_deltas(pairs)
    chunks = [
        chunk.tolist()
        for chunk in np.array_split(
            np.arange(args.iterations, dtype=int), min(args.workers, args.iterations)
        )
    ]
    if len(chunks) == 1:
        results = [sample_chunk(pairs, args.seed, chunks[0])]
    else:
        with ProcessPoolExecutor(max_workers=len(chunks)) as pool:
            results = [
                future.result()
                for future in [
                    pool.submit(sample_chunk, pairs, args.seed, chunk) for chunk in chunks
                ]
            ]
    merged: dict[tuple[str, str, str], list[float]] = {}
    for result in results:
        for key, values in result.items():
            merged.setdefault(key, []).extend(values)
    rows = []
    for key, values in sorted(merged.items()):
        dataset, condition, metric = key
        array = np.asarray(values, dtype=np.float64)
        rows.append(
            {
                "dataset": dataset,
                "scenario": SCENARIO,
                "condition": condition,
                "baseline": BASELINE,
                "metric": metric,
                "delta": points[key],
                "bootstrap_mean": float(array.mean()),
                "ci95_low": float(np.quantile(array, 0.025)),
                "ci95_high": float(np.quantile(array, 0.975)),
                "iterations": args.iterations,
            }
        )
    output = pd.DataFrame(rows)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(args.output, index=False)
    update_report(output, args.report)
    print(output[output["dataset"].eq("Macro-3")].to_string(index=False))


def update_report(output: pd.DataFrame, report_path: Path) -> None:
    marker = "## Paired cluster bootstrap for Scenario A"
    report = report_path.read_text(encoding="utf-8").split(marker)[0].rstrip()
    macro = output[
        output["dataset"].eq("Macro-3")
        & output["metric"].eq("real_positive_ap")
    ]
    iterations = int(output["iterations"].iloc[0])
    lines = [
        report,
        "",
        marker,
        "",
        "Deltas are paired against R0 on the same videos. Resampling is clustered by video ID, "
        f"stratified by dataset and generator, and repeated {iterations:,} times; windows are never sampling units.",
        "",
        "| Condition | Macro AP delta | 95% CI |",
        "|---|---:|---:|",
    ]
    for condition in CONDITIONS:
        row = macro[macro["condition"].eq(condition)].iloc[0]
        lines.append(
            f"| {condition} | {row.delta:+.4f} | "
            f"[{row.ci95_low:+.4f}, {row.ci95_high:+.4f}] |"
        )
    lines.extend(
        [
            "",
            "CRF23 and drop10 have small point losses whose intervals cross zero; this is evidence "
            "of limited observed change, not an equivalence test. CRF35, resize, drop25, repeat25, "
            "and 4fps have fully negative AP intervals. The positive scene-cut interval reflects "
            "better class separation on this subset and must not be read as monotonic anomaly response.",
        ]
    )
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--scores",
        type=Path,
        default=ROOT / "results/u0_robustness/analysis/per_video_scores.csv",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "results/research_summary/u0_robustness_bootstrap_deltas.csv",
    )
    parser.add_argument(
        "--locked-scores",
        type=Path,
        default=ROOT / "release/u0/final_video_scores.csv",
    )
    parser.add_argument(
        "--report", type=Path, default=ROOT / "reports/u0_robustness.md"
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--iterations", type=int, default=1000)
    parser.add_argument("--workers", type=int, default=8)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
