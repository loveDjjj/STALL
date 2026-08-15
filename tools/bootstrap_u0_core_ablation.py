#!/usr/bin/env python3
"""Paired video-cluster bootstrap for the five locked U0 core claims."""

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


COMPARISONS = {
    "u0_vs_original_stall": ("A10", "A0"),
    "add_k3_coverage": ("A10", "A9"),
    "add_local_to_k3_global": ("A10", "A7"),
    "add_global_to_k3_local": ("A10", "A8"),
    "add_patch_spatial_to_k1_d2": ("A6", "A5"),
}
METRICS = ("auc", "fake_positive_ap", "real_positive_ap")
CONFIGS = sorted({config for pair in COMPARISONS.values() for config in pair})


def point_deltas(
    pairs_by_dataset: dict[str, dict[str, pd.DataFrame]]
) -> dict[tuple[str, str, str], float]:
    result = {}
    macro_values = {comparison: {metric: [] for metric in METRICS} for comparison in COMPARISONS}
    for dataset, pairs in pairs_by_dataset.items():
        dataset_values = {
            comparison: {metric: [] for metric in METRICS}
            for comparison in COMPARISONS
        }
        for pair in pairs.values():
            values = {config: binary_metrics(pair, config) for config in CONFIGS}
            for comparison, (new, base) in COMPARISONS.items():
                for metric in METRICS:
                    dataset_values[comparison][metric].append(
                        values[new][metric] - values[base][metric]
                    )
        for comparison in COMPARISONS:
            for metric in METRICS:
                value = float(np.mean(dataset_values[comparison][metric]))
                result[(dataset, comparison, metric)] = value
                macro_values[comparison][metric].append(value)
    for comparison in COMPARISONS:
        for metric in METRICS:
            result[("Macro-3", comparison, metric)] = float(
                np.mean(macro_values[comparison][metric])
            )
    return result


def sample_chunk(
    pairs_by_dataset: dict[str, dict[str, pd.DataFrame]],
    seed: int,
    iteration_indices: list[int],
) -> dict[tuple[str, str, str], list[float]]:
    samples: dict[tuple[str, str, str], list[float]] = {}
    for iteration in iteration_indices:
        macro_values = {
            comparison: {metric: [] for metric in METRICS}
            for comparison in COMPARISONS
        }
        for dataset, pairs in pairs_by_dataset.items():
            real_ids = sorted(
                {
                    str(video_id)
                    for pair in pairs.values()
                    for video_id in pair.loc[pair["subset"].eq("real"), "video_id"]
                }
            )
            rng = np.random.default_rng(stable_seed(seed, dataset, str(iteration), "real"))
            counts = pd.Series(
                rng.choice(real_ids, size=len(real_ids), replace=True)
            ).value_counts().astype(int).to_dict()
            dataset_values = {
                comparison: {metric: [] for metric in METRICS}
                for comparison in COMPARISONS
            }
            for generator, pair in pairs.items():
                real = repeat_by_count(pair[pair["subset"].eq("real")], counts)
                fake = pair[pair["subset"].ne("real")]
                fake_rng = np.random.default_rng(
                    stable_seed(seed, dataset, str(iteration), str(generator), "fake")
                )
                fake = fake.iloc[fake_rng.integers(0, len(fake), size=len(fake))]
                sample = pd.concat([real, fake], ignore_index=True)
                values = {config: binary_metrics(sample, config) for config in CONFIGS}
                for comparison, (new, base) in COMPARISONS.items():
                    for metric in METRICS:
                        dataset_values[comparison][metric].append(
                            values[new][metric] - values[base][metric]
                        )
            for comparison in COMPARISONS:
                for metric in METRICS:
                    value = float(np.mean(dataset_values[comparison][metric]))
                    samples.setdefault((dataset, comparison, metric), []).append(value)
                    macro_values[comparison][metric].append(value)
        for comparison in COMPARISONS:
            for metric in METRICS:
                samples.setdefault(("Macro-3", comparison, metric), []).append(
                    float(np.mean(macro_values[comparison][metric]))
                )
    return samples


def run(args: argparse.Namespace) -> None:
    scores = pd.read_csv(args.scores, float_precision="round_trip")
    required = {"video_id", "dataset", "subset", "source_model", *CONFIGS}
    missing = required - set(scores.columns)
    if missing:
        raise ValueError(f"ablation score file missing columns: {sorted(missing)}")
    pairs = {
        str(dataset): pairwise_frames(frame, args.seed)
        for dataset, frame in scores.groupby("dataset", sort=True)
    }
    points = point_deltas(pairs)
    indices = np.arange(args.iterations, dtype=int)
    chunks = [
        chunk.tolist()
        for chunk in np.array_split(indices, min(args.workers, args.iterations))
    ]
    with ProcessPoolExecutor(max_workers=len(chunks)) as pool:
        futures = [pool.submit(sample_chunk, pairs, args.seed, chunk) for chunk in chunks]
        chunk_results = [future.result() for future in futures]
    merged: dict[tuple[str, str, str], list[float]] = {}
    for chunk in chunk_results:
        for key, values in chunk.items():
            merged.setdefault(key, []).extend(values)
    rows = []
    for key, values in sorted(merged.items()):
        dataset, comparison, metric = key
        array = np.asarray(values, dtype=np.float64)
        new, base = COMPARISONS[comparison]
        rows.append(
            {
                "dataset": dataset,
                "comparison": comparison,
                "new_config": new,
                "base_config": base,
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
    macro = output[output["dataset"] == "Macro-3"]
    marker = "## Paired cluster bootstrap"
    report = args.report.read_text(encoding="utf-8").split(marker)[0].rstrip()
    lines = [
        report,
        "",
        marker,
        "",
        "Resampling is paired by score and clustered by video ID; windows are never sampling units.",
        "",
        "| Comparison | Metric | Delta | 95% CI |",
        "|---|---|---:|---:|",
    ]
    for row in macro.sort_values(["comparison", "metric"]).itertuples(index=False):
        lines.append(
            f"| {row.comparison} | {row.metric} | {row.delta:+.4f} | "
            f"[{row.ci95_low:+.4f}, {row.ci95_high:+.4f}] |"
        )
    args.report.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(output[output["dataset"] == "Macro-3"].to_string(index=False))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--scores",
        type=Path,
        default=ROOT / "results/u0_core_ablation/per_video_ablation_scores.csv",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "results/u0_core_ablation/core_ablation_bootstrap.csv",
    )
    parser.add_argument(
        "--report", type=Path, default=ROOT / "reports/u0_core_ablation.md"
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--iterations", type=int, default=1000)
    parser.add_argument("--workers", type=int, default=8)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
