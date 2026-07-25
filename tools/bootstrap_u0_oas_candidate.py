#!/usr/bin/env python3
"""Paired video-cluster bootstrap for OAS versus stable U0."""

from __future__ import annotations

import argparse
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
TOOLS = ROOT / "tools"
if str(TOOLS) not in sys.path:
    sys.path.insert(0, str(TOOLS))

from audit_u0_metric_protocol import binary_metrics, repeat_by_count, stable_seed
from build_multi_order_baselines import pairwise_frames


METRICS = ("auc", "fake_positive_ap", "real_positive_ap")


def point_deltas(pairs_by_dataset: dict[str, dict[str, pd.DataFrame]]) -> dict:
    result = {}
    macro = {metric: [] for metric in METRICS}
    for dataset, pairs in pairs_by_dataset.items():
        values = {metric: [] for metric in METRICS}
        for pair in pairs.values():
            new = binary_metrics(pair, "S1")
            base = binary_metrics(pair, "S0")
            for metric in METRICS:
                values[metric].append(new[metric] - base[metric])
        for metric in METRICS:
            value = float(np.mean(values[metric]))
            result[(dataset, metric)] = value
            macro[metric].append(value)
    for metric in METRICS:
        result[("Macro-3", metric)] = float(np.mean(macro[metric]))
    return result


def sample_chunk(
    pairs_by_dataset: dict[str, dict[str, pd.DataFrame]],
    seed: int,
    iterations: list[int],
) -> dict[tuple[str, str], list[float]]:
    output: dict[tuple[str, str], list[float]] = {}
    for iteration in iterations:
        macro = {metric: [] for metric in METRICS}
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
            counts = (
                pd.Series(real_rng.choice(real_ids, size=len(real_ids), replace=True))
                .value_counts()
                .astype(int)
                .to_dict()
            )
            dataset_values = {metric: [] for metric in METRICS}
            for generator, pair in pairs.items():
                real = repeat_by_count(pair[pair["subset"].eq("real")], counts)
                fake = pair[pair["subset"].ne("real")]
                fake_rng = np.random.default_rng(
                    stable_seed(seed, dataset, str(iteration), str(generator), "fake")
                )
                fake = fake.iloc[fake_rng.integers(0, len(fake), size=len(fake))]
                sample = pd.concat([real, fake], ignore_index=True)
                new = binary_metrics(sample, "S1")
                base = binary_metrics(sample, "S0")
                for metric in METRICS:
                    dataset_values[metric].append(new[metric] - base[metric])
            for metric in METRICS:
                value = float(np.mean(dataset_values[metric]))
                output.setdefault((dataset, metric), []).append(value)
                macro[metric].append(value)
        for metric in METRICS:
            output.setdefault(("Macro-3", metric), []).append(float(np.mean(macro[metric])))
    return output


def run(args: argparse.Namespace) -> None:
    scores = pd.read_csv(args.scores, float_precision="round_trip")
    pairs = {
        str(dataset): pairwise_frames(frame, args.seed)
        for dataset, frame in scores.groupby("dataset", sort=True)
    }
    points = point_deltas(pairs)
    indices = np.arange(args.iterations, dtype=int)
    chunks = [
        part.tolist()
        for part in np.array_split(indices, min(args.workers, args.iterations))
    ]
    with ProcessPoolExecutor(max_workers=len(chunks)) as pool:
        results = [
            future.result()
            for future in [
                pool.submit(sample_chunk, pairs, args.seed, chunk) for chunk in chunks
            ]
        ]
    merged: dict[tuple[str, str], list[float]] = {}
    for result in results:
        for key, values in result.items():
            merged.setdefault(key, []).extend(values)
    rows = []
    for (dataset, metric), values in sorted(merged.items()):
        array = np.asarray(values, dtype=np.float64)
        rows.append(
            {
                "dataset": dataset,
                "metric": metric,
                "delta": points[(dataset, metric)],
                "bootstrap_mean": float(array.mean()),
                "ci95_low": float(np.quantile(array, 0.025)),
                "ci95_high": float(np.quantile(array, 0.975)),
                "iterations": args.iterations,
            }
        )
    output = pd.DataFrame(rows)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(args.output, index=False)

    macro = output[output["dataset"].eq("Macro-3")]
    report = args.report.read_text(encoding="utf-8").split("## Paired cluster bootstrap")[0].rstrip()
    lines = [
        report,
        "",
        "## Paired cluster bootstrap",
        "",
        "| Metric | Delta | 95% CI |",
        "|---|---:|---:|",
    ]
    for row in macro.itertuples(index=False):
        lines.append(
            f"| {row.metric} | {row.delta:+.6f} | "
            f"[{row.ci95_low:+.6f}, {row.ci95_high:+.6f}] |"
        )
    ap = macro[macro["metric"].eq("real_positive_ap")].iloc[0]
    dataset_metrics = pd.read_csv(args.dataset_metrics)
    indexed = dataset_metrics.set_index(["config", "dataset"])
    dataset_deltas = [
        float(indexed.loc[("S1", dataset), "ap"] - indexed.loc[("S0", dataset), "ap"])
        for dataset in ("comgenvid", "videofeedback", "genvideo")
    ]
    generator_metrics = pd.read_csv(args.generator_metrics)
    generator = generator_metrics.pivot_table(
        index=["dataset", "generator"], columns="config", values="ap"
    )
    nondecline = int((generator["S1"] >= generator["S0"]).sum())
    seed_metrics = pd.read_csv(args.seed_metrics)
    seed_macro = seed_metrics[seed_metrics["dataset"].eq("Macro-3")]
    variance_not_increased = (
        float(seed_macro["S1_ap"].std(ddof=1))
        <= float(seed_macro["S0_ap"].std(ddof=1))
    )
    admitted = (
        ap.delta >= 0.003
        and min(dataset_deltas) >= -0.003
        and nondecline >= 12
        and ap.ci95_low > 0
        and variance_not_increased
    )
    lines.extend(
        [
            "",
            "## Internal decision",
            "",
            (
                "OAS passes the internal metric gates and remains pending locked external validation."
                if admitted
                else "OAS is rejected by the predeclared internal gates; S0 remains the locked method and no further covariance search is allowed."
            ),
        ]
    )
    args.report.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(output.to_string(index=False))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--scores", type=Path, default=ROOT / "results/u0_oas_candidate/per_video_scores.csv"
    )
    parser.add_argument(
        "--dataset-metrics",
        type=Path,
        default=ROOT / "results/u0_oas_candidate/dataset_metrics.csv",
    )
    parser.add_argument(
        "--generator-metrics",
        type=Path,
        default=ROOT / "results/u0_oas_candidate/generator_metrics.csv",
    )
    parser.add_argument(
        "--seed-metrics",
        type=Path,
        default=ROOT / "results/u0_oas_candidate/seed_metrics.csv",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "results/u0_oas_candidate/bootstrap_deltas.csv",
    )
    parser.add_argument(
        "--report", type=Path, default=ROOT / "reports/u0_oas_covariance_candidate.md"
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--iterations", type=int, default=1000)
    parser.add_argument("--workers", type=int, default=8)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
