#!/usr/bin/env python3
"""Audit locked U0 score orientation and aggregation-level metrics."""

from __future__ import annotations

import argparse
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_curve


ROOT = Path(__file__).resolve().parents[1]
TOOLS = ROOT / "tools"
for directory in (ROOT / "src", TOOLS):
    if str(directory) not in sys.path:
        sys.path.insert(0, str(directory))

from alpha_stalled.metrics import (
    binary_metrics,
    pairwise_frames,
    repeat_by_count,
    stable_seed,
)


METRICS = (
    "auc",
    "fake_positive_ap",
    "real_positive_ap",
    "balanced_accuracy_at_0p5",
    "fake_tpr_at_1pct_real_fpr",
    "fake_tpr_at_5pct_real_fpr",
    "eer",
)


def tpr_at_fpr(fake_label: np.ndarray, anomaly_score: np.ndarray, target: float) -> float:
    fpr, tpr, _ = roc_curve(fake_label, anomaly_score)
    return tpr_from_curve(fpr, tpr, target)


def tpr_from_curve(fpr: np.ndarray, tpr: np.ndarray, target: float) -> float:
    eligible = tpr[fpr <= target]
    return float(eligible.max()) if len(eligible) else 0.0


def metric_row(scope: str, dataset: str, frame: pd.DataFrame, **extra: object) -> dict:
    return {
        "scope": scope,
        "dataset": dataset,
        "n_real": int(frame["subset"].eq("real").sum()),
        "n_fake": int(frame["subset"].ne("real").sum()),
        **extra,
        **binary_metrics(frame),
    }


def point_metrics(scores: pd.DataFrame, seed: int) -> tuple[pd.DataFrame, dict]:
    rows: list[dict] = []
    pairs_by_dataset: dict[str, dict[str, pd.DataFrame]] = {}
    for dataset, frame in scores.groupby("dataset", sort=True):
        pairs = pairwise_frames(frame, seed)
        pairs_by_dataset[str(dataset)] = pairs
        generator_values = []
        for generator, pair in pairs.items():
            row = metric_row(
                "generator_pair", str(dataset), pair, generator=str(generator)
            )
            rows.append(row)
            generator_values.append(row)
        dataset_row = {
            "scope": "generator_pairwise_dataset_macro",
            "dataset": str(dataset),
            "generator": "ALL",
            "n_real": int(sum(row["n_real"] for row in generator_values)),
            "n_fake": int(sum(row["n_fake"] for row in generator_values)),
        }
        dataset_row.update(
            {metric: float(np.mean([row[metric] for row in generator_values])) for metric in METRICS}
        )
        rows.append(dataset_row)
        rows.append(metric_row("unique_video_pooled", str(dataset), frame, generator="ALL"))

    table = pd.DataFrame(rows)
    pairwise_dataset = table[table["scope"] == "generator_pairwise_dataset_macro"]
    macro = {
        "scope": "generator_pairwise_macro3",
        "dataset": "Macro-3",
        "generator": "ALL",
        "n_real": int(pairwise_dataset["n_real"].sum()),
        "n_fake": int(pairwise_dataset["n_fake"].sum()),
    }
    macro.update({metric: float(pairwise_dataset[metric].mean()) for metric in METRICS})
    pooled = table[table["scope"] == "unique_video_pooled"]
    pooled_macro = {
        "scope": "unique_video_pooled_macro3",
        "dataset": "Macro-3",
        "generator": "ALL",
        "n_real": int(pooled["n_real"].sum()),
        "n_fake": int(pooled["n_fake"].sum()),
    }
    pooled_macro.update({metric: float(pooled[metric].mean()) for metric in METRICS})
    generators = table[table["scope"] == "generator_pair"]
    generator_macro = {
        "scope": "generator_macro20",
        "dataset": "All-20",
        "generator": "ALL",
        "n_real": int(generators["n_real"].sum()),
        "n_fake": int(generators["n_fake"].sum()),
    }
    generator_macro.update({metric: float(generators[metric].mean()) for metric in METRICS})
    table = pd.concat(
        [table, pd.DataFrame([macro, pooled_macro, generator_macro])], ignore_index=True
    )
    return table, pairs_by_dataset


def bootstrap_sample_chunk(
    scores: pd.DataFrame,
    pairs_by_dataset: dict[str, dict[str, pd.DataFrame]],
    seed: int,
    iteration_indices: list[int],
) -> dict[tuple[str, str], list[float]]:
    samples: dict[tuple[str, str], list[float]] = {}
    for iteration in iteration_indices:
        pair_iteration: list[dict] = []
        pooled_iteration: list[dict] = []
        pairwise_dataset_iteration: list[dict] = []
        for dataset, dataset_scores in scores.groupby("dataset", sort=True):
            dataset = str(dataset)
            pairs = pairs_by_dataset[dataset]
            real_ids = sorted(
                {
                    str(video_id)
                    for pair in pairs.values()
                    for video_id in pair.loc[pair["subset"].eq("real"), "video_id"]
                }
            )
            real_rng = np.random.default_rng(stable_seed(seed, dataset, str(iteration), "real"))
            real_draw = real_rng.choice(real_ids, size=len(real_ids), replace=True)
            real_counts = pd.Series(real_draw).value_counts().astype(int).to_dict()
            generator_rows = []
            for generator, pair in pairs.items():
                real = pair[pair["subset"].eq("real")]
                fake = pair[pair["subset"].ne("real")]
                fake_rng = np.random.default_rng(
                    stable_seed(seed, dataset, str(iteration), str(generator), "fake")
                )
                fake_sample = fake.iloc[
                    fake_rng.integers(0, len(fake), size=len(fake))
                ]
                sample = pd.concat(
                    [repeat_by_count(real, real_counts), fake_sample], ignore_index=True
                )
                values = binary_metrics(sample)
                generator_rows.append(values)
                pair_iteration.append(values)
            dataset_values = {
                metric: float(np.mean([row[metric] for row in generator_rows]))
                for metric in METRICS
            }
            for metric, value in dataset_values.items():
                samples.setdefault((f"generator_pairwise:{dataset}", metric), []).append(value)
            pairwise_dataset_iteration.append(dataset_values)

            real = dataset_scores[dataset_scores["subset"].eq("real")]
            fake = dataset_scores[dataset_scores["subset"].ne("real")]
            pooled_rng = np.random.default_rng(stable_seed(seed, dataset, str(iteration), "pooled"))
            pooled_sample = pd.concat(
                [
                    real.iloc[pooled_rng.integers(0, len(real), size=len(real))],
                    fake.iloc[pooled_rng.integers(0, len(fake), size=len(fake))],
                ],
                ignore_index=True,
            )
            pooled_values = binary_metrics(pooled_sample)
            pooled_iteration.append(pooled_values)
            for metric, value in pooled_values.items():
                samples.setdefault((f"unique_video_pooled:{dataset}", metric), []).append(value)

        for metric in METRICS:
            samples.setdefault(("generator_pairwise:Macro-3", metric), []).append(
                float(np.mean([row[metric] for row in pairwise_dataset_iteration]))
            )
            samples.setdefault(("generator_macro:All-20", metric), []).append(
                float(np.mean([row[metric] for row in pair_iteration]))
            )
            samples.setdefault(("unique_video_pooled:Macro-3", metric), []).append(
                float(np.mean([row[metric] for row in pooled_iteration]))
            )
    return samples


def cluster_bootstrap(
    scores: pd.DataFrame,
    pairs_by_dataset: dict[str, dict[str, pd.DataFrame]],
    seed: int,
    iterations: int,
    workers: int = 1,
) -> pd.DataFrame:
    if iterations < 1 or workers < 1:
        raise ValueError("bootstrap iterations and workers must be positive")
    indices = np.arange(iterations, dtype=int)
    chunks = [chunk.tolist() for chunk in np.array_split(indices, min(workers, iterations))]
    if len(chunks) == 1:
        chunk_results = [bootstrap_sample_chunk(scores, pairs_by_dataset, seed, chunks[0])]
    else:
        with ProcessPoolExecutor(max_workers=len(chunks)) as pool:
            futures = [
                pool.submit(
                    bootstrap_sample_chunk, scores, pairs_by_dataset, seed, chunk
                )
                for chunk in chunks
            ]
            chunk_results = [future.result() for future in futures]
    samples: dict[tuple[str, str], list[float]] = {}
    for chunk in chunk_results:
        for key, values in chunk.items():
            samples.setdefault(key, []).extend(values)
    rows = []
    for (scope, metric), values in sorted(samples.items()):
        array = np.asarray(values, dtype=np.float64)
        rows.append(
            {
                "scope": scope,
                "metric": metric,
                "bootstrap_iterations": iterations,
                "mean": float(array.mean()),
                "std": float(array.std(ddof=1)) if len(array) > 1 else 0.0,
                "ci95_low": float(np.quantile(array, 0.025)),
                "ci95_high": float(np.quantile(array, 0.975)),
            }
        )
    return pd.DataFrame(rows)


def write_report(metrics: pd.DataFrame, bootstrap: pd.DataFrame, path: Path) -> None:
    selected = metrics[
        metrics["scope"].isin(
            [
                "generator_pairwise_dataset_macro",
                "generator_pairwise_macro3",
                "unique_video_pooled",
                "unique_video_pooled_macro3",
                "generator_macro20",
            ]
        )
    ]
    lines = [
        "# U0 metric protocol audit",
        "",
        "Stored `S` is higher-is-real. Historical AP is real-positive AP. "
        "Fake-positive AP uses `1-S` with fake label 1; the two AP values are not interchangeable.",
        "",
        "| Scope | Dataset | AUC | Fake-positive AP | Real-positive AP | BAcc@0.5 | TPR@1% | TPR@5% | EER |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in selected.itertuples(index=False):
        lines.append(
            f"| {row.scope} | {row.dataset} | {row.auc:.4f} | "
            f"{row.fake_positive_ap:.4f} | {row.real_positive_ap:.4f} | "
            f"{row.balanced_accuracy_at_0p5:.4f} | "
            f"{row.fake_tpr_at_1pct_real_fpr:.4f} | "
            f"{row.fake_tpr_at_5pct_real_fpr:.4f} | {row.eer:.4f} |"
        )
    ci_scopes = {
        "generator_pairwise:Macro-3",
        "unique_video_pooled:Macro-3",
        "generator_macro:All-20",
    }
    ci_metrics = {"auc", "fake_positive_ap", "real_positive_ap"}
    ci = bootstrap[
        bootstrap["scope"].isin(ci_scopes) & bootstrap["metric"].isin(ci_metrics)
    ]
    lines.extend(
        [
            "",
            "The original STALL paper states that generated video is the positive class for AP. "
            "This repository's frozen Alpha-STALLED main table instead uses real-positive AP for "
            "continuity with its historical evaluation scripts. Fake-positive AP is therefore the "
            "orientation aligned with STALL Table 1; real-positive AP is an internal endpoint, and "
            "the two must be named explicitly. AUC is unchanged when both score and label orientation "
            "are reversed. Pooled AP is prevalence-sensitive and must not be compared numerically "
            "with balanced pairwise AP.",
            "",
            "| Bootstrap scope | Metric | Mean | Std | 95% CI |",
            "|---|---|---:|---:|---:|",
        ]
    )
    for row in ci.sort_values(["scope", "metric"]).itertuples(index=False):
        lines.append(
            f"| {row.scope} | {row.metric} | {row.mean:.4f} | {row.std:.4f} | "
            f"[{row.ci95_low:.4f}, {row.ci95_high:.4f}] |"
        )
    lines.extend(
        [
            "",
            "Bootstrap resamples video IDs as clusters. Repeated real videos in generator-pairwise "
            "evaluation share one resampling multiplicity; windows are never bootstrap units.",
            "",
            f"Machine-readable bootstrap rows: {len(bootstrap)}.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run(args: argparse.Namespace) -> None:
    scores = pd.read_csv(args.scores, float_precision="round_trip")
    required = {"video_id", "dataset", "subset", "source_model", "S"}
    missing = required - set(scores.columns)
    if missing:
        raise ValueError(f"score file missing columns: {sorted(missing)}")
    if scores["video_id"].duplicated().any():
        raise ValueError("metric input must contain one row per video ID")
    if not np.isfinite(scores["S"].to_numpy(dtype=np.float64)).all():
        raise ValueError("metric input contains non-finite scores")
    metrics, pairs = point_metrics(scores, args.seed)
    bootstrap = cluster_bootstrap(
        scores, pairs, args.seed, args.bootstrap_iterations, args.workers
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    metrics.to_csv(args.output_dir / "metric_protocol_metrics.csv", index=False)
    bootstrap.to_csv(args.output_dir / "metric_protocol_bootstrap.csv", index=False)
    write_report(metrics, bootstrap, args.report)
    print(metrics[metrics["dataset"].isin(["Macro-3", "All-20"])].to_string(index=False))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--scores", type=Path, default=ROOT / "release/u0/final_video_scores.csv"
    )
    parser.add_argument(
        "--output-dir", type=Path, default=ROOT / "results/u0_metric_protocol"
    )
    parser.add_argument(
        "--report", type=Path, default=ROOT / "reports/u0_metric_protocol_audit.md"
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--bootstrap-iterations", type=int, default=1000)
    parser.add_argument("--workers", type=int, default=8)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
