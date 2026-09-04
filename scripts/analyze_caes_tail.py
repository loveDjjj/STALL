#!/usr/bin/env python3
"""分析三selector的Tail主对照与5-fold crossfit效应并执行配对bootstrap。"""

from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from evaluation.metrics import binary_metrics
from evaluation.tables import _balanced_real_pair
from temporal_selection.evaluation import align_selector_scores


SELECTORS = ("uniform", "feature_change", "real_anomaly")


def _variant(selector: str, aggregation: str, mode: str) -> str:
    return f"{selector}__{aggregation}__{mode}"


COMPARISONS = tuple(
    (
        _variant(selector, "cvar_10", "standard"),
        _variant(selector, "mean", "standard"),
        f"{selector}:cvar10_vs_mean",
    )
    for selector in SELECTORS
) + (
    (
        _variant("real_anomaly", "mean", "crossfit5"),
        _variant("real_anomaly", "mean", "standard"),
        "real_anomaly:crossfit_vs_standard_mean",
    ),
    (
        _variant("real_anomaly", "cvar_10", "crossfit5"),
        _variant("real_anomaly", "cvar_10", "standard"),
        "real_anomaly:crossfit_vs_standard_cvar10",
    ),
)


def _seed(seed: int, *parts: str) -> int:
    digest = hashlib.sha256(
        "\0".join((str(seed), *parts)).encode("utf-8")
    ).digest()
    return int.from_bytes(digest[:8], "little") % (2**32)


def _bootstrap(
    scores: pd.DataFrame, *, seed: int, iterations: int
) -> pd.DataFrame:
    long = scores.drop(columns=["selector"]).rename(columns={"variant": "selector"})
    aligned = align_selector_scores(
        long, baseline=_variant("uniform", "mean", "standard")
    )
    rows = []
    macro_samples = {}
    for dataset, frame in aligned.groupby("dataset", sort=True):
        real = frame[frame["subset"].eq("real")]
        pairs = [
            _balanced_real_pair(real, fake, seed)
            for _, fake in frame[frame["subset"].eq("annotated")].groupby(
                "source_model", sort=True
            )
        ]
        for candidate, baseline, label in COMPARISONS:
            rng = np.random.default_rng(_seed(seed, dataset, label))
            samples = {
                "auc": np.empty(iterations),
                "ap_real": np.empty(iterations),
            }
            for iteration in range(iterations):
                deltas = {"auc": [], "ap_real": []}
                for pair in pairs:
                    real_pair = pair[pair["subset"].eq("real")]
                    fake_pair = pair[pair["subset"].eq("annotated")]
                    sampled = pd.concat([
                        real_pair.iloc[rng.integers(0, len(real_pair), len(real_pair))],
                        fake_pair.iloc[rng.integers(0, len(fake_pair), len(fake_pair))],
                    ], ignore_index=True)
                    left = binary_metrics(sampled, f"final_score__{candidate}")
                    right = binary_metrics(sampled, f"final_score__{baseline}")
                    deltas["auc"].append(left["auc"] - right["auc"])
                    deltas["ap_real"].append(
                        left["real_positive_ap"] - right["real_positive_ap"]
                    )
                for metric in samples:
                    samples[metric][iteration] = float(np.mean(deltas[metric]))
            for metric, values in samples.items():
                macro_samples.setdefault((label, metric), []).append(values)
                rows.append({
                    "comparison": label,
                    "candidate": candidate,
                    "baseline": baseline,
                    "scope": "generator_pairwise_dataset_macro",
                    "dataset": dataset,
                    "metric": metric,
                    "delta_mean": float(values.mean()),
                    "ci95_low": float(np.quantile(values, 0.025)),
                    "ci95_high": float(np.quantile(values, 0.975)),
                    "bootstrap_iterations": iterations,
                })
    for (label, metric), values in macro_samples.items():
        samples = np.mean(np.stack(values), axis=0)
        candidate, baseline, _ = next(item for item in COMPARISONS if item[2] == label)
        rows.append({
            "comparison": label,
            "candidate": candidate,
            "baseline": baseline,
            "scope": "generator_pairwise_macro3",
            "dataset": "Macro-3",
            "metric": metric,
            "delta_mean": float(samples.mean()),
            "ci95_low": float(np.quantile(samples, 0.025)),
            "ci95_high": float(np.quantile(samples, 0.975)),
            "bootstrap_iterations": iterations,
        })
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run-dir", type=Path, default=ROOT / "results/runs/caes_tail_matrix"
    )
    parser.add_argument(
        "--output-dir", type=Path,
        default=ROOT / "results/analysis/caes_tail",
    )
    parser.add_argument("--bootstrap-iterations", type=int, default=1000)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    videos = pd.read_csv(
        args.run_dir / "video_scores.csv", float_precision="round_trip"
    )
    pairwise = pd.read_csv(
        args.run_dir / "pairwise_metrics.csv", float_precision="round_trip"
    )
    bootstrap = _bootstrap(
        videos, seed=42, iterations=args.bootstrap_iterations
    )
    bootstrap.to_csv(args.output_dir / "paired_bootstrap.csv", index=False)

    indexed = pairwise.set_index(["variant", "dataset"])
    comparison_rows = []
    for candidate, baseline, label in COMPARISONS:
        for dataset in [
            item for item in pairwise["dataset"].drop_duplicates()
            if item == "Macro-3" or item in {"comgenvid", "videofeedback", "genvideo"}
        ]:
            if (candidate, dataset) not in indexed.index or (baseline, dataset) not in indexed.index:
                continue
            left = indexed.loc[(candidate, dataset)]
            right = indexed.loc[(baseline, dataset)]
            comparison_rows.append({
                "comparison": label,
                "candidate": candidate,
                "baseline": baseline,
                "dataset": dataset,
                "auc_delta": float(left["auc"] - right["auc"]),
                "ap_delta": float(left["ap_real"] - right["ap_real"]),
            })
    pd.DataFrame(comparison_rows).to_csv(
        args.output_dir / "comparison_deltas.csv", index=False
    )

    controls = videos[
        videos["selector"].isin(["uniform", "feature_change"])
    ].pivot(
        index=["selector", "aggregation", "video_id"],
        columns="calibration_mode", values="final_score",
    )
    control = controls.assign(
        abs_difference=(controls["standard"] - controls["crossfit5"]).abs()
    ).groupby(level=[0, 1])["abs_difference"].agg(["max", "mean"]).reset_index()
    control.to_csv(args.output_dir / "crossfit_controls.csv", index=False)
    print(
        pairwise[pairwise["dataset"].eq("Macro-3")][
            ["selector", "aggregation", "calibration_mode", "auc", "ap_real"]
        ].to_string(index=False)
    )
    print("\nPrimary paired bootstrap：")
    print(
        bootstrap[bootstrap["dataset"].eq("Macro-3")][
            ["comparison", "metric", "delta_mean", "ci95_low", "ci95_high"]
        ].to_string(index=False)
    )


if __name__ == "__main__":
    main()
