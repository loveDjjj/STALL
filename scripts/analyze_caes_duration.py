#!/usr/bin/env python3
"""按预注册固定时长区间评估三种CAES selector，保持生成器内共享配对。"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from temporal_selection.evaluation import (
    build_matched_selector_pairwise_table,
    paired_selector_bootstrap,
)


SELECTORS = ("uniform", "feature_change", "real_anomaly")
DURATION_BINS = (
    ("2_to_4", 2.0, 4.0),
    ("4_to_8", 4.0, 8.0),
    ("8_to_16", 8.0, 16.0),
    ("16_plus", 16.0, float("inf")),
)


def _attach_duration(scores: pd.DataFrame, manifest_root: Path) -> pd.DataFrame:
    pieces = []
    for dataset, frame in scores.groupby("dataset", sort=False):
        manifest = pd.read_csv(manifest_root / f"{dataset}_evaluation.csv")
        duration = manifest[["video_path", "duration_seconds"]].drop_duplicates()
        merged = frame.merge(
            duration, on="video_path", how="left", validate="many_to_one"
        )
        if merged["duration_seconds"].isna().any():
            raise ValueError(f"{dataset}有视频无法关联duration")
        pieces.append(merged)
    return pd.concat(pieces, ignore_index=True)


def _eligible_bin(
    frame: pd.DataFrame,
    lower: float,
    upper: float,
    minimum_per_class: int,
) -> tuple[pd.DataFrame, list[dict]]:
    selected = frame[
        frame["duration_seconds"].ge(lower)
        & frame["duration_seconds"].lt(upper)
    ].copy()
    baseline = selected[selected["selector"].eq("uniform")]
    keep_fake = set()
    cells = []
    for dataset, dataset_frame in baseline.groupby("dataset", sort=True):
        real_count = int(dataset_frame["subset"].eq("real").sum())
        if real_count < minimum_per_class:
            continue
        fake = dataset_frame[dataset_frame["subset"].eq("annotated")]
        for generator, generator_frame in fake.groupby("source_model", sort=True):
            fake_count = len(generator_frame)
            if fake_count < minimum_per_class:
                continue
            keep_fake.add((dataset, generator))
            cells.append({
                "dataset": dataset,
                "generator": generator,
                "n_real_available": real_count,
                "n_fake_available": fake_count,
            })
    keep_datasets = {item[0] for item in keep_fake}
    mask = (
        selected["subset"].eq("real")
        & selected["dataset"].isin(keep_datasets)
    ) | selected.apply(
        lambda row: (
            str(row["dataset"]), str(row["source_model"])
        ) in keep_fake,
        axis=1,
    )
    return selected[mask].reset_index(drop=True), cells


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run-dir", type=Path, default=ROOT / "results/runs/caes_stage_fs"
    )
    parser.add_argument(
        "--output-dir", type=Path,
        default=ROOT / "results/analysis/caes_duration",
    )
    parser.add_argument("--minimum-per-class", type=int, default=20)
    parser.add_argument("--bootstrap-iterations", type=int, default=1000)
    args = parser.parse_args()
    scores = pd.read_csv(
        args.run_dir / "video_scores.csv", float_precision="round_trip"
    )
    scores = scores[scores["selector"].isin(SELECTORS)].copy()
    manifest_root = ROOT / "data/manifests/development"
    scores = _attach_duration(scores, manifest_root)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    metric_tables, bootstrap_tables, cell_rows = [], [], []
    for label, lower, upper in DURATION_BINS:
        selected, cells = _eligible_bin(
            scores, lower, upper, args.minimum_per_class
        )
        if selected.empty:
            continue
        table = build_matched_selector_pairwise_table(selected, seed=42)
        table.loc[table["dataset"].eq("Macro-3"), "dataset"] = "Eligible-Macro"
        table.loc[
            table["scope"].eq("generator_pairwise_macro3"), "scope"
        ] = "eligible_dataset_macro"
        table.insert(0, "duration_bin", label)
        metric_tables.append(table)
        bootstrap = paired_selector_bootstrap(
            selected, seed=42, iterations=args.bootstrap_iterations
        )
        bootstrap.loc[
            bootstrap["dataset"].eq("Macro-3"), "dataset"
        ] = "Eligible-Macro"
        bootstrap.loc[
            bootstrap["scope"].eq("generator_pairwise_macro3"), "scope"
        ] = "eligible_dataset_macro"
        bootstrap.insert(0, "duration_bin", label)
        bootstrap_tables.append(bootstrap)
        for item in cells:
            cell_rows.append({"duration_bin": label, **item})
    metrics = pd.concat(metric_tables, ignore_index=True)
    bootstrap = pd.concat(bootstrap_tables, ignore_index=True)
    metrics.to_csv(args.output_dir / "pairwise_metrics.csv", index=False)
    bootstrap.to_csv(args.output_dir / "paired_bootstrap.csv", index=False)
    pd.DataFrame(cell_rows).to_csv(
        args.output_dir / "eligible_generator_cells.csv", index=False
    )
    (args.output_dir / "protocol.json").write_text(
        json.dumps({
            "duration_bins_seconds": [
                {
                    "name": name,
                    "lower_inclusive": low,
                    "upper_exclusive": None if high == float("inf") else high,
                }
                for name, low, high in DURATION_BINS
            ],
            "minimum_real_and_fake_per_generator_cell": args.minimum_per_class,
            "selectors": list(SELECTORS),
            "pair_identity_source": "uniform",
            "bootstrap_iterations": args.bootstrap_iterations,
            "note": "固定结构区间，不依据fake结果选择边界",
        }, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(
        metrics[metrics["dataset"].eq("Eligible-Macro")][
            ["duration_bin", "selector", "auc", "ap_real", "n_generators"]
        ].to_string(index=False)
    )


if __name__ == "__main__":
    main()
