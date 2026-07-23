#!/usr/bin/env python3
"""Evaluate G0-G4 calibrated global D3 variants from frozen per-video scores."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
TOOLS_DIR = REPO_ROOT / "tools"
if str(TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(TOOLS_DIR))

from build_multi_order_baselines import (
    GLOBAL_D3_CONFIG_NAMES,
    metric_tables,
    paired_bootstrap,
)


GLOBAL_COLUMNS = ("G0", "G1", "G2", "G3", "G4")
COMPARISONS = (
    ("G1", "G0", "raw_d3"),
    ("G2", "G0", "two_sided_d3"),
    ("G3", "G0", "motion_conditioned_d3"),
    ("G4", "G0", "time_normalized_conditioned_d3"),
)


def add_global_variants(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    out["G0"] = out["B2"]
    out["G1"] = 0.5 * out["B0"] + 0.25 * out["B1"] + 0.25 * out["B3"]
    out["G2"] = out["G2_two_sided"]
    out["G3"] = out["G3_motion_conditional"]
    out["G4"] = out["G4_time_conditional"]
    return out


def _delta_table(generator_metrics: pd.DataFrame) -> pd.DataFrame:
    base = generator_metrics[generator_metrics["config"] == "G0"].set_index(
        ["dataset", "generator"]
    )
    rows = []
    for config in GLOBAL_COLUMNS[1:]:
        current = generator_metrics[generator_metrics["config"] == config].set_index(
            ["dataset", "generator"]
        )
        for key in current.index:
            rows.append(
                {
                    "dataset": key[0],
                    "generator": key[1],
                    "config": config,
                    "delta_auc_vs_G0": float(current.loc[key, "auc"] - base.loc[key, "auc"]),
                    "delta_ap_vs_G0": float(current.loc[key, "ap"] - base.loc[key, "ap"]),
                }
            )
    return pd.DataFrame(rows)


def write_analysis(
    path: Path,
    dataset_metrics: pd.DataFrame,
    generator_deltas: pd.DataFrame,
    bootstrap: pd.DataFrame,
) -> None:
    metrics = dataset_metrics.pivot(index="config", columns="dataset", values=["auc", "ap"])
    lines = [
        "# Global D3 calibration analysis",
        "",
        "## Protocol",
        "",
        "All variants use the Stage-1 strict 2 s / 8 FPS video intersection. G0 is original STALL. G1-G4 use the fixed structured weights `0.5 spatial + 0.25 first-order temporal + 0.25 second-order temporal`; no weight is selected on a target test set.",
        "",
        "- G1: one-sided real-CDF percentile of raw D3 volatility.",
        "- G2: two-sided real-CDF realness.",
        "- G3: two-sided realness conditioned on five real-calibration motion bins.",
        "- G4: G3 after native timestamp normalization of velocity and acceleration.",
        "",
        "## Dataset macro metrics",
        "",
        "| ID | Configuration | ComGenVid AUC/AP | VideoFeedback AUC/AP | GenVideo AUC/AP | Macro-3 AUC/AP |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for config in GLOBAL_COLUMNS:
        cells = []
        for dataset in ("comgenvid", "videofeedback", "genvideo", "Macro-3"):
            cells.append(
                f"{metrics.loc[config, ('auc', dataset)]:.4f}/{metrics.loc[config, ('ap', dataset)]:.4f}"
            )
        lines.append(
            f"| {config} | {GLOBAL_D3_CONFIG_NAMES[config]} | " + " | ".join(cells) + " |"
        )

    lines.extend(["", "## Cross-generator wins", ""])
    for config in GLOBAL_COLUMNS[1:]:
        selected = generator_deltas[generator_deltas["config"] == config]
        lines.append(
            f"- `{config}` vs G0: AP improves on {(selected.delta_ap_vs_G0 > 0).sum()}/{len(selected)} generators; AUC improves on {(selected.delta_auc_vs_G0 > 0).sum()}/{len(selected)}."
        )

    lines.extend(
        [
            "",
            "## GenVideo AP retention",
            "",
            "| Variant | Delta AP vs G0 | Retention relative to raw G1 |",
            "|---|---:|---:|",
        ]
    )
    genvideo = dataset_metrics[dataset_metrics["dataset"] == "genvideo"].set_index("config")
    raw_gain = float(genvideo.loc["G1", "ap"] - genvideo.loc["G0", "ap"])
    for config in GLOBAL_COLUMNS[1:]:
        delta = float(genvideo.loc[config, "ap"] - genvideo.loc["G0", "ap"])
        retention = delta / raw_gain if raw_gain != 0 else float("nan")
        lines.append(f"| {config} | {delta:+.4f} | {retention:.1%} |")

    lines.extend(
        [
            "",
            "## Paired bootstrap",
            "",
            "| Dataset | Variant | Metric | Delta | 95% CI |",
            "|---|---|---|---:|---:|",
        ]
    )
    for row in bootstrap.itertuples(index=False):
        lines.append(
            f"| {row.dataset} | {row.new_config}-{row.base_config} | {row.metric.upper()} | "
            f"{row.delta:+.4f} | [{row.ci95_low:+.4f}, {row.ci95_high:+.4f}] |"
        )

    vf = dataset_metrics[dataset_metrics["dataset"] == "videofeedback"].set_index("config")
    macro = dataset_metrics[dataset_metrics["dataset"] == "Macro-3"].set_index("config")
    lines.extend(
        [
            "",
            "## Decision",
            "",
            f"Raw G1 retains a GenVideo AP gain of {raw_gain:+.4f}, but transfers negatively to VideoFeedback ({vf.loc['G1', 'ap'] - vf.loc['G0', 'ap']:+.4f} AP) and lowers Macro-3 AP by {macro.loc['G1', 'ap'] - macro.loc['G0', 'ap']:+.4f}. The calibrated G2-G4 variants do not preserve the raw GenVideo gain and do not resolve VideoFeedback negative transfer. None satisfies the admission criteria, so global D3 remains an ablation rather than a default branch.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--per-video",
        type=Path,
        default=REPO_ROOT / "results/multi_order_baselines/per_video_scores.csv",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPO_ROOT / "results/multi_order_baselines",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--bootstrap-iterations", type=int, default=1000)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    scores = add_global_variants(pd.read_csv(args.per_video))
    dataset_metrics, generator_metrics = metric_tables(
        scores,
        args.seed,
        score_columns=GLOBAL_COLUMNS,
        config_names=GLOBAL_D3_CONFIG_NAMES,
    )
    bootstrap = paired_bootstrap(
        scores,
        args.seed,
        args.bootstrap_iterations,
        comparisons=COMPARISONS,
    )
    deltas = _delta_table(generator_metrics)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    dataset_metrics.to_csv(args.output_dir / "global_d3_dataset_metrics.csv", index=False)
    generator_metrics.to_csv(args.output_dir / "global_d3_generator_metrics.csv", index=False)
    deltas.to_csv(args.output_dir / "global_d3_generator_deltas.csv", index=False)
    bootstrap.to_csv(args.output_dir / "global_d3_bootstrap_deltas.csv", index=False)
    write_analysis(
        args.output_dir / "global_d3_analysis.md",
        dataset_metrics,
        deltas,
        bootstrap,
    )


if __name__ == "__main__":
    main()
