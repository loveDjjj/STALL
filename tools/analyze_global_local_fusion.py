#!/usr/bin/env python3
"""Analyze fixed global-local fusion after global and local model selection."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
TOOLS_DIR = REPO_ROOT / "tools"
if str(TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(TOOLS_DIR))

from build_multi_order_baselines import _auc_ap, paired_bootstrap, pairwise_frames


KEY_COLUMNS = ["dataset", "subset", "source_model", "filename"]
LOCAL_CANDIDATES = ("L0", "L1", "L2", "L3", "P0")
ALPHAS = (0.2, 0.4, 0.5, 0.6, 0.8)


def load_scores(global_path: Path, local_path: Path) -> pd.DataFrame:
    global_scores = pd.read_csv(global_path)
    local_scores = pd.read_csv(local_path)
    keep_global = KEY_COLUMNS + ["B2", "B8"]
    missing = set(keep_global).difference(global_scores.columns)
    if missing:
        raise ValueError(f"global score CSV missing: {sorted(missing)}")
    merged = global_scores[keep_global].merge(
        local_scores,
        on=KEY_COLUMNS,
        how="inner",
        validate="one_to_one",
    )
    if len(merged) != len(global_scores) or len(merged) != len(local_scores):
        raise ValueError(
            f"global/local protocol mismatch: global={len(global_scores)} local={len(local_scores)} merged={len(merged)}"
        )
    return merged


def fusion_metrics(scores: pd.DataFrame, seed: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    generator_rows = []
    for dataset, dataset_scores in scores.groupby("dataset", sort=False):
        pairs = pairwise_frames(dataset_scores, seed)
        for generator, pair in pairs.items():
            for local in LOCAL_CANDIDATES:
                for alpha in ALPHAS:
                    pair_score = pair.copy()
                    pair_score["fusion"] = alpha * pair_score["B2"] + (1.0 - alpha) * pair_score[local]
                    auc, ap = _auc_ap(pair_score, "fusion")
                    generator_rows.append(
                        {
                            "dataset": dataset,
                            "generator": generator,
                            "local": local,
                            "alpha_global": alpha,
                            "auc": auc,
                            "ap": ap,
                        }
                    )
    generator = pd.DataFrame(generator_rows)
    dataset = (
        generator.groupby(["dataset", "local", "alpha_global"], as_index=False)
        .agg(n_generators=("generator", "nunique"), auc=("auc", "mean"), ap=("ap", "mean"))
    )
    return dataset, generator


def choose_protocols(dataset_metrics: pd.DataFrame) -> pd.DataFrame:
    rows = []
    datasets = tuple(dataset_metrics["dataset"].unique())

    worst = (
        dataset_metrics.groupby(["local", "alpha_global"], as_index=False)
        .agg(selection_score=("ap", "min"))
        .sort_values(["selection_score", "local", "alpha_global"], ascending=[False, True, True])
        .iloc[0]
    )
    for dataset in datasets:
        result = dataset_metrics[
            (dataset_metrics["dataset"] == dataset)
            & (dataset_metrics["local"] == worst["local"])
            & (dataset_metrics["alpha_global"] == worst["alpha_global"])
        ].iloc[0]
        rows.append(
            {
                "selection": "worst_case",
                "target_dataset": dataset,
                "local": worst["local"],
                "alpha_global": float(worst["alpha_global"]),
                "selection_ap": float(worst["selection_score"]),
                "target_auc": float(result["auc"]),
                "target_ap": float(result["ap"]),
            }
        )

    fixed_local = (
        dataset_metrics[dataset_metrics["alpha_global"] == 0.6]
        .groupby("local", as_index=False)
        .agg(selection_score=("ap", "mean"))
        .sort_values(["selection_score", "local"], ascending=[False, True])
        .iloc[0]
    )
    for dataset in datasets:
        result = dataset_metrics[
            (dataset_metrics["dataset"] == dataset)
            & (dataset_metrics["local"] == fixed_local["local"])
            & (dataset_metrics["alpha_global"] == 0.6)
        ].iloc[0]
        rows.append(
            {
                "selection": "fixed_alpha_0.6",
                "target_dataset": dataset,
                "local": fixed_local["local"],
                "alpha_global": 0.6,
                "selection_ap": float(fixed_local["selection_score"]),
                "target_auc": float(result["auc"]),
                "target_ap": float(result["ap"]),
            }
        )

    for target in datasets:
        train = dataset_metrics[dataset_metrics["dataset"] != target]
        selected = (
            train.groupby(["local", "alpha_global"], as_index=False)
            .agg(selection_score=("ap", "mean"))
            .sort_values(["selection_score", "local", "alpha_global"], ascending=[False, True, True])
            .iloc[0]
        )
        result = dataset_metrics[
            (dataset_metrics["dataset"] == target)
            & (dataset_metrics["local"] == selected["local"])
            & (dataset_metrics["alpha_global"] == selected["alpha_global"])
        ].iloc[0]
        rows.append(
            {
                "selection": "leave_one_dataset_out",
                "target_dataset": target,
                "local": selected["local"],
                "alpha_global": float(selected["alpha_global"]),
                "selection_ap": float(selected["selection_score"]),
                "target_auc": float(result["auc"]),
                "target_ap": float(result["ap"]),
            }
        )

        oracle = (
            dataset_metrics[dataset_metrics["dataset"] == target]
            .sort_values(["ap", "auc", "local", "alpha_global"], ascending=[False, False, True, True])
            .iloc[0]
        )
        rows.append(
            {
                "selection": "target_oracle_diagnostic",
                "target_dataset": target,
                "local": oracle["local"],
                "alpha_global": float(oracle["alpha_global"]),
                "selection_ap": float(oracle["ap"]),
                "target_auc": float(oracle["auc"]),
                "target_ap": float(oracle["ap"]),
            }
        )
    return pd.DataFrame(rows)


def add_main_score(scores: pd.DataFrame, selections: pd.DataFrame) -> tuple[pd.DataFrame, str, float]:
    fixed = selections[selections["selection"] == "fixed_alpha_0.6"]
    local = str(fixed.iloc[0]["local"])
    if fixed["local"].nunique() != 1:
        raise ValueError("fixed-alpha selection is not unique")
    out = scores.copy()
    out["final_selected"] = 0.6 * out["B2"] + 0.4 * out[local]
    return out, local, 0.6


def write_analysis(
    path: Path,
    dataset_metrics: pd.DataFrame,
    selections: pd.DataFrame,
    bootstrap: pd.DataFrame,
    selected_local: str,
) -> None:
    fixed = selections[selections["selection"] == "fixed_alpha_0.6"]
    lines = [
        "# Global-local fusion analysis",
        "",
        "## Protocol",
        "",
        "Global D3 G1-G4 failed the Stage-2 admission criteria, so the global branch is frozen as original STALL (B2). Fixed fusion scans only the specified alpha grid `{0.2, 0.4, 0.5, 0.6, 0.8}`. Local models are the leakage-free Stage-3 scores.",
        "",
        f"The fixed-alpha main rule selects `{selected_local}` by three-dataset mean AP at alpha 0.6, then uses `0.6 * B2 + 0.4 * {selected_local}` without target-specific tuning.",
        "",
        "## Selection protocols",
        "",
        "| Protocol | Target | Local | Alpha | Selection AP | Target AUC/AP |",
        "|---|---|---|---:|---:|---:|",
    ]
    for row in selections.itertuples(index=False):
        lines.append(
            f"| {row.selection} | {row.target_dataset} | {row.local} | {row.alpha_global:.1f} | "
            f"{row.selection_ap:.4f} | {row.target_auc:.4f}/{row.target_ap:.4f} |"
        )

    lines.extend(
        [
            "",
            "## Fixed alpha 0.6",
            "",
            "| Dataset | Local | AUC | AP |",
            "|---|---|---:|---:|",
        ]
    )
    for row in fixed.itertuples(index=False):
        lines.append(
            f"| {row.target_dataset} | {row.local} | {row.target_auc:.4f} | {row.target_ap:.4f} |"
        )

    lines.extend(
        [
            "",
            "## Paired bootstrap for selected fixed fusion",
            "",
            "| Dataset | Comparison | Metric | Delta | 95% CI |",
            "|---|---|---|---:|---:|",
        ]
    )
    for row in bootstrap.itertuples(index=False):
        lines.append(
            f"| {row.dataset} | {row.new_config}-{row.base_config} | {row.metric.upper()} | "
            f"{row.delta:+.4f} | [{row.ci95_low:+.4f}, {row.ci95_high:+.4f}] |"
        )
    lines.extend(
        [
            "",
            "## Reliability fusion decision",
            "",
            "A per-video inverse-variance reliability weight is not reported because the frozen protocol currently contains one compact 2 s patch window per video. Estimating variance from a single scalar would be invalid, and `|score - 0.5|` is explicitly disallowed. Reliability fusion therefore does not enter the method until multiple independently cached windows are available.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--global-scores",
        type=Path,
        default=REPO_ROOT / "results/multi_order_baselines/per_video_scores.csv",
    )
    parser.add_argument(
        "--local-scores",
        type=Path,
        default=REPO_ROOT / "results/multi_order_baselines/local_d2_per_video_scores.csv",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPO_ROOT / "results/multi_order_baselines",
    )
    parser.add_argument("--bootstrap-iterations", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    scores = load_scores(args.global_scores, args.local_scores)
    dataset_metrics, generator_metrics = fusion_metrics(scores, args.seed)
    selections = choose_protocols(dataset_metrics)
    selected, selected_local, _ = add_main_score(scores, selections)
    bootstrap = paired_bootstrap(
        selected,
        args.seed,
        args.bootstrap_iterations,
        comparisons=(
            ("final_selected", "B2", "selected_vs_original_stall"),
            ("final_selected", "B8", "selected_vs_current_alpha_stalled"),
        ),
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    dataset_metrics.to_csv(args.output_dir / "fusion_dataset_metrics.csv", index=False)
    generator_metrics.to_csv(args.output_dir / "fusion_generator_metrics.csv", index=False)
    selections.to_csv(args.output_dir / "fusion_selections.csv", index=False)
    bootstrap.to_csv(args.output_dir / "fusion_bootstrap_deltas.csv", index=False)
    selected[
        KEY_COLUMNS + ["B2", "B8", selected_local, "final_selected"]
    ].to_csv(args.output_dir / "fusion_selected_per_video_scores.csv", index=False)
    write_analysis(
        args.output_dir / "fusion_analysis.md",
        dataset_metrics,
        selections,
        bootstrap,
        selected_local,
    )


if __name__ == "__main__":
    main()
