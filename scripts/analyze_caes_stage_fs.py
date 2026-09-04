#!/usr/bin/env python3
"""汇总CAES Stage FS的分支、低FPR、覆盖和effective-K审计。"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from evaluation.tables import build_metric_tables
from temporal_selection.evaluation import build_matched_selector_pairwise_table


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run-dir", type=Path, default=ROOT / "results/runs/caes_stage_fs"
    )
    parser.add_argument(
        "--output-dir", type=Path, default=ROOT / "results/analysis/caes_stage_fs"
    )
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    videos = pd.read_csv(
        args.run_dir / "video_scores.csv", float_precision="round_trip"
    )
    diagnostics = pd.read_csv(args.run_dir / "selection_diagnostics.csv")
    pairwise = pd.read_csv(
        args.run_dir / "pairwise_metrics.csv", float_precision="round_trip"
    )
    baseline = pairwise[pairwise["selector"].eq("uniform")][
        ["scope", "dataset", "auc", "ap_real"]
    ].rename(columns={"auc": "baseline_auc", "ap_real": "baseline_ap_real"})
    deltas = pairwise.merge(
        baseline, on=["scope", "dataset"], how="left", validate="many_to_one"
    )
    deltas["auc_delta_vs_uniform"] = deltas["auc"] - deltas["baseline_auc"]
    deltas["ap_delta_vs_uniform"] = deltas["ap_real"] - deltas["baseline_ap_real"]
    deltas.to_csv(args.output_dir / "pairwise_deltas.csv", index=False)

    branch_rows = []
    for branch in ("global_score", "local_score"):
        branch_scores = videos.copy()
        branch_scores["final_score"] = branch_scores[branch]
        table = build_matched_selector_pairwise_table(branch_scores)
        table.insert(0, "branch", branch.removesuffix("_score"))
        branch_rows.append(table)
    pd.concat(branch_rows, ignore_index=True).to_csv(
        args.output_dir / "branch_pairwise_metrics.csv", index=False
    )

    deployment_rows = []
    for selector, frame in videos.groupby("selector", sort=False):
        dataset, _ = build_metric_tables(frame.drop(columns="selector"), selector)
        dataset.insert(0, "selector", selector)
        deployment_rows.append(dataset)
    deployment = pd.concat(deployment_rows, ignore_index=True)
    baseline_deployment = deployment[deployment["selector"].eq("uniform")].drop(
        columns=["selector", "run_name"]
    )
    deployment_delta = deployment.merge(
        baseline_deployment,
        on="dataset",
        suffixes=("", "__uniform"),
        how="left",
        validate="many_to_one",
    )
    for metric in (
        "auc", "ap_real", "tpr_at_0_1pct_fpr", "tpr_at_1pct_fpr",
        "fpr_at_95pct_tpr",
    ):
        deployment_delta[f"{metric}_delta_vs_uniform"] = (
            deployment_delta[metric] - deployment_delta[f"{metric}__uniform"]
        )
    deployment_delta.to_csv(
        args.output_dir / "deployment_deltas.csv", index=False
    )

    evaluation = diagnostics[diagnostics["split"].eq("evaluation")]
    coverage = evaluation.groupby(["selector", "dataset"], as_index=False).agg(
        videos=("video_id", "size"),
        effective_k_mean=("effective_k", "mean"),
        center_span_mean_seconds=("selected_center_span_seconds", "mean"),
        minimum_distance_mean_seconds=(
            "selected_pairwise_distance_min_seconds", "mean"
        ),
        selected_unique_frames_mean=("selected_unique_dense_frames", "mean"),
    )
    coverage.to_csv(args.output_dir / "selection_coverage.csv", index=False)

    wide_k = evaluation.pivot(
        index=["dataset", "video_id"], columns="selector", values="effective_k"
    )
    selectors = [item for item in wide_k.columns if item != "uniform"]
    parity = wide_k[selectors].eq(wide_k["uniform"], axis=0).all(axis=1)
    audit = wide_k.assign(all_selector_k_matches_uniform=parity).reset_index()
    audit.to_csv(args.output_dir / "effective_k_audit.csv", index=False)

    parity_tables = []
    parity_counts = []
    for selector in selectors:
        selector_parity = wide_k[selector].eq(wide_k["uniform"])
        keep_ids = {
            video_id
            for (_dataset, video_id), matches in selector_parity.items()
            if bool(matches)
        }
        parity_scores = videos[
            videos["selector"].isin(["uniform", selector])
            & videos["video_id"].isin(keep_ids)
        ].copy()
        full_table = build_matched_selector_pairwise_table(parity_scores)
        parity_baseline = full_table[
            full_table["selector"].eq("uniform")
        ][["scope", "dataset", "auc", "ap_real"]].rename(columns={
            "auc": "uniform_auc", "ap_real": "uniform_ap_real"
        })
        table = full_table[full_table["selector"].eq(selector)].copy().merge(
            parity_baseline,
            on=["scope", "dataset"],
            how="left",
            validate="one_to_one",
        )
        table["auc_delta_vs_uniform"] = table["auc"] - table["uniform_auc"]
        table["ap_delta_vs_uniform"] = table["ap_real"] - table["uniform_ap_real"]
        table.insert(0, "audited_selector", selector)
        parity_tables.append(table)
        parity_counts.append({
            "selector": selector,
            "matching_videos": len(keep_ids),
            "total_videos": wide_k.index.nunique(),
            "mismatching_videos": wide_k.index.nunique() - len(keep_ids),
        })
    pd.concat(parity_tables, ignore_index=True).to_csv(
        args.output_dir / "pairwise_metrics_equal_effective_k.csv", index=False
    )
    pd.DataFrame(parity_counts).to_csv(
        args.output_dir / "effective_k_parity_counts.csv", index=False
    )
    print(
        deltas[deltas["dataset"].eq("Macro-3")][
            ["selector", "auc", "ap_real", "auc_delta_vs_uniform", "ap_delta_vs_uniform"]
        ].to_string(index=False)
    )
    print(
        "\neffective-K逐selector一致性：\n"
        + pd.DataFrame(parity_counts).to_string(index=False)
    )


if __name__ == "__main__":
    main()
