#!/usr/bin/env python3
"""汇总 Stage 4 S0-S2，并检验 shrinkage covariance 的稳定性。"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT / "src", ROOT / "scripts"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from analyze_correspondence import _align, _pairwise_bootstrap
from evaluation.tables import build_metric_tables, build_pairwise_metric_table, normalize_scores


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bootstrap-iterations", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    baseline_dir = ROOT / "results/runs/alpha_stall_full_d2_k3_no_spatial_refit"
    matrix_dir = ROOT / "results/runs/stage4_statistics_matrix"
    progress = json.loads((matrix_dir / "progress.json").read_text(encoding="utf-8"))
    if progress.get("status") != "completed":
        raise ValueError("Stage 4 statistics matrix 尚未完成")
    baseline = normalize_scores(pd.read_csv(
        baseline_dir / "video_scores.csv", float_precision="round_trip"
    ))
    matrix = pd.read_csv(matrix_dir / "video_scores.csv", float_precision="round_trip")
    scores = {"s0_empirical_d2": baseline}
    for variant, frame in matrix.groupby("variant", sort=False):
        scores[str(variant)] = normalize_scores(frame.drop(columns="variant"))
    aligned = _align(scores)
    output = ROOT / "results/analysis/stage4_statistics"
    output.mkdir(parents=True, exist_ok=True)

    pairwise_rows, branch_rows, dataset_rows, generator_rows = [], [], [], []
    for variant, frame in scores.items():
        pairwise = build_pairwise_metric_table(frame, variant, args.seed)
        pairwise.insert(0, "variant", variant)
        pairwise_rows.append(pairwise)
        for branch in ("global_score", "local_score"):
            branch_frame = frame.copy()
            branch_frame["final_score"] = branch_frame[branch]
            table = build_pairwise_metric_table(branch_frame, variant, args.seed)
            table.insert(0, "branch", branch.removesuffix("_score"))
            table.insert(0, "variant", variant)
            branch_rows.append(table)
        dataset, generator = build_metric_tables(frame, variant)
        dataset.insert(0, "variant", variant)
        generator.insert(0, "variant", variant)
        dataset_rows.append(dataset)
        generator_rows.append(generator)
    pairwise = pd.concat(pairwise_rows, ignore_index=True)
    pairwise.to_csv(output / "pairwise_metrics.csv", index=False)
    pd.concat(branch_rows, ignore_index=True).to_csv(
        output / "branch_pairwise_metrics.csv", index=False
    )
    pd.concat(dataset_rows, ignore_index=True).to_csv(
        output / "deployment_metrics.csv", index=False
    )
    pd.concat(generator_rows, ignore_index=True).to_csv(
        output / "generator_metrics.csv", index=False
    )

    macro = pairwise[pairwise["dataset"].eq("Macro-3")].set_index("variant")
    comparisons = (
        ("s1_ledoit_wolf_d2", "s0_empirical_d2"),
        ("s2_oas_d2", "s0_empirical_d2"),
        ("s2_oas_d2", "s1_ledoit_wolf_d2"),
    )
    decisions, bootstraps = [], []
    for candidate, baseline_name in comparisons:
        auc_delta = float(macro.loc[candidate, "auc"] - macro.loc[baseline_name, "auc"])
        ap_delta = float(macro.loc[candidate, "ap_real"] - macro.loc[baseline_name, "ap_real"])
        decisions.append({
            "comparison": f"{candidate}_vs_{baseline_name}",
            "macro_auc_delta": auc_delta,
            "macro_ap_delta": ap_delta,
            "passes_mainline_gate": bool(max(auc_delta, ap_delta) >= 0.005),
        })
        bootstraps.append(_pairwise_bootstrap(
            aligned,
            candidate,
            baseline_variant=baseline_name,
            seed=args.seed,
            iterations=args.bootstrap_iterations,
        ))
    decision_table = pd.DataFrame(decisions)
    decision_table.to_csv(output / "gate_decision.csv", index=False)
    pd.concat(bootstraps, ignore_index=True).to_csv(
        output / "paired_bootstrap.csv", index=False
    )

    shrinkage_rows = []
    for candidate_dir in sorted((matrix_dir / "candidates").iterdir()):
        for path in sorted((candidate_dir / "fitted_local").glob("*_local_params.npz")):
            with np.load(path, allow_pickle=False) as data:
                value = float(data["patch_temporal_shrinkage"][0])
            shrinkage_rows.append({
                "variant": candidate_dir.name,
                "dataset": path.name.removesuffix("_local_params.npz"),
                "shrinkage": value,
            })
    pd.DataFrame(shrinkage_rows).to_csv(output / "shrinkage_coefficients.csv", index=False)
    print(decision_table.to_string(index=False))
    print(f"结果已写入：{output}")


if __name__ == "__main__":
    main()
