#!/usr/bin/env python3
"""汇总 Stage 3 D0-D3，并检验 speed-conditioned likelihood 的增益。"""

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


def _load_scores() -> dict[str, pd.DataFrame]:
    baseline = normalize_scores(pd.read_csv(
        ROOT / "results/runs/alpha_stall_full_d2_k3_no_spatial_refit/video_scores.csv",
        float_precision="round_trip",
    ))
    stage2 = pd.read_csv(
        ROOT / "results/runs/stage2_trajectory_matrix/video_scores.csv",
        float_precision="round_trip",
    )
    stage3 = pd.read_csv(
        ROOT / "results/runs/stage3_conditional_matrix/video_scores.csv",
        float_precision="round_trip",
    )
    return {
        "d0_unconditioned_d2": baseline,
        "d1_conditioned_d2": normalize_scores(
            stage3[stage3["variant"].eq("d1_conditioned_d2")].drop(columns="variant")
        ),
        "d2_unconditioned_geometry": normalize_scores(
            stage2[stage2["variant"].eq("t5_geometry")].drop(columns="variant")
        ),
        "d3_conditioned_geometry": normalize_scores(
            stage3[stage3["variant"].eq("d3_conditioned_geometry")].drop(columns="variant")
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bootstrap-iterations", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    progress = json.loads((
        ROOT / "results/runs/stage3_conditional_matrix/progress.json"
    ).read_text(encoding="utf-8"))
    if progress.get("status") != "completed":
        raise ValueError("Stage 3 尚未完成")

    scores = _load_scores()
    aligned = _align(scores)
    output = ROOT / "results/analysis/stage3_conditional"
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
    pairwise_table = pd.concat(pairwise_rows, ignore_index=True)
    pairwise_table.to_csv(output / "pairwise_metrics.csv", index=False)
    pd.concat(branch_rows, ignore_index=True).to_csv(
        output / "branch_pairwise_metrics.csv", index=False
    )
    pd.concat(dataset_rows, ignore_index=True).to_csv(
        output / "deployment_metrics.csv", index=False
    )
    pd.concat(generator_rows, ignore_index=True).to_csv(
        output / "generator_metrics.csv", index=False
    )

    comparisons = (
        ("d1_conditioned_d2", "d0_unconditioned_d2"),
        ("d3_conditioned_geometry", "d2_unconditioned_geometry"),
        ("d3_conditioned_geometry", "d0_unconditioned_d2"),
    )
    macro = pairwise_table[pairwise_table["dataset"].eq("Macro-3")].set_index("variant")
    decisions, bootstraps = [], []
    for candidate, baseline in comparisons:
        auc_count = 0
        ap_count = 0
        for dataset in ("comgenvid", "videofeedback", "genvideo"):
            rows = pairwise_table[pairwise_table["dataset"].eq(dataset)].set_index("variant")
            auc_count += int(rows.loc[candidate, "auc"] > rows.loc[baseline, "auc"])
            ap_count += int(rows.loc[candidate, "ap_real"] > rows.loc[baseline, "ap_real"])
        auc_delta = float(macro.loc[candidate, "auc"] - macro.loc[baseline, "auc"])
        ap_delta = float(macro.loc[candidate, "ap_real"] - macro.loc[baseline, "ap_real"])
        decisions.append({
            "comparison": f"{candidate}_vs_{baseline}",
            "macro_auc_delta": auc_delta,
            "macro_ap_delta": ap_delta,
            "datasets_with_auc_improvement": auc_count,
            "datasets_with_ap_improvement": ap_count,
            "passes_conditional_gate": bool(
                (auc_delta >= 0.005 and auc_count >= 2)
                or (ap_delta >= 0.005 and ap_count >= 2)
            ),
        })
        bootstraps.append(_pairwise_bootstrap(
            aligned,
            candidate,
            baseline_variant=baseline,
            seed=args.seed,
            iterations=args.bootstrap_iterations,
        ))
    decisions = pd.DataFrame(decisions)
    decisions.to_csv(output / "gate_decision.csv", index=False)
    pd.concat(bootstraps, ignore_index=True).to_csv(
        output / "paired_bootstrap.csv", index=False
    )

    boundary_rows = []
    run_root = ROOT / "results/runs/stage3_conditional_matrix/candidates"
    for candidate in ("d1_conditioned_d2", "d3_conditioned_geometry"):
        for path in sorted((run_root / candidate / "fitted_local").glob(
            "*_conditional_local_params.npz"
        )):
            with np.load(path, allow_pickle=False) as data:
                boundaries = np.asarray(data["boundaries"], dtype=np.float64)
            boundary_rows.append({
                "variant": candidate,
                "dataset": path.name.removesuffix("_conditional_local_params.npz"),
                "q33_speed": float(boundaries[0]),
                "q67_speed": float(boundaries[1]),
            })
    pd.DataFrame(boundary_rows).to_csv(output / "real_speed_boundaries.csv", index=False)
    print(decisions.to_string(index=False))
    print(f"结果已写入：{output}")


if __name__ == "__main__":
    main()
