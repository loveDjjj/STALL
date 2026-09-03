#!/usr/bin/env python3
"""汇总 Stage 2 T0-T5，并按预注册门槛判断 trajectory geometry。"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT / "src", ROOT / "scripts"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from analyze_correspondence import _align, _pairwise_bootstrap
from evaluation.tables import build_metric_tables, build_pairwise_metric_table, normalize_scores


BASELINE_RUN = "alpha_stall_full_d2_k3_no_spatial_refit"
MATRIX_RUN = "stage2_trajectory_matrix"
VARIANTS = (
    "t0_d2",
    "t1_curvature",
    "t2_speed_ratio",
    "t3_path_chord",
    "t4_d2_curvature",
    "t5_geometry",
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bootstrap-iterations", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    baseline_dir = ROOT / "results" / "runs" / BASELINE_RUN
    matrix_dir = ROOT / "results" / "runs" / MATRIX_RUN
    matrix_progress = json.loads((matrix_dir / "progress.json").read_text(encoding="utf-8"))
    if matrix_progress.get("status") != "completed":
        raise ValueError("Stage 2 trajectory matrix 尚未完成")
    baseline = normalize_scores(
        pd.read_csv(baseline_dir / "video_scores.csv", float_precision="round_trip")
    )
    matrix = pd.read_csv(matrix_dir / "video_scores.csv", float_precision="round_trip")
    all_scores = {"t0_d2": baseline}
    for variant, frame in matrix.groupby("variant", sort=False):
        all_scores[str(variant)] = normalize_scores(frame.drop(columns="variant"))
    if set(all_scores) != set(VARIANTS):
        raise ValueError(f"Stage 2 候选不完整：{sorted(all_scores)}")
    aligned = _align(all_scores)

    output = ROOT / "results" / "analysis" / "stage2_trajectory"
    output.mkdir(parents=True, exist_ok=True)
    pairwise_rows, branch_rows, deployment_rows, generator_rows = [], [], [], []
    for variant in VARIANTS:
        frame = all_scores[variant]
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
        deployment_rows.append(dataset)
        generator_rows.append(generator)
    pairwise_table = pd.concat(pairwise_rows, ignore_index=True)
    pairwise_table.to_csv(output / "pairwise_metrics.csv", index=False)
    pd.concat(branch_rows, ignore_index=True).to_csv(
        output / "branch_pairwise_metrics.csv", index=False
    )
    pd.concat(deployment_rows, ignore_index=True).to_csv(
        output / "deployment_metrics.csv", index=False
    )
    pd.concat(generator_rows, ignore_index=True).to_csv(
        output / "generator_metrics.csv", index=False
    )

    macro = pairwise_table[pairwise_table["dataset"].eq("Macro-3")].set_index("variant")
    decisions, bootstraps = [], []
    for variant in VARIANTS[1:]:
        auc_improvements = 0
        ap_improvements = 0
        for dataset in ("comgenvid", "videofeedback", "genvideo"):
            candidate = pairwise_table[
                pairwise_table["variant"].eq(variant)
                & pairwise_table["dataset"].eq(dataset)
            ].iloc[0]
            baseline_row = pairwise_table[
                pairwise_table["variant"].eq("t0_d2")
                & pairwise_table["dataset"].eq(dataset)
            ].iloc[0]
            auc_improvements += int(candidate["auc"] > baseline_row["auc"])
            ap_improvements += int(candidate["ap_real"] > baseline_row["ap_real"])
        auc_delta = float(macro.loc[variant, "auc"] - macro.loc["t0_d2", "auc"])
        ap_delta = float(macro.loc[variant, "ap_real"] - macro.loc["t0_d2", "ap_real"])
        decisions.append({
            "variant": variant,
            "macro_auc_delta_vs_t0": auc_delta,
            "macro_ap_delta_vs_t0": ap_delta,
            "datasets_with_auc_improvement": auc_improvements,
            "datasets_with_ap_improvement": ap_improvements,
            "passes_mainline_gate": bool(
                (auc_delta >= 0.005 and auc_improvements >= 2)
                or (ap_delta >= 0.005 and ap_improvements >= 2)
            ),
            "passes_low_dim_retention_gate": bool(
                variant == "t5_geometry" and auc_delta >= -0.002 and ap_delta >= -0.002
            ),
        })
        bootstraps.append(
            _pairwise_bootstrap(
                aligned,
                variant,
                baseline_variant="t0_d2",
                seed=args.seed,
                iterations=args.bootstrap_iterations,
            )
        )
    decision_table = pd.DataFrame(decisions)
    decision_table.to_csv(output / "gate_decision.csv", index=False)
    pd.concat(bootstraps, ignore_index=True).to_csv(
        output / "paired_bootstrap.csv", index=False
    )

    manifest = json.loads((matrix_dir / "run_manifest.json").read_text(encoding="utf-8"))
    datasets = manifest["pipeline"]["datasets"]
    pd.DataFrame([{
        "matrix_run": MATRIX_RUN,
        "candidate_count": len(VARIANTS) - 1,
        "evaluation_videos_per_candidate": len(baseline),
        "score_seconds_all_candidates": sum(
            float(item["score_seconds_all_candidates"]) for item in datasets.values()
        ),
        "peak_vram_gib": max(float(item["primary_peak_vram_gib"]) for item in datasets.values()),
        "result_size_bytes": sum(
            path.stat().st_size for path in matrix_dir.rglob("*") if path.is_file()
        ),
    }]).to_csv(output / "efficiency.csv", index=False)
    print(decision_table.to_string(index=False))
    print(f"结果已写入：{output}")


if __name__ == "__main__":
    main()
