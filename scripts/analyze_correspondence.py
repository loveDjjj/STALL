#!/usr/bin/env python3
"""汇总 Stage 1 C0-C3，并按预注册门槛判断 correspondence 是否继续。"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from evaluation.metrics import binary_metrics
from evaluation.tables import (
    _balanced_real_pair,
    build_metric_tables,
    build_pairwise_metric_table,
    normalize_scores,
)


RUNS = {
    "c0": "alpha_stall_full_d2_k3_no_spatial_refit",
    "c1": "stage1_c1_hard_local_d2_k3",
    "c2": "stage1_c2_soft_local_d2_k3",
    "c3": "stage1_c3_soft_local_d2_k3",
}
KEYS = ["video_id", "dataset", "subset", "source_model", "video_path"]


def _load_run(variant: str, run_name: str) -> tuple[pd.DataFrame, dict]:
    directory = ROOT / "results" / "runs" / run_name
    progress = json.loads((directory / "progress.json").read_text(encoding="utf-8"))
    if progress.get("status") != "completed":
        raise ValueError(f"{variant} 尚未完成：{run_name}")
    scores = normalize_scores(
        pd.read_csv(directory / "video_scores.csv", float_precision="round_trip")
    )
    manifest = json.loads((directory / "run_manifest.json").read_text(encoding="utf-8"))
    return scores, manifest


def _align(all_scores: dict[str, pd.DataFrame]) -> pd.DataFrame:
    merged = None
    for variant, scores in all_scores.items():
        score_columns = [
            column for column in ("final_score", "global_score", "local_score")
            if column in scores
        ]
        selected = scores[KEYS + score_columns].rename(
            columns={column: f"{column}_{variant}" for column in score_columns}
        )
        merged = selected if merged is None else merged.merge(
            selected, on=KEYS, how="inner", validate="one_to_one"
        )
    expected = {variant: len(scores) for variant, scores in all_scores.items()}
    if len(set(expected.values())) != 1 or len(merged) != next(iter(expected.values())):
        raise ValueError(f"C0-C3 视频身份不一致：counts={expected}, matched={len(merged)}")
    return merged


def _seed(seed: int, *parts: str) -> int:
    digest = hashlib.sha256(
        "\0".join((str(seed), *parts)).encode("utf-8")
    ).digest()
    return int.from_bytes(digest[:8], "little") % (2**32)


def _pairwise_bootstrap(
    aligned: pd.DataFrame,
    variant: str,
    *,
    seed: int,
    iterations: int,
) -> pd.DataFrame:
    """按论文真实配对规则，对 AUC/AP 数据集宏平均与 Macro-3 做成对重采样。"""

    rows = []
    dataset_samples: dict[str, dict[str, list[float]]] = {}
    for dataset, dataset_frame in aligned.groupby("dataset", sort=True):
        real = dataset_frame[dataset_frame["subset"].eq("real")]
        pairs = []
        for _, fake in dataset_frame[
            dataset_frame["subset"].eq("annotated")
        ].groupby("source_model", sort=True):
            pairs.append(_balanced_real_pair(real, fake, seed))
        rng = np.random.default_rng(_seed(seed, variant, dataset))
        samples = {"auc": [], "ap_real": []}
        candidate_column = f"final_score_{variant}"
        baseline_column = "final_score_c0"
        for _ in range(iterations):
            deltas = {"auc": [], "ap_real": []}
            for pair in pairs:
                sampled = pair.iloc[rng.integers(0, len(pair), len(pair))]
                if sampled["subset"].nunique() < 2:
                    continue
                candidate = binary_metrics(sampled, candidate_column)
                baseline = binary_metrics(sampled, baseline_column)
                deltas["auc"].append(candidate["auc"] - baseline["auc"])
                deltas["ap_real"].append(
                    candidate["real_positive_ap"] - baseline["real_positive_ap"]
                )
            for metric in samples:
                samples[metric].append(float(np.mean(deltas[metric])))
        dataset_samples[dataset] = samples
        for metric, values in samples.items():
            array = np.asarray(values, dtype=float)
            rows.append({
                "variant": variant,
                "scope": "generator_pairwise_dataset_macro",
                "dataset": dataset,
                "metric": metric,
                "delta_mean": float(array.mean()),
                "ci95_low": float(np.quantile(array, 0.025)),
                "ci95_high": float(np.quantile(array, 0.975)),
                "bootstrap_iterations": iterations,
            })
    for metric in ("auc", "ap_real"):
        macro = np.mean(
            np.stack(
                [dataset_samples[name][metric] for name in sorted(dataset_samples)]
            ),
            axis=0,
        )
        rows.append({
            "variant": variant,
            "scope": "generator_pairwise_macro3",
            "dataset": "Macro-3",
            "metric": metric,
            "delta_mean": float(macro.mean()),
            "ci95_low": float(np.quantile(macro, 0.025)),
            "ci95_high": float(np.quantile(macro, 0.975)),
            "bootstrap_iterations": iterations,
        })
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bootstrap-iterations", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    scores, manifests = {}, {}
    for variant, run_name in RUNS.items():
        scores[variant], manifests[variant] = _load_run(variant, run_name)
    aligned = _align(scores)
    output_dir = ROOT / "results" / "analysis" / "stage1_correspondence"
    output_dir.mkdir(parents=True, exist_ok=True)

    pairwise_rows = []
    branch_pairwise_rows = []
    deployment_rows = []
    generator_rows = []
    for variant, frame in scores.items():
        pairwise = build_pairwise_metric_table(frame, RUNS[variant], args.seed)
        pairwise.insert(0, "variant", variant)
        pairwise_rows.append(pairwise)
        for branch in ("global_score", "local_score"):
            if branch not in frame:
                continue
            branch_frame = frame.copy()
            branch_frame["final_score"] = branch_frame[branch]
            branch_table = build_pairwise_metric_table(
                branch_frame, RUNS[variant], args.seed
            )
            branch_table.insert(0, "branch", branch.removesuffix("_score"))
            branch_table.insert(0, "variant", variant)
            branch_pairwise_rows.append(branch_table)
        dataset, generator = build_metric_tables(frame, RUNS[variant])
        dataset.insert(0, "variant", variant)
        generator.insert(0, "variant", variant)
        deployment_rows.append(dataset)
        generator_rows.append(generator)
    pairwise_table = pd.concat(pairwise_rows, ignore_index=True)
    pairwise_table.to_csv(output_dir / "pairwise_metrics.csv", index=False)
    pd.concat(branch_pairwise_rows, ignore_index=True).to_csv(
        output_dir / "branch_pairwise_metrics.csv", index=False
    )
    pd.concat(deployment_rows, ignore_index=True).to_csv(
        output_dir / "deployment_metrics.csv", index=False
    )
    pd.concat(generator_rows, ignore_index=True).to_csv(
        output_dir / "generator_metrics.csv", index=False
    )

    macro = pairwise_table[pairwise_table["dataset"].eq("Macro-3")].set_index("variant")
    deltas = []
    bootstrap_rows = []
    for variant in ("c1", "c2", "c3"):
        auc_improvements = 0
        ap_improvements = 0
        for dataset in sorted(aligned["dataset"].unique()):
            candidate = pairwise_table[
                pairwise_table["variant"].eq(variant)
                & pairwise_table["dataset"].eq(dataset)
            ].iloc[0]
            baseline = pairwise_table[
                pairwise_table["variant"].eq("c0")
                & pairwise_table["dataset"].eq(dataset)
            ].iloc[0]
            auc_improvements += int(candidate["auc"] > baseline["auc"])
            ap_improvements += int(candidate["ap_real"] > baseline["ap_real"])
        auc_delta = float(macro.loc[variant, "auc"] - macro.loc["c0", "auc"])
        ap_delta = float(macro.loc[variant, "ap_real"] - macro.loc["c0", "ap_real"])
        deltas.append({
            "variant": variant,
            "macro_auc_delta_vs_c0": auc_delta,
            "macro_ap_delta_vs_c0": ap_delta,
            "datasets_with_auc_improvement": auc_improvements,
            "datasets_with_ap_improvement": ap_improvements,
            "passes_gate": bool(
                (auc_delta >= 0.005 and auc_improvements >= 2)
                or (ap_delta >= 0.005 and ap_improvements >= 2)
            ),
        })
        bootstrap_rows.append(
            _pairwise_bootstrap(
                aligned,
                variant,
                seed=args.seed,
                iterations=args.bootstrap_iterations,
            )
        )
    delta_table = pd.DataFrame(deltas)
    delta_table.to_csv(output_dir / "gate_decision.csv", index=False)
    pd.concat(bootstrap_rows, ignore_index=True).to_csv(
        output_dir / "paired_bootstrap.csv", index=False
    )

    efficiency = []
    for variant, manifest in manifests.items():
        datasets = manifest.get("pipeline", {}).get("datasets", {})
        efficiency.append({
            "variant": variant,
            "calibration_score_seconds": sum(
                float(value.get("calibration_score_seconds", 0.0)) for value in datasets.values()
            ),
            "evaluation_score_seconds": sum(
                float(value.get("evaluation_score_seconds", 0.0)) for value in datasets.values()
            ),
            "peak_vram_gib": max(
                [float(value.get("primary_peak_vram_gib", 0.0)) for value in datasets.values()]
                or [0.0]
            ),
        })
    pd.DataFrame(efficiency).to_csv(output_dir / "efficiency.csv", index=False)
    print(delta_table.to_string(index=False))
    print(f"结果已写入：{output_dir}")


if __name__ == "__main__":
    main()
