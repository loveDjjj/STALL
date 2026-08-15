#!/usr/bin/env python3
"""Audit full-coverage and paper-count evaluation protocols without rescoring videos."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from alpha_stalled.duration_aware_metrics import (
    BRANCHES,
    DATASETS,
    PAPER_FAKE_COUNTS,
    balanced_pair_frame,
    balanced_real_ids,
    duration_matched_real,
    metric,
    proportional_allocation,
    select_paper_fake_cohort,
    stable_rank,
)


def aggregate_metric_rows(rows: list[dict], scope: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    generators = pd.DataFrame(rows)
    datasets = (
        generators.groupby(["branch", "dataset"], as_index=False)
        .agg(
            auc=("auc", "mean"),
            ap_raw=("ap_raw", "mean"),
            ap_std50=("ap_std50", "mean"),
            fake_ap_std50=("fake_ap_std50", "mean"),
            generators=("source_model", "size"),
            fake_videos=("n_fake", "sum"),
        )
    )
    macro3 = (
        datasets.groupby("branch", as_index=False)
        .agg(
            auc=("auc", "mean"),
            ap_raw=("ap_raw", "mean"),
            ap_std50=("ap_std50", "mean"),
            fake_ap_std50=("fake_ap_std50", "mean"),
            generators=("generators", "sum"),
            fake_videos=("fake_videos", "sum"),
        )
    )
    macro3["dataset"] = "Macro-3"
    all_generators = (
        generators.groupby("branch", as_index=False)
        .agg(
            auc=("auc", "mean"),
            ap_raw=("ap_raw", "mean"),
            ap_std50=("ap_std50", "mean"),
            fake_ap_std50=("fake_ap_std50", "mean"),
            generators=("source_model", "size"),
            fake_videos=("n_fake", "sum"),
        )
    )
    all_generators["dataset"] = "All-23"
    summary = pd.concat([datasets, macro3, all_generators], ignore_index=True)
    generators.insert(0, "scope", scope)
    summary.insert(0, "scope", scope)
    return generators, summary


def coverage_metrics(
    scores: pd.DataFrame, fake_cohort: pd.DataFrame, scope: str
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    for (dataset, source_model), fake in fake_cohort.groupby(
        ["dataset", "source_model"], sort=True
    ):
        real = scores[
            scores["dataset"].eq(dataset) & scores["subset"].eq("real")
        ]
        chosen_real = duration_matched_real(
            real, fake, f"{scope}:{dataset}:{source_model}:real-duration"
        )
        for branch in BRANCHES:
            values = metric(chosen_real[branch].to_numpy(), fake[branch].to_numpy())
            rows.append(
                {
                    "dataset": dataset,
                    "source_model": source_model,
                    "branch": branch,
                    "n_real": len(chosen_real),
                    "n_fake": len(fake),
                    **values,
                }
            )
    return aggregate_metric_rows(rows, scope)


def balanced_metrics(
    scores: pd.DataFrame,
    fake_cohort: pd.DataFrame,
    scope: str,
    seed: int,
    selection_protocol: str | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    for (dataset, source_model), fake in fake_cohort.groupby(
        ["dataset", "source_model"], sort=True
    ):
        real = scores[
            scores["dataset"].eq(dataset) & scores["subset"].eq("real")
        ]
        protocol = selection_protocol or scope
        chosen_real, chosen_fake = balanced_pair_frame(
            real, fake, f"{seed}:{protocol}:{dataset}:{source_model}"
        )
        for branch in BRANCHES:
            values = metric(
                chosen_real[branch].to_numpy(), chosen_fake[branch].to_numpy()
            )
            rows.append(
                {
                    "dataset": dataset,
                    "source_model": source_model,
                    "branch": branch,
                    "n_real": len(chosen_real),
                    "n_fake": len(chosen_fake),
                    **values,
                }
            )
    return aggregate_metric_rows(rows, scope)


def seed_sensitivity(
    scores: pd.DataFrame, fake: pd.DataFrame, seeds: range
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    for seed in seeds:
        _, summary = balanced_metrics(
            scores, fake, "full23_balanced", seed, selection_protocol="full23"
        )
        selected = summary[summary["dataset"].isin([*DATASETS, "Macro-3"])]
        selected = selected.copy()
        selected.insert(0, "seed", seed)
        rows.append(selected)
    metrics = pd.concat(rows, ignore_index=True)
    macro = metrics[metrics["dataset"].eq("Macro-3")]
    pivot = macro.pivot(
        index="seed", columns="branch", values=["auc", "ap_std50", "fake_ap_std50"]
    )
    deltas = pd.DataFrame(
        {
            "seed": pivot.index,
            "delta_auc_S_minus_G": pivot[("auc", "S")] - pivot[("auc", "G")],
            "delta_ap_S_minus_G": pivot[("ap_std50", "S")]
            - pivot[("ap_std50", "G")],
            "delta_fake_ap_S_minus_G": pivot[("fake_ap_std50", "S")]
            - pivot[("fake_ap_std50", "G")],
        }
    ).reset_index(drop=True)
    return metrics, deltas


def distribution_summary(scores: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for branch in BRANCHES:
        real = scores[scores["subset"].eq("real")]
        fake = scores[scores["subset"].eq("annotated")]
        groups = [
            ("real", real.groupby(["dataset", "protocol_duration_sec"], sort=True)),
            ("fake", fake.groupby(["dataset", "source_model"], sort=True)),
        ]
        for subset, grouped in groups:
            for keys, frame in grouped:
                values = frame[branch].to_numpy(dtype=np.float64)
                dataset = str(keys[0])
                source = str(keys[1]) if subset == "fake" else f"duration_{int(keys[1])}s"
                rows.append(
                    {
                        "subset": subset,
                        "dataset": dataset,
                        "source_or_duration": source,
                        "branch": branch,
                        "n": len(values),
                        "mean": float(np.mean(values)),
                        "std": float(np.std(values)),
                        "q05": float(np.quantile(values, 0.05)),
                        "q25": float(np.quantile(values, 0.25)),
                        "median": float(np.quantile(values, 0.50)),
                        "q75": float(np.quantile(values, 0.75)),
                        "q95": float(np.quantile(values, 0.95)),
                    }
                )
    return pd.DataFrame(rows)


def failure_analysis(
    fake: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    output = fake[
        [
            "video_id",
            "dataset",
            "source_model",
            "filename",
            "duration_seconds",
            "protocol_duration_sec",
            "effective_k",
            "G",
            "L",
            "S",
        ]
    ].copy()
    output["delta_S_minus_G"] = output["S"] - output["G"]
    output["failure_type"] = np.select(
        [
            (output["G"] >= 0.5) & (output["S"] < 0.5),
            (output["G"] < 0.5) & (output["S"] >= 0.5),
            (output["G"] >= 0.5) & (output["S"] >= 0.5),
        ],
        ["local_rescue", "local_harm", "both_miss"],
        default="both_detect",
    )
    summary = (
        output.groupby(["dataset", "source_model", "failure_type"], as_index=False)
        .agg(videos=("video_id", "size"), mean_delta=("delta_S_minus_G", "mean"))
    )
    totals = output.groupby(["dataset", "source_model"])["video_id"].size().rename("total")
    summary = summary.merge(totals, on=["dataset", "source_model"])
    summary["rate"] = summary["videos"] / summary["total"]
    priority = output[output["failure_type"].isin(["local_harm", "both_miss"])].copy()
    priority = priority.sort_values(
        ["failure_type", "S", "delta_S_minus_G"],
        ascending=[True, False, False],
    )
    return output, summary, priority.head(500)


def report(
    coverage_summary: pd.DataFrame,
    paper_summary: pd.DataFrame,
    balanced_summary: pd.DataFrame,
    seed_delta: pd.DataFrame,
    fake: pd.DataFrame,
    output: Path,
) -> None:
    def row(frame: pd.DataFrame, dataset: str, branch: str) -> pd.Series:
        return frame[(frame["dataset"] == dataset) & (frame["branch"] == branch)].iloc[0]

    lines = [
        "# Full-coverage and paper-count protocol audit",
        "",
        "This analysis reuses frozen duration-aware per-video scores. No DINO inference, "
        "whitening fit, CDF fit, or generated-video parameter selection is performed.",
        "",
        "## Full-coverage results",
        "",
        "All 45,185 generated videos are included. AP_std50 uses sample weights so real and "
        "fake each contribute total weight 0.5; raw AP is retained only as a prevalence diagnostic. "
        "AP_fake matches the positive-class convention stated in the original STALL paper, while "
        "AP_real preserves this repository's historical convention.",
        "",
        "| Scope | Global AUC/AP_fake/AP_real | Alpha AUC/AP_fake/AP_real | Delta AUC/AP_fake/AP_real |",
        "|---|---:|---:|---:|",
    ]
    for dataset in (*DATASETS, "Macro-3", "All-23"):
        g = row(coverage_summary, dataset, "G")
        s = row(coverage_summary, dataset, "S")
        lines.append(
            f"| {dataset} | {g.auc:.4f}/{g.fake_ap_std50:.4f}/{g.ap_std50:.4f} | "
            f"{s.auc:.4f}/{s.fake_ap_std50:.4f}/{s.ap_std50:.4f} | "
            f"{s.auc-g.auc:+.4f}/{s.fake_ap_std50-g.fake_ap_std50:+.4f}/"
            f"{s.ap_std50-g.ap_std50:+.4f} |"
        )
    lines.extend(
        [
            "",
            "## Paper-count-matched fake cohort",
            "",
            f"The deterministic cohort contains {len(fake):,} generated videos, matching the "
            "per-generator counts in the STALL supplementary tables. Video identities cannot "
            "be claimed identical because the original complete manifests are unavailable. "
            "Target-domain calibration remains disjoint, so the original paper's full real counts "
            "cannot simultaneously be reused for Alpha-STALLED without leakage.",
            "",
            "| Scope | Global AUC/AP_fake/AP_real | Alpha AUC/AP_fake/AP_real | Delta AUC/AP_fake/AP_real |",
            "|---|---:|---:|---:|",
        ]
    )
    for dataset in (*DATASETS, "Macro-3", "All-23"):
        g = row(paper_summary, dataset, "G")
        s = row(paper_summary, dataset, "S")
        lines.append(
            f"| {dataset} | {g.auc:.4f}/{g.fake_ap_std50:.4f}/{g.ap_std50:.4f} | "
            f"{s.auc:.4f}/{s.fake_ap_std50:.4f}/{s.ap_std50:.4f} | "
            f"{s.auc-g.auc:+.4f}/{s.fake_ap_std50-g.fake_ap_std50:+.4f}/"
            f"{s.ap_std50-g.ap_std50:+.4f} |"
        )
    lines.extend(
        [
            "",
            "## Balanced primary and seed sensitivity",
            "",
            "The balanced row uses the current disjoint-real constraints. One hundred deterministic "
            "selection seeds change only which eligible real/fake identities enter each generator "
            "comparison; scores and parameters stay frozen.",
            "",
        ]
    )
    g = row(balanced_summary, "Macro-3", "G")
    s = row(balanced_summary, "Macro-3", "S")
    lines.extend(
        [
            f"Seed 42 Macro-3 AUC/AP_fake/AP_real is Global "
            f"{g.auc:.4f}/{g.fake_ap_std50:.4f}/{g.ap_std50:.4f} and Alpha "
            f"{s.auc:.4f}/{s.fake_ap_std50:.4f}/{s.ap_std50:.4f}.",
            "",
            f"Across 100 seeds, Alpha-minus-Global AP_real has mean "
            f"{seed_delta.delta_ap_S_minus_G.mean():+.4f}, standard deviation "
            f"{seed_delta.delta_ap_S_minus_G.std(ddof=1):.4f}, and range "
            f"[{seed_delta.delta_ap_S_minus_G.min():+.4f}, "
            f"{seed_delta.delta_ap_S_minus_G.max():+.4f}].",
            f"Alpha-minus-Global AP_fake has mean "
            f"{seed_delta.delta_fake_ap_S_minus_G.mean():+.4f}, standard deviation "
            f"{seed_delta.delta_fake_ap_S_minus_G.std(ddof=1):.4f}, and range "
            f"[{seed_delta.delta_fake_ap_S_minus_G.min():+.4f}, "
            f"{seed_delta.delta_fake_ap_S_minus_G.max():+.4f}].",
            "",
            "Bootstrap and selection-seed sensitivity answer different questions. Bootstrap estimates "
            "finite-sample uncertainty within a fixed cohort; seed sensitivity measures dependence on "
            "which eligible identities were selected. Neither creates coverage of unseen videos.",
            "",
            "## Artifacts",
            "",
            "- `full_coverage_generator_metrics.csv` and `full_coverage_summary.csv`",
            "- `paper_count_fake_manifest.csv` and `paper_count_summary.csv`",
            "- `balanced_seed_metrics.csv` and `balanced_seed_deltas.csv`",
            "- `score_distribution_summary.csv`, `all_fake_scores.csv`, and `failure_cases.csv`",
        ]
    )
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run(args: argparse.Namespace) -> None:
    scores = pd.read_csv(args.scores, float_precision="round_trip")
    scores = scores[
        scores["candidate"].eq("ncustom")
        & scores["protocol_split"].eq("evaluation")
    ].copy()
    if scores.duplicated(["video_id", "protocol_duration_sec"]).any():
        raise ValueError("duplicate candidate/video/duration score rows")
    fake = scores[scores["subset"].eq("annotated")].copy()
    if fake["video_id"].duplicated().any() or len(fake) != 45185:
        raise ValueError(f"expected 45,185 unique generated scores, found {len(fake)}")

    paper_fake = select_paper_fake_cohort(fake)
    full_generators, full_summary = coverage_metrics(scores, fake, "full_coverage")
    paper_generators, paper_summary = coverage_metrics(
        scores, paper_fake, "paper_count_fake_cohort"
    )
    balanced_generators, balanced_summary = balanced_metrics(
        scores,
        fake,
        "full23_balanced",
        args.primary_seed,
        selection_protocol="full23",
    )
    seed_metrics, seed_deltas = seed_sensitivity(
        scores, fake, range(args.seed_count)
    )
    distributions = distribution_summary(scores)
    all_fake, failure_summary, failures = failure_analysis(fake)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    artifacts = {
        "full_coverage_generator_metrics.csv": full_generators,
        "full_coverage_summary.csv": full_summary,
        "paper_count_fake_manifest.csv": paper_fake[
            [
                "video_id",
                "dataset",
                "source_model",
                "filename",
                "protocol_duration_sec",
                "paper_rank",
            ]
        ],
        "paper_count_generator_metrics.csv": paper_generators,
        "paper_count_summary.csv": paper_summary,
        "balanced_seed42_generator_metrics.csv": balanced_generators,
        "balanced_seed42_summary.csv": balanced_summary,
        "balanced_seed_metrics.csv": seed_metrics,
        "balanced_seed_deltas.csv": seed_deltas,
        "score_distribution_summary.csv": distributions,
        "all_fake_scores.csv": all_fake,
        "failure_summary.csv": failure_summary,
        "failure_cases.csv": failures,
    }
    for name, frame in artifacts.items():
        frame.to_csv(args.output_dir / name, index=False)
    report(
        full_summary,
        paper_summary,
        balanced_summary,
        seed_deltas,
        paper_fake,
        args.report,
    )
    print(full_summary[full_summary["dataset"].isin([*DATASETS, "Macro-3", "All-23"])].to_string(index=False))
    print(f"wrote {len(artifacts)} artifacts and {args.report}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--scores",
        type=Path,
        default=ROOT / "results/duration_aware_23source/per_video_scores.csv",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "results/full_coverage_paper_protocol",
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=ROOT / "reports/full_coverage_paper_protocol.md",
    )
    parser.add_argument("--primary-seed", type=int, default=42)
    parser.add_argument("--seed-count", type=int, default=100)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
