#!/usr/bin/env python3
"""Compare Global/Alpha and original-K1/uniform-K3 on identical full-23 scores."""

from __future__ import annotations

import argparse
import glob
import hashlib
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score


ROOT = Path(__file__).resolve().parents[1]
import sys

SRC_DIR = ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from alpha_stalled.calibration import (  # noqa: E402
    cdf_with_positive_infinity as cdf_temporal,
)
from alpha_stalled.duration_aware_metrics import (  # noqa: E402
    DATASETS,
    balanced_real_ids,
    balanced_pair_frame,
    duration_matched_real,
    metric,
    select_paper_fake_cohort,
)
from alpha_stalled.parameters import global_references as global_vatex_references  # noqa: E402
from alpha_stalled.whitening import empirical_cdf_right_inclusive, stable_sorted  # noqa: E402


CONFIGS = ("K1_STALL", "K1_G", "K1_S", "K3_G", "K3_S")
CONTRASTS = {
    "duration_calibration_at_K1": ("K1_G", "K1_STALL"),
    "local_at_K1": ("K1_S", "K1_G"),
    "local_at_K3": ("K3_S", "K3_G"),
    "window_for_Alpha": ("K3_S", "K1_S"),
    "total_K3_Alpha_vs_K1_Global": ("K3_S", "K1_G"),
    "total_K3_Alpha_vs_original_STALL": ("K3_S", "K1_STALL"),
}


def load_k1_raw(directory: Path) -> pd.DataFrame:
    paths = [directory.parent / "original_k1_raw_reused.csv"] + sorted(
        Path(path) for path in glob.glob(str(directory / "*.csv"))
    )
    if not all(path.is_file() for path in paths):
        missing = [str(path) for path in paths if not path.is_file()]
        raise FileNotFoundError(f"missing K1 raw files: {missing}")
    frame = pd.concat(
        [pd.read_csv(path, float_precision="round_trip") for path in paths],
        ignore_index=True,
    )
    if len(frame) != 75011 or frame["k1_task_id"].duplicated().any():
        raise ValueError(f"expected 75,011 unique K1 raw tasks, found {len(frame)}")
    return frame


def selected_ids(size_membership: pd.DataFrame, dataset: str, size: int) -> set[str]:
    selected = size_membership[
        size_membership["dataset"].eq(dataset)
        & size_membership["calibration_size"].eq(size)
    ]["video_id"].astype(str)
    if len(selected) != size:
        raise ValueError(f"{dataset}: expected {size} selected calibration IDs")
    return set(selected)


def original_stall_scores(
    raw: pd.DataFrame, vatex_spatial: np.ndarray, vatex_t1: np.ndarray
) -> pd.DataFrame:
    output = raw.copy()
    spatial = empirical_cdf_right_inclusive(
        output["global_spatial_raw"].to_numpy(), vatex_spatial
    )
    temporal = cdf_temporal(output["global_t1_raw"].to_numpy(), vatex_t1)
    output["K1_STALL"] = 0.5 * spatial + 0.5 * temporal
    return output


def calibrate_k1(
    raw: pd.DataFrame,
    size_membership: pd.DataFrame,
    sizes: dict[str, int],
    vatex_spatial: np.ndarray,
    vatex_t1: np.ndarray,
) -> pd.DataFrame:
    outputs = []
    for (dataset, duration), frame in raw.groupby(
        ["dataset", "protocol_duration_sec"], sort=False
    ):
        calibration_ids = selected_ids(size_membership, str(dataset), sizes[str(dataset)])
        reference_mask = (
            frame["protocol_split"].eq("calibration")
            & frame["video_id"].astype(str).isin(calibration_ids)
        )
        reference = frame[reference_mask]
        if reference["video_id"].nunique() != len(calibration_ids):
            raise ValueError(f"{dataset}/{duration}s: incomplete K1 calibration")
        original_global_spatial_ref = vatex_spatial
        original_global_t1_ref = vatex_t1
        if int(duration) == 2:
            global_spatial_ref, global_t1_ref = vatex_spatial, vatex_t1
        else:
            global_spatial_ref = stable_sorted(reference["global_spatial_raw"].to_numpy())
            finite_t1 = reference["global_t1_raw"].to_numpy(dtype=np.float64)
            global_t1_ref = stable_sorted(finite_t1[np.isfinite(finite_t1)])
        target = frame.copy()
        target["original_global_spatial"] = empirical_cdf_right_inclusive(
            target["global_spatial_raw"].to_numpy(), original_global_spatial_ref
        )
        target["original_global_t1"] = cdf_temporal(
            target["global_t1_raw"].to_numpy(), original_global_t1_ref
        )
        target["K1_STALL"] = (
            0.5 * target["original_global_spatial"]
            + 0.5 * target["original_global_t1"]
        )
        target["global_spatial"] = empirical_cdf_right_inclusive(
            target["global_spatial_raw"].to_numpy(), global_spatial_ref
        )
        target["global_t1"] = cdf_temporal(
            target["global_t1_raw"].to_numpy(), global_t1_ref
        )
        target["patch_spatial"] = empirical_cdf_right_inclusive(
            target["patch_spatial_raw"].to_numpy(),
            stable_sorted(reference["patch_spatial_raw"].to_numpy()),
        )
        target["patch_d2"] = empirical_cdf_right_inclusive(
            target["patch_d2_raw"].to_numpy(),
            stable_sorted(reference["patch_d2_raw"].to_numpy()),
        )
        target["G_k"] = 0.5 * target["global_spatial"] + 0.5 * target["global_t1"]
        target["L_k"] = 0.1 * target["patch_spatial"] + 0.9 * target["patch_d2"]
        scored_reference = target[reference_mask]
        evaluation = target[target["protocol_split"].eq("evaluation")].copy()
        evaluation["K1_G"] = empirical_cdf_right_inclusive(
            evaluation["G_k"].to_numpy(), stable_sorted(scored_reference["G_k"].to_numpy())
        )
        evaluation["K1_L"] = empirical_cdf_right_inclusive(
            evaluation["L_k"].to_numpy(), stable_sorted(scored_reference["L_k"].to_numpy())
        )
        evaluation["K1_S"] = 0.6 * evaluation["K1_G"] + 0.4 * evaluation["K1_L"]
        outputs.append(evaluation)
    result = pd.concat(outputs, ignore_index=True)
    if result.duplicated(["video_id", "protocol_duration_sec"]).any():
        raise ValueError("duplicate calibrated K1 evaluation scores")
    return result


def attach_k3(k1: pd.DataFrame, path: Path) -> pd.DataFrame:
    k3 = pd.read_csv(path, float_precision="round_trip")
    k3 = k3[
        k3["candidate"].eq("ncustom")
        & k3["protocol_split"].eq("evaluation")
    ][["video_id", "protocol_duration_sec", "G", "L", "S", "effective_k"]].rename(
        columns={"G": "K3_G", "L": "K3_L", "S": "K3_S"}
    )
    result = k1.merge(
        k3, on=["video_id", "protocol_duration_sec"], validate="one_to_one"
    )
    expected = 69211
    if len(result) != expected:
        raise ValueError(f"expected {expected} K1/K3 evaluation task rows, found {len(result)}")
    return result


def aggregate(rows: list[dict], scope: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    generators = pd.DataFrame(rows)
    datasets = (
        generators.groupby(["config", "dataset"], as_index=False)
        .agg(
            auc=("auc", "mean"),
            ap_std50=("ap_std50", "mean"),
            fake_ap_std50=("fake_ap_std50", "mean"),
            ap_raw=("ap_raw", "mean"),
            generators=("source_model", "size"),
            fake_videos=("n_fake", "sum"),
        )
    )
    macro = (
        datasets.groupby("config", as_index=False)
        .agg(
            auc=("auc", "mean"),
            ap_std50=("ap_std50", "mean"),
            fake_ap_std50=("fake_ap_std50", "mean"),
            ap_raw=("ap_raw", "mean"),
            generators=("generators", "sum"),
            fake_videos=("fake_videos", "sum"),
        )
    )
    macro["dataset"] = "Macro-3"
    all23 = (
        generators.groupby("config", as_index=False)
        .agg(
            auc=("auc", "mean"),
            ap_std50=("ap_std50", "mean"),
            fake_ap_std50=("fake_ap_std50", "mean"),
            ap_raw=("ap_raw", "mean"),
            generators=("source_model", "size"),
            fake_videos=("n_fake", "sum"),
        )
    )
    all23["dataset"] = "All-23"
    generators.insert(0, "scope", scope)
    summary = pd.concat([datasets, macro, all23], ignore_index=True)
    summary.insert(0, "scope", scope)
    return generators, summary


def evaluate(
    scores: pd.DataFrame,
    fake: pd.DataFrame,
    scope: str,
    balanced: bool,
    seed: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, list[dict]]:
    rows = []
    paired_groups = []
    for (dataset, source_model), fake_group in fake.groupby(
        ["dataset", "source_model"], sort=True
    ):
        real = scores[scores["dataset"].eq(dataset) & scores["subset"].eq("real")]
        if balanced:
            real_group, fake_group = balanced_pair_frame(
                real, fake_group, f"{seed}:full23:{dataset}:{source_model}"
            )
        else:
            real_group = duration_matched_real(
                real, fake_group, f"{scope}:{dataset}:{source_model}:real-duration"
            )
        group_payload = {"dataset": dataset, "source_model": source_model}
        for config in CONFIGS:
            values = metric(real_group[config].to_numpy(), fake_group[config].to_numpy())
            rows.append(
                {
                    **group_payload,
                    "config": config,
                    "n_real": len(real_group),
                    "n_fake": len(fake_group),
                    **values,
                }
            )
        if balanced:
            paired_groups.append(
                {
                    **group_payload,
                    "real_ids": real_group["video_id"].astype(str).to_numpy(),
                    "fake_ids": fake_group["video_id"].astype(str).to_numpy(),
                    "real": {config: real_group[config].to_numpy() for config in CONFIGS},
                    "fake": {config: fake_group[config].to_numpy() for config in CONFIGS},
                }
            )
    generators, summary = aggregate(rows, scope)
    return generators, summary, paired_groups


def original_paper_size_evaluation(
    scores: pd.DataFrame, paper_ids: set[str]
) -> tuple[pd.DataFrame, pd.DataFrame]:
    fake = scores[
        scores["subset"].eq("annotated")
        & scores["video_id"].astype(str).isin(paper_ids)
    ].copy()
    real = scores[scores["subset"].eq("real")].copy()
    rows = []
    for (dataset, source_model), fake_group in fake.groupby(
        ["dataset", "source_model"], sort=True
    ):
        real_pool = real[real["dataset"].eq(dataset)].copy()
        real_metadata = real_pool[["video_id", "source_model"]].drop_duplicates("video_id")
        maximum = 1698 if dataset == "comgenvid" else (3722 if dataset == "videofeedback" else 1400)
        n = min(len(fake_group), maximum)
        namespace = f"original-paper-size-v1:{dataset}:{source_model}"
        real_ids = balanced_real_ids(real_pool, n, namespace + ":real")
        chosen_fake = fake_group.copy()
        chosen_fake["_rank"] = chosen_fake["video_id"].map(
            lambda value: hashlib.sha256(
                f"{namespace}:fake\0{value}".encode("utf-8")
            ).hexdigest()
        )
        chosen_fake = chosen_fake.sort_values("_rank").head(n).reset_index(drop=True)
        assignment = pd.DataFrame(
            {
                "video_id": real_ids,
                "protocol_duration_sec": chosen_fake["protocol_duration_sec"].astype(int),
            }
        )
        chosen_real = assignment.merge(
            real_pool,
            on=["video_id", "protocol_duration_sec"],
            validate="one_to_one",
        )
        if len(chosen_real) != n or len(real_metadata) < n:
            raise ValueError(f"{dataset}/{source_model}: incomplete paper-size real cohort")
        values = metric(
            chosen_real["K1_STALL"].to_numpy(), chosen_fake["K1_STALL"].to_numpy()
        )
        rows.append(
            {
                "dataset": dataset,
                "source_model": source_model,
                "config": "K1_STALL",
                "n_real": n,
                "n_fake": n,
                **values,
            }
        )
    return aggregate(rows, "original_paper_size_k1_stall")


def bootstrap(groups: list[dict], iterations: int, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    samples = {
        (name, metric_name): []
        for name in CONTRASTS
        for metric_name in ("ap_real", "ap_fake")
    }
    for _ in range(iterations):
        per_contrast = {
            key: {dataset: [] for dataset in DATASETS} for key in samples
        }
        for dataset in DATASETS:
            dataset_groups = [group for group in groups if group["dataset"] == dataset]
            real_universe = sorted(
                {video_id for group in dataset_groups for video_id in group["real_ids"]}
            )
            real_draw = rng.choice(real_universe, size=len(real_universe), replace=True)
            real_counts = pd.Series(real_draw).value_counts().astype(int).to_dict()
            for group in dataset_groups:
                repeat = np.asarray(
                    [real_counts.get(video_id, 0) for video_id in group["real_ids"]],
                    dtype=int,
                )
                real_indices = np.repeat(np.arange(len(repeat)), repeat)
                if len(real_indices) == 0:
                    raise ValueError("cluster bootstrap produced an empty real class")
                n_fake = len(group["fake_ids"])
                fake_indices = rng.integers(0, n_fake, n_fake)
                values = {}
                for config in CONFIGS:
                    values[config] = metric(
                        group["real"][config][real_indices],
                        group["fake"][config][fake_indices],
                    )
                for name, (new, base) in CONTRASTS.items():
                    per_contrast[(name, "ap_real")][dataset].append(
                        values[new]["ap_std50"] - values[base]["ap_std50"]
                    )
                    per_contrast[(name, "ap_fake")][dataset].append(
                        values[new]["fake_ap_std50"]
                        - values[base]["fake_ap_std50"]
                    )
        for key in samples:
            samples[key].append(
                float(
                    np.mean(
                        [np.mean(per_contrast[key][dataset]) for dataset in DATASETS]
                    )
                )
            )
    rows = []
    for (name, metric_name), values in samples.items():
        array = np.asarray(values)
        rows.append(
            {
                "contrast": name,
                "metric": metric_name,
                "new_config": CONTRASTS[name][0],
                "base_config": CONTRASTS[name][1],
                "iterations": iterations,
                "mean_delta_ap": float(array.mean()),
                "ci95_low": float(np.quantile(array, 0.025)),
                "ci95_high": float(np.quantile(array, 0.975)),
            }
        )
    return pd.DataFrame(rows)


def seed_sensitivity(scores: pd.DataFrame, fake: pd.DataFrame, count: int) -> pd.DataFrame:
    rows = []
    for seed in range(count):
        _, summary, _ = evaluate(scores, fake, "balanced_seed", True, seed)
        macro = summary[summary["dataset"].eq("Macro-3")].set_index("config")
        row = {"seed": seed}
        for name, (new, base) in CONTRASTS.items():
            row[f"{name}_delta_ap"] = float(
                macro.loc[new, "ap_std50"] - macro.loc[base, "ap_std50"]
            )
            row[f"{name}_delta_fake_ap"] = float(
                macro.loc[new, "fake_ap_std50"]
                - macro.loc[base, "fake_ap_std50"]
            )
        rows.append(row)
    return pd.DataFrame(rows)


def write_report(
    summaries: pd.DataFrame,
    paper_size_summary: pd.DataFrame,
    bootstrap_frame: pd.DataFrame,
    seeds: pd.DataFrame,
    output: Path,
) -> None:
    lines = [
        "# Original-window K1 x Global-Local factorial evaluation",
        "",
        "K1 uses the public STALL fixed-seed contiguous 1s/2s window stored in the source "
        "indexes. K3 uses deterministic beginning/middle/end coverage. K1_STALL applies the "
        "original VATEX component percentiles and no second video-level CDF. The four causal "
        "cells K1_G/K1_S/K3_G/K3_S share ncustom duration-aware real-only calibration, the same "
        "video identities, and the same score direction.",
        "",
    ]
    lines.extend(
        [
            "## Original-paper-size standalone STALL reproduction",
            "",
            "This row uses the paper's per-generator fake counts and real-pool caps. It may use "
            "target real videos that belong to Alpha's Local calibration, which is valid for the "
            "VATEX-calibrated Global-only STALL reproduction but makes the row ineligible for a "
            "causal comparison against Alpha. The local ComGenVid index contains 1,698 rather than "
            "the paper's 1,700 real files, so both ComGenVid pairs use 1,698.",
            "",
            "| Scope | Reproduced K1 STALL AUC/AP_fake/AP_real | Published STALL AUC/AP_fake |",
            "|---|---:|---:|",
        ]
    )
    published = {
        "comgenvid": "0.85/0.86",
        "videofeedback": "0.83/0.85",
        "genvideo": "0.80/0.80",
        "All-23": "0.82/0.82",
    }
    indexed = paper_size_summary.set_index(["config", "dataset"])
    for dataset in (*DATASETS, "All-23"):
        row = indexed.loc[("K1_STALL", dataset)]
        lines.append(
            f"| {dataset} | {row.auc:.4f}/{row.fake_ap_std50:.4f}/{row.ap_std50:.4f} "
            f"| {published[dataset]} |"
        )
    lines.append("")
    for scope in ("full_coverage", "paper_count_fake_cohort", "balanced_seed42"):
        lines.extend(
            [
                f"## {scope}",
                "",
                "Values are AUC/AP_fake/AP_real. AP_fake follows the original STALL paper; "
                "AP_real is retained for continuity with this repository's historical tables.",
                "",
                "| Config | ComGenVid | VideoFeedback | GenVideo | Macro-3 | All-23 |",
                "|---|---:|---:|---:|---:|---:|",
            ]
        )
        frame = summaries[summaries["scope"].eq(scope)].set_index(["config", "dataset"])
        for config in CONFIGS:
            cells = []
            for dataset in (*DATASETS, "Macro-3", "All-23"):
                row = frame.loc[(config, dataset)]
                cells.append(
                    f"{row.auc:.4f}/{row.fake_ap_std50:.4f}/{row.ap_std50:.4f}"
                )
            lines.append(f"| {config} | " + " | ".join(cells) + " |")
        lines.append("")
    lines.extend(
        [
            "## Paired bootstrap",
            "",
            "| Contrast | Metric | Mean AP delta | 95% CI |",
            "|---|---|---:|---:|",
        ]
    )
    for row in bootstrap_frame.itertuples(index=False):
        lines.append(
            f"| {row.contrast} | {row.metric} | {row.mean_delta_ap:+.4f} | "
            f"[{row.ci95_low:+.4f}, {row.ci95_high:+.4f}] |"
        )
    lines.extend(["", "## Selection-seed sensitivity", ""])
    for name in CONTRASTS:
        values = seeds[f"{name}_delta_ap"]
        fake_values = seeds[f"{name}_delta_fake_ap"]
        lines.append(
            f"- {name} AP_real: mean {values.mean():+.4f}, std {values.std(ddof=1):.4f}, "
            f"range [{values.min():+.4f}, {values.max():+.4f}]."
        )
        lines.append(
            f"- {name} AP_fake: mean {fake_values.mean():+.4f}, "
            f"std {fake_values.std(ddof=1):.4f}, "
            f"range [{fake_values.min():+.4f}, {fake_values.max():+.4f}]."
        )
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run(args: argparse.Namespace) -> None:
    args.output_dir.mkdir(parents=True, exist_ok=True)
    raw = load_k1_raw(args.k1_raw_dir)
    membership = pd.read_csv(args.size_membership)
    sizes = {"comgenvid": 600, "videofeedback": 400, "genvideo": 1500}
    import yaml

    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    vatex_spatial, vatex_t1 = global_vatex_references(config)
    original_all = original_stall_scores(raw, vatex_spatial, vatex_t1)
    k1 = calibrate_k1(raw, membership, sizes, vatex_spatial, vatex_t1)
    scores = attach_k3(k1, args.k3_scores)
    fake = scores[scores["subset"].eq("annotated")].copy()
    if len(fake) != 45185 or fake["video_id"].duplicated().any():
        raise ValueError("factorial evaluation does not cover 45,185 unique fakes")
    paper_ids = set(pd.read_csv(args.paper_manifest)["video_id"].astype(str))
    paper_size_generators, paper_size_summary = original_paper_size_evaluation(
        original_all, paper_ids
    )
    paper_fake = fake[fake["video_id"].astype(str).isin(paper_ids)].copy()
    if len(paper_fake) != 36641:
        raise ValueError("paper-count cohort alignment failed")

    artifacts = []
    groups = []
    for scope, cohort, balanced in (
        ("full_coverage", fake, False),
        ("paper_count_fake_cohort", paper_fake, False),
        ("balanced_seed42", fake, True),
    ):
        generators, summary, selected_groups = evaluate(
            scores, cohort, scope, balanced, args.primary_seed
        )
        generators.to_csv(args.output_dir / f"factorial_{scope}_generator_metrics.csv", index=False)
        artifacts.append(summary)
        if balanced:
            groups = selected_groups
    summaries = pd.concat(artifacts, ignore_index=True)
    boot = bootstrap(groups, args.bootstrap_iterations, args.primary_seed)
    seeds = seed_sensitivity(scores, fake, args.seed_count)
    scores.to_csv(args.output_dir / "original_k1_per_video_scores.csv", index=False)
    paper_size_generators.to_csv(
        args.output_dir / "original_paper_size_k1_stall_generator_metrics.csv", index=False
    )
    paper_size_summary.to_csv(
        args.output_dir / "original_paper_size_k1_stall_summary.csv", index=False
    )
    summaries.to_csv(args.output_dir / "factorial_summary.csv", index=False)
    boot.to_csv(args.output_dir / "factorial_bootstrap.csv", index=False)
    seeds.to_csv(args.output_dir / "factorial_seed_sensitivity.csv", index=False)
    write_report(summaries, paper_size_summary, boot, seeds, args.report)
    print(summaries[summaries["dataset"].eq("Macro-3")].to_string(index=False))
    print(boot.to_string(index=False))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--k1-raw-dir",
        type=Path,
        default=ROOT / "results/full_coverage_paper_protocol/original_k1_raw",
    )
    parser.add_argument(
        "--k3-scores",
        type=Path,
        default=ROOT / "results/duration_aware_23source/per_video_scores.csv",
    )
    parser.add_argument(
        "--size-membership",
        type=Path,
        default=ROOT / "results/duration_aware_23source/calibration_size_membership.csv",
    )
    parser.add_argument(
        "--paper-manifest",
        type=Path,
        default=ROOT / "results/full_coverage_paper_protocol/paper_count_fake_manifest.csv",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "configs/alpha_stalled_u0_locked.yaml",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "results/full_coverage_paper_protocol",
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=ROOT / "reports/original_k1_full23_factorial.md",
    )
    parser.add_argument("--primary-seed", type=int, default=42)
    parser.add_argument("--seed-count", type=int, default=100)
    parser.add_argument("--bootstrap-iterations", type=int, default=1000)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
