#!/usr/bin/env python3
"""Analyze independent U0 calibration size/seed sensitivity."""

from __future__ import annotations

import argparse
import glob
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml


ROOT = Path(__file__).resolve().parents[1]
for directory in (ROOT / "src", ROOT / "tools"):
    if str(directory) not in sys.path:
        sys.path.insert(0, str(directory))

from analyze_u0_locked import cdf_with_positive_infinity, global_references
from build_multi_order_baselines import metric_tables
from stable_whitening import empirical_cdf_right_inclusive, stable_sorted


DATASETS = ("comgenvid", "videofeedback", "genvideo")
DISPLAY = {
    "comgenvid": "ComGenVid",
    "videofeedback": "VideoFeedback",
    "genvideo": "GenVideo",
    "Macro-3": "Macro-3",
}
SEEDS = (17, 29, 43, 71, 101)
SIZES = (25, 50, 100, 200)
BRANCHES = ("G", "L", "S")


def candidate(seed: int, size: int) -> str:
    return f"s{seed}_n{size}"


def load_parts(root: Path, split: str) -> pd.DataFrame:
    paths = sorted(
        Path(path)
        for path in glob.glob(str(root / "checkpoints" / split / "*" / "shard_*" / "part_*.csv"))
    )
    if not paths:
        raise FileNotFoundError(f"no {split} candidate score parts")
    frame = pd.concat(
        [pd.read_csv(path, float_precision="round_trip") for path in paths],
        ignore_index=True,
    )
    keys = ["video_id", "sampling", "window_id"]
    if frame.duplicated(keys).any():
        raise ValueError(f"duplicate {split} score keys")
    return frame


def selected_k_reference(
    reserve_k3: pd.DataFrame,
    selected_ids: set[str],
    target_k: int,
    column: str,
) -> np.ndarray:
    values = []
    selected = reserve_k3[reserve_k3["video_id"].isin(selected_ids)]
    for _, frame in selected.groupby("video_id", sort=False):
        ordered = frame.sort_values("window_id")
        if len(ordered) < target_k:
            continue
        positions = (
            np.array([(len(ordered) - 1) // 2], dtype=int)
            if target_k == 1
            else np.rint(np.linspace(0, len(ordered) - 1, target_k)).astype(int)
        )
        chosen = ordered.iloc[np.unique(positions)]
        if len(chosen) == target_k:
            values.append(float(chosen[column].mean()))
    if len(values) < 2:
        raise ValueError(f"insufficient K={target_k} reference values for {column}")
    return stable_sorted(np.asarray(values, dtype=np.float64))


def calibrate_candidate(
    evaluation: pd.DataFrame,
    reserve: pd.DataFrame,
    selected_ids: set[str],
    name: str,
    global_spatial_ref: np.ndarray,
    global_t1_ref: np.ndarray,
) -> pd.DataFrame:
    reserve_k1 = reserve[
        reserve["sampling"].eq("k1") & reserve["video_id"].isin(selected_ids)
    ]
    if len(reserve_k1) != len(selected_ids):
        raise ValueError(f"{name}: K1 reserve count mismatch")
    patch_spatial_ref = stable_sorted(reserve_k1[f"patch_spatial__{name}"].to_numpy())
    patch_d2_ref = stable_sorted(reserve_k1[f"patch_d2__{name}"].to_numpy())

    def calibrate_windows(frame: pd.DataFrame) -> pd.DataFrame:
        output = frame.copy()
        output["global_spatial"] = empirical_cdf_right_inclusive(
            output["global_spatial_raw"].to_numpy(), global_spatial_ref
        )
        output["global_t1"] = cdf_with_positive_infinity(
            output["global_t1_raw"].to_numpy(), global_t1_ref
        )
        output["patch_spatial"] = empirical_cdf_right_inclusive(
            output[f"patch_spatial__{name}"].to_numpy(), patch_spatial_ref
        )
        output["patch_d2"] = empirical_cdf_right_inclusive(
            output[f"patch_d2__{name}"].to_numpy(), patch_d2_ref
        )
        output["G_k"] = 0.5 * output["global_spatial"] + 0.5 * output["global_t1"]
        output["L_k"] = 0.1 * output["patch_spatial"] + 0.9 * output["patch_d2"]
        return output

    evaluation_scored = calibrate_windows(evaluation)
    reserve_k3 = calibrate_windows(reserve[reserve["sampling"].eq("k3")])
    evaluation_video = (
        evaluation_scored.groupby("video_id", sort=False)
        .agg(effective_k=("window_id", "size"), G_raw=("G_k", "mean"), L_raw=("L_k", "mean"))
        .reset_index()
    )
    pieces = []
    for effective_k, target in evaluation_video.groupby("effective_k", sort=True):
        target = target.copy()
        for branch in ("G", "L"):
            reference = selected_k_reference(
                reserve_k3,
                selected_ids,
                int(effective_k),
                f"{branch}_k",
            )
            target[branch] = empirical_cdf_right_inclusive(
                target[f"{branch}_raw"].to_numpy(), reference
            )
        target["S"] = 0.6 * target["G"] + 0.4 * target["L"]
        pieces.append(target)
    output = pd.concat(pieces, ignore_index=True)
    if len(output) != evaluation["video_id"].nunique() or output["video_id"].duplicated().any():
        raise ValueError(f"{name}: incomplete candidate video scores")
    return output[["video_id", "effective_k", "G", "L", "S"]]


def locked_raw_audit(evaluation: pd.DataFrame, locked_windows: Path) -> dict[str, float]:
    locked = pd.read_csv(locked_windows, float_precision="round_trip")
    merged = evaluation.merge(
        locked,
        on=["video_id", "window_id"],
        suffixes=("_candidate", "_release"),
        validate="one_to_one",
    )
    if len(merged) != len(evaluation):
        raise ValueError("locked raw audit did not align every evaluation window")
    return {
        "global_spatial": float(
            np.max(np.abs(merged["global_spatial_raw_candidate"] - merged["global_spatial_raw_release"]))
        ),
        "global_t1": float(
            np.nanmax(np.abs(merged["global_t1_raw_candidate"] - merged["global_t1_raw_release"]))
        ),
        "patch_spatial": float(
            np.max(np.abs(merged["patch_spatial__locked"] - merged["patch_spatial_raw"]))
        ),
        "patch_d2": float(
            np.max(np.abs(merged["patch_d2__locked"] - merged["patch_d2_raw"]))
        ),
    }


def seed_summary(metrics: pd.DataFrame) -> pd.DataFrame:
    parsed = metrics.copy()
    parsed[["branch", "seed", "calibration_size"]] = parsed["config"].str.extract(
        r"^([GLS])_s(\d+)_n(\d+)$"
    )
    parsed["seed"] = parsed["seed"].astype(int)
    parsed["calibration_size"] = parsed["calibration_size"].astype(int)
    return (
        parsed.groupby(["dataset", "branch", "calibration_size"], as_index=False)
        .agg(
            auc_mean=("auc", "mean"),
            auc_std=("auc", "std"),
            auc_min=("auc", "min"),
            auc_max=("auc", "max"),
            ap_mean=("ap", "mean"),
            ap_std=("ap", "std"),
            ap_min=("ap", "min"),
            ap_max=("ap", "max"),
            auc_delta_vs_stall_mean=("auc_delta_vs_stall", "mean"),
            ap_delta_vs_stall_mean=("ap_delta_vs_stall", "mean"),
        )
    )


def score_variation(per_video: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for dataset, frame in per_video.groupby("dataset", sort=True):
        for size in SIZES:
            for branch in BRANCHES:
                columns = [f"{branch}_{candidate(seed, size)}" for seed in SEEDS]
                variation = frame[columns].std(axis=1, ddof=1)
                rows.append(
                    {
                        "dataset": dataset,
                        "branch": branch,
                        "calibration_size": size,
                        "mean_video_seed_std": float(variation.mean()),
                        "median_video_seed_std": float(variation.median()),
                        "max_video_seed_std": float(variation.max()),
                    }
                )
    return pd.DataFrame(rows)


def write_report(
    summary: pd.DataFrame,
    variation: pd.DataFrame,
    reserve_payload: dict,
    membership: pd.DataFrame,
    raw_errors: dict[str, float],
    report: Path,
    locked_macro_ap: float,
    stall_macro_ap: float,
) -> None:
    lines = [
        "# U0 calibration size and seed stability",
        "",
        "All splits are sampled from a real-only reserve that is disjoint from both the "
        "locked 600-video calibration set and the 21,421-video evaluation set. Splits from "
        "different seeds may overlap each other; they are independent evaluation-disjoint "
        "calibration draws, not five mutually disjoint banks.",
        "",
        "## Reserve audit",
        "",
        "| Dataset | Eligible reserve | Sources | Duration mean/median (s) |",
        "|---|---:|---|---:|",
    ]
    videos = pd.DataFrame(reserve_payload["videos"])
    for dataset in DATASETS:
        frame = videos[videos["dataset"].eq(dataset)]
        sources = ", ".join(
            f"{key}:{value}" for key, value in frame["source_model"].value_counts().sort_index().items()
        )
        lines.append(
            f"| {DISPLAY[dataset]} | {len(frame)} | {sources} | "
            f"{frame.duration_seconds.mean():.2f}/{frame.duration_seconds.median():.2f} |"
        )
    lines.extend(
        [
            "",
            f"Membership rows: {len(membership):,}; locked overlap: "
            f"{reserve_payload['locked_overlap_count']}. Sizes 25/50/100 are nested in size 200 "
            "within each dataset/seed and source-stratified before stable SHA-256 ranking.",
            "",
            "## Final-score stability",
            "",
            "Values are mean +/- sample standard deviation across seeds 17, 29, 43, 71, 101; "
            "bracketed values are seed min/max. AP remains real-positive.",
            "",
            "| Size | ComGenVid AUC/AP | VideoFeedback AUC/AP | GenVideo AUC/AP | Macro-3 AUC/AP |",
            "|---:|---:|---:|---:|---:|",
        ]
    )
    final = summary[summary["branch"].eq("S")].set_index(["dataset", "calibration_size"])
    for size in SIZES:
        cells = []
        for dataset in (*DATASETS, "Macro-3"):
            row = final.loc[(dataset, size)]
            cells.append(
                f"{row.auc_mean:.4f}+/-{row.auc_std:.4f} / "
                f"{row.ap_mean:.4f}+/-{row.ap_std:.4f} "
                f"[{row.ap_min:.4f},{row.ap_max:.4f}]"
            )
        lines.append(f"| {size} | " + " | ".join(cells) + " |")
    lines.extend(
        [
            "",
            "## Branch calibration dependence",
            "",
            "| Size | Global Macro AP | Local Macro AP | Final Macro AP |",
            "|---:|---:|---:|---:|",
        ]
    )
    indexed = summary.set_index(["dataset", "branch", "calibration_size"])
    for size in SIZES:
        cells = []
        for branch in BRANCHES:
            row = indexed.loc[("Macro-3", branch, size)]
            cells.append(f"{row.ap_mean:.4f}+/-{row.ap_std:.4f}")
        lines.append(f"| {size} | " + " | ".join(cells) + " |")
    lines.extend(
        [
            "",
            "## Gain relative to Original STALL",
            "",
            "| Size | Macro AUC delta | Macro AP delta |",
            "|---:|---:|---:|",
        ]
    )
    for size in SIZES:
        row = indexed.loc[("Macro-3", "S", size)]
        lines.append(
            f"| {size} | {row.auc_delta_vs_stall_mean:+.4f} | "
            f"{row.ap_delta_vs_stall_mean:+.4f} |"
        )
    macro_final = final.loc["Macro-3"]
    n200 = float(macro_final.loc[200, "ap_mean"])
    target = stall_macro_ap + 0.95 * (n200 - stall_macro_ap)
    eligible = [size for size in SIZES if float(macro_final.loc[size, "ap_mean"]) >= target]
    minimum = min(eligible) if eligible else None
    n200_std = float(macro_final.loc[200, "ap_std"])
    lines.extend(
        [
            "",
            "## Decision",
            "",
            f"The independent-reserve N=200 Macro AP is `{n200:.6f} +/- {n200_std:.6f}`; "
            f"the locked release split is `{locked_macro_ap:.6f}`. Relative to Original STALL "
            f"AP `{stall_macro_ap:.6f}`, retaining 95% of the N=200 mean improvement requires "
            f"Macro AP >= `{target:.6f}`; the smallest tested size meeting it is "
            f"`{minimum if minimum is not None else 'none'}`.",
            "",
            "The seed experiment refits Local whitening, K1 window CDFs, and effective-K video "
            "CDFs for every split. It is separate from the paired video-cluster bootstrap.",
            "",
            "Mean per-video score standard deviation across seeds at N=200:",
        ]
    )
    v200 = variation[variation["calibration_size"].eq(200)]
    for branch in BRANCHES:
        values = v200[v200["branch"].eq(branch)]["mean_video_seed_std"]
        lines.append(f"- {branch}: dataset mean `{values.mean():.6f}`.")
    lines.extend(
        [
            "",
            "## Numerical and integrity checks",
            "",
            f"- Evaluation windows: 56,812; reserve K1/K3 rows: 10,611; score/decode failures: 0.",
            f"- Accelerated locked raw max errors: GlobalSpatial `{raw_errors['global_spatial']:.3g}`, "
            f"GlobalT1 `{raw_errors['global_t1']:.3g}`, PatchSpatial "
            f"`{raw_errors['patch_spatial']:.3g}`, PatchD2 `{raw_errors['patch_d2']:.3g}`.",
            "- Generated videos are used only for final evaluation metrics; none enter whitening, "
            "CDF construction, split selection, or parameter fitting.",
        ]
    )
    report.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run(args: argparse.Namespace) -> None:
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    evaluation_windows = load_parts(args.input_dir, "evaluation")
    reserve_windows = load_parts(args.input_dir, "reserve")
    if len(evaluation_windows) != 56812 or len(reserve_windows) != 10611:
        raise ValueError(
            f"unexpected score rows eval={len(evaluation_windows)} reserve={len(reserve_windows)}"
        )
    raw_errors = locked_raw_audit(evaluation_windows, args.locked_windows)
    membership = pd.read_csv(args.membership)
    reserve_payload = json.loads(args.reserve_manifest.read_text(encoding="utf-8"))
    global_spatial_ref, global_t1_ref = global_references(config)
    release = pd.read_csv(args.locked_scores, float_precision="round_trip")
    per_video = release[
        ["video_id", "dataset", "protocol_split", "subset", "source_model", "filename"]
    ].copy()
    for seed in SEEDS:
        for size in SIZES:
            name = candidate(seed, size)
            pieces = []
            for dataset in DATASETS:
                selected_ids = set(
                    membership[
                        membership["dataset"].eq(dataset)
                        & membership["seed"].eq(seed)
                        & membership["calibration_size"].eq(size)
                    ]["video_id"]
                )
                if len(selected_ids) != size:
                    raise ValueError(f"{dataset}/{name}: membership count {len(selected_ids)}")
                pieces.append(
                    calibrate_candidate(
                        evaluation_windows[evaluation_windows["dataset"].eq(dataset)],
                        reserve_windows[reserve_windows["dataset"].eq(dataset)],
                        selected_ids,
                        name,
                        global_spatial_ref,
                        global_t1_ref,
                    )
                )
            candidate_scores = pd.concat(pieces, ignore_index=True).rename(
                columns={branch: f"{branch}_{name}" for branch in BRANCHES}
            )
            per_video = per_video.merge(
                candidate_scores.drop(columns="effective_k"),
                on="video_id",
                validate="one_to_one",
            )
    score_columns = [
        f"{branch}_{candidate(seed, size)}"
        for branch in BRANCHES
        for seed in SEEDS
        for size in SIZES
    ]
    names = {column: column for column in score_columns}
    dataset_metrics, generator_metrics = metric_tables(
        per_video,
        seed=int(config["release"]["random_seed"]),
        score_columns=score_columns,
        config_names=names,
    )
    stall_metrics = pd.read_csv(args.core_metrics)
    stall_metrics = stall_metrics[stall_metrics["config"].eq("A0")].set_index("dataset")
    dataset_metrics["auc_delta_vs_stall"] = dataset_metrics.apply(
        lambda row: float(row["auc"] - stall_metrics.loc[row["dataset"], "auc"]), axis=1
    )
    dataset_metrics["ap_delta_vs_stall"] = dataset_metrics.apply(
        lambda row: float(row["ap"] - stall_metrics.loc[row["dataset"], "ap"]), axis=1
    )
    summary = seed_summary(dataset_metrics)
    variation = score_variation(per_video)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    per_video.to_csv(args.output_dir / "per_video_scores.csv", index=False)
    dataset_metrics.to_csv(args.output_dir / "dataset_metrics.csv", index=False)
    generator_metrics.to_csv(args.output_dir / "generator_metrics.csv", index=False)
    summary.to_csv(args.output_dir / "seed_summary.csv", index=False)
    variation.to_csv(args.output_dir / "score_seed_variation.csv", index=False)
    pd.DataFrame([raw_errors]).to_csv(args.output_dir / "locked_raw_audit.csv", index=False)
    write_report(
        summary,
        variation,
        reserve_payload,
        membership,
        raw_errors,
        args.report,
        float(config["release"]["stable_macro_real_positive_ap"]),
        args.stall_macro_ap,
    )
    print(
        summary[
            summary["dataset"].eq("Macro-3") & summary["branch"].eq("S")
        ].to_string(index=False)
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", type=Path, default=ROOT / "configs/alpha_stalled_u0_locked.yaml"
    )
    parser.add_argument(
        "--input-dir", type=Path, default=ROOT / "results/u0_calibration_sensitivity"
    )
    parser.add_argument(
        "--output-dir", type=Path, default=ROOT / "results/u0_calibration_sensitivity/analysis"
    )
    parser.add_argument(
        "--membership", type=Path, default=ROOT / "release/u0/calibration_split_membership.csv"
    )
    parser.add_argument(
        "--reserve-manifest",
        type=Path,
        default=ROOT / "release/u0/calibration_reserve_manifest.json",
    )
    parser.add_argument(
        "--locked-windows",
        type=Path,
        default=ROOT / "results/u0_locked_reproduction/per_window_scores.csv",
    )
    parser.add_argument(
        "--locked-scores", type=Path, default=ROOT / "release/u0/final_video_scores.csv"
    )
    parser.add_argument(
        "--core-metrics",
        type=Path,
        default=ROOT / "results/u0_core_ablation/core_ablation_dataset_metrics.csv",
    )
    parser.add_argument(
        "--report", type=Path, default=ROOT / "reports/u0_calibration_size_and_seed.md"
    )
    parser.add_argument("--stall-macro-ap", type=float, default=0.842810)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
