#!/usr/bin/env python3
"""Analyze U0 score response, temporal coverage, and PatchD2 localization."""

from __future__ import annotations

import argparse
import glob
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
from analyze_u0_robustness import selected_k_reference
from stable_whitening import empirical_cdf_right_inclusive, stable_sorted
from u0_injections import CONDITIONS


DATASETS = ("comgenvid", "videofeedback", "genvideo")
KEY_COLUMNS = [
    "video_id",
    "dataset",
    "protocol_split",
    "subset",
    "source_model",
    "filename",
]


def load_parts(root: Path) -> pd.DataFrame:
    paths = sorted(
        Path(path)
        for path in glob.glob(str(root / "checkpoints" / "*" / "shard_*" / "part_*.csv"))
    )
    if not paths:
        raise FileNotFoundError("no injection score parts")
    frame = pd.concat(
        [pd.read_csv(path, float_precision="round_trip") for path in paths],
        ignore_index=True,
    )
    if frame.duplicated(["video_id", "sampling", "condition", "window_id"]).any():
        raise ValueError("duplicate injection score keys")
    return frame


def load_calibration_k1(directory: Path) -> pd.DataFrame:
    paths = sorted(directory.glob("*.csv"))
    if not paths:
        raise FileNotFoundError(directory)
    frame = pd.concat(
        [pd.read_csv(path, float_precision="round_trip") for path in paths],
        ignore_index=True,
    )
    if len(frame) != 600 or frame["video_id"].nunique() != 600:
        raise ValueError("locked K1 calibration references are incomplete")
    return frame


def calibrate_windows(
    frame: pd.DataFrame,
    patch_spatial_ref: np.ndarray,
    patch_d2_ref: np.ndarray,
    global_spatial_ref: np.ndarray,
    global_t1_ref: np.ndarray,
) -> pd.DataFrame:
    output = frame.copy()
    output["global_spatial"] = empirical_cdf_right_inclusive(
        output["global_spatial_raw"].to_numpy(), global_spatial_ref
    )
    output["global_t1"] = cdf_with_positive_infinity(
        output["global_t1_raw"].to_numpy(), global_t1_ref
    )
    output["patch_spatial"] = empirical_cdf_right_inclusive(
        output["patch_spatial_raw"].to_numpy(), patch_spatial_ref
    )
    output["patch_d2"] = empirical_cdf_right_inclusive(
        output["patch_d2_raw"].to_numpy(), patch_d2_ref
    )
    output["G_k"] = 0.5 * output["global_spatial"] + 0.5 * output["global_t1"]
    output["L_k"] = 0.1 * output["patch_spatial"] + 0.9 * output["patch_d2"]
    return output


def build_video_scores(
    raw: pd.DataFrame,
    calibration_k1: pd.DataFrame,
    locked_k3: pd.DataFrame,
    global_spatial_ref: np.ndarray,
    global_t1_ref: np.ndarray,
) -> pd.DataFrame:
    frames = []
    for dataset in DATASETS:
        target_dataset = raw[raw["dataset"].eq(dataset)]
        calibration = calibration_k1[calibration_k1["dataset"].eq(dataset)]
        spatial_ref = stable_sorted(calibration["patch_spatial_raw"].to_numpy())
        d2_ref = stable_sorted(calibration["patch_d2_raw"].to_numpy())
        calibrated_k1 = calibrate_windows(
            calibration, spatial_ref, d2_ref, global_spatial_ref, global_t1_ref
        )
        k1_g_ref = stable_sorted(calibrated_k1["G_k"].to_numpy())
        k1_l_ref = stable_sorted(calibrated_k1["L_k"].to_numpy())
        locked_dataset = locked_k3[
            locked_k3["dataset"].eq(dataset) & locked_k3["protocol_split"].eq("calibration")
        ]
        for condition in CONDITIONS:
            for sampling in ("k1", "k3"):
                current = target_dataset[
                    target_dataset["condition"].eq(condition)
                    & target_dataset["sampling"].eq(sampling)
                ]
                current = calibrate_windows(
                    current, spatial_ref, d2_ref, global_spatial_ref, global_t1_ref
                )
                video = (
                    current.groupby(KEY_COLUMNS, sort=False, observed=True)
                    .agg(
                        duration_seconds=("duration_seconds", "first"),
                        effective_k=("effective_k", "first"),
                        window_count=("window_id", "size"),
                        target_k3_window=("target_k3_window", "first"),
                        affected_native_overlap=("affected_native_overlap", "max"),
                        G_raw=("G_k", "mean"),
                        L_raw=("L_k", "mean"),
                    )
                    .reset_index()
                )
                pieces = []
                for effective_k, group in video.groupby("effective_k", sort=True):
                    group = group.copy()
                    if sampling == "k1":
                        references = {"G": k1_g_ref, "L": k1_l_ref}
                    else:
                        references = {
                            branch: selected_k_reference(
                                locked_dataset, int(effective_k), f"{branch}_k"
                            )
                            for branch in ("G", "L")
                        }
                    for branch in ("G", "L"):
                        group[branch] = empirical_cdf_right_inclusive(
                            group[f"{branch}_raw"].to_numpy(), references[branch]
                        )
                    group["S"] = 0.6 * group["G"] + 0.4 * group["L"]
                    group["condition"] = condition
                    group["sampling"] = sampling
                    pieces.append(group)
                frames.append(pd.concat(pieces, ignore_index=True))
    return pd.concat(frames, ignore_index=True)


def reproduction_errors(raw: pd.DataFrame, per_video: pd.DataFrame, locked_windows: pd.DataFrame, core: pd.DataFrame, release: pd.DataFrame) -> pd.DataFrame:
    columns = ["global_spatial_raw", "global_t1_raw", "patch_spatial_raw", "patch_d2_raw"]
    rows = []
    for sampling, reference in (
        ("k1", core[["video_id", *columns]]),
        ("k3", locked_windows[["video_id", "window_id", *columns]]),
    ):
        current = raw[raw["sampling"].eq(sampling) & raw["condition"].eq("L0_original")]
        keys = ["video_id"] if sampling == "k1" else ["video_id", "window_id"]
        compared = current[keys + columns].merge(
            reference, on=keys, suffixes=("_new", "_locked"), validate="one_to_one"
        )
        for column in columns:
            rows.append(
                {
                    "level": f"raw_{sampling}",
                    "component": column,
                    "max_abs_error": float(
                        np.max(np.abs(compared[f"{column}_new"] - compared[f"{column}_locked"]))
                    ),
                }
            )
    baseline = per_video[per_video["condition"].eq("L0_original")]
    for sampling, reference, column in (
        ("k1", core[["video_id", "A9"]], "A9"),
        ("k3", release[["video_id", "S"]], "S"),
    ):
        expected = reference.rename(columns={column: "S_expected"})
        compared = baseline[baseline["sampling"].eq(sampling)][["video_id", "S"]].merge(
            expected, on="video_id", validate="one_to_one"
        )
        rows.append(
            {
                "level": f"final_{sampling}",
                "component": "S",
                "max_abs_error": float(
                    np.max(np.abs(compared["S"] - compared["S_expected"]))
                ),
            }
        )
    return pd.DataFrame(rows)


def response_summary(per_video: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    baseline = per_video[per_video["condition"].eq("L0_original")][
        ["video_id", "sampling", "G", "L", "S"]
    ].rename(columns={branch: f"{branch}_baseline" for branch in ("G", "L", "S")})
    merged = per_video.merge(baseline, on=["video_id", "sampling"], validate="many_to_one")
    rows = []
    for (dataset, sampling, condition), frame in merged.groupby(
        ["dataset", "sampling", "condition"], sort=True
    ):
        for branch in ("G", "L", "S"):
            response = frame[f"{branch}_baseline"] - frame[branch]
            rows.append(
                {
                    "dataset": dataset,
                    "sampling": sampling,
                    "condition": condition,
                    "branch": branch,
                    "count": len(frame),
                    "mean_score_drop": float(response.mean()),
                    "median_score_drop": float(response.median()),
                    "fraction_score_decreased": float((response > 0).mean()),
                    "fraction_below_0_5": float((frame[branch] < 0.5).mean()),
                }
            )
    summary = pd.DataFrame(rows)
    macro = summary.groupby(["sampling", "condition", "branch"], as_index=False).agg(
        count=("count", "sum"),
        mean_score_drop=("mean_score_drop", "mean"),
        median_score_drop=("median_score_drop", "mean"),
        fraction_score_decreased=("fraction_score_decreased", "mean"),
        fraction_below_0_5=("fraction_below_0_5", "mean"),
    )
    macro.insert(0, "dataset", "Macro-3")
    return merged, pd.concat([summary, macro], ignore_index=True)


def coverage_summary(merged: pd.DataFrame) -> pd.DataFrame:
    plan_overlap = merged[merged["sampling"].eq("k1") & merged["condition"].eq("L1_local_freeze4")][
        ["video_id", "affected_native_overlap"]
    ].rename(columns={"affected_native_overlap": "k1_overlap"})
    frame = merged.merge(plan_overlap, on="video_id", validate="many_to_one")
    frame["coverage_group"] = np.select(
        [frame["k1_overlap"].eq(0), frame["k1_overlap"].eq(4)],
        ["none", "full"],
        default="partial",
    )
    rows = []
    for (condition, coverage, sampling), group in frame[
        frame["condition"].isin(CONDITIONS[1:7])
    ].groupby(["condition", "coverage_group", "sampling"], sort=True):
        response = group["S_baseline"] - group["S"]
        rows.append(
            {
                "condition": condition,
                "coverage_group": coverage,
                "sampling": sampling,
                "count": len(group),
                "mean_final_score_drop": float(response.mean()),
                "fraction_final_score_decreased": float((response > 0).mean()),
            }
        )
    return pd.DataFrame(rows)


def localization_summary(raw: pd.DataFrame) -> pd.DataFrame:
    target = raw[raw["is_target_window"].fillna(False)].copy()
    columns = [
        "heatmap_auprc",
        "inside_anomaly",
        "outside_anomaly",
        "inside_minus_outside",
        "topk_patch_hit_rate",
        "temporal_hit",
    ]
    frames = []
    for (dataset, condition), group in target.groupby(["dataset", "condition"], sort=True):
        row = {"dataset": dataset, "condition": condition, "count": len(group)}
        row.update({column: float(group[column].mean()) for column in columns})
        frames.append(row)
    summary = pd.DataFrame(frames)
    macro = summary.groupby("condition", as_index=False).agg(
        count=("count", "sum"), **{column: (column, "mean") for column in columns}
    )
    macro.insert(0, "dataset", "Macro-3")
    return pd.concat([summary, macro], ignore_index=True)


def write_report(response: pd.DataFrame, coverage: pd.DataFrame, localization: pd.DataFrame, errors: pd.DataFrame, report: Path) -> None:
    indexed = response.set_index(["dataset", "sampling", "condition", "branch"])
    localize = localization.set_index(["dataset", "condition"])
    lines = [
        "# U0 localization and synthetic injection",
        "",
        "The 300 real videos, target K3 windows, native affected frame indices, center-25% region, "
        "scene-cut donors, and anomaly definitions were locked before scoring.",
        "",
        "## Score response",
        "",
        "Positive values are realness-score drops, so larger values indicate stronger anomaly response.",
        "",
        "| Condition | K1 Global/Local/Final drop | K3 Global/Local/Final drop |",
        "|---|---:|---:|",
    ]
    for condition in CONDITIONS[1:]:
        cells = []
        for sampling in ("k1", "k3"):
            values = [
                indexed.loc[("Macro-3", sampling, condition, branch), "mean_score_drop"]
                for branch in ("G", "L", "S")
            ]
            cells.append("/".join(f"{value:+.4f}" for value in values))
        lines.append(f"| {condition} | {cells[0]} | {cells[1]} |")
    lines.extend(
        [
            "",
            "## PatchD2 localization",
            "",
            "| Condition | Patch-time AUPRC | Inside-outside anomaly | Top-k patch hit | Temporal hit |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for condition in CONDITIONS[1:7]:
        row = localize.loc[("Macro-3", condition)]
        lines.append(
            f"| {condition} | {row.heatmap_auprc:.4f} | {row.inside_minus_outside:.4f} | "
            f"{row.topk_patch_hit_rate:.4f} | {row.temporal_hit:.4f} |"
        )
    zero = coverage[coverage["coverage_group"].eq("none")]
    k1_zero = float(zero[zero["sampling"].eq("k1")]["mean_final_score_drop"].mean())
    k3_zero = float(zero[zero["sampling"].eq("k3")]["mean_final_score_drop"].mean())
    scene = {
        sampling: indexed.loc[("Macro-3", sampling, "L7_scene_cut", "S")]
        for sampling in ("k1", "k3")
    }
    lines.extend(
        [
            "",
            "## Coverage and difficult negative",
            "",
            f"- K1 has zero native-frame overlap for 165/300 locked injections. Across L1-L6 in "
            f"that subset, mean K1/K3 final-score drops are `{k1_zero:+.4f}/{k3_zero:+.4f}`.",
            f"- Scene-cut K1/K3 final-score drops are "
            f"`{scene['k1'].mean_score_drop:+.4f}/{scene['k3'].mean_score_drop:+.4f}`; fractions below "
            f"0.5 are `{scene['k1'].fraction_below_0_5:.4f}/{scene['k3'].fraction_below_0_5:.4f}`.",
            f"- Maximum R0 reproduction error across raw and final scores is `{errors.max_abs_error.max():.3g}`.",
            "- Per-video scores, localization rows, coverage groups, and compressed heatmap arrays are retained.",
        ]
    )
    report.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run(args: argparse.Namespace) -> None:
    config = yaml.safe_load(args.config.read_text())
    raw = load_parts(args.scores_dir)
    calibration_k1 = load_calibration_k1(args.calibration_k1_dir)
    locked_windows = pd.read_csv(args.locked_windows, float_precision="round_trip")
    core = pd.read_csv(args.core_scores, float_precision="round_trip")
    release = pd.read_csv(args.release_scores, float_precision="round_trip")
    global_spatial_ref, global_t1_ref = global_references(config)
    per_video = build_video_scores(
        raw, calibration_k1, locked_windows, global_spatial_ref, global_t1_ref
    )
    selected_ids = set(raw["video_id"])
    errors = reproduction_errors(
        raw,
        per_video,
        locked_windows,
        core[core["video_id"].isin(selected_ids)],
        release[release["video_id"].isin(selected_ids)],
    )
    if errors["max_abs_error"].max() > args.reproduction_tolerance:
        raise ValueError(f"L0 does not reproduce locked K1/K3 scores:\n{errors}")
    merged, response = response_summary(per_video)
    coverage = coverage_summary(merged)
    localization = localization_summary(raw)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    per_video.to_csv(args.output_dir / "per_video_scores.csv", index=False)
    response.to_csv(args.output_dir / "score_response.csv", index=False)
    coverage.to_csv(args.output_dir / "coverage_summary.csv", index=False)
    localization.to_csv(args.output_dir / "localization_summary.csv", index=False)
    errors.to_csv(args.output_dir / "l0_reproduction_errors.csv", index=False)
    write_report(response, coverage, localization, errors, args.report)
    print(response[response["dataset"].eq("Macro-3")].to_string(index=False))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--scores-dir", type=Path, default=ROOT / "results/u0_injection"
    )
    parser.add_argument(
        "--config", type=Path, default=ROOT / "configs/alpha_stalled_u0_locked.yaml"
    )
    parser.add_argument(
        "--calibration-k1-dir",
        type=Path,
        default=ROOT / "results/u0_locked_reproduction/calibration_raw",
    )
    parser.add_argument(
        "--locked-windows",
        type=Path,
        default=ROOT / "results/u0_locked_reproduction/per_window_scores.csv",
    )
    parser.add_argument(
        "--core-scores",
        type=Path,
        default=ROOT / "results/u0_core_ablation/per_video_ablation_scores.csv",
    )
    parser.add_argument(
        "--release-scores", type=Path, default=ROOT / "release/u0/final_video_scores.csv"
    )
    parser.add_argument(
        "--output-dir", type=Path, default=ROOT / "results/u0_injection/analysis"
    )
    parser.add_argument(
        "--report", type=Path, default=ROOT / "reports/u0_localization_and_injection.md"
    )
    parser.add_argument("--reproduction-tolerance", type=float, default=1e-7)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
