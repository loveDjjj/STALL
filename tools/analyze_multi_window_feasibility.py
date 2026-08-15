#!/usr/bin/env python3
"""Audit deterministic multi-window coverage on the frozen evaluation protocol."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from alpha_stalled.sampling import (
    WINDOW_FRAMES,
    current_window,
    nonoverlap_windows,
    parse_indices,
    uniform_windows,
)
from alpha_stalled.legacy_local_d2_protocol import (
    KEY_COLUMNS,
    build_strict_eval_index,
    dataset_specs,
)


DURATION_BINS = (0.0, 2.0, 4.0, 6.0, 10.0, 20.0, float("inf"))
DURATION_LABELS = ("<2", "2-4", "4-6", "6-10", "10-20", "20+")
EXCLUSION_KEYS = [
    "dataset",
    "protocol_split",
    "subset",
    "source_model",
    "filename",
    "sampling",
]

def window_sets(row: pd.Series) -> dict[str, list[list[int]]]:
    downsample = parse_indices(row["downsample_idxs"])
    return {
        "K1_current": current_window(row["2_sec_idxs"]),
        "K3_uniform": uniform_windows(downsample, 3),
        "K5_uniform": uniform_windows(downsample, 5),
        "all_nonoverlap": nonoverlap_windows(downsample),
    }


def load_window_exclusions(path: Path) -> dict[tuple[str, ...], set[tuple[int, ...]]]:
    if not path.exists():
        return {}
    frame = pd.read_csv(path)
    lookup: dict[tuple[str, ...], set[tuple[int, ...]]] = {}
    for row in frame.to_dict("records"):
        key = tuple(str(row[column]) for column in EXCLUSION_KEYS)
        lookup.setdefault(key, set()).add(tuple(parse_indices(row["frame_indices"])))
    return lookup


def apply_window_exclusions(
    row: pd.Series,
    sets: dict[str, list[list[int]]],
    exclusions: dict[tuple[str, ...], set[tuple[int, ...]]],
) -> dict[str, list[list[int]]]:
    prefix = (
        str(row["dataset"]),
        str(row["protocol_split"]),
        str(row["subset"]),
        str(row["source_model"]),
        Path(str(row["video_path"])).name,
    )
    return {
        sampling: [
            window
            for window in windows
            if tuple(window) not in exclusions.get((*prefix, sampling), set())
        ]
        for sampling, windows in sets.items()
    }


def _with_filename(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    out["filename"] = out["video_path"].map(lambda value: Path(str(value)).name)
    for column in KEY_COLUMNS:
        out[column] = out[column].astype(str)
    return out


def build_manifest(root: Path, exclusion_path: Path) -> pd.DataFrame:
    stage1 = pd.read_csv(root / "results/multi_order_baselines/per_video_scores.csv")
    frames: list[pd.DataFrame] = []
    for spec in dataset_specs(root):
        evaluation = build_strict_eval_index(spec, stage1)
        evaluation["protocol_split"] = "evaluation"
        calibration = pd.read_csv(spec.calib_index)
        calibration["protocol_split"] = "calibration"
        for frame in (evaluation, calibration):
            frame["dataset"] = spec.name
            frames.append(frame)
    manifest = pd.concat(frames, ignore_index=True)
    exclusions = load_window_exclusions(exclusion_path)
    rows: list[dict] = []
    for _, series in manifest.iterrows():
        sets = apply_window_exclusions(series, window_sets(series), exclusions)
        output = {
            "dataset": series["dataset"],
            "protocol_split": series["protocol_split"],
            "subset": series["subset"],
            "source_model": series["source_model"],
            "filename": Path(str(series["video_path"])).name,
            "video_path": series["video_path"],
            "duration_seconds": float(series["duration_seconds"]),
            "downsample_frames": len(parse_indices(series["downsample_idxs"])),
        }
        for name, windows in sets.items():
            output[f"effective_{name}"] = len(windows)
            output[f"indices_{name}"] = json.dumps(windows, separators=(",", ":"))
        rows.append(output)
    out = pd.DataFrame(rows)
    out["duration_bin"] = pd.cut(
        out["duration_seconds"],
        bins=DURATION_BINS,
        labels=DURATION_LABELS,
        right=False,
    ).astype(str)
    return out


def build_effective_summary(manifest: pd.DataFrame) -> pd.DataFrame:
    effective = [column for column in manifest.columns if column.startswith("effective_")]
    long = manifest.melt(
        id_vars=["dataset", "protocol_split", "subset"],
        value_vars=effective,
        var_name="sampling",
        value_name="effective_k",
    )
    long["sampling"] = long["sampling"].str.removeprefix("effective_")
    return (
        long.groupby(["dataset", "protocol_split", "subset", "sampling", "effective_k"], observed=True)
        .size()
        .rename("videos")
        .reset_index()
        .sort_values(["dataset", "protocol_split", "subset", "sampling", "effective_k"])
    )


def build_generator_summary(manifest: pd.DataFrame) -> pd.DataFrame:
    evaluation = manifest[manifest["protocol_split"] == "evaluation"].copy()
    return (
        evaluation.groupby(["dataset", "subset", "source_model"], observed=True)
        .agg(
            videos=("filename", "size"),
            duration_mean=("duration_seconds", "mean"),
            duration_median=("duration_seconds", "median"),
            duration_p10=("duration_seconds", lambda values: np.quantile(values, 0.1)),
            duration_p90=("duration_seconds", lambda values: np.quantile(values, 0.9)),
            k3_mean=("effective_K3_uniform", "mean"),
            k3_ge3=("effective_K3_uniform", lambda values: int((values >= 3).sum())),
            k5_mean=("effective_K5_uniform", "mean"),
            all_mean=("effective_all_nonoverlap", "mean"),
        )
        .reset_index()
    )


def build_duration_summary(manifest: pd.DataFrame) -> pd.DataFrame:
    return (
        manifest.groupby(["dataset", "protocol_split", "subset", "duration_bin"], observed=True)
        .agg(
            videos=("filename", "size"),
            k3_ge3=("effective_K3_uniform", lambda values: int((values >= 3).sum())),
            k5_ge5=("effective_K5_uniform", lambda values: int((values >= 5).sum())),
        )
        .reset_index()
    )


def write_report(
    manifest: pd.DataFrame,
    effective: pd.DataFrame,
    generator: pd.DataFrame,
    output: Path,
) -> None:
    evaluation = manifest[manifest["protocol_split"] == "evaluation"]
    lines = [
        "# Multi-window feasibility audit",
        "",
        "This audit uses the exact strict Stage-1 evaluation intersection and the disjoint 200-real calibration split per dataset.",
        "All windows contain 16 consecutive positions from each video's frozen 8 FPS downsample index.",
        "Uniform configurations deduplicate identical mapped windows; all-non-overlap excludes incomplete tails.",
        "Windows listed in `configs/multi_window_exclusions.csv` are explicitly removed when the container index references a native frame that no decoder can produce; the video remains in the intersection if another strict window is valid.",
        "",
        "## Evaluation coverage",
        "",
        "| dataset | videos | duration mean | duration median | K3 mean | K3>=3 | K5 mean | all-window mean |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for dataset, group in evaluation.groupby("dataset", sort=True):
        lines.append(
            f"| {dataset} | {len(group)} | {group.duration_seconds.mean():.2f} | "
            f"{group.duration_seconds.median():.2f} | {group.effective_K3_uniform.mean():.2f} | "
            f"{int((group.effective_K3_uniform >= 3).sum())} | {group.effective_K5_uniform.mean():.2f} | "
            f"{group.effective_all_nonoverlap.mean():.2f} |"
        )
    lines.extend(
        [
            "",
            "## Effective-K distribution",
            "",
            effective.to_markdown(index=False),
            "",
            "## Per-source duration and window coverage",
            "",
            generator.to_markdown(index=False, floatfmt=".3f"),
        ]
    )
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=REPO_ROOT)
    parser.add_argument(
        "--window-exclusions",
        type=Path,
        default=REPO_ROOT / "configs/multi_window_exclusions.csv",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPO_ROOT / "results/multi_window_joint_typicality",
    )
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest = build_manifest(args.root, args.window_exclusions)
    effective = build_effective_summary(manifest)
    generator = build_generator_summary(manifest)
    duration = build_duration_summary(manifest)
    manifest.to_csv(args.output_dir / "multi_window_manifest.csv", index=False)
    effective.to_csv(args.output_dir / "effective_k_distribution.csv", index=False)
    generator.to_csv(args.output_dir / "generator_duration_summary.csv", index=False)
    duration.to_csv(args.output_dir / "duration_histogram.csv", index=False)
    write_report(manifest, effective, generator, args.output_dir / "multi_window_feasibility.md")
    print(f"videos={len(manifest)} evaluation={(manifest.protocol_split == 'evaluation').sum()}")
    print(f"saved -> {args.output_dir}")


if __name__ == "__main__":
    main()
