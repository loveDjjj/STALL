#!/usr/bin/env python3
"""Build the locked U0 real-calibration source-to-test matrix."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import yaml


ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from alpha_stalled.metrics import metric_tables
from alpha_stalled.parameters import global_references
from alpha_stalled.release_io import resolve_required_video as resolve_video
from alpha_stalled.u0_calibration_experiments import (
    DATASETS,
    calibrate_cross,
    load_parts,
)


SOURCES = ("comgenvid", "videofeedback", "genvideo", "pooled200", "pooled600")
DISPLAY = {
    "comgenvid": "ComGenVid",
    "videofeedback": "VideoFeedback",
    "genvideo": "GenVideo",
    "pooled200": "Pooled-200",
    "pooled600": "Pooled-600",
    "Macro-3": "Macro-3",
}


def probe_calibration_videos(release_dir: Path) -> pd.DataFrame:
    payload = json.loads((release_dir / "calibration_manifest.json").read_text())
    rows = []
    for item in payload["videos"]:
        path = resolve_video(item["video_path"])
        capture = cv2.VideoCapture(str(path))
        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        capture.release()
        if width <= 0 or height <= 0:
            raise ValueError(f"failed to probe resolution: {path}")
        rows.append(
            {
                "video_id": item["video_id"],
                "dataset": item["dataset"],
                "source_model": item["source_model"],
                "duration_seconds": float(item["duration_seconds"]),
                "width": width,
                "height": height,
                "pixels": width * height,
            }
        )
    return pd.DataFrame(rows)


def write_report(
    metrics: pd.DataFrame,
    shifts: pd.DataFrame,
    bank_metadata: pd.DataFrame,
    bank_manifest: dict,
    diagonal_error: float,
    report: Path,
) -> None:
    lines = [
        "# U0 cross-dataset real calibration",
        "",
        "The detector configuration is fixed. Only the real calibration bank changes; no generated "
        "video enters whitening, CDFs, source selection, or parameter fitting.",
        "",
        "## Calibration banks",
        "",
        "| Bank | Videos | Dataset composition | Duration mean/median (s) | Resolution median (px) |",
        "|---|---:|---|---:|---:|",
    ]
    for source in SOURCES:
        ids = set(bank_manifest["banks"][source]["video_ids"])
        frame = bank_metadata[bank_metadata["video_id"].isin(ids)]
        composition = ", ".join(
            f"{DISPLAY[key]}:{value}"
            for key, value in frame["dataset"].value_counts().sort_index().items()
        )
        lines.append(
            f"| {DISPLAY[source]} | {len(frame)} | {composition} | "
            f"{frame.duration_seconds.mean():.2f}/{frame.duration_seconds.median():.2f} | "
            f"{int(frame.pixels.median()):,} |"
        )
    indexed = metrics.set_index(["config", "dataset"])
    for branch, title in (("G", "Global"), ("L", "Local"), ("S", "Final")):
        lines.extend(
            [
                "",
                f"## {title} AUC/AP matrix",
                "",
                "| Calibration source | ComGenVid | VideoFeedback | GenVideo | Macro-3 |",
                "|---|---:|---:|---:|---:|",
            ]
        )
        for source in SOURCES:
            cells = []
            config = f"{branch}_{source}"
            for dataset in (*DATASETS, "Macro-3"):
                row = indexed.loc[(config, dataset)]
                cells.append(f"{row.auc:.4f}/{row.ap:.4f}")
            lines.append(f"| {DISPLAY[source]} | " + " | ".join(cells) + " |")

    target_ap = []
    off_ap = []
    dependence = {}
    for branch in ("G", "L", "S"):
        target_values = []
        off_values = []
        for dataset in DATASETS:
            target_values.append(float(indexed.loc[(f"{branch}_{dataset}", dataset), "ap"]))
            for source in DATASETS:
                if source != dataset:
                    off_values.append(float(indexed.loc[(f"{branch}_{source}", dataset), "ap"]))
        dependence[branch] = (float(np.mean(target_values)), float(np.mean(off_values)))
    pooled200 = float(indexed.loc[("S_pooled200", "Macro-3"), "ap"])
    pooled600 = float(indexed.loc[("S_pooled600", "Macro-3"), "ap"])
    target_macro = float(
        np.mean([indexed.loc[(f"S_{dataset}", dataset), "ap"] for dataset in DATASETS])
    )
    lines.extend(
        [
            "",
            "## Distribution shift",
            "",
            "Machine-readable real/fake score means, standard deviations, and quantiles for every "
            "source-target-branch combination are stored in `score_distribution_shift.csv`.",
            "",
            "## Answers",
            "",
            f"1. Target-domain real calibration gives diagonal mean final AP `{target_macro:.4f}`. "
            f"The mean off-domain final AP is `{dependence['S'][1]:.4f}`.",
            f"2. Global target/off-domain AP is `{dependence['G'][0]:.4f}/{dependence['G'][1]:.4f}`; "
            f"Local is `{dependence['L'][0]:.4f}/{dependence['L'][1]:.4f}`. The larger gap identifies "
            "the more domain-dependent branch.",
            f"3. Pooled-200/Pooled-600 Macro AP is `{pooled200:.4f}/{pooled600:.4f}` versus target-bank "
            f"diagonal `{target_macro:.4f}`.",
            "4. Positioning must follow the matrix: use target-domain real calibration unless the "
            "pooled bank is empirically close enough across all three targets; do not call the detector "
            "target-data-free.",
            "",
            "## Integrity",
            "",
            f"- Diagonal reconstructed locked-score max error: `{diagonal_error:.3g}`.",
            f"- Distribution rows: {len(shifts):,}; calibration videos probed: {len(bank_metadata)}.",
        ]
    )
    report.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run(args: argparse.Namespace) -> None:
    config = yaml.safe_load(args.config.read_text())
    evaluation = load_parts(args.scores_dir, "evaluation")
    calibration = load_parts(args.scores_dir, "locked_calibration")
    bank_manifest = json.loads(args.bank_manifest.read_text())
    global_spatial_ref, global_t1_ref = global_references(config)
    release = pd.read_csv(args.locked_scores, float_precision="round_trip")
    per_video = release[
        ["video_id", "dataset", "protocol_split", "subset", "source_model", "filename"]
    ].copy()
    for source in SOURCES:
        selected_ids = set(bank_manifest["banks"][source]["video_ids"])
        candidate_name = f"current_{source}"
        pieces = []
        for dataset in DATASETS:
            pieces.append(
                calibrate_cross(
                    evaluation[evaluation["dataset"].eq(dataset)],
                    calibration,
                    selected_ids,
                    candidate_name,
                    global_spatial_ref,
                    global_t1_ref,
                )
            )
        scores = pd.concat(pieces, ignore_index=True).rename(
            columns={branch: f"{branch}_{source}" for branch in ("G", "L", "S")}
        )
        per_video = per_video.merge(scores, on="video_id", validate="one_to_one")

    diagonal = np.empty(len(per_video), dtype=np.float64)
    for dataset in DATASETS:
        mask = per_video["dataset"].eq(dataset)
        diagonal[mask] = per_video.loc[mask, f"S_{dataset}"]
    diagonal_frame = per_video[["video_id"]].copy()
    diagonal_frame["reconstructed"] = diagonal
    diagonal_frame = diagonal_frame.merge(
        release[["video_id", "S"]], on="video_id", validate="one_to_one"
    )
    diagonal_error = float(
        np.max(np.abs(diagonal_frame["reconstructed"] - diagonal_frame["S"]))
    )
    if diagonal_error > 1e-15:
        raise ValueError(f"target-domain diagonal does not reproduce locked U0: {diagonal_error}")

    columns = [f"{branch}_{source}" for branch in ("G", "L", "S") for source in SOURCES]
    names = {column: column for column in columns}
    dataset_metrics, generator_metrics = metric_tables(
        per_video,
        seed=int(config["release"]["random_seed"]),
        score_columns=columns,
        config_names=names,
    )
    shift_rows = []
    for dataset, frame in per_video.groupby("dataset", sort=True):
        for source in SOURCES:
            for branch in ("G", "L", "S"):
                column = f"{branch}_{source}"
                for subset, group in frame.groupby("subset", sort=True):
                    values = group[column].to_numpy(dtype=np.float64)
                    shift_rows.append(
                        {
                            "calibration_source": source,
                            "test_dataset": dataset,
                            "branch": branch,
                            "subset": subset,
                            "count": len(values),
                            "mean": float(values.mean()),
                            "std": float(values.std(ddof=1)),
                            "q05": float(np.quantile(values, 0.05)),
                            "median": float(np.median(values)),
                            "q95": float(np.quantile(values, 0.95)),
                        }
                    )
    shifts = pd.DataFrame(shift_rows)
    bank_metadata = probe_calibration_videos(args.release_dir)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    per_video.to_csv(args.output_dir / "per_video_scores.csv", index=False)
    dataset_metrics.to_csv(args.output_dir / "dataset_metrics.csv", index=False)
    generator_metrics.to_csv(args.output_dir / "generator_metrics.csv", index=False)
    shifts.to_csv(args.output_dir / "score_distribution_shift.csv", index=False)
    bank_metadata.to_csv(args.output_dir / "calibration_video_metadata.csv", index=False)
    write_report(
        dataset_metrics,
        shifts,
        bank_metadata,
        bank_manifest,
        diagonal_error,
        args.report,
    )
    print(
        dataset_metrics[
            dataset_metrics["dataset"].eq("Macro-3")
            & dataset_metrics["config"].str.startswith("S_")
        ].to_string(index=False)
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", type=Path, default=ROOT / "configs/alpha_stalled_u0_locked.yaml"
    )
    parser.add_argument(
        "--scores-dir", type=Path, default=ROOT / "results/u0_cross_and_oas/scores"
    )
    parser.add_argument(
        "--bank-manifest", type=Path, default=ROOT / "release/u0/cross_calibration_banks.json"
    )
    parser.add_argument("--release-dir", type=Path, default=ROOT / "release/u0")
    parser.add_argument(
        "--locked-scores", type=Path, default=ROOT / "release/u0/final_video_scores.csv"
    )
    parser.add_argument(
        "--output-dir", type=Path, default=ROOT / "results/u0_cross_dataset_calibration"
    )
    parser.add_argument(
        "--report", type=Path, default=ROOT / "reports/u0_cross_dataset_calibration.md"
    )
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
