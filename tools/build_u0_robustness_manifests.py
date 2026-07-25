#!/usr/bin/env python3
"""Lock generator-balanced robustness and real-only injection subsets."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]

from build_u0_release_manifests import resolve_video, write_json


SEED = 20260725


def stable_rank(video_id: str, purpose: str) -> str:
    return hashlib.sha256(f"{SEED}\0{purpose}\0{video_id}".encode()).hexdigest()


def select(frame: pd.DataFrame, count: int, purpose: str) -> pd.DataFrame:
    ranked = frame.copy()
    ranked["_rank"] = ranked["video_id"].map(lambda value: stable_rank(value, purpose))
    result = ranked.sort_values("_rank").head(count).drop(columns="_rank")
    if len(result) != count:
        raise ValueError(f"{purpose}: requested {count}, found {len(result)}")
    return result


def run(args: argparse.Namespace) -> None:
    evaluation_payload = json.loads(args.evaluation_manifest.read_text())
    calibration_payload = json.loads(args.calibration_manifest.read_text())
    evaluation = pd.DataFrame(evaluation_payload["videos"])
    calibration = pd.DataFrame(calibration_payload["videos"])
    selected = []
    for dataset, frame in evaluation.groupby("dataset", sort=True):
        real = frame[frame["subset"].eq("real")]
        # Preserve source representation when a dataset has multiple real sources.
        real_pieces = []
        source_counts = real["source_model"].value_counts().sort_index()
        exact = source_counts * (args.real_per_dataset / source_counts.sum())
        allocation = exact.astype(int)
        for source in sorted(
            source_counts.index,
            key=lambda key: (-(exact.loc[key] - allocation.loc[key]), key),
        ):
            if allocation.sum() >= args.real_per_dataset:
                break
            allocation[source] += 1
        for source, count in allocation.items():
            real_pieces.append(
                select(
                    real[real["source_model"].eq(source)],
                    int(count),
                    f"robustness-real-{dataset}-{source}",
                )
            )
        selected.append(pd.concat(real_pieces, ignore_index=True))
        fake = frame[frame["subset"].eq("annotated")]
        for generator, group in fake.groupby("source_model", sort=True):
            selected.append(
                select(
                    group,
                    args.fake_per_generator,
                    f"robustness-fake-{dataset}-{generator}",
                )
            )
    robustness = pd.concat(selected, ignore_index=True)
    if robustness["video_id"].duplicated().any():
        raise ValueError("robustness subset contains duplicate video IDs")
    missing = [
        item for item in robustness.to_dict("records") if not resolve_video(item["video_path"]).is_file()
    ]
    if missing:
        raise FileNotFoundError(f"missing robustness videos: {len(missing)}")
    calibration_ids = set(calibration["video_id"])
    if set(robustness["video_id"]) & calibration_ids:
        raise ValueError("robustness evaluation subset overlaps calibration")

    injection_pieces = []
    real_subset = robustness[robustness["subset"].eq("real")]
    for dataset, frame in real_subset.groupby("dataset", sort=True):
        injection_pieces.append(
            select(frame, args.injection_real_per_dataset, f"injection-{dataset}")
        )
    injection = pd.concat(injection_pieces, ignore_index=True)
    if set(injection["video_id"]) - set(robustness["video_id"]):
        raise ValueError("injection subset is not nested in robustness reals")

    robustness_records = robustness.sort_values(
        ["dataset", "subset", "source_model", "video_id"]
    ).to_dict("records")
    injection_records = injection.sort_values(["dataset", "video_id"]).to_dict("records")
    write_json(
        args.robustness_output,
        {
            "schema_version": "u0_robustness_subset_v1",
            "locked_before_perturbation_metrics": True,
            "selection_seed": SEED,
            "real_per_dataset": args.real_per_dataset,
            "fake_per_generator": args.fake_per_generator,
            "video_count": len(robustness_records),
            "dataset_counts": robustness.groupby("dataset").size().astype(int).to_dict(),
            "generator_counts": [
                {"dataset": dataset, "generator": generator, "count": int(count)}
                for (dataset, generator), count in robustness[
                    robustness["subset"].eq("annotated")
                ].groupby(["dataset", "source_model"]).size().items()
            ],
            "calibration_overlap_count": 0,
            "videos": robustness_records,
        },
    )
    write_json(
        args.injection_output,
        {
            "schema_version": "u0_injection_subset_v1",
            "locked_before_injection_metrics": True,
            "selection_seed": SEED,
            "real_per_dataset": args.injection_real_per_dataset,
            "video_count": len(injection_records),
            "calibration_overlap_count": 0,
            "videos": injection_records,
        },
    )
    print(
        f"robustness={len(robustness_records)} injection={len(injection_records)} "
        f"calibration_overlap=0"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--evaluation-manifest",
        type=Path,
        default=ROOT / "release/u0/evaluation_manifest.json",
    )
    parser.add_argument(
        "--calibration-manifest",
        type=Path,
        default=ROOT / "release/u0/calibration_manifest.json",
    )
    parser.add_argument("--real-per-dataset", type=int, default=200)
    parser.add_argument("--fake-per-generator", type=int, default=50)
    parser.add_argument("--injection-real-per-dataset", type=int, default=100)
    parser.add_argument(
        "--robustness-output",
        type=Path,
        default=ROOT / "release/u0/robustness_subset_manifest.json",
    )
    parser.add_argument(
        "--injection-output",
        type=Path,
        default=ROOT / "release/u0/injection_subset_manifest.json",
    )
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
