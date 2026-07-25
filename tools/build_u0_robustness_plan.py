#!/usr/bin/env python3
"""Lock perturbation donors and injection geometry before robustness metrics."""

from __future__ import annotations

import argparse
import glob
import hashlib
import json
import sys
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
for directory in (ROOT / "src", ROOT / "tools"):
    if str(directory) not in sys.path:
        sys.path.insert(0, str(directory))

from build_u0_release_manifests import sha256_file, write_json
from u0_perturbations import CONDITIONS


SEED = 20260725


def stable_rank(video_id: str, purpose: str) -> str:
    return hashlib.sha256(f"{SEED}\0{purpose}\0{video_id}".encode("utf-8")).hexdigest()


def donor_map(frame: pd.DataFrame, purpose: str) -> dict[str, str]:
    mapping: dict[str, str] = {}
    keys = ["dataset", "subset", "source_model"]
    for group_key, group in frame.groupby(keys, sort=True):
        name = "|".join(str(value) for value in group_key)
        ordered = sorted(
            group["video_id"].astype(str), key=lambda value: stable_rank(value, f"{purpose}|{name}")
        )
        if len(ordered) < 2:
            raise ValueError(f"scene-cut donor group has fewer than two videos: {name}")
        for index, video_id in enumerate(ordered):
            mapping[video_id] = ordered[(index + 1) % len(ordered)]
    if set(mapping) != set(frame["video_id"].astype(str)):
        raise ValueError("incomplete scene-cut donor mapping")
    if any(video_id == donor for video_id, donor in mapping.items()):
        raise ValueError("scene-cut mapping contains a self donor")
    return dict(sorted(mapping.items()))


def run(args: argparse.Namespace) -> None:
    robustness_payload = json.loads(args.robustness_manifest.read_text())
    calibration_payload = json.loads(args.calibration_manifest.read_text())
    injection_payload = json.loads(args.injection_manifest.read_text())
    robustness = pd.DataFrame(robustness_payload["videos"])
    calibration = pd.DataFrame(calibration_payload["videos"])
    injection_ids = {item["video_id"] for item in injection_payload["videos"]}
    if not injection_ids.issubset(set(robustness["video_id"])):
        raise ValueError("injection videos are not nested in the robustness subset")
    frame_payload = json.loads(args.frame_indices.read_text())
    k1_paths = sorted(Path(path) for path in glob.glob(str(args.k1_raw_dir / "*.csv")))
    if not k1_paths:
        raise FileNotFoundError(f"no locked K1 score shards: {args.k1_raw_dir}")
    k1_frame = pd.concat(
        [pd.read_csv(path, usecols=["video_id", "frame_indices"]) for path in k1_paths],
        ignore_index=True,
    )
    k1_windows = {
        row.video_id: [int(value) for value in json.loads(row.frame_indices)]
        for row in k1_frame.itertuples(index=False)
        if row.video_id in injection_ids
    }
    if set(k1_windows) != injection_ids:
        raise ValueError("locked K1 windows are incomplete for the injection subset")
    target_windows = {}
    affected_indices = {}
    k1_overlap = {}
    for video_id in sorted(injection_ids):
        k3 = frame_payload["videos"][video_id]
        target = int(stable_rank(video_id, "injection-target-window")[:16], 16) % len(k3)
        affected = [int(value) for value in k3[target][6:10]]
        k1 = set(k1_windows[video_id])
        target_windows[video_id] = target
        affected_indices[video_id] = affected
        k1_overlap[video_id] = len(k1.intersection(affected))

    write_json(
        args.robustness_plan,
        {
            "schema_version": "u0_robustness_plan_v1",
            "locked_before_perturbation_metrics": True,
            "selection_seed": SEED,
            "conditions": list(CONDITIONS),
            "definitions": {
                "R0_original": "locked sampled frames without perturbation",
                "R1_h264_crf23": "per-window libx264 yuv420p medium CRF 23 roundtrip at 8 FPS",
                "R2_h264_crf35": "per-window libx264 yuv420p medium CRF 35 roundtrip at 8 FPS",
                "R3_resize_half_restore": "0.5 area downscale followed by cubic restore",
                "R4_drop10": "drop round(0.10*T) hash-ranked sampled frames without replacement",
                "R5_drop25": "drop round(0.25*T) hash-ranked sampled frames without replacement",
                "R6_repeat10": "replace round(0.10*T) hash-ranked positions by the preceding frame",
                "R7_repeat25": "replace round(0.25*T) hash-ranked positions by the preceding frame",
                "R8_scene_cut": (
                    "first half target plus second half fixed same-group donor; donor windows are "
                    "matched by normalized temporal position, with the middle donor window used for K=1"
                ),
                "R9_4fps": "take positions 0,2,...,14 from each locked 16-frame window",
            },
            "evaluation_scene_cut_donor": donor_map(robustness, "robustness-evaluation"),
            "calibration_scene_cut_donor": donor_map(calibration, "robustness-calibration"),
            "input_sha256": {
                str(args.robustness_manifest.relative_to(ROOT)): sha256_file(args.robustness_manifest),
                str(args.calibration_manifest.relative_to(ROOT)): sha256_file(args.calibration_manifest),
            },
        },
    )
    write_json(
        args.injection_plan,
        {
            "schema_version": "u0_injection_plan_v1",
            "locked_before_injection_metrics": True,
            "selection_seed": SEED,
            "video_ids": sorted(injection_ids),
            "spatial_region": {
                "definition": "center 50% width by center 50% height",
                "area_fraction": 0.25,
                "patch_grid_bounds_14x14": [3, 10, 3, 10],
            },
            "temporal_region": {
                "affected_zero_based_frames": [6, 7, 8, 9],
                "reference_frame": 5,
            },
            "target_k3_window": target_windows,
            "affected_native_frame_indices": affected_indices,
            "k1_overlap_count": k1_overlap,
            "conditions": {
                "L0_original": "no injected anomaly",
                "L1_local_freeze4": "copy the center region from frame 5 into frames 6-9",
                "L2_local_repeat1": "copy the center region from frame 7 into frame 8",
                "L3_local_flicker": "multiply center-region brightness in frames 6-9 by 0.6/1.4 alternately",
                "L4_local_affine_jitter": "translate the center region by alternating +/-4% region width",
                "L5_local_temporal_shift": "copy center-region content from frames 8-11 into frames 6-9",
                "L6_full_frame_freeze4": "copy full frame 5 into frames 6-9",
                "L7_scene_cut": "first half target plus second half fixed same-source real donor",
            },
            "scene_cut_donor": {
                video_id: donor
                for video_id, donor in donor_map(
                    robustness[robustness["video_id"].isin(injection_ids)], "injection"
                ).items()
            },
            "input_sha256": {
                str(args.injection_manifest.relative_to(ROOT)): sha256_file(args.injection_manifest),
                str(args.frame_indices.relative_to(ROOT)): sha256_file(args.frame_indices),
                **{
                    str(path.relative_to(ROOT)): sha256_file(path)
                    for path in k1_paths
                },
            },
        },
    )
    print(
        f"locked robustness plan: evaluation={len(robustness)} calibration={len(calibration)} "
        f"injection={len(injection_ids)}"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--robustness-manifest",
        type=Path,
        default=ROOT / "release/u0/robustness_subset_manifest.json",
    )
    parser.add_argument(
        "--calibration-manifest",
        type=Path,
        default=ROOT / "release/u0/calibration_manifest.json",
    )
    parser.add_argument(
        "--injection-manifest",
        type=Path,
        default=ROOT / "release/u0/injection_subset_manifest.json",
    )
    parser.add_argument(
        "--frame-indices",
        type=Path,
        default=ROOT / "release/u0/frame_indices.json",
    )
    parser.add_argument(
        "--k1-raw-dir",
        type=Path,
        default=ROOT / "results/u0_core_ablation/k1_raw",
    )
    parser.add_argument(
        "--robustness-plan",
        type=Path,
        default=ROOT / "release/u0/robustness_perturbation_plan.json",
    )
    parser.add_argument(
        "--injection-plan",
        type=Path,
        default=ROOT / "release/u0/injection_plan.json",
    )
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
