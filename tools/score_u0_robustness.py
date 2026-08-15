#!/usr/bin/env python3
"""Score the prelocked U0 robustness conditions at raw-component level."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml


ROOT = Path(__file__).resolve().parents[1]
for directory in (ROOT / "src", ROOT / "tools"):
    if str(directory) not in sys.path:
        sys.path.insert(0, str(directory))

from alpha_stalled.video_io import decode_selected_frames
from alpha_stalled.release_io import resolve_required_video as resolve_video, video_id_shard
from alpha_stalled.parameters import load_raw_params
from alpha_stalled.artifacts import checkpoint_completed_ids
from alpha_stalled.global_branch import score_global_raw
from alpha_stalled.local_branch import score_local_raw
from stall_patch import PatchSTALL
from u0_perturbations import (
    CONDITIONS,
    h264_roundtrip,
    insert_scene_cut,
    resize_half_restore,
    temporal_perturbation,
)


KEY_COLUMNS = [
    "video_id",
    "dataset",
    "protocol_split",
    "subset",
    "source_model",
    "filename",
]


def load_protocol(
    split: str,
    dataset: str,
    release_dir: Path,
    robustness_manifest: Path,
    plan_path: Path,
) -> tuple[
    pd.DataFrame,
    dict[str, dict],
    dict[str, dict[str, list[list[int]]]],
    dict[str, dict[str, list[list[int]]]],
    dict[str, str],
]:
    calibration_payload = json.loads((release_dir / "calibration_manifest.json").read_text())
    evaluation_payload = json.loads((release_dir / "evaluation_manifest.json").read_text())
    all_rows = {
        item["video_id"]: item
        for payload in (calibration_payload, evaluation_payload)
        for item in payload["videos"]
    }
    frames = json.loads((release_dir / "frame_indices.json").read_text())
    if split == "calibration":
        records = [item for item in calibration_payload["videos"] if item["dataset"] == dataset]
        donor_key = "calibration_scene_cut_donor"
    else:
        robustness = json.loads(robustness_manifest.read_text())
        records = [item for item in robustness["videos"] if item["dataset"] == dataset]
        donor_key = "evaluation_scene_cut_donor"
    plan = json.loads(plan_path.read_text())
    donors = {video_id: donor for video_id, donor in plan[donor_key].items() if video_id in {x['video_id'] for x in records}}
    rows = pd.DataFrame(records)
    if rows["video_id"].duplicated().any() or len(donors) != len(rows):
        raise ValueError("robustness rows or donor map are incomplete")
    all_windows: dict[str, dict[str, list[list[int]]]] = {}
    for video_id, k3_windows in frames["videos"].items():
        all_windows[video_id] = {"k3": k3_windows}
    for video_id, k1_window in frames["calibration_reference_windows"].items():
        all_windows[video_id]["k1"] = [k1_window]
    windows: dict[str, dict[str, list[list[int]]]] = {}
    for item in records:
        groups = all_windows[item["video_id"]]
        windows[item["video_id"]] = (
            {"k1": groups["k1"], "k3": groups["k3"]}
            if split == "calibration"
            else {"k3": groups["k3"]}
        )
    return rows, all_rows, windows, all_windows, donors


def donor_window_id(target_id: int, target_k: int, donor_k: int) -> int:
    if target_k < 1 or donor_k < 1:
        raise ValueError("effective K must be positive")
    if target_k == 1:
        return (donor_k - 1) // 2
    return int(np.rint((target_id / float(target_k - 1)) * (donor_k - 1)))


def decode_windows(item: dict, windows: list[list[int]], seek_gap: int) -> list[np.ndarray]:
    unique = sorted({int(index) for window in windows for index in window})
    frames = decode_selected_frames(resolve_video(item["video_path"]), unique, seek_gap)
    positions = {value: index for index, value in enumerate(unique)}
    return [frames[[positions[int(value)] for value in window]] for window in windows]


def embed_sequence(extractor: PatchSTALL, frames: np.ndarray, frame_batch_size: int) -> dict:
    output = extractor.frames_to_global_patch_embeddings(
        [frames], batch_size=frame_batch_size
    )[0]
    if tuple(output["grid_size"]) != (14, 14):
        raise ValueError(f"unexpected patch grid {output['grid_size']}")
    return {"global": output["global"], "patch": output["patch"]}


def embed_window_list(
    extractor: PatchSTALL, windows: list[np.ndarray], frame_batch_size: int
) -> list[dict]:
    lengths = [len(window) for window in windows]
    combined = np.concatenate(windows, axis=0)
    embedded = embed_sequence(extractor, combined, frame_batch_size)
    output = []
    offset = 0
    for length in lengths:
        output.append(
            {
                "global": embedded["global"][offset : offset + length],
                "patch": embedded["patch"][offset : offset + length],
            }
        )
        offset += length
    return output


def original_embeddings(
    extractor: PatchSTALL,
    item: dict,
    windows: list[list[int]],
    seek_gap: int,
    frame_batch_size: int,
) -> tuple[list[np.ndarray], list[dict]]:
    unique = sorted({int(index) for window in windows for index in window})
    frames = decode_selected_frames(resolve_video(item["video_path"]), unique, seek_gap)
    positions = {value: index for index, value in enumerate(unique)}
    window_positions = [[positions[int(value)] for value in window] for window in windows]
    embedded = embed_sequence(extractor, frames, frame_batch_size)
    pixel_windows = [frames[positions] for positions in window_positions]
    embedding_windows = [
        {"global": embedded["global"][positions], "patch": embedded["patch"][positions]}
        for positions in window_positions
    ]
    return pixel_windows, embedding_windows


def temporal_embeddings(source: dict, condition: str, key: str) -> dict:
    positions = np.arange(len(source["global"]), dtype=np.int64)[:, None, None, None]
    transformed = temporal_perturbation(positions, condition, key).reshape(-1).astype(int)
    return {
        "global": source["global"][transformed],
        "patch": source["patch"][transformed],
    }


@torch.inference_mode()
def score_condition(
    windows: list[dict], params: dict, device: str
) -> list[dict[str, float]]:
    global_batch = torch.from_numpy(np.stack([item["global"] for item in windows]).astype(np.float32))
    patch_batch = torch.from_numpy(np.stack([item["patch"] for item in windows]).astype(np.float32))
    global_raw = score_global_raw(
        global_batch,
        params["global_spatial"],
        params["global_t1"],
        device=device,
    )
    local_raw = score_local_raw(
        patch_batch,
        params["patch_spatial"],
        params["patch_d2"],
        device=device,
    )
    return [
        {
            "global_spatial_raw": global_raw.spatial[index],
            "global_t1_raw": global_raw.temporal_t1[index],
            "patch_spatial_raw": local_raw.patch_spatial[index],
            "patch_d2_raw": local_raw.patch_temporal[index],
        }
        for index in range(len(windows))
    ]


def score_video(
    item: dict,
    donor: dict,
    windows: list[list[int]],
    donor_windows: list[list[int]],
    sampling: str,
    extractor: PatchSTALL,
    params: dict,
    score_device: str,
    seek_gap: int,
    frame_batch_size: int,
) -> list[dict]:
    pixel_windows, original = original_embeddings(
        extractor, item, windows, seek_gap, frame_batch_size
    )
    condition_embeddings: dict[str, list[dict]] = {"R0_original": original}
    resized = [resize_half_restore(window) for window in pixel_windows]
    condition_embeddings["R3_resize_half_restore"] = embed_window_list(
        extractor, resized, frame_batch_size
    )
    for condition, crf in (("R1_h264_crf23", 23), ("R2_h264_crf35", 35)):
        variants = [h264_roundtrip(window, crf=crf) for window in pixel_windows]
        condition_embeddings[condition] = embed_window_list(
            extractor, variants, frame_batch_size
        )
    donor_pixels = decode_windows(donor, donor_windows, seek_gap)
    scene_windows = []
    for window_id, target in enumerate(pixel_windows):
        selected = donor_window_id(window_id, len(pixel_windows), len(donor_pixels))
        scene_windows.append(insert_scene_cut(target, donor_pixels[selected]))
    condition_embeddings["R8_scene_cut"] = embed_window_list(
        extractor, scene_windows, frame_batch_size
    )
    for condition in ("R4_drop10", "R5_drop25", "R6_repeat10", "R7_repeat25", "R9_4fps"):
        condition_embeddings[condition] = [
            temporal_embeddings(source, condition, f"{item['video_id']}/{window_id}")
            for window_id, source in enumerate(original)
        ]

    rows = []
    for condition in CONDITIONS:
        scored = score_condition(condition_embeddings[condition], params, score_device)
        for window_id, raw in enumerate(scored):
            rows.append(
                {
                    **{column: item[column] for column in KEY_COLUMNS},
                    "video_path": item["video_path"],
                    "duration_seconds": float(item["duration_seconds"]),
                    "effective_k": len(windows),
                    "sampling": sampling,
                    "condition": condition,
                    "window_id": window_id,
                    "input_frame_count": len(condition_embeddings[condition][window_id]["global"]),
                    "frame_indices": json.dumps(windows[window_id], separators=(",", ":")),
                    **raw,
                }
            )
    return rows


def completed_ids(checkpoint: Path) -> tuple[set[str], int]:
    completed, parts = checkpoint_completed_ids(checkpoint, cast_str=True)
    return {str(value) for value in completed}, len(parts)


def run(args: argparse.Namespace) -> None:
    config = yaml.safe_load(args.config.read_text())
    rows, all_rows, windows, all_windows, donors = load_protocol(
        args.split,
        args.dataset,
        args.release_dir,
        args.robustness_manifest,
        args.plan,
    )
    rows = rows[
        rows["video_id"].map(
            lambda value: video_id_shard(str(value), args.num_shards)
            == args.shard_index
        )
    ].reset_index(drop=True)
    if args.debug_videos is not None:
        rows = rows.head(args.debug_videos)
    checkpoint = (
        args.output_dir
        / "checkpoints"
        / args.split
        / args.dataset
        / f"shard_{args.shard_index:02d}_of_{args.num_shards:02d}"
    )
    checkpoint.mkdir(parents=True, exist_ok=True)
    completed, part_index = completed_ids(checkpoint)
    pending = rows[~rows["video_id"].isin(completed)].reset_index(drop=True)
    print(
        f"split={args.split} dataset={args.dataset} shard={args.shard_index}/{args.num_shards} "
        f"expected={len(rows)} completed={len(completed)} pending={len(pending)}",
        flush=True,
    )
    params = load_raw_params(config, args.dataset)
    extractor = PatchSTALL(args.extract_device, data_dict=None, load_dino=True)
    buffer = []
    failures = []
    started = time.perf_counter()
    for index, row in pending.iterrows():
        item = row.to_dict()
        donor_id = donors[item["video_id"]]
        donor = all_rows[donor_id]
        try:
            video_rows = []
            for sampling, sampling_windows in windows[item["video_id"]].items():
                video_rows.extend(
                    score_video(
                        item,
                        donor,
                        sampling_windows,
                        all_windows[donor_id][sampling],
                        sampling,
                        extractor,
                        params,
                        args.score_device,
                        args.seek_gap,
                        args.frame_batch_size,
                    )
                )
            buffer.extend(video_rows)
        except Exception as error:
            failures.append({**{column: item[column] for column in KEY_COLUMNS}, "error": f"{type(error).__name__}: {error}"})
        processed = index + 1
        if len(buffer) >= args.checkpoint_rows or processed == len(pending):
            if buffer:
                path = checkpoint / f"part_{part_index:06d}.csv"
                temporary = path.with_suffix(".tmp.csv")
                pd.DataFrame(buffer).to_csv(temporary, index=False)
                temporary.replace(path)
                buffer.clear()
                part_index += 1
        if processed == 1 or processed % 10 == 0 or processed == len(pending):
            print(
                f"processed={processed}/{len(pending)} failures={len(failures)} "
                f"elapsed={time.perf_counter()-started:.1f}s",
                flush=True,
            )
    pd.DataFrame(failures, columns=[*KEY_COLUMNS, "error"]).to_csv(
        checkpoint / "failures.csv", index=False
    )
    if failures:
        raise RuntimeError(f"robustness scoring failures: {len(failures)}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", choices=("calibration", "evaluation"), required=True)
    parser.add_argument(
        "--dataset", choices=("comgenvid", "videofeedback", "genvideo"), required=True
    )
    parser.add_argument("--shard-index", type=int, required=True)
    parser.add_argument("--num-shards", type=int, default=4)
    parser.add_argument("--extract-device", default="cuda:0")
    parser.add_argument("--score-device", default="cuda:0")
    parser.add_argument("--frame-batch-size", type=int, default=32)
    parser.add_argument("--seek-gap", type=int, default=48)
    parser.add_argument("--checkpoint-rows", type=int, default=1000)
    parser.add_argument("--debug-videos", type=int)
    parser.add_argument(
        "--config", type=Path, default=ROOT / "configs/alpha_stalled_u0_locked.yaml"
    )
    parser.add_argument("--release-dir", type=Path, default=ROOT / "release/u0")
    parser.add_argument(
        "--robustness-manifest",
        type=Path,
        default=ROOT / "release/u0/robustness_subset_manifest.json",
    )
    parser.add_argument(
        "--plan", type=Path, default=ROOT / "release/u0/robustness_perturbation_plan.json"
    )
    parser.add_argument(
        "--output-dir", type=Path, default=ROOT / "results/u0_robustness"
    )
    args = parser.parse_args()
    if not 0 <= args.shard_index < args.num_shards:
        parser.error("shard-index must be in [0, num-shards)")
    return args


if __name__ == "__main__":
    run(parse_args())
