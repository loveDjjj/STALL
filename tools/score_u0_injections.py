#!/usr/bin/env python3
"""Score prelocked synthetic anomalies and PatchD2 localization heatmaps."""

from __future__ import annotations

import argparse
import glob
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml
from sklearn.metrics import average_precision_score


ROOT = Path(__file__).resolve().parents[1]
for directory in (ROOT / "src", ROOT / "tools"):
    if str(directory) not in sys.path:
        sys.path.insert(0, str(directory))

from score_multi_window import decode_selected_frames
from score_u0_locked_windows import load_raw_params, resolve_video, stable_shard
from score_u0_robustness import donor_window_id, embed_sequence, score_condition
from stable_whitening import l2_normalized_second_order
from stall_patch import PatchSTALL
from u0_injections import CONDITIONS, InjectionResult, inject


KEY_COLUMNS = [
    "video_id",
    "dataset",
    "protocol_split",
    "subset",
    "source_model",
    "filename",
]


def load_k1_windows(directory: Path, selected: set[str]) -> dict[str, list[list[int]]]:
    paths = sorted(Path(path) for path in glob.glob(str(directory / "*.csv")))
    if not paths:
        raise FileNotFoundError(f"no K1 score shards: {directory}")
    frame = pd.concat(
        [pd.read_csv(path, usecols=["video_id", "frame_indices"]) for path in paths],
        ignore_index=True,
    )
    result = {
        row.video_id: [[int(value) for value in json.loads(row.frame_indices)]]
        for row in frame.itertuples(index=False)
        if row.video_id in selected
    }
    if set(result) != selected:
        raise ValueError("K1 windows are incomplete for injection videos")
    return result


def decode_frame_map(item: dict, indices: set[int], seek_gap: int) -> dict[int, np.ndarray]:
    ordered = sorted(int(value) for value in indices)
    frames = decode_selected_frames(resolve_video(item["video_path"]), ordered, seek_gap)
    return {value: frames[index] for index, value in enumerate(ordered)}


def embeddings_from_frame_map(
    extractor: PatchSTALL,
    source: dict[int, np.ndarray],
    windows: list[list[int]],
    replacements: dict[int, np.ndarray],
    frame_batch_size: int,
) -> list[dict]:
    unique = sorted({int(value) for window in windows for value in window})
    frames = np.stack([replacements.get(value, source[value]) for value in unique])
    embedded = embed_sequence(extractor, frames, frame_batch_size)
    positions = {value: index for index, value in enumerate(unique)}
    return [
        {
            "global": embedded["global"][[positions[int(value)] for value in window]],
            "patch": embedded["patch"][[positions[int(value)] for value in window]],
        }
        for window in windows
    ]


@torch.inference_mode()
def patch_d2_anomaly(patch: np.ndarray, params: dict, device: str) -> np.ndarray:
    features = l2_normalized_second_order(
        torch.from_numpy(patch[None].astype(np.float32))
    )[0]
    target = torch.device(device)
    values = features.reshape(-1, features.shape[-1]).to(target, dtype=torch.float64)
    parameter = params["patch_d2"]
    mean = torch.as_tensor(parameter.mean, dtype=torch.float64, device=target)
    whitening = torch.as_tensor(parameter.whitening, dtype=torch.float64, device=target)
    white = torch.mm(values - mean, whitening)
    anomaly = 0.5 * torch.sum(white * white, dim=1)
    return anomaly.reshape(features.shape[0], features.shape[1]).cpu().numpy()


def d2_labels(mask: np.ndarray) -> np.ndarray:
    return np.stack([mask[index : index + 3].any(axis=0) for index in range(14)]).reshape(14, 196)


def localization_metrics(
    anomaly: np.ndarray, result: InjectionResult
) -> dict[str, float]:
    if not result.positive_anomaly:
        return {
            "heatmap_auprc": np.nan,
            "inside_anomaly": np.nan,
            "outside_anomaly": np.nan,
            "inside_minus_outside": np.nan,
            "topk_patch_hit_rate": np.nan,
            "temporal_hit": np.nan,
        }
    labels = d2_labels(result.patch_mask)
    flat_label = labels.reshape(-1)
    flat_score = anomaly.reshape(-1)
    spatial_label = labels.any(axis=0)
    spatial_score = anomaly.max(axis=0)
    top_k = int(spatial_label.sum())
    top_positions = np.argsort(spatial_score, kind="mergesort")[-top_k:]
    temporal_label = labels.any(axis=1)
    temporal_score = anomaly.max(axis=1)
    return {
        "heatmap_auprc": float(average_precision_score(flat_label, flat_score)),
        "inside_anomaly": float(flat_score[flat_label].mean()),
        "outside_anomaly": float(flat_score[~flat_label].mean()) if (~flat_label).any() else np.nan,
        "inside_minus_outside": (
            float(flat_score[flat_label].mean() - flat_score[~flat_label].mean())
            if (~flat_label).any()
            else np.nan
        ),
        "topk_patch_hit_rate": float(spatial_label[top_positions].mean()),
        "temporal_hit": float(temporal_label[int(np.argmax(temporal_score))]),
    }


def save_heatmaps(
    path: Path,
    heatmaps: list[np.ndarray],
    labels: list[np.ndarray],
    target_window: int,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp.npz")
    np.savez_compressed(
        temporary,
        conditions=np.asarray(CONDITIONS),
        anomaly=np.stack(heatmaps).astype(np.float32),
        labels=np.stack(labels).astype(bool),
        target_window=np.asarray([target_window], dtype=np.int32),
    )
    temporary.replace(path)


def completed_ids(checkpoint: Path) -> tuple[set[str], int]:
    paths = sorted(checkpoint.glob("part_*.csv"))
    completed: set[str] = set()
    for path in paths:
        completed.update(pd.read_csv(path, usecols=["video_id"])["video_id"].astype(str))
    return completed, len(paths)


def run(args: argparse.Namespace) -> None:
    config = yaml.safe_load(args.config.read_text())
    subset = json.loads(args.subset_manifest.read_text())
    plan = json.loads(args.plan.read_text())
    release_evaluation = json.loads((args.release_dir / "evaluation_manifest.json").read_text())
    all_rows = {item["video_id"]: item for item in release_evaluation["videos"]}
    selected_records = [item for item in subset["videos"] if item["dataset"] == args.dataset]
    selected_ids = {item["video_id"] for item in selected_records}
    k1_windows = load_k1_windows(args.k1_raw_dir, selected_ids)
    k3_all = json.loads((args.release_dir / "frame_indices.json").read_text())["videos"]
    rows = pd.DataFrame(selected_records)
    rows = rows[
        rows["video_id"].map(lambda value: stable_shard(str(value), args.num_shards) == args.shard_index)
    ].reset_index(drop=True)
    if args.debug_videos is not None:
        rows = rows.head(args.debug_videos)
    checkpoint = (
        args.output_dir
        / "checkpoints"
        / args.dataset
        / f"shard_{args.shard_index:02d}_of_{args.num_shards:02d}"
    )
    checkpoint.mkdir(parents=True, exist_ok=True)
    completed, part_index = completed_ids(checkpoint)
    pending = rows[~rows["video_id"].isin(completed)].reset_index(drop=True)
    print(
        f"dataset={args.dataset} shard={args.shard_index}/{args.num_shards} "
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
        video_id = item["video_id"]
        k3_windows = k3_all[video_id]
        target_window = int(plan["target_k3_window"][video_id])
        target_indices = [int(value) for value in k3_windows[target_window]]
        all_indices = {
            int(value)
            for windows in (k1_windows[video_id], k3_windows)
            for window in windows
            for value in window
        }
        try:
            source = decode_frame_map(item, all_indices, args.seek_gap)
            target_pixels = np.stack([source[value] for value in target_indices])
            donor_id = plan["scene_cut_donor"][video_id]
            donor = all_rows[donor_id]
            donor_k3 = k3_all[donor_id]
            donor_index = donor_window_id(target_window, len(k3_windows), len(donor_k3))
            donor_indices = [int(value) for value in donor_k3[donor_index]]
            donor_source = decode_frame_map(donor, set(donor_indices), args.seek_gap)
            donor_pixels = np.stack([donor_source[value] for value in donor_indices])
            results = {
                condition: inject(
                    target_pixels,
                    condition,
                    donor=donor_pixels if condition == "L7_scene_cut" else None,
                )
                for condition in CONDITIONS
            }
            video_rows = []
            heatmaps = []
            heatmap_labels = []
            for condition in CONDITIONS:
                result = results[condition]
                condition_affected_indices = {
                    target_indices[position]
                    for position, affected in enumerate(result.temporal_mask)
                    if affected
                }
                replacements = {
                    native_index: result.frames[position]
                    for position, native_index in enumerate(target_indices)
                }
                embedded_by_sampling = {}
                for sampling, sampling_windows in (
                    ("k1", k1_windows[video_id]),
                    ("k3", k3_windows),
                ):
                    embedded = embeddings_from_frame_map(
                        extractor,
                        source,
                        sampling_windows,
                        replacements,
                        args.frame_batch_size,
                    )
                    embedded_by_sampling[sampling] = embedded
                    scored = score_condition(embedded, params, args.score_device)
                    for window_id, raw in enumerate(scored):
                        window_indices = set(int(value) for value in sampling_windows[window_id])
                        video_rows.append(
                            {
                                **{column: item[column] for column in KEY_COLUMNS},
                                "video_path": item["video_path"],
                                "duration_seconds": float(item["duration_seconds"]),
                                "sampling": sampling,
                                "effective_k": len(sampling_windows),
                                "condition": condition,
                                "window_id": window_id,
                                "target_k3_window": target_window,
                                "is_target_window": sampling == "k3" and window_id == target_window,
                                "affected_native_overlap": len(
                                    window_indices.intersection(condition_affected_indices)
                                ),
                                "frame_indices": json.dumps(sampling_windows[window_id], separators=(",", ":")),
                                **raw,
                            }
                        )
                target_patch = embedded_by_sampling["k3"][target_window]["patch"]
                anomaly = patch_d2_anomaly(target_patch, params, args.score_device)
                labels = d2_labels(result.patch_mask)
                heatmaps.append(anomaly)
                heatmap_labels.append(labels)
                summary = localization_metrics(anomaly, result)
                for target_row in video_rows:
                    if (
                        target_row["condition"] == condition
                        and target_row["sampling"] == "k3"
                        and target_row["window_id"] == target_window
                    ):
                        target_row.update(summary)
            heatmap_path = (
                args.output_dir
                / "heatmaps"
                / args.dataset
                / f"shard_{args.shard_index:02d}_of_{args.num_shards:02d}"
                / f"{video_id}.npz"
            )
            save_heatmaps(heatmap_path, heatmaps, heatmap_labels, target_window)
            buffer.extend(video_rows)
        except Exception as error:
            failures.append(
                {
                    **{column: item[column] for column in KEY_COLUMNS},
                    "error": f"{type(error).__name__}: {error}",
                }
            )
        processed = index + 1
        if len(buffer) >= args.checkpoint_rows or processed == len(pending):
            if buffer:
                path = checkpoint / f"part_{part_index:06d}.csv"
                temporary = path.with_suffix(".tmp.csv")
                pd.DataFrame(buffer).to_csv(temporary, index=False)
                temporary.replace(path)
                buffer.clear()
                part_index += 1
        if processed == 1 or processed % 5 == 0 or processed == len(pending):
            print(
                f"processed={processed}/{len(pending)} failures={len(failures)} "
                f"elapsed={time.perf_counter()-started:.1f}s",
                flush=True,
            )
    pd.DataFrame(failures, columns=[*KEY_COLUMNS, "error"]).to_csv(
        checkpoint / "failures.csv", index=False
    )
    if failures:
        raise RuntimeError(f"injection scoring failures: {len(failures)}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
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
        "--subset-manifest",
        type=Path,
        default=ROOT / "release/u0/injection_subset_manifest.json",
    )
    parser.add_argument(
        "--plan", type=Path, default=ROOT / "release/u0/injection_plan.json"
    )
    parser.add_argument(
        "--k1-raw-dir", type=Path, default=ROOT / "results/u0_core_ablation/k1_raw"
    )
    parser.add_argument(
        "--output-dir", type=Path, default=ROOT / "results/u0_injection"
    )
    args = parser.parse_args()
    if not 0 <= args.shard_index < args.num_shards:
        parser.error("shard-index must be in [0, num-shards)")
    return args


if __name__ == "__main__":
    run(parse_args())
