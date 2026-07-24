#!/usr/bin/env python3
"""Stream DINOv3 extraction and frozen Alpha-STALLED scoring for many windows."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from eval_patch_fast import FastPatchScorer
from stall import STALL
from stall_patch import PatchSTALL


KEY_COLUMNS = ["dataset", "protocol_split", "subset", "source_model", "filename"]
SAMPLINGS = ("K1_current", "K3_uniform", "K5_uniform", "all_nonoverlap")
LOCAL_PARAMS = {
    "comgenvid": Path("/tmp/alpha_stalled_local_d2_work/comgenvid_L0.npz"),
    "videofeedback": Path("/tmp/alpha_stalled_local_d2_work/videofeedback_L0.npz"),
    "genvideo": Path("/tmp/alpha_stalled_local_d2_work/genvideo_L0.npz"),
}


def stable_shard(row: pd.Series, num_shards: int) -> int:
    key = "|".join(str(row[column]) for column in KEY_COLUMNS)
    return int(hashlib.sha256(key.encode("utf-8")).hexdigest()[:16], 16) % num_shards


def video_key(row: pd.Series | dict) -> tuple[str, ...]:
    return tuple(str(row[column]) for column in KEY_COLUMNS)


def resolve_video_path(value: str, repo_root: Path = REPO_ROOT) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    candidates = (Path.cwd() / path, repo_root / path, repo_root.parent / path)
    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()
    raise FileNotFoundError(f"video not found: {value}")


def _decode_spans(targets: list[int], seek_gap: int) -> list[tuple[int, int]]:
    if not targets:
        return []
    spans: list[tuple[int, int]] = []
    start = previous = targets[0]
    for index in targets[1:]:
        if index - previous > seek_gap:
            spans.append((start, previous))
            start = index
        previous = index
    spans.append((start, previous))
    return spans


def decode_selected_frames(
    video_path: Path,
    frame_indices: list[int],
    seek_gap: int = 64,
) -> np.ndarray:
    """Decode selected native frames with one sequential pass per nearby span."""
    targets = sorted(set(int(index) for index in frame_indices))
    if not targets:
        raise ValueError("frame_indices is empty")
    target_set = set(targets)
    decoded: dict[int, np.ndarray] = {}
    cap = cv2.VideoCapture(str(video_path), cv2.CAP_FFMPEG)
    if not cap.isOpened():
        cap.release()
        cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise ValueError(f"cannot open video: {video_path}")
    try:
        for start, end in _decode_spans(targets, seek_gap):
            cap.set(cv2.CAP_PROP_POS_FRAMES, start)
            for index in range(start, end + 1):
                ok, frame = cap.read()
                if not ok:
                    break
                if index in target_set:
                    decoded[index] = frame
    finally:
        cap.release()
    missing = [index for index in targets if index not in decoded]
    if missing:
        raise ValueError(f"missing {len(missing)} decoded frames, first={missing[:5]}")
    return np.stack([decoded[index] for index in targets])


def load_windows(row: pd.Series, sampling: str) -> list[list[int]]:
    value = row[f"indices_{sampling}"]
    windows = json.loads(value)
    parsed = [[int(index) for index in window] for window in windows]
    if not parsed or any(len(window) != 16 for window in parsed):
        raise ValueError(f"invalid {sampling} windows for {video_key(row)}")
    if len({tuple(window) for window in parsed}) != len(parsed):
        raise ValueError(f"duplicate {sampling} windows for {video_key(row)}")
    return parsed


def decode_manifest_row(row: pd.Series, sampling: str, seek_gap: int) -> dict:
    windows = load_windows(row, sampling)
    unique_indices = sorted({index for window in windows for index in window})
    frames = decode_selected_frames(resolve_video_path(str(row["video_path"])), unique_indices, seek_gap)
    positions = {index: position for position, index in enumerate(unique_indices)}
    window_positions = [[positions[index] for index in window] for window in windows]
    return {
        "row": row,
        "windows": windows,
        "window_positions": window_positions,
        "unique_indices": unique_indices,
        "frames": frames,
    }


def decode_manifest_row_with_retries(
    row: pd.Series,
    sampling: str,
    seek_gap: int,
    attempts: int,
    retry_delay: float = 0.25,
) -> dict:
    if attempts < 1:
        raise ValueError("decode attempts must be positive")
    for attempt in range(attempts):
        try:
            return decode_manifest_row(row, sampling, seek_gap)
        except Exception:
            if attempt + 1 == attempts:
                raise
            time.sleep(retry_delay * (attempt + 1))
    raise AssertionError("unreachable")


def load_completed(checkpoint_dir: Path) -> tuple[set[tuple[str, ...]], list[Path]]:
    parts = sorted(checkpoint_dir.glob("part_*.csv"))
    completed: set[tuple[str, ...]] = set()
    for part in parts:
        frame = pd.read_csv(
            part, usecols=KEY_COLUMNS, float_precision="round_trip"
        )
        completed.update(tuple(str(value) for value in row) for row in frame.drop_duplicates().itertuples(index=False, name=None))
    return completed, parts


def score_batch(
    decoded: list[dict],
    extractor: PatchSTALL,
    global_scorer: STALL,
    local_scorer: FastPatchScorer,
    frame_batch_size: int,
) -> list[dict]:
    outputs = extractor.frames_to_global_patch_embeddings(
        [item["frames"] for item in decoded],
        batch_size=frame_batch_size,
    )
    global_windows: list[np.ndarray] = []
    patch_windows: list[np.ndarray] = []
    metadata: list[tuple[dict, int]] = []
    for item, output in zip(decoded, outputs):
        global_emb = output["global"]
        patch_emb = output["patch"]
        for window_id, positions in enumerate(item["window_positions"]):
            global_windows.append(global_emb[positions])
            patch_windows.append(patch_emb[positions])
            metadata.append((item, window_id))
    global_batch = np.stack(global_windows)
    patch_batch = np.stack(patch_windows)
    global_scores = global_scorer._scores_from_embs(global_batch)
    local_scores = local_scorer.score_batch(
        patch_batch,
        patch_temp_mode="same_grid_second_order",
        patch_spat_weight=0.1,
        patch_temp_weight=0.9,
        aggregation=local_scorer.aggregation_config.get("mode", "bottomk_mean"),
        bottomk_ratio=local_scorer.params_bottomk_ratio,
        temporal_run_length=local_scorer.params_temporal_run_length,
        patch_region_size=local_scorer.params_patch_region_size,
        global_batch=global_batch,
    )
    rows: list[dict] = []
    for index, (item, window_id) in enumerate(metadata):
        source = item["row"]
        gs = float(global_scores["spat_percentile"][index])
        gt1 = float(global_scores["temp_percentile"][index])
        global_final = float(global_scores["final_score"][index])
        ls = float(local_scores["patch_spat_percentile"][index])
        ld2 = float(local_scores["patch_temp_percentile"][index])
        local_final = float(local_scores["patch_final_score"][index])
        rows.append(
            {
                **{column: source[column] for column in KEY_COLUMNS},
                "video_path": source["video_path"],
                "duration_seconds": source["duration_seconds"],
                "sampling": source["active_sampling"],
                "effective_k": len(item["windows"]),
                "unique_frame_count": len(item["unique_indices"]),
                "window_id": window_id,
                "frame_indices": json.dumps(item["windows"][window_id], separators=(",", ":")),
                "global_spatial": gs,
                "global_t1": gt1,
                "G_k": global_final,
                "patch_spatial": ls,
                "patch_d2": ld2,
                "L_k": local_final,
                "S_k": 0.6 * global_final + 0.4 * local_final,
            }
        )
    return rows


def run(args: argparse.Namespace) -> None:
    manifest = pd.read_csv(args.manifest)
    manifest = manifest[manifest["dataset"] == args.dataset].copy()
    if args.protocol_split != "all":
        manifest = manifest[manifest["protocol_split"] == args.protocol_split].copy()
    manifest["active_sampling"] = args.sampling
    manifest = manifest[
        manifest.apply(lambda row: stable_shard(row, args.num_shards) == args.shard_index, axis=1)
    ].reset_index(drop=True)
    if args.debug_videos is not None:
        manifest = manifest.head(args.debug_videos).copy()

    checkpoint_dir = (
        args.checkpoint_dir
        / args.sampling
        / args.dataset
        / f"shard_{args.shard_index:02d}_of_{args.num_shards:02d}"
    )
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    completed, parts = load_completed(checkpoint_dir)
    pending = manifest[
        ~manifest.apply(lambda row: video_key(row) in completed, axis=1)
    ].reset_index(drop=True)
    print(
        f"dataset={args.dataset} sampling={args.sampling} shard={args.shard_index}/{args.num_shards} "
        f"expected={len(manifest)} completed={len(completed)} pending={len(pending)}",
        flush=True,
    )

    local_params = args.local_params or LOCAL_PARAMS[args.dataset]
    if not local_params.exists():
        raise FileNotFoundError(
            f"missing leakage-free L0 params: {local_params}; run tools/run_local_d2_residuals.py first"
        )
    global_data = np.load(args.global_params, allow_pickle=True)
    global_scorer = STALL(args.device, global_data, load_dino=False)
    local_scorer = FastPatchScorer(str(local_params), device=args.device)
    local_scorer.validate("same_grid_second_order")
    extractor = PatchSTALL(args.device, data_dict=None, load_dino=True)

    started = time.perf_counter()
    next_part = len(parts)
    failures: list[dict] = []
    for start in range(0, len(pending), args.video_batch_size):
        batch = [row for _, row in pending.iloc[start : start + args.video_batch_size].iterrows()]
        with ThreadPoolExecutor(max_workers=min(args.decode_workers, len(batch))) as executor:
            futures = [
                executor.submit(
                    decode_manifest_row_with_retries,
                    row,
                    args.sampling,
                    args.seek_gap,
                    args.decode_attempts,
                )
                for row in batch
            ]
            decoded: list[dict] = []
            for row, future in zip(batch, futures):
                try:
                    decoded.append(future.result())
                except Exception as exc:
                    failures.append(
                        {**{column: row[column] for column in KEY_COLUMNS}, "error": str(exc)}
                    )
        if decoded:
            rows = score_batch(
                decoded,
                extractor,
                global_scorer,
                local_scorer,
                args.frame_batch_size,
            )
            part = checkpoint_dir / f"part_{next_part:06d}.csv"
            temp = part.with_suffix(".tmp.csv")
            pd.DataFrame(rows).to_csv(temp, index=False)
            temp.replace(part)
            next_part += 1
        processed = min(start + args.video_batch_size, len(pending))
        if processed == len(pending) or processed % max(args.video_batch_size * 10, 1) == 0:
            elapsed = time.perf_counter() - started
            rate = processed / elapsed if elapsed else 0.0
            print(
                f"processed={processed}/{len(pending)} failures={len(failures)} "
                f"videos_per_sec={rate:.3f}",
                flush=True,
            )

    if local_scorer._executor is not None:
        local_scorer._executor.shutdown(wait=True)
    if failures:
        pd.DataFrame(failures).to_csv(checkpoint_dir / "failures.csv", index=False)

    completed, parts = load_completed(checkpoint_dir)
    expected = {video_key(row) for _, row in manifest.iterrows()}
    missing = expected - completed
    if missing:
        raise RuntimeError(f"incomplete shard: missing={len(missing)}, failures={len(failures)}")
    merged = pd.concat(
        [pd.read_csv(part, float_precision="round_trip") for part in parts],
        ignore_index=True,
    )
    merged = merged.drop_duplicates(KEY_COLUMNS + ["window_id"], keep="last")
    merged = merged.sort_values(KEY_COLUMNS + ["window_id"]).reset_index(drop=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temp = args.output.with_suffix(".tmp.csv")
    merged.to_csv(temp, index=False)
    temp.replace(args.output)
    elapsed = time.perf_counter() - started
    print(f"complete videos={len(expected)} windows={len(merged)} elapsed_sec={elapsed:.1f}")
    print(f"saved -> {args.output}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=tuple(LOCAL_PARAMS), required=True)
    parser.add_argument("--sampling", choices=SAMPLINGS, required=True)
    parser.add_argument(
        "--protocol-split",
        choices=("all", "calibration", "evaluation"),
        default="all",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=REPO_ROOT / "results/multi_window_joint_typicality/multi_window_manifest.csv",
    )
    parser.add_argument(
        "--global-params",
        type=Path,
        default=REPO_ROOT / "precomputed/stall_params_vatex_dino_v3.npz",
    )
    parser.add_argument("--local-params", type=Path)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--video-batch-size", type=int, default=4)
    parser.add_argument("--frame-batch-size", type=int, default=32)
    parser.add_argument("--decode-workers", type=int, default=4)
    parser.add_argument("--decode-attempts", type=int, default=3)
    parser.add_argument("--seek-gap", type=int, default=64)
    parser.add_argument("--debug-videos", type=int)
    parser.add_argument(
        "--checkpoint-dir",
        type=Path,
        default=Path("/tmp/alpha_stalled_multi_window"),
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.num_shards < 1 or not 0 <= args.shard_index < args.num_shards:
        parser.error("require num_shards >= 1 and 0 <= shard_index < num_shards")
    if args.decode_attempts < 1:
        parser.error("decode-attempts must be positive")
    return args


if __name__ == "__main__":
    run(parse_args())
