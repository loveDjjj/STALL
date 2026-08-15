#!/usr/bin/env python3
"""Score original fixed-seed K1 windows from compact caches or source video."""

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
SRC_DIR = ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from alpha_stalled.video_io import decode_selected_frames  # noqa: E402
from alpha_stalled.release_io import (  # noqa: E402
    resolve_required_video as resolve_video,
    video_id_shard,
)
from alpha_stalled.parameters import load_raw_params  # noqa: E402
from alpha_stalled.global_branch import score_global_raw  # noqa: E402
from alpha_stalled.local_branch import score_local_raw  # noqa: E402
from alpha_stalled.whitening import (  # noqa: E402
    StableGaussianParams,
)
from stall_patch import PatchSTALL  # noqa: E402


CUSTOM_SIZE = {"comgenvid": 600, "videofeedback": 400, "genvideo": 1500}
KEY_COLUMNS = [
    "k1_task_id",
    "video_id",
    "dataset",
    "protocol_split",
    "subset",
    "source_model",
    "filename",
    "video_path",
    "duration_seconds",
    "protocol_duration_sec",
    "sampling",
    "frame_indices",
]


def local_param_path(params_dir: Path, dataset: str, duration: int) -> Path:
    return params_dir / f"{dataset}_{duration}s_n{CUSTOM_SIZE[dataset]}.npz"


def load_params(config: dict, params_dir: Path, dataset: str, duration: int) -> dict:
    params = load_raw_params(config, dataset)
    path = local_param_path(params_dir, dataset, duration)
    params["patch_spatial"] = StableGaussianParams.from_npz(
        str(path), "mu_patch_spat", "W_patch_spat", "calib_patch_spat_scores"
    )
    params["patch_d2"] = StableGaussianParams.from_npz(
        str(path), "mu_patch_temp", "W_patch_temp", "calib_patch_temp_scores"
    )
    return params


def score_batch(
    global_batch: torch.Tensor,
    patch_batch: torch.Tensor,
    params: dict,
    device: str,
) -> dict[str, np.ndarray]:
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
    return {
        "global_spatial_raw": global_raw.spatial,
        "global_t1_raw": global_raw.temporal_t1,
        "patch_spatial_raw": local_raw.patch_spatial,
        "patch_d2_raw": local_raw.patch_temporal,
    }


def load_cache(row: pd.Series) -> tuple[torch.Tensor, torch.Tensor]:
    path = ROOT / str(row["compact_cache_path"])
    payload = torch.load(path, weights_only=True, map_location="cpu")
    expected = [int(value) for value in json.loads(str(row["frame_indices"]))]
    cached = [int(value) for value in payload.get("frame_indices", [])]
    if cached != expected:
        raise ValueError(f"compact cache frame mismatch: {path}")
    frames = int(row["protocol_duration_sec"]) * 8
    if tuple(payload["global"].shape) != (frames, 1024):
        raise ValueError(f"invalid global compact cache: {path}")
    if tuple(payload["patch"].shape) != (frames, 196, 1024):
        raise ValueError(f"invalid patch compact cache: {path}")
    return payload["global"].float(), payload["patch"].float()


def decode_video(row: pd.Series) -> np.ndarray:
    indices = [int(value) for value in json.loads(str(row["frame_indices"]))]
    return decode_selected_frames(resolve_video(str(row["video_path"])), indices, 96)


def completed_ids(checkpoint: Path) -> tuple[set[str], list[Path]]:
    parts = sorted(checkpoint.glob("part_*.csv"))
    completed: set[str] = set()
    for path in parts:
        completed.update(pd.read_csv(path, usecols=["k1_task_id"])["k1_task_id"].astype(str))
    return completed, parts


def run(args: argparse.Namespace) -> None:
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    tasks = pd.read_csv(args.input, float_precision="round_trip")
    tasks = tasks[
        tasks["dataset"].eq(args.dataset)
        & tasks["video_id"].map(
            lambda value: video_id_shard(str(value), args.num_shards)
            == args.shard_index
        )
    ].copy()
    checkpoint = (
        args.output_dir
        / "original_k1_checkpoints"
        / args.input_mode
        / args.dataset
        / f"shard_{args.shard_index:02d}_of_{args.num_shards:02d}"
    )
    checkpoint.mkdir(parents=True, exist_ok=True)
    completed, parts = completed_ids(checkpoint)
    pending = tasks[~tasks["k1_task_id"].astype(str).isin(completed)].copy()
    params = {
        duration: load_params(config, args.params_dir, args.dataset, int(duration))
        for duration in sorted(tasks["protocol_duration_sec"].unique())
    }
    extractor = (
        PatchSTALL(args.extract_device, data_dict=None, load_dino=True)
        if args.input_mode == "video" and len(pending)
        else None
    )
    print(
        f"mode={args.input_mode} dataset={args.dataset} shard={args.shard_index}/"
        f"{args.num_shards} tasks={len(tasks)} pending={len(pending)}",
        flush=True,
    )
    buffer: list[dict] = []
    part_index = len(parts)
    started = time.perf_counter()
    for start in range(0, len(pending), args.batch_size):
        group = pending.iloc[start : start + args.batch_size]
        loaded = []
        if args.input_mode == "cache":
            for _, row in group.iterrows():
                global_features, patch_features = load_cache(row)
                loaded.append((row, global_features, patch_features))
        else:
            assert extractor is not None
            source_rows = [row for _, row in group.iterrows()]
            decoded = [decode_video(row) for row in source_rows]
            embedded = extractor.frames_to_global_patch_embeddings(
                decoded, batch_size=args.frame_batch_size
            )
            for row, output in zip(source_rows, embedded):
                loaded.append(
                    (
                        row,
                        torch.from_numpy(np.asarray(output["global"], dtype=np.float32)),
                        torch.from_numpy(np.asarray(output["patch"], dtype=np.float32)),
                    )
                )
        for duration, items in pd.Series(
            [int(item[0]["protocol_duration_sec"]) for item in loaded]
        ).groupby(lambda index: int(loaded[index][0]["protocol_duration_sec"])):
            positions = list(items.index)
            subset = [loaded[index] for index in positions]
            global_batch = torch.stack([item[1] for item in subset])
            patch_batch = torch.stack([item[2] for item in subset])
            raw = score_batch(global_batch, patch_batch, params[int(duration)], args.score_device)
            for index, (row, _, _) in enumerate(subset):
                buffer.append(
                    {
                        **{column: row[column] for column in KEY_COLUMNS},
                        **{name: values[index] for name, values in raw.items()},
                        "score_source": f"original_k1_{args.input_mode}",
                    }
                )
        processed = min(start + args.batch_size, len(pending))
        if len(buffer) >= args.checkpoint_rows or processed == len(pending):
            if buffer:
                path = checkpoint / f"part_{part_index:06d}.csv"
                temporary = path.with_suffix(".tmp.csv")
                pd.DataFrame(buffer).to_csv(temporary, index=False)
                temporary.replace(path)
                buffer.clear()
                part_index += 1
        if start == 0 or processed == len(pending) or processed % 200 == 0:
            print(
                f"processed={processed}/{len(pending)} elapsed={time.perf_counter()-started:.1f}s",
                flush=True,
            )
    completed, parts = completed_ids(checkpoint)
    missing = set(tasks["k1_task_id"].astype(str)) - completed
    if missing:
        raise RuntimeError(f"incomplete K1 shard: {len(missing)} tasks missing")
    merged = pd.concat(
        [pd.read_csv(path, float_precision="round_trip") for path in parts],
        ignore_index=True,
    )
    output = (
        args.output_dir
        / "original_k1_raw"
        / f"{args.input_mode}_{args.dataset}_shard{args.shard_index:02d}_of_{args.num_shards:02d}.csv"
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    merged.sort_values("k1_task_id").to_csv(output, index=False)
    print(f"wrote {len(merged)} tasks -> {output}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--input-mode", choices=("cache", "video"), required=True)
    parser.add_argument("--dataset", choices=tuple(CUSTOM_SIZE), required=True)
    parser.add_argument("--shard-index", type=int, required=True)
    parser.add_argument("--num-shards", type=int, default=2)
    parser.add_argument("--extract-device", default="cuda")
    parser.add_argument("--score-device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--frame-batch-size", type=int, default=32)
    parser.add_argument("--checkpoint-rows", type=int, default=500)
    parser.add_argument(
        "--config", type=Path, default=ROOT / "configs/alpha_stalled_u0_locked.yaml"
    )
    parser.add_argument(
        "--params-dir",
        type=Path,
        default=ROOT / "results/duration_aware_23source/params",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "results/full_coverage_paper_protocol",
    )
    args = parser.parse_args()
    if args.num_shards < 1 or not 0 <= args.shard_index < args.num_shards:
        parser.error("require num_shards >= 1 and valid shard index")
    return args


if __name__ == "__main__":
    run(parse_args())
