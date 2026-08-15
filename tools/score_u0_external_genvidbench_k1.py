#!/usr/bin/env python3
"""Score locked current K1 windows from GenVidBench patch-token caches."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import pandas as pd
import torch
import yaml


ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from alpha_stalled.parameters import load_raw_params
from alpha_stalled.release_io import video_id_shard
from alpha_stalled.u0_protocol import KEY_COLUMNS
from alpha_stalled.u0_scoring import score_raw_batch


DATASET = "genvidbench_pair1"


def load_manifest(release_dir: Path, num_shards: int, shard: int) -> tuple[pd.DataFrame, dict]:
    records = []
    for name in ("calibration_manifest.json", "evaluation_manifest.json"):
        records.extend(
            json.loads((release_dir / name).read_text(encoding="utf-8"))["videos"]
        )
    references = json.loads(
        (release_dir / "frame_indices.json").read_text(encoding="utf-8")
    )["calibration_reference_windows"]
    frame = pd.DataFrame(records)
    frame = frame[
        frame["video_id"].map(lambda value: video_id_shard(value, num_shards) == shard)
    ].reset_index(drop=True)
    return frame, references


def cache_path(row: pd.Series, cache_root: Path) -> Path:
    return (
        cache_root
        / str(row["subset"])
        / str(row["source_model"])
        / f"{Path(str(row['filename'])).stem}_2s.pt"
    )


def load_batch(
    rows: list[pd.Series], cache_root: Path, calibration_references: dict
) -> tuple[torch.Tensor, torch.Tensor, list[list[int]]]:
    global_features = []
    patch_features = []
    frame_indices = []
    for row in rows:
        path = cache_path(row, cache_root)
        if not path.is_file():
            raise FileNotFoundError(path)
        payload = torch.load(path, weights_only=True, map_location="cpu")
        indices = [int(value) for value in payload["frame_indices"]]
        if len(indices) != 16 or len(set(indices)) != 16:
            raise ValueError(f"external K1 cache is not a strict 16-frame window: {path}")
        if row["protocol_split"] == "calibration":
            expected = [int(value) for value in calibration_references[row["video_id"]]]
            if indices != expected:
                raise ValueError(f"external calibration K1 cache differs from lock: {path}")
        if tuple(payload["global"].shape) != (16, 1024):
            raise ValueError(f"invalid global cache shape: {path}")
        if tuple(payload["patch"].shape) != (16, 196, 1024):
            raise ValueError(f"invalid patch cache shape: {path}")
        global_features.append(payload["global"].float())
        patch_features.append(payload["patch"].float())
        frame_indices.append(indices)
    return torch.stack(global_features), torch.stack(patch_features), frame_indices


def run(args: argparse.Namespace) -> None:
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    manifest, calibration_references = load_manifest(
        args.release_dir, args.num_shards, args.shard_index
    )
    output = (
        args.output_dir
        / "k1_raw"
        / f"{DATASET}_shard{args.shard_index:02d}_of_{args.num_shards:02d}.csv"
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    completed = set()
    if output.is_file():
        previous = pd.read_csv(output, float_precision="round_trip")
        completed = set(previous["video_id"])
    else:
        previous = pd.DataFrame()
    pending = manifest[~manifest["video_id"].isin(completed)].reset_index(drop=True)
    params = load_raw_params(config, DATASET, args.local_params)
    rows = []
    started = time.perf_counter()
    for start in range(0, len(pending), args.batch_size):
        sources = [
            row for _, row in pending.iloc[start : start + args.batch_size].iterrows()
        ]
        global_batch, patch_batch, indices = load_batch(
            sources, args.cache_root, calibration_references
        )
        raw = score_raw_batch(global_batch, patch_batch, params, args.device)
        motion = torch.linalg.vector_norm(
            global_batch[:, 1:] - global_batch[:, :-1], dim=-1
        ).mean(dim=1).numpy()
        for index, source in enumerate(sources):
            rows.append(
                {
                    **{column: source[column] for column in KEY_COLUMNS},
                    "video_path": source["video_path"],
                    "duration_seconds": float(source["duration_seconds"]),
                    "frame_indices": json.dumps(indices[index], separators=(",", ":")),
                    "mean_global_motion": float(motion[index]),
                    **{key: value[index] for key, value in raw.items()},
                }
            )
        completed_count = min(start + args.batch_size, len(pending))
        if start == 0 or completed_count == len(pending) or completed_count % 100 == 0:
            print(
                f"external K1 shard={args.shard_index} processed={completed_count}/{len(pending)} "
                f"elapsed={time.perf_counter()-started:.1f}s",
                flush=True,
            )
    merged = pd.concat([previous, pd.DataFrame(rows)], ignore_index=True)
    expected = set(manifest["video_id"])
    if set(merged["video_id"]) != expected or merged["video_id"].duplicated().any():
        raise ValueError("external K1 shard output is incomplete or duplicated")
    temporary = output.with_suffix(".tmp.csv")
    merged.sort_values(KEY_COLUMNS).to_csv(temporary, index=False)
    temporary.replace(output)
    print(f"wrote {len(merged)} external K1 raw scores -> {output}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", type=Path, default=ROOT / "configs/alpha_stalled_u0_locked.yaml"
    )
    parser.add_argument(
        "--release-dir", type=Path, default=ROOT / "release/u0_external_genvidbench"
    )
    parser.add_argument(
        "--cache-root",
        type=Path,
        default=ROOT / "cache/patch_embeddings/genvidbench_pair1_ms_vript",
    )
    parser.add_argument(
        "--local-params",
        type=Path,
        default=ROOT / "release/u0_external_genvidbench/params/region1_mean.npz",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=ROOT / "results/u0_external_genvidbench"
    )
    parser.add_argument("--num-shards", type=int, default=4)
    parser.add_argument("--shard-index", type=int, required=True)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    if args.num_shards < 1 or not 0 <= args.shard_index < args.num_shards:
        parser.error("require num_shards >= 1 and valid shard-index")
    return args


if __name__ == "__main__":
    run(parse_args())
