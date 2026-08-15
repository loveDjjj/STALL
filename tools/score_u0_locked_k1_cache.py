#!/usr/bin/env python3
"""Score unified U0 K1 evaluation windows from locked patch-token caches."""

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

from alpha_stalled.global_branch import score_global_raw
from alpha_stalled.local_branch import score_local_raw
from alpha_stalled.parameters import load_raw_params
from alpha_stalled.release_io import video_id, video_id_shard
from alpha_stalled.u0_analysis import (
    calibrate_raw,
    calibration_raw_references,
)
from alpha_stalled.u0_scoring import score_raw_batch


CACHE_ROOTS = {
    "comgenvid": ROOT / "cache/patch_embeddings/comgenvid",
    "videofeedback": ROOT / "cache/patch_embeddings/videofeedback",
    "genvideo": ROOT / "cache/patch_embeddings/genvideo",
}
KEY_COLUMNS = [
    "video_id",
    "dataset",
    "protocol_split",
    "subset",
    "source_model",
    "filename",
]


def cache_path(row: pd.Series) -> Path:
    return (
        CACHE_ROOTS[str(row["dataset"])]
        / str(row["subset"])
        / str(row["source_model"])
        / f"{Path(str(row['filename'])).stem}_2s.pt"
    )


def load_manifest(source_path: Path, dataset: str, num_shards: int, shard: int) -> pd.DataFrame:
    source = pd.read_csv(source_path, float_precision="round_trip")
    source = source[
        (source["dataset"] == dataset) & (source["protocol_split"] == "evaluation")
    ].copy()
    source["video_id"] = source.apply(video_id, axis=1)
    source = source[
        source["video_id"].map(lambda value: video_id_shard(value, num_shards) == shard)
    ].reset_index(drop=True)
    return source


def load_batch(rows: list[pd.Series]) -> tuple[torch.Tensor, torch.Tensor]:
    global_features = []
    patch_features = []
    for row in rows:
        path = cache_path(row)
        if not path.is_file():
            raise FileNotFoundError(path)
        payload = torch.load(path, weights_only=True, map_location="cpu")
        expected = json.loads(str(row["indices_K1_current"]))
        if len(expected) != 1 or [int(value) for value in payload["frame_indices"]] != [
            int(value) for value in expected[0]
        ]:
            raise ValueError(f"K1 cache frame indices differ from lock: {path}")
        if tuple(payload["global"].shape) != (16, 1024):
            raise ValueError(f"invalid global cache shape: {path}")
        if tuple(payload["patch"].shape) != (16, 196, 1024):
            raise ValueError(f"invalid patch cache shape: {path}")
        if tuple(int(value) for value in payload["grid_size"]) != (14, 14):
            raise ValueError(f"invalid patch grid: {path}")
        global_features.append(payload["global"].float())
        patch_features.append(payload["patch"].float())
    return torch.stack(global_features), torch.stack(patch_features)


def run(args: argparse.Namespace) -> None:
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    manifest = load_manifest(
        args.source_manifest, args.dataset, args.num_shards, args.shard_index
    )
    params = load_raw_params(config, args.dataset)
    references = calibration_raw_references(
        args.calibration_raw_dir, args.num_shards, args.dataset, config
    )
    rows = []
    started = time.perf_counter()
    for start in range(0, len(manifest), args.batch_size):
        source_rows = [
            row for _, row in manifest.iloc[start : start + args.batch_size].iterrows()
        ]
        global_batch, patch_batch = load_batch(source_rows)
        raw = score_raw_batch(global_batch, patch_batch, params, args.device)
        calibrated = calibrate_raw(raw, references)
        for index, source in enumerate(source_rows):
            rows.append(
                {
                    **{column: source[column] for column in KEY_COLUMNS},
                    "video_path": source["video_path"],
                    "duration_seconds": float(source["duration_seconds"]),
                    "frame_indices": json.dumps(
                        json.loads(str(source["indices_K1_current"]))[0],
                        separators=(",", ":"),
                    ),
                    **{key: value[index] for key, value in raw.items()},
                    **{key: value[index] for key, value in calibrated.items()},
                }
            )
        completed = min(start + args.batch_size, len(manifest))
        if start == 0 or completed == len(manifest) or completed % 200 == 0:
            print(
                f"{args.dataset} shard={args.shard_index} "
                f"processed={completed}/{len(manifest)} "
                f"elapsed={time.perf_counter()-started:.1f}s",
                flush=True,
            )
    output = (
        args.output_dir
        / "k1_raw"
        / f"{args.dataset}_shard{args.shard_index:02d}_of_{args.num_shards:02d}.csv"
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame(rows).sort_values(KEY_COLUMNS)
    if len(frame) != len(manifest) or frame["video_id"].duplicated().any():
        raise ValueError("K1 cache score output is incomplete or duplicated")
    temporary = output.with_suffix(".tmp.csv")
    frame.to_csv(temporary, index=False)
    temporary.replace(output)
    print(f"wrote {len(frame)} K1 cache scores -> {output}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset", choices=("comgenvid", "videofeedback", "genvideo"), required=True
    )
    parser.add_argument("--shard-index", type=int, required=True)
    parser.add_argument("--num-shards", type=int, default=2)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument(
        "--config", type=Path, default=ROOT / "configs/alpha_stalled_u0_locked.yaml"
    )
    parser.add_argument(
        "--source-manifest",
        type=Path,
        default=ROOT / "results/multi_window_joint_typicality/multi_window_manifest.csv",
    )
    parser.add_argument(
        "--calibration-raw-dir",
        type=Path,
        default=ROOT / "results/u0_locked_reproduction/calibration_raw",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=ROOT / "results/u0_core_ablation"
    )
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
