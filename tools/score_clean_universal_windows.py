#!/usr/bin/env python3
"""Score U0-U5 and universal layer-17 evidence in one K=3 DINO traversal."""

from __future__ import annotations

import argparse
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
TOOLS_DIR = REPO_ROOT / "tools"
for path in (SRC_DIR, TOOLS_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from eval_patch_fast import FastPatchScorer
from score_multi_window import (
    KEY_COLUMNS,
    decode_manifest_row_with_retries,
    stable_shard,
    video_key,
)
from stall_patch import PatchSTALL


WINDOW_KEYS = [*KEY_COLUMNS, "window_id"]


def param_path(params_dir: Path, dataset: str, region: int, aggregation: str) -> Path:
    return params_dir / f"{dataset}_region{region}_{aggregation}.npz"


def load_completed(checkpoint_dir: Path) -> tuple[set[tuple[str, ...]], list[Path]]:
    parts = sorted(checkpoint_dir.glob("part_*.csv"))
    completed: set[tuple[str, ...]] = set()
    for part in parts:
        frame = pd.read_csv(part, usecols=KEY_COLUMNS)
        completed.update(
            tuple(str(value) for value in row)
            for row in frame.drop_duplicates().itertuples(index=False, name=None)
        )
    return completed, parts


@torch.inference_mode()
def score_region_aggregations(
    patch_batch: np.ndarray,
    region: int,
    mean_scorer: FastPatchScorer,
    bottom_scorer: FastPatchScorer,
) -> dict[str, np.ndarray]:
    if not np.array_equal(mean_scorer.mu_temp, bottom_scorer.mu_temp):
        raise ValueError(f"region {region}: mean/bottom temporal means differ")
    if not np.array_equal(mean_scorer.W_temp, bottom_scorer.W_temp):
        raise ValueError(f"region {region}: mean/bottom whitening differs")
    device = mean_scorer.device
    patch = torch.as_tensor(patch_batch, dtype=torch.float32, device=device)
    _, _, mu_temp, W_temp = mean_scorer._params_for_device(device)
    temporal = mean_scorer.temporal_features(
        patch, "same_grid_second_order", region
    )
    likelihood = mean_scorer.log_likelihood_from_white(
        torch.matmul(temporal - mu_temp, W_temp)
    )
    raw_mean = likelihood.reshape(likelihood.shape[0], -1).mean(dim=1)
    raw_bottom = mean_scorer.bottomk_mean(likelihood, 0.2)
    mean_np = raw_mean.cpu().numpy()
    bottom_np = raw_bottom.cpu().numpy()
    return {
        "mean_raw": mean_np.astype(np.float32),
        "mean_temporal": mean_scorer.percentile(
            mean_np, mean_scorer.calib_temp
        ).astype(np.float32),
        "bottom20_raw": bottom_np.astype(np.float32),
        "bottom20_temporal": bottom_scorer.percentile(
            bottom_np, bottom_scorer.calib_temp
        ).astype(np.float32),
    }


def score_batch(
    decoded: list[dict],
    extractor: PatchSTALL,
    region_scorers: dict[int, dict[str, FastPatchScorer]],
    layer17_scorer: FastPatchScorer,
    frame_batch_size: int,
) -> list[dict]:
    outputs = extractor.frames_to_layer_patch_embeddings(
        [item["frames"] for item in decoded],
        layers=(17, 23),
        batch_size=frame_batch_size,
    )
    layer17_windows = []
    layer23_windows = []
    metadata = []
    for item, output in zip(decoded, outputs):
        for window_id, positions in enumerate(item["window_positions"]):
            layer17_windows.append(output["layers"][17][positions])
            layer23_windows.append(output["layers"][23][positions])
            metadata.append((item, window_id))
    layer17_batch = np.stack(layer17_windows)
    layer23_batch = np.stack(layer23_windows)
    layer17 = layer17_scorer.score_temporal_batch(
        layer17_batch,
        patch_temp_mode="same_grid_second_order",
        aggregation="mean",
        bottomk_ratio=0.2,
        temporal_run_length=layer17_scorer.params_temporal_run_length,
        patch_region_size=1,
    )
    regions = {
        region: score_region_aggregations(
            layer23_batch,
            region,
            scorers["mean"],
            scorers["bottom20"],
        )
        for region, scorers in region_scorers.items()
    }
    rows = []
    for index, (item, window_id) in enumerate(metadata):
        source = item["row"]
        row = {
            **{column: source[column] for column in KEY_COLUMNS},
            "window_id": window_id,
            "frame_indices": json.dumps(
                item["windows"][window_id], separators=(",", ":")
            ),
            "layer17_raw": float(layer17["patch_temp_raw"][index]),
            "layer17_temporal": float(layer17["patch_temp_percentile"][index]),
        }
        for region, scores in regions.items():
            for aggregation in ("mean", "bottom20"):
                row[f"region{region}_{aggregation}_raw"] = float(
                    scores[f"{aggregation}_raw"][index]
                )
                row[f"region{region}_{aggregation}_temporal"] = float(
                    scores[f"{aggregation}_temporal"][index]
                )
        rows.append(row)
    return rows


def run(args: argparse.Namespace) -> None:
    manifest = pd.read_csv(args.manifest)
    manifest = manifest[
        (manifest["dataset"] == args.dataset)
        & manifest.apply(
            lambda row: stable_shard(row, args.num_shards) == args.shard_index,
            axis=1,
        )
    ].reset_index(drop=True)
    if args.debug_videos is not None:
        manifest = manifest.head(args.debug_videos)
    checkpoint_dir = args.checkpoint_dir / args.dataset / f"shard_{args.shard_index}"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    completed, parts = load_completed(checkpoint_dir)
    pending = manifest[
        ~manifest.apply(lambda row: video_key(row) in completed, axis=1)
    ].reset_index(drop=True)
    print(
        f"dataset={args.dataset} shard={args.shard_index}/{args.num_shards} "
        f"expected={len(manifest)} completed={len(completed)} pending={len(pending)}",
        flush=True,
    )

    region_scorers = {
        region: {
            aggregation: FastPatchScorer(
                str(param_path(args.params_dir, args.dataset, region, aggregation)),
                device=args.device,
            )
            for aggregation in ("mean", "bottom20")
        }
        for region in (1, 2, 3)
    }
    for region, scorers in region_scorers.items():
        for name, scorer in scorers.items():
            scorer.validate("same_grid_second_order")
            if scorer.params_patch_region_size != region:
                raise ValueError(f"region mismatch: {region}/{name}")
    layer17_path = (
        args.params_dir / f"{args.dataset}_layer17_region1_mean.npz"
    )
    layer17_scorer = FastPatchScorer(str(layer17_path), device=args.device)
    layer17_scorer.validate("same_grid_second_order")
    if layer17_scorer.params_patch_region_size != 1:
        raise ValueError("layer17 clean-universal params are not region 1")
    extractor = PatchSTALL(args.device, data_dict=None, load_dino=True)
    failures = []
    next_part = len(parts)
    started = time.perf_counter()
    for start in range(0, len(pending), args.video_batch_size):
        source_rows = [
            row
            for _, row in pending.iloc[
                start : start + args.video_batch_size
            ].iterrows()
        ]
        decoded = []
        with ThreadPoolExecutor(max_workers=min(args.decode_workers, len(source_rows))) as pool:
            futures = [
                pool.submit(
                    decode_manifest_row_with_retries,
                    row,
                    "K3_uniform",
                    args.seek_gap,
                    args.decode_attempts,
                )
                for row in source_rows
            ]
            for row, future in zip(source_rows, futures):
                try:
                    decoded.append(future.result())
                except Exception as error:
                    failures.append(
                        {
                            **{column: row[column] for column in KEY_COLUMNS},
                            "error": repr(error),
                        }
                    )
        if decoded:
            frame = pd.DataFrame(
                score_batch(
                    decoded,
                    extractor,
                    region_scorers,
                    layer17_scorer,
                    args.frame_batch_size,
                )
            )
            frame.to_csv(checkpoint_dir / f"part_{next_part:06d}.csv", index=False)
            next_part += 1
        processed = min(start + args.video_batch_size, len(pending))
        if start == 0 or processed == len(pending) or next_part % 50 == 0:
            print(
                f"processed={processed}/{len(pending)} "
                f"elapsed={time.perf_counter()-started:.1f}s failures={len(failures)}",
                flush=True,
            )
    parts = sorted(checkpoint_dir.glob("part_*.csv"))
    merged = pd.concat(
        [pd.read_csv(path, float_precision="round_trip") for path in parts],
        ignore_index=True,
    )
    if merged.duplicated(WINDOW_KEYS).any():
        raise ValueError("duplicate clean-universal window keys")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    merged.sort_values(WINDOW_KEYS).to_csv(args.output, index=False)
    if failures:
        pd.DataFrame(failures).to_csv(
            args.output.with_suffix(".failures.csv"), index=False
        )
        raise RuntimeError(f"clean-universal scoring had {len(failures)} failures")
    print(f"wrote {len(merged)} windows to {args.output}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset", choices=("comgenvid", "videofeedback", "genvideo"), required=True
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=REPO_ROOT
        / "results/multi_window_joint_typicality/multi_window_manifest.csv",
    )
    parser.add_argument(
        "--params-dir",
        type=Path,
        default=REPO_ROOT / "results/clean_universal/params",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--checkpoint-dir",
        type=Path,
        default=REPO_ROOT / "results/clean_universal/checkpoints",
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--video-batch-size", type=int, default=4)
    parser.add_argument("--frame-batch-size", type=int, default=32)
    parser.add_argument("--decode-workers", type=int, default=4)
    parser.add_argument("--decode-attempts", type=int, default=3)
    parser.add_argument("--seek-gap", type=int, default=64)
    parser.add_argument("--debug-videos", type=int)
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
