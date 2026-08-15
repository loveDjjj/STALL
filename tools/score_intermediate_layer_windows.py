#!/usr/bin/env python3
"""Score calibrated D2 at DINO layers 11/17/23 from one K=3 traversal."""

from __future__ import annotations

import argparse
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
TOOLS_DIR = REPO_ROOT / "tools"
for path in (SRC_DIR, TOOLS_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from eval_patch_fast import FastPatchScorer
from fit_intermediate_layer_params import LAYERS, SPECS
from score_unified_multiscale_windows import PARAMS as REGION_PARAMS
from alpha_stalled.legacy_window_scoring import (
    KEY_COLUMNS,
    decode_manifest_row_with_retries,
    stable_shard,
    video_key,
)
from stall_patch import PatchSTALL


WINDOW_KEYS = [*KEY_COLUMNS, "window_id"]


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


def score_batch(
    decoded: list[dict],
    extractor: PatchSTALL,
    layer_scorers: dict[int, FastPatchScorer],
    region_scorers: dict[int, FastPatchScorer],
    frame_batch_size: int,
) -> list[dict]:
    outputs = extractor.frames_to_layer_patch_embeddings(
        [item["frames"] for item in decoded],
        layers=LAYERS,
        batch_size=frame_batch_size,
    )
    patches = {layer: [] for layer in LAYERS}
    metadata = []
    for item, output in zip(decoded, outputs):
        for window_id, positions in enumerate(item["window_positions"]):
            for layer in LAYERS:
                patches[layer].append(output["layers"][layer][positions])
            metadata.append((item, window_id))

    layer_scores = {}
    for layer, scorer in layer_scorers.items():
        layer_scores[layer] = scorer.score_temporal_batch(
            np.stack(patches[layer]),
            patch_temp_mode="same_grid_second_order",
            aggregation=scorer.aggregation_config["mode"],
            bottomk_ratio=scorer.params_bottomk_ratio,
            temporal_run_length=scorer.params_temporal_run_length,
            patch_region_size=scorer.params_patch_region_size,
        )
    final_patch = np.stack(patches[23])
    region_scores = {}
    for region, scorer in region_scorers.items():
        region_scores[region] = scorer.score_temporal_batch(
            final_patch,
            patch_temp_mode="same_grid_second_order",
            aggregation=scorer.aggregation_config["mode"],
            bottomk_ratio=scorer.params_bottomk_ratio,
            temporal_run_length=scorer.params_temporal_run_length,
            patch_region_size=region,
        )
    rows = []
    for index, (item, window_id) in enumerate(metadata):
        source = item["row"]
        rows.append(
            {
                **{column: source[column] for column in KEY_COLUMNS},
                "window_id": window_id,
                "frame_indices": json.dumps(
                    item["windows"][window_id], separators=(",", ":")
                ),
                **{
                    f"layer{layer}_raw": float(result["patch_temp_raw"][index])
                    for layer, result in layer_scores.items()
                },
                **{
                    f"layer{layer}_temporal": float(
                        result["patch_temp_percentile"][index]
                    )
                    for layer, result in layer_scores.items()
                },
                **{
                    f"region{region}_raw": float(result["patch_temp_raw"][index])
                    for region, result in region_scores.items()
                },
                **{
                    f"region{region}_temporal": float(
                        result["patch_temp_percentile"][index]
                    )
                    for region, result in region_scores.items()
                },
            }
        )
    return rows


def run(args: argparse.Namespace) -> None:
    manifest = pd.read_csv(args.manifest)
    manifest = manifest[
        (manifest["dataset"] == args.dataset)
        & manifest.apply(
            lambda row: stable_shard(row, args.num_shards) == args.shard_index, axis=1
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

    base = Path(SPECS[args.dataset]["base_params"])
    param_paths = {
        11: args.params_dir / f"{args.dataset}_layer11.npz",
        17: args.params_dir / f"{args.dataset}_layer17.npz",
        23: base,
    }
    for path in param_paths.values():
        if not path.exists():
            raise FileNotFoundError(path)
    layer_scorers = {
        layer: FastPatchScorer(str(path), device=args.device)
        for layer, path in param_paths.items()
    }
    for layer, scorer in layer_scorers.items():
        scorer.validate("same_grid_second_order")
        if scorer.params_patch_region_size != int(SPECS[args.dataset]["region"]):
            raise ValueError(f"{args.dataset}/layer{layer}: region mismatch")
    region_scorers = {
        region: FastPatchScorer(str(path), device=args.device)
        for region, path in REGION_PARAMS[args.dataset].items()
    }
    for region, scorer in region_scorers.items():
        scorer.validate("same_grid_second_order")
        if scorer.params_patch_region_size != region:
            raise ValueError(f"{args.dataset}/region{region}: region mismatch")
    extractor = PatchSTALL(args.device, data_dict=None, load_dino=True)
    failures = []
    next_part = len(parts)
    started = time.perf_counter()
    for start in range(0, len(pending), args.video_batch_size):
        source_rows = [
            row for _, row in pending.iloc[start : start + args.video_batch_size].iterrows()
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
                        {**{column: row[column] for column in KEY_COLUMNS}, "error": repr(error)}
                    )
        if decoded:
            pd.DataFrame(
                score_batch(
                    decoded,
                    extractor,
                    layer_scorers,
                    region_scorers,
                    args.frame_batch_size,
                )
            ).to_csv(checkpoint_dir / f"part_{next_part:06d}.csv", index=False)
            next_part += 1
        processed = min(start + args.video_batch_size, len(pending))
        if start == 0 or processed == len(pending) or next_part % 50 == 0:
            print(
                f"processed={processed}/{len(pending)} elapsed={time.perf_counter()-started:.1f}s "
                f"failures={len(failures)}",
                flush=True,
            )
    parts = sorted(checkpoint_dir.glob("part_*.csv"))
    merged = pd.concat(
        [pd.read_csv(path, float_precision="round_trip") for path in parts],
        ignore_index=True,
    )
    if merged.duplicated(WINDOW_KEYS).any():
        raise ValueError("duplicate intermediate-layer window keys")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    merged.sort_values(WINDOW_KEYS).to_csv(args.output, index=False)
    if failures:
        pd.DataFrame(failures).to_csv(args.output.with_suffix(".failures.csv"), index=False)
        raise RuntimeError(f"intermediate-layer scoring had {len(failures)} failures")
    print(f"wrote {len(merged)} windows to {args.output}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=tuple(SPECS), required=True)
    parser.add_argument("--manifest", type=Path, default=REPO_ROOT / "results/multi_window_joint_typicality/multi_window_manifest.csv")
    parser.add_argument("--params-dir", type=Path, default=REPO_ROOT / "results/unified_multiscale_layers/layer_params")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--checkpoint-dir", type=Path, default=REPO_ROOT / "results/unified_multiscale_layers/checkpoints_layers")
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
