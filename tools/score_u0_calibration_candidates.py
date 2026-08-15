#!/usr/bin/env python3
"""Score all calibration-size candidates in one locked DINO traversal."""

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
import yaml


ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from alpha_stalled.video_io import decode_selected_frames
from alpha_stalled.release_io import resolve_required_video as resolve_video, video_id_shard
from alpha_stalled.parameters import load_raw_params
from alpha_stalled.artifacts import checkpoint_completed_ids
from alpha_stalled.global_branch import global_t1_features
from alpha_stalled.local_branch import local_d2_features
from alpha_stalled.whitening import (
    GaussianMeanCandidateScorerFloat64,
    StableGaussianParams,
    score_gaussian_aggregate_float64,
)
from stall_patch import PatchSTALL


KEY_COLUMNS = [
    "video_id",
    "dataset",
    "protocol_split",
    "subset",
    "source_model",
    "filename",
]


def load_candidate_params(
    config: dict, dataset: str, directory: Path, candidate_set: str
) -> tuple[list[str], list[StableGaussianParams], list[StableGaussianParams], dict]:
    locked = load_raw_params(config, dataset)
    if candidate_set == "cross_oas":
        specifications: list[tuple[str, Path]] = []
        # Put the target-domain current model first so direct validation checks
        # the exact locked target scorer used by the downstream comparison.
        ordered_datasets = [dataset, *[name for name in ("comgenvid", "videofeedback", "genvideo") if name != dataset]]
        for source in ordered_datasets:
            specifications.append(
                (
                    f"current_{source}",
                    ROOT / config["local_branch"]["params_by_dataset"][source]["path"],
                )
            )
        specifications.extend(
            [
                ("current_pooled200", directory / "cross/pooled200.npz"),
                ("current_pooled600", directory / "cross/pooled600.npz"),
            ]
        )
        for source in ("comgenvid", "videofeedback", "genvideo"):
            specifications.append(
                (f"oas_{source}_locked200", directory / "oas" / f"{source}_locked200.npz")
            )
            for seed in (17, 29, 43, 71, 101):
                specifications.append(
                    (
                        f"oas_{source}_s{seed}_n200",
                        directory / "oas" / f"{source}_seed{seed}_n200.npz",
                    )
                )
        names = []
        spatial = []
        temporal = []
        for name, path in specifications:
            if not path.is_file():
                raise FileNotFoundError(path)
            names.append(name)
            spatial.append(
                StableGaussianParams.from_npz(
                    str(path), "mu_patch_spat", "W_patch_spat", "calib_patch_spat_scores"
                )
            )
            temporal.append(
                StableGaussianParams.from_npz(
                    str(path), "mu_patch_temp", "W_patch_temp", "calib_patch_temp_scores"
                )
            )
        return names, spatial, temporal, locked

    names = ["locked"]
    spatial = [locked["patch_spatial"]]
    temporal = [locked["patch_d2"]]
    for seed in (17, 29, 43, 71, 101):
        for size in (25, 50, 100, 200):
            path = directory / f"{dataset}_seed{seed}_n{size}.npz"
            if not path.is_file():
                raise FileNotFoundError(path)
            names.append(f"s{seed}_n{size}")
            spatial.append(
                StableGaussianParams.from_npz(
                    str(path),
                    mean_key="mu_patch_spat",
                    whitening_key="W_patch_spat",
                    calibration_key="calib_patch_spat_scores",
                )
            )
            temporal.append(
                StableGaussianParams.from_npz(
                    str(path),
                    mean_key="mu_patch_temp",
                    whitening_key="W_patch_temp",
                    calibration_key="calib_patch_temp_scores",
                )
            )
    return names, spatial, temporal, locked


def load_rows(
    split: str,
    dataset: str,
    release_dir: Path,
    reserve_manifest: Path,
    remaining_real_manifest: Path,
) -> tuple[pd.DataFrame, dict[str, dict[str, list[list[int]]]]]:
    window_map: dict[str, dict[str, list[list[int]]]] = {}
    if split in {"evaluation", "locked_calibration"}:
        manifest_name = (
            "evaluation_manifest.json"
            if split == "evaluation"
            else "calibration_manifest.json"
        )
        payload = json.loads((release_dir / manifest_name).read_text())
        frame_payload = json.loads((release_dir / "frame_indices.json").read_text())
        frame_indices = frame_payload["videos"]
        records = [item for item in payload["videos"] if item["dataset"] == dataset]
        for item in records:
            window_map[item["video_id"]] = {"k3": frame_indices[item["video_id"]]}
            if split == "locked_calibration":
                window_map[item["video_id"]]["k1"] = [
                    frame_payload["calibration_reference_windows"][item["video_id"]]
                ]
    elif split == "reserve":
        payload = json.loads(reserve_manifest.read_text())
        records = [item for item in payload["videos"] if item["dataset"] == dataset]
        for item in records:
            window_map[item["video_id"]] = {
                "k1": [item["k1_window"]],
                "k3": item["k3_windows"],
            }
            item["protocol_split"] = "calibration_reserve"
            item["video_path"] = item["source_path"]
    else:
        payload = json.loads(remaining_real_manifest.read_text())
        records = [item for item in payload["videos"] if item["dataset"] == dataset]
        for item in records:
            window_map[item["video_id"]] = {"k3": item["k3_windows"]}
            item["video_path"] = item["source_path"]
    rows = pd.DataFrame(records)
    if rows["video_id"].duplicated().any() or set(rows["video_id"]) != set(window_map):
        raise ValueError("candidate manifest has duplicate or mismatched video IDs")
    return rows, window_map


def decode_row(
    row: pd.Series,
    window_groups: dict[str, list[list[int]]],
    seek_gap: int,
    attempts: int,
) -> dict:
    flattened = [
        (sampling, window_id, [int(value) for value in window])
        for sampling, windows in window_groups.items()
        for window_id, window in enumerate(windows)
    ]
    unique_indices = sorted({value for _, _, window in flattened for value in window})
    error = None
    for attempt in range(attempts):
        try:
            frames = decode_selected_frames(
                resolve_video(str(row["video_path"])), unique_indices, seek_gap
            )
            positions = {value: index for index, value in enumerate(unique_indices)}
            return {
                "row": row,
                "windows": flattened,
                "positions": [[positions[value] for value in window] for _, _, window in flattened],
                "frames": frames,
                "unique_indices": unique_indices,
            }
        except Exception as caught:
            error = caught
            if attempt + 1 < attempts:
                time.sleep(0.25 * (attempt + 1))
    raise RuntimeError(f"decode failed after {attempts} attempts: {error}") from error


class CandidateScorers:
    def __init__(
        self,
        names: list[str],
        spatial: list[StableGaussianParams],
        temporal: list[StableGaussianParams],
        locked: dict,
        device: str,
    ) -> None:
        self.names = names
        self.spatial_params = spatial
        self.temporal_params = temporal
        self.locked = locked
        self.spatial = GaussianMeanCandidateScorerFloat64(
            spatial, spatial[0].mean, device=device
        )
        self.temporal = GaussianMeanCandidateScorerFloat64(
            temporal, temporal[0].mean, device=device
        )
        self.device = device

    @torch.inference_mode()
    def score(
        self,
        global_batch: torch.Tensor,
        patch_batch: torch.Tensor,
        validate_direct: bool,
    ) -> tuple[dict[str, np.ndarray], dict[str, float]]:
        global_spatial, _ = score_gaussian_aggregate_float64(
            global_batch,
            self.locked["global_spatial"],
            aggregation="max",
            device=self.device,
            compute_percentile=False,
        )
        global_delta, zero = global_t1_features(global_batch)
        global_t1, _ = score_gaussian_aggregate_float64(
            global_delta,
            self.locked["global_t1"],
            aggregation="min",
            device=self.device,
            invalid_mask=zero,
            compute_percentile=False,
        )
        spatial = self.spatial.score(patch_batch)
        d2 = local_d2_features(patch_batch)
        temporal = self.temporal.score(d2)
        output = {
            "global_spatial_raw": global_spatial,
            "global_t1_raw": global_t1,
        }
        for index, name in enumerate(self.names):
            output[f"patch_spatial__{name}"] = spatial[:, index]
            output[f"patch_d2__{name}"] = temporal[:, index]
        errors: dict[str, float] = {}
        if validate_direct:
            direct_spatial, _ = score_gaussian_aggregate_float64(
                patch_batch,
                self.spatial_params[0],
                aggregation="mean",
                device=self.device,
                compute_percentile=False,
            )
            direct_temporal, _ = score_gaussian_aggregate_float64(
                d2,
                self.temporal_params[0],
                aggregation="mean",
                device=self.device,
                compute_percentile=False,
            )
            errors = {
                "patch_spatial": float(np.max(np.abs(direct_spatial - spatial[:, 0]))),
                "patch_d2": float(np.max(np.abs(direct_temporal - temporal[:, 0]))),
            }
        return output, errors


def completed_videos(directory: Path) -> tuple[set[str], list[Path]]:
    completed, parts = checkpoint_completed_ids(directory, cast_str=True)
    return {str(value) for value in completed}, parts


def run(args: argparse.Namespace) -> None:
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    rows, window_map = load_rows(
        args.split,
        args.dataset,
        args.release_dir,
        args.reserve_manifest,
        args.remaining_real_manifest,
    )
    rows = rows[
        rows["video_id"].map(
            lambda value: video_id_shard(str(value), args.num_shards)
            == args.shard_index
        )
    ].reset_index(drop=True)
    if args.debug_videos is not None:
        rows = rows.head(args.debug_videos).copy()
    checkpoint = (
        args.output_dir
        / "checkpoints"
        / args.split
        / args.dataset
        / f"shard_{args.shard_index:02d}_of_{args.num_shards:02d}"
    )
    checkpoint.mkdir(parents=True, exist_ok=True)
    completed, parts = completed_videos(checkpoint)
    pending = rows[~rows["video_id"].isin(completed)].reset_index(drop=True)
    print(
        f"split={args.split} dataset={args.dataset} shard={args.shard_index}/"
        f"{args.num_shards} expected={len(rows)} completed={len(completed)} "
        f"pending={len(pending)}",
        flush=True,
    )
    names, spatial, temporal, locked = load_candidate_params(
        config, args.dataset, args.params_dir, args.candidate_set
    )
    scorers = CandidateScorers(names, spatial, temporal, locked, args.score_device)
    extractor = PatchSTALL(args.extract_device, data_dict=None, load_dino=True)
    failures = []
    buffer: list[dict] = []
    max_errors = {"patch_spatial": 0.0, "patch_d2": 0.0}
    validated = 0
    part_index = len(parts)
    started = time.perf_counter()
    for start in range(0, len(pending), args.video_batch_size):
        source_rows = [
            row for _, row in pending.iloc[start : start + args.video_batch_size].iterrows()
        ]
        decoded = []
        with ThreadPoolExecutor(max_workers=min(args.decode_workers, len(source_rows))) as pool:
            futures = [
                pool.submit(
                    decode_row,
                    row,
                    window_map[str(row["video_id"])],
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
        extracted = [
            extractor.frames_to_global_patch_embeddings(
                [item["frames"]], batch_size=args.frame_batch_size
            )[0]
            for item in decoded
        ]
        global_windows = []
        patch_windows = []
        metadata = []
        for item, output in zip(decoded, extracted):
            for position, (sampling, window_id, indices) in zip(
                item["positions"], item["windows"]
            ):
                global_windows.append(output["global"][position])
                patch_windows.append(output["patch"][position])
                metadata.append((item, sampling, window_id, indices))
        if metadata:
            global_batch = torch.from_numpy(np.stack(global_windows).astype(np.float32))
            patch_batch = torch.from_numpy(np.stack(patch_windows).astype(np.float32))
            validate = validated < args.validate_direct_windows
            scores, errors = scorers.score(global_batch, patch_batch, validate)
            if validate:
                validated += len(metadata)
                for key, value in errors.items():
                    max_errors[key] = max(max_errors[key], value)
            for index, (item, sampling, window_id, indices) in enumerate(metadata):
                source = item["row"]
                buffer.append(
                    {
                        **{column: source[column] for column in KEY_COLUMNS},
                        "video_path": source["video_path"],
                        "duration_seconds": float(source["duration_seconds"]),
                        "sampling": sampling,
                        "window_id": window_id,
                        "frame_indices": json.dumps(indices, separators=(",", ":")),
                        **{key: value[index] for key, value in scores.items()},
                    }
                )
        processed = min(start + args.video_batch_size, len(pending))
        if len(buffer) >= args.checkpoint_rows or processed == len(pending):
            if buffer:
                path = checkpoint / f"part_{part_index:06d}.csv"
                temporary = path.with_suffix(".tmp.csv")
                pd.DataFrame(buffer).to_csv(temporary, index=False)
                temporary.replace(path)
                part_index += 1
                buffer.clear()
        if start == 0 or processed == len(pending) or processed % 100 == 0:
            print(
                f"processed={processed}/{len(pending)} elapsed={time.perf_counter()-started:.1f}s "
                f"direct_errors={max_errors}",
                flush=True,
            )
    failure_path = checkpoint / "failures.csv"
    pd.DataFrame(failures, columns=[*KEY_COLUMNS, "error"]).to_csv(failure_path, index=False)
    if failures:
        raise RuntimeError(f"{len(failures)} videos failed; see {failure_path}")
    if max(max_errors.values()) > args.direct_tolerance:
        raise ValueError(f"candidate sufficient-statistic validation failed: {max_errors}")
    print(f"complete direct_errors={max_errors}", flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--split",
        choices=(
            "evaluation",
            "locked_calibration",
            "reserve",
            "independent_remaining_real",
        ),
        required=True,
    )
    parser.add_argument(
        "--candidate-set",
        choices=("calibration_sensitivity", "cross_oas"),
        default="calibration_sensitivity",
    )
    parser.add_argument(
        "--dataset", choices=("comgenvid", "videofeedback", "genvideo"), required=True
    )
    parser.add_argument("--shard-index", type=int, required=True)
    parser.add_argument("--num-shards", type=int, default=2)
    parser.add_argument("--extract-device", default="cuda:0")
    parser.add_argument("--score-device", default="cuda:0")
    parser.add_argument("--video-batch-size", type=int, default=1)
    parser.add_argument("--frame-batch-size", type=int, default=32)
    parser.add_argument("--decode-workers", type=int, default=2)
    parser.add_argument("--seek-gap", type=int, default=48)
    parser.add_argument("--decode-attempts", type=int, default=3)
    parser.add_argument("--checkpoint-rows", type=int, default=1000)
    parser.add_argument("--validate-direct-windows", type=int, default=8)
    parser.add_argument("--direct-tolerance", type=float, default=1e-7)
    parser.add_argument("--debug-videos", type=int)
    parser.add_argument(
        "--config", type=Path, default=ROOT / "configs/alpha_stalled_u0_locked.yaml"
    )
    parser.add_argument("--release-dir", type=Path, default=ROOT / "release/u0")
    parser.add_argument(
        "--reserve-manifest",
        type=Path,
        default=ROOT / "release/u0/calibration_reserve_manifest.json",
    )
    parser.add_argument(
        "--remaining-real-manifest",
        type=Path,
        default=ROOT / "release/u0/independent_remaining_real_manifest.json",
    )
    parser.add_argument(
        "--params-dir",
        type=Path,
        default=ROOT / "results/u0_calibration_sensitivity/params",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "results/u0_calibration_sensitivity",
    )
    args = parser.parse_args()
    if not 0 <= args.shard_index < args.num_shards:
        parser.error("shard-index must be in [0, num-shards)")
    return args


if __name__ == "__main__":
    run(parse_args())
