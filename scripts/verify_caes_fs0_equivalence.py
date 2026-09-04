#!/usr/bin/env python3
"""验证CAES按需提取的FS0窗口与630GB严格缓存数值等价。"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from config import load_config, validate_config
from data.cache_contract import prepare_feature_cache
from data.manifest import load_manifest
from data.packed_cache import PackedCacheReader
from data.sampling import parse_indices
from data.video import decode_all_frames, decode_indexed_frames
from features import AlphaStallFeatureExtractor
from pipeline import (
    _choose_calibration,
    _load_cache_payload,
    _load_global_parameters,
)
from temporal_selection.calibration import load_frozen_local_reference
from temporal_selection.dense_scoring import (
    request_feature_arrays,
    score_fixed_windows,
    selected_window_requests,
    union_frame_indices,
)
from temporal_selection.manifest import read_window_manifests


DATASETS = ("comgenvid", "videofeedback", "genvideo")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/benchmark.yaml")
    parser.add_argument("--window-dir", type=Path, default=ROOT / "results/caes/window_selection_seed17")
    parser.add_argument("--detector-reference-dir", type=Path, default=ROOT / "precomputed/caes_detector")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "results/caes/fs0_equivalence")
    parser.add_argument("--datasets", nargs="+", choices=DATASETS, default=list(DATASETS))
    parser.add_argument("--samples-per-group", type=int, default=1)
    parser.add_argument("--device", default="cuda:1")
    parser.add_argument("--max-feature-abs", type=float, default=5e-6)
    parser.add_argument("--max-raw-score-abs", type=float, default=5e-3)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def _source_path(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def _decode(path: Path, indices: list[int]):
    try:
        return decode_indexed_frames(path, indices, require_all=True)
    except ValueError as error:
        frames = decode_all_frames(path, require_open=True)
        if not indices or max(indices) >= len(frames):
            raise ValueError(f"FS0回退解码仍缺帧：{path}") from error
        return frames[indices]


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _sample_rows(frame: pd.DataFrame, count: int) -> pd.DataFrame:
    if count < 1:
        raise ValueError("--samples-per-group必须为正数")
    return (
        frame.groupby(["subset", "source_model"], sort=True, group_keys=False)
        .head(count)
        .reset_index(drop=True)
    )


def _max_abs(left: np.ndarray, right: np.ndarray) -> float:
    if left.shape != right.shape:
        raise ValueError(f"FS0特征shape不一致：{left.shape} != {right.shape}")
    return float(np.max(np.abs(left.astype(np.float64) - right.astype(np.float64))))


def _score_abs(left: float, right: float) -> float:
    if np.isposinf(left) and np.isposinf(right):
        return 0.0
    difference = abs(float(left) - float(right))
    if not np.isfinite(difference):
        raise ValueError(f"FS0 raw score非有限或无效差异：{left}, {right}")
    return difference


def main() -> None:
    args = parse_args()
    if args.output_dir.exists() and not args.overwrite:
        raise FileExistsError(f"FS0等价审计目录已存在：{args.output_dir}")
    if args.output_dir.exists() and args.overwrite:
        import shutil
        shutil.rmtree(args.output_dir)
    args.output_dir.mkdir(parents=True)
    config = load_config(args.config)
    validate_config(config)
    cache_root = ROOT / config["runtime"]["cache_dir"]
    context = prepare_feature_cache(
        cache_root,
        expected_contract=None,
        policy="strict",
        create=False,
        required_cache_kind="patch_embeddings",
    )
    reader = PackedCacheReader(cache_root)
    model = AlphaStallFeatureExtractor(args.device)
    global_parameters = _load_global_parameters(ROOT, config)
    manifest_root = ROOT / config["data"]["development_manifests"]
    records = []
    sampled_ids = []
    manifest_hashes = {}

    for dataset in args.datasets:
        calibration = load_manifest(str(manifest_root / f"{dataset}_calibration.csv"))
        evaluation = load_manifest(str(manifest_root / f"{dataset}_evaluation.csv"))
        calibration = calibration[
            calibration["downsample_idxs"].map(
                lambda value: len(parse_indices(value)) >= 16
            )
        ].reset_index(drop=True)
        evaluation = evaluation[
            evaluation["downsample_idxs"].map(
                lambda value: len(parse_indices(value)) >= 16
            )
        ].reset_index(drop=True)
        selected_calibration = _choose_calibration(
            calibration,
            dataset,
            int(config["calibration"]["real_videos_per_dataset"]),
            int(config["calibration"]["seed"]),
        ).head(args.samples_per_group)
        sampled = pd.concat([
            selected_calibration.assign(split="calibration"),
            _sample_rows(evaluation, args.samples_per_group).assign(split="evaluation"),
        ], ignore_index=True)
        manifests = {}
        for split in ("calibration", "evaluation"):
            path = args.window_dir / "manifests" / dataset / f"uniform_{split}.jsonl"
            manifest_hashes[f"{dataset}/{split}"] = _sha256(path)
            manifests[split] = {
                item.video_id: item for item in read_window_manifests(path)
            }
        local_parameters = load_frozen_local_reference(
            args.detector_reference_dir / f"{dataset}_local_d2.npz"
        ).params

        for _, row in sampled.iterrows():
            video_id = f"{dataset}:{row['video_path']}"
            split = str(row["split"])
            manifest = manifests[split][video_id]
            requests = selected_window_requests({"uniform": manifest})
            union = union_frame_indices(requests)
            cached = _load_cache_payload(
                ROOT, cache_root, row, context, reader
            )
            if union != list(cached["frame_indices"]):
                raise ValueError(f"FS0窗口并集与strict cache索引不一致：{video_id}")
            cached_global, cached_patch = request_feature_arrays(
                requests,
                extracted_frame_indices=list(cached["frame_indices"]),
                global_features=cached["global"].numpy(),
                patch_features=cached["patch"].numpy(),
            )
            frames = _decode(_source_path(str(row["video_path"])), union)
            extracted = model.frames_to_global_patch_embeddings(
                [frames], batch_size=8
            )[0]
            dense_global, dense_patch = request_feature_arrays(
                requests,
                extracted_frame_indices=union,
                global_features=extracted["global"],
                patch_features=extracted["patch"],
            )
            cached_scores = score_fixed_windows(
                requests,
                global_windows=cached_global,
                patch_windows=cached_patch,
                global_parameters=global_parameters,
                local_parameters=local_parameters,
                device=args.device,
            )
            dense_scores = score_fixed_windows(
                requests,
                global_windows=dense_global,
                patch_windows=dense_patch,
                global_parameters=global_parameters,
                local_parameters=local_parameters,
                device=args.device,
            )
            for cached_score, dense_score in zip(cached_scores, dense_scores):
                if cached_score["candidate_id"] != dense_score["candidate_id"]:
                    raise ValueError("FS0评分窗口顺序漂移")
                records.append({
                    "dataset": dataset,
                    "split": split,
                    "subset": str(row["subset"]),
                    "source_model": str(row["source_model"]),
                    "video_id": video_id,
                    "candidate_id": cached_score["candidate_id"],
                    "global_feature_max_abs": _max_abs(cached_global, dense_global),
                    "patch_feature_max_abs": _max_abs(cached_patch, dense_patch),
                    "global_spatial_raw_max_abs": _score_abs(
                        cached_score["global_spatial_raw"],
                        dense_score["global_spatial_raw"],
                    ),
                    "global_t1_raw_max_abs": _score_abs(
                        cached_score["global_t1_raw"], dense_score["global_t1_raw"]
                    ),
                    "patch_temporal_raw_max_abs": _score_abs(
                        cached_score["patch_temporal_raw"], dense_score["patch_temporal_raw"]
                    ),
                })
            sampled_ids.append(video_id)
            print(f"[{dataset}] FS0等价审计：{video_id}", flush=True)

    table = pd.DataFrame(records)
    feature_max = float(table[
        ["global_feature_max_abs", "patch_feature_max_abs"]
    ].to_numpy().max())
    raw_max = float(table[
        ["global_spatial_raw_max_abs", "global_t1_raw_max_abs", "patch_temporal_raw_max_abs"]
    ].to_numpy().max())
    table.to_csv(args.output_dir / "window_differences.csv", index=False)
    audit = {
        "schema_version": "caes_fs0_equivalence_v1",
        "status": "passed" if (
            feature_max <= args.max_feature_abs
            and raw_max <= args.max_raw_score_abs
        ) else "failed",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "cache_contract_sha256": context.contract_sha256,
        "window_manifest_sha256": manifest_hashes,
        "sample_video_ids": sampled_ids,
        "sample_windows": len(table),
        "feature_max_abs": feature_max,
        "raw_score_max_abs": raw_max,
        "thresholds": {
            "max_feature_abs": args.max_feature_abs,
            "max_raw_score_abs": args.max_raw_score_abs,
        },
        "window_differences_sha256": _sha256(
            args.output_dir / "window_differences.csv"
        ),
    }
    (args.output_dir / "audit.json").write_text(
        json.dumps(audit, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    if audit["status"] != "passed":
        raise ValueError(f"FS0 on-demand与strict cache超过数值门槛：{audit}")
    print(f"[通过] feature max={feature_max:.3e}, raw max={raw_max:.3e}")


if __name__ == "__main__":
    main()
