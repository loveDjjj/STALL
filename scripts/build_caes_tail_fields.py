#!/usr/bin/env python3
"""为三种selector构建可恢复的Local D2位置似然场小缓存。"""

from __future__ import annotations

import argparse
import hashlib
import json
import shlex
import shutil
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import torch


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from artifacts import write_progress
from config import config_digest, dump_config, load_config, validate_config
from data.manifest import load_manifest
from data.sampling import parse_indices
from data.video import decode_all_frames, decode_indexed_frames
from features import AlphaStallFeatureExtractor
from pipeline import _choose_calibration, _ensure_disjoint, _load_global_parameters
from tail_evidence import merge_tail_window_requests, score_tail_windows
from temporal_selection.calibration import load_frozen_local_reference
from temporal_selection.manifest import read_window_manifests


DATASETS = ("comgenvid", "videofeedback", "genvideo")
SELECTORS = ("uniform", "feature_change", "real_anomaly")
SHARD_SCHEMA = "caes_tail_field_shard_v1"
RUN_SCHEMA = "caes_tail_field_run_v1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/benchmark.yaml")
    parser.add_argument(
        "--standard-window-dir", type=Path,
        default=ROOT / "results/caes/window_selection_seed17",
    )
    parser.add_argument(
        "--crossfit-window-dir", type=Path,
        default=ROOT / "results/caes/crossfit5_seed17",
    )
    parser.add_argument(
        "--detector-reference-dir", type=Path,
        default=ROOT / "precomputed/caes_detector",
    )
    parser.add_argument(
        "--output-dir", type=Path,
        default=ROOT / "results/caes/tail_fields_v1",
    )
    parser.add_argument("--device", default="cuda:1")
    parser.add_argument("--datasets", nargs="+", choices=DATASETS, default=list(DATASETS))
    parser.add_argument("--chunk-videos", type=int, default=16)
    parser.add_argument("--frame-batch-size", type=int, default=8)
    parser.add_argument("--score-window-batch-size", type=int, default=48)
    parser.add_argument("--decode-workers", type=int, default=8)
    parser.add_argument("--limit-evaluation-per-source", type=int)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _canonical_digest(value: object) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _atomic_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def _atomic_torch(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp.pt")
    torch.save(value, temporary)
    temporary.replace(path)


def _source_path(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def _decode(path: Path, indices: list[int]):
    try:
        return decode_indexed_frames(path, indices, require_all=True)
    except ValueError as error:
        frames = decode_all_frames(path, require_open=True)
        if not indices or max(indices) >= len(frames):
            raise ValueError(f"Tail顺序回退仍缺帧：{path}") from error
        return frames[indices]


def _load_mapping(path: Path, expected_sha: str) -> dict:
    if _sha256(path) != expected_sha:
        raise ValueError(f"Tail输入WindowManifest哈希漂移：{path}")
    values = read_window_manifests(path)
    return {item.video_id: item for item in values}


def _feature_arrays(requests, union, output):
    positions = {frame: index for index, frame in enumerate(union)}
    global_windows, patch_windows = [], []
    for request in requests:
        try:
            selected = [positions[frame] for frame in request.frame_indices]
        except KeyError as error:
            raise ValueError("Tail dense提取缺少selected frame") from error
        global_windows.append(output["global"][selected])
        patch_windows.append(output["patch"][selected])
    return np.stack(global_windows), np.stack(patch_windows)


def _score_split(
    *,
    dataset: str,
    split: str,
    rows: pd.DataFrame,
    standard: dict[str, dict],
    crossfit_real_anomaly: dict | None,
    output_dir: Path,
    run_identity_sha256: str,
    model,
    global_parameters,
    local_parameters,
    args,
    progress: dict,
) -> list[dict]:
    expected_ids = [f"{dataset}:{path}" for path in rows["video_path"].astype(str)]
    if len(expected_ids) != len(set(expected_ids)):
        raise ValueError(f"{dataset}/{split}输入含重复video_id")
    for selector, mapping in standard.items():
        if not set(expected_ids).issubset(mapping):
            raise ValueError(f"{dataset}/{split}/{selector}缺少输入视频")
    if crossfit_real_anomaly is not None and set(expected_ids) != set(crossfit_real_anomaly):
        raise ValueError(f"{dataset}/calibration crossfit视频身份不一致")
    ordered = rows.assign(video_id=expected_ids).reset_index(drop=True)
    shard_summaries = []
    total_unique_frames = 0
    started = time.monotonic()
    for start in range(0, len(ordered), args.chunk_videos):
        stop = min(start + args.chunk_videos, len(ordered))
        shard_index = start // args.chunk_videos
        shard_path = output_dir / "shards" / dataset / split / f"shard-{shard_index:05d}.pt"
        video_ids = ordered.iloc[start:stop]["video_id"].tolist()
        if shard_path.is_file():
            payload = torch.load(shard_path, weights_only=True)
            if not (
                payload.get("schema_version") == SHARD_SCHEMA
                and payload.get("run_identity_sha256") == run_identity_sha256
                and payload.get("video_ids") == video_ids
                and payload.get("dataset") == dataset
                and payload.get("split") == split
            ):
                raise ValueError(f"Tail shard恢复身份不一致：{shard_path}")
            total_unique_frames += int(payload["dense_unique_frames"])
            shard_summaries.append({
                "path": shard_path.relative_to(output_dir).as_posix(),
                "sha256": _sha256(shard_path),
                "videos": len(video_ids),
                "unique_windows": int(payload["local_likelihood_fields"].shape[0]),
                "uses": len(payload["records"]),
                "dense_unique_frames": int(payload["dense_unique_frames"]),
                "resumed": True,
            })
            continue

        batch_rows = [row for _, row in ordered.iloc[start:stop].iterrows()]

        def prepare_video(row):
            video_id = str(row["video_id"])
            manifests = {
                ("standard", selector): mapping[video_id]
                for selector, mapping in standard.items()
            }
            if crossfit_real_anomaly is not None:
                manifests[("crossfit5", "real_anomaly")] = crossfit_real_anomaly[video_id]
            requests = merge_tail_window_requests(manifests)
            union = sorted({frame for request in requests for frame in request.frame_indices})
            if not requests or not union:
                raise ValueError(f"Tail没有可评分窗口：{video_id}")
            frames = _decode(_source_path(str(row["video_path"])), union)
            return row, video_id, requests, union, frames

        with ThreadPoolExecutor(
            max_workers=min(args.decode_workers, len(batch_rows))
        ) as executor:
            prepared = list(executor.map(prepare_video, batch_rows))
        outputs = model.frames_to_global_patch_embeddings(
            [item[4] for item in prepared], batch_size=args.frame_batch_size
        )
        flat_requests, request_metadata = [], []
        global_items, patch_items = [], []
        shard_dense_frames = 0
        for (row, video_id, requests, union, _frames), output in zip(prepared, outputs):
            global_windows, patch_windows = _feature_arrays(requests, union, output)
            flat_requests.extend(requests)
            global_items.extend(global_windows)
            patch_items.extend(patch_windows)
            request_metadata.extend([(row, video_id)] * len(requests))
            shard_dense_frames += len(union)

        fields = []
        records = []
        for offset in range(0, len(flat_requests), args.score_window_batch_size):
            end = min(offset + args.score_window_batch_size, len(flat_requests))
            scored = score_tail_windows(
                np.stack(global_items[offset:end]),
                np.stack(patch_items[offset:end]),
                global_parameters=global_parameters,
                local_parameters=local_parameters,
                device=args.device,
            )
            field_offset = sum(len(item) for item in fields)
            fields.append(scored.local_likelihood_field)
            for local_index, request in enumerate(flat_requests[offset:end]):
                row, video_id = request_metadata[offset + local_index]
                field_index = field_offset + local_index
                for use in request.uses:
                    records.append({
                        "field_index": field_index,
                        "video_id": video_id,
                        "dataset": dataset,
                        "split": split,
                        "subset": str(row["subset"]),
                        "source_model": str(row["source_model"]),
                        "video_path": str(row["video_path"]),
                        "selector": use.selector,
                        "calibration_mode": use.calibration_mode,
                        "candidate_id": use.candidate.candidate_id,
                        "selection_rank": use.selection.rank,
                        "start_seconds": use.candidate.start_seconds,
                        "end_seconds": use.candidate.end_seconds,
                        "frame_indices": list(request.frame_indices),
                        "global_spatial_raw": float(scored.global_spatial_raw[local_index]),
                        "global_spatial": float(scored.global_spatial[local_index]),
                        "global_t1_raw": float(scored.global_t1_raw[local_index]),
                        "global_t1": float(scored.global_t1[local_index]),
                        **{
                            f"local_raw__{name}": float(values[local_index])
                            for name, values in scored.local_raw.items()
                        },
                    })
        grouped = {}
        for record in records:
            key = (
                record["video_id"], record["selector"], record["calibration_mode"]
            )
            grouped.setdefault(key, []).append(record)
        records = []
        for key in sorted(grouped):
            values = sorted(grouped[key], key=lambda item: item["start_seconds"])
            for window_id, record in enumerate(values):
                record["window_id"] = window_id
                records.append(record)
        field_tensor = torch.from_numpy(np.concatenate(fields, axis=0))
        if field_tensor.dtype != torch.float32 or field_tensor.ndim != 3:
            raise ValueError("Tail field存储dtype/shape无效")
        _atomic_torch(shard_path, {
            "schema_version": SHARD_SCHEMA,
            "run_identity_sha256": run_identity_sha256,
            "dataset": dataset,
            "split": split,
            "video_ids": video_ids,
            "dense_unique_frames": shard_dense_frames,
            "local_likelihood_fields": field_tensor,
            "records": records,
        })
        total_unique_frames += shard_dense_frames
        shard_summaries.append({
            "path": shard_path.relative_to(output_dir).as_posix(),
            "sha256": _sha256(shard_path),
            "videos": len(video_ids),
            "unique_windows": len(field_tensor),
            "uses": len(records),
            "dense_unique_frames": shard_dense_frames,
            "resumed": False,
        })
        progress.update({
            "status": "running",
            "phase": "tail_field_extraction",
            "dataset": dataset,
            "split": split,
            "completed_videos": stop,
            "total_videos": len(ordered),
            "dense_unique_frames": total_unique_frames,
            "updated_at": datetime.now(timezone.utc).isoformat(),
        })
        write_progress(output_dir, progress)
        elapsed = max(time.monotonic() - started, 1e-6)
        print(
            f"[{dataset}/{split}] Tail fields {stop}/{len(ordered)}，"
            f"{stop / elapsed:.2f}视频/秒",
            flush=True,
        )
    return shard_summaries


def main() -> None:
    args = parse_args()
    if args.resume and args.overwrite:
        raise ValueError("--resume与--overwrite不能同时使用")
    if args.chunk_videos < 1 or args.decode_workers < 1:
        raise ValueError("chunk/decode workers必须为正数")
    if args.frame_batch_size != 8 or args.score_window_batch_size != 48:
        raise ValueError("Tail第一轮固定frame batch=8、score window batch=48")
    for name in ("config", "standard_window_dir", "crossfit_window_dir", "detector_reference_dir", "output_dir"):
        setattr(args, name, getattr(args, name).resolve())
    config = load_config(args.config)
    validate_config(config)
    standard_path = args.standard_window_dir / "run_manifest.json"
    crossfit_path = args.crossfit_window_dir / "run_manifest.json"
    standard_run = json.loads(standard_path.read_text(encoding="utf-8"))
    crossfit_run = json.loads(crossfit_path.read_text(encoding="utf-8"))
    if standard_run.get("status") != "completed" or crossfit_run.get("status") != "completed":
        raise ValueError("Tail要求standard与crossfit WindowManifest均已完成")
    local_references = {
        dataset: load_frozen_local_reference(
            args.detector_reference_dir / f"{dataset}_local_d2.npz"
        ) for dataset in args.datasets
    }
    identity = {
        "schema_version": RUN_SCHEMA,
        "config_hash": config_digest(config),
        "standard_window_run_sha256": _sha256(standard_path),
        "crossfit_window_run_sha256": _sha256(crossfit_path),
        "datasets": list(args.datasets),
        "selectors": list(SELECTORS),
        "calibration_modes": ["standard", "crossfit5"],
        "local_reference_sha256": {
            dataset: reference.digest() for dataset, reference in local_references.items()
        },
        "chunk_videos": args.chunk_videos,
        "frame_batch_size": args.frame_batch_size,
        "score_window_batch_size": args.score_window_batch_size,
        "decode_workers": args.decode_workers,
        "limit_evaluation_per_source": args.limit_evaluation_per_source,
        "implementation_sha256": {
            "script": _sha256(Path(__file__)),
            "tail_evidence": _sha256(ROOT / "src/tail_evidence.py"),
            "features": _sha256(ROOT / "src/features.py"),
        },
    }
    identity_sha = _canonical_digest(identity)
    if args.output_dir.exists() and args.overwrite:
        shutil.rmtree(args.output_dir)
    if args.output_dir.exists() and not args.resume:
        raise FileExistsError(f"Tail field目录已存在：{args.output_dir}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    identity_path = args.output_dir / "run_identity.json"
    expected_identity = {**identity, "identity_sha256": identity_sha}
    if args.resume:
        if json.loads(identity_path.read_text(encoding="utf-8")) != expected_identity:
            raise ValueError("Tail恢复运行身份发生变化")
    else:
        _atomic_json(identity_path, expected_identity)
        dump_config(args.output_dir / "resolved_config.yaml", config)
        (args.output_dir / "command.txt").write_text(
            "命令：" + " ".join(shlex.quote(item) for item in [sys.executable, *sys.argv]) + "\n",
            encoding="utf-8",
        )
    progress = {"status": "running", "phase": "initializing", "run_identity_sha256": identity_sha}
    write_progress(args.output_dir, progress)
    model = AlphaStallFeatureExtractor(args.device)
    global_parameters = _load_global_parameters(ROOT, config)
    manifest_root = ROOT / config["data"]["development_manifests"]
    index = {"schema_version": RUN_SCHEMA, "status": "running", "identity_sha256": identity_sha, "datasets": {}}

    for dataset in args.datasets:
        calibration = load_manifest(str(manifest_root / f"{dataset}_calibration.csv"))
        evaluation = load_manifest(str(manifest_root / f"{dataset}_evaluation.csv"))
        _ensure_disjoint(calibration, evaluation, dataset)
        calibration = calibration[
            calibration["downsample_idxs"].map(lambda value: len(parse_indices(value)) >= 16)
        ].reset_index(drop=True)
        evaluation = evaluation[
            evaluation["downsample_idxs"].map(lambda value: len(parse_indices(value)) >= 16)
        ].reset_index(drop=True)
        calibration = _choose_calibration(
            calibration, dataset,
            int(config["calibration"]["real_videos_per_dataset"]),
            int(config["calibration"]["seed"]),
        )
        if args.limit_evaluation_per_source is not None:
            evaluation = evaluation.groupby(
                ["subset", "source_model"], sort=True, group_keys=False
            ).head(args.limit_evaluation_per_source).reset_index(drop=True)

        standard_mappings = {}
        for split in ("calibration", "evaluation"):
            standard_mappings[split] = {}
            for selector in SELECTORS:
                path = args.standard_window_dir / "manifests" / dataset / f"{selector}_{split}.jsonl"
                expected_sha = standard_run["datasets"][dataset]["splits"][split]["selectors"][selector]["sha256"]
                standard_mappings[split][selector] = _load_mapping(path, expected_sha)
        cf_path = args.crossfit_window_dir / "manifests" / dataset / "real_anomaly_calibration.jsonl"
        cf_sha = crossfit_run["datasets"][dataset]["manifests"]["real_anomaly"]["sha256"]
        crossfit_mapping = _load_mapping(cf_path, cf_sha)
        dataset_index = {}
        dataset_index["calibration"] = _score_split(
            dataset=dataset, split="calibration", rows=calibration,
            standard=standard_mappings["calibration"],
            crossfit_real_anomaly=crossfit_mapping,
            output_dir=args.output_dir, run_identity_sha256=identity_sha,
            model=model, global_parameters=global_parameters,
            local_parameters=local_references[dataset].params,
            args=args, progress=progress,
        )
        dataset_index["evaluation"] = _score_split(
            dataset=dataset, split="evaluation", rows=evaluation,
            standard=standard_mappings["evaluation"],
            crossfit_real_anomaly=None,
            output_dir=args.output_dir, run_identity_sha256=identity_sha,
            model=model, global_parameters=global_parameters,
            local_parameters=local_references[dataset].params,
            args=args, progress=progress,
        )
        index["datasets"][dataset] = dataset_index
        _atomic_json(args.output_dir / "index.json", index)
    index["status"] = "completed"
    index["completed_at"] = datetime.now(timezone.utc).isoformat()
    _atomic_json(args.output_dir / "index.json", index)
    progress.update({"status": "completed", "phase": "completed", "completed_at": index["completed_at"]})
    write_progress(args.output_dir, progress)
    print(f"[完成] CAES Tail likelihood fields：{args.output_dir}")


if __name__ == "__main__":
    main()
