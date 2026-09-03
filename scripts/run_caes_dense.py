#!/usr/bin/env python3
"""按FS1-FS5 WindowManifest执行可恢复的selected-window dense评分。"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import shlex
import shutil
import signal
import subprocess
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from artifacts import write_csv, write_progress
from config import config_digest, dump_config, load_config, validate_config
from data.manifest import load_manifest
from data.sampling import parse_indices, uniform_windows
from data.video import decode_all_frames, decode_indexed_frames
from evaluation.tables import build_metric_tables, build_pairwise_metric_table
from features import AlphaStallFeatureExtractor
from pipeline import (
    _calibrate_and_aggregate,
    _calibrate_component,
    _choose_calibration,
    _ensure_disjoint,
    _load_global_parameters,
    _weighted,
)
from runner import _Tee
from temporal_selection.calibration import load_frozen_local_reference
from temporal_selection.dense_scoring import (
    request_feature_arrays,
    score_fixed_windows,
    selected_window_requests,
    union_frame_indices,
)
from temporal_selection.evaluation import (
    evaluate_selector_gate,
    paired_selector_bootstrap,
)
from temporal_selection.manifest import read_window_manifests


SOURCE_RUN = "alpha_stall_full_d2_k3_no_spatial_refit"
SELECTORS = (
    "uniform",
    "random",
    "feature_change",
    "real_anomaly",
    "real_anomaly_nms",
    "stratified_real_anomaly",
)
ADAPTIVE_SELECTORS = SELECTORS[1:]
DATASETS = ("comgenvid", "videofeedback", "genvideo")
RAW_SHARD_SCHEMA = "caes_raw_score_shard_v2"
RUN_IDENTITY_SCHEMA = "caes_dense_identity_v1"


class RunTerminated(RuntimeError):
    """把外部终止信号转换为可恢复的显式中断。"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/benchmark.yaml")
    parser.add_argument("--window-dir", type=Path, default=ROOT / "results/caes/window_selection_seed17")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "results/runs/caes_stage_fs")
    parser.add_argument("--detector-reference-dir", type=Path, default=ROOT / "precomputed/caes_detector")
    parser.add_argument("--device", default="cuda:1")
    parser.add_argument("--datasets", nargs="+", choices=DATASETS, default=list(DATASETS))
    parser.add_argument("--chunk-videos", type=int, default=64)
    parser.add_argument("--frame-batch-size", type=int, default=8)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def _source_path(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _canonical_digest(payload: object) -> str:
    canonical = json.dumps(
        payload, ensure_ascii=True, sort_keys=True, separators=(",", ":")
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _decode(path: Path, indices: list[int]):
    try:
        return decode_indexed_frames(path, indices, require_all=True)
    except ValueError as error:
        frames = decode_all_frames(path, require_open=True)
        if not indices or max(indices) >= len(frames):
            raise ValueError(
                f"selected-window顺序回退仍缺帧：{path}，解码={len(frames)}"
            ) from error
        return frames[indices]


def _atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def _detector_contract(config: dict) -> dict:
    """只提取Stage FS必须冻结的检测器字段，运行资源不属于方法定义。"""

    method = config["method"]
    local = method["local"]
    correspondence = local.get("correspondence", {})
    return {
        "global": {
            key: method["global"][key]
            for key in (
                "enabled", "parameter_source", "parameters_sha256",
                "spatial_weight", "temporal_weight",
            )
        },
        "local": {
            "enabled": local["enabled"],
            "spatial_enabled": local["spatial_enabled"],
            "temporal_enabled": local["temporal_enabled"],
            "temporal_order": local["temporal_order"],
            # C0 resolved config早于模块化字段；这些是当时执行代码的等价默认值。
            "dynamics": local.get("dynamics", "finite_difference"),
            "correspondence_type": correspondence.get("type", "same_grid"),
            "correspondence_confidence": correspondence.get("confidence", "none"),
            # Spatial关闭时历史0.1/0.9与当前0/1都会由_weighted归一成D2-only。
            "effective_spatial_weight": (
                local["spatial_weight"] if local["spatial_enabled"] else 0.0
            ),
            "effective_temporal_weight": (
                1.0 if not local["spatial_enabled"] else local["temporal_weight"]
            ),
        },
        "fusion": method["fusion"],
        "sampling": config["sampling"],
        "calibration": {
            key: config["calibration"][key]
            for key in (
                "real_videos_per_dataset", "seed", "disjoint_from_evaluation",
                "video_level",
            )
        },
    }


def _load_manifests(
    window_dir: Path,
    window_run: dict,
    dataset: str,
    split: str,
) -> dict[str, dict[str, object]]:
    output = {}
    expected_ids = None
    split_contract = window_run["datasets"][dataset]["splits"][split]["selectors"]
    for selector in SELECTORS:
        path = window_dir / "manifests" / dataset / f"{selector}_{split}.jsonl"
        expected_sha = split_contract[selector]["sha256"]
        if _sha256(path) != expected_sha:
            raise ValueError(f"WindowManifest内容哈希漂移：{path}")
        values = read_window_manifests(path)
        mapping = {item.video_id: item for item in values}
        if len(values) != int(split_contract[selector]["videos"]):
            raise ValueError(f"WindowManifest视频数与run manifest不一致：{path}")
        if expected_ids is None:
            expected_ids = set(mapping)
        elif set(mapping) != expected_ids:
            raise ValueError(f"{dataset}/{split} selector视频身份不一致")
        output[selector] = mapping
    return output


def _verify_uniform_manifests(
    dataset: str,
    rows: pd.DataFrame,
    manifests: dict[str, dict[str, object]],
) -> None:
    """逐视频确认FS0仍调用历史uniform_windows，而不是近似实现。"""

    uniform = manifests["uniform"]
    for _, row in rows.iterrows():
        video_id = f"{dataset}:{row['video_path']}"
        item = uniform[video_id]
        candidates = {candidate.candidate_id: candidate for candidate in item.candidates}
        actual = [
            list(candidates[selected.candidate_id].frame_indices)
            for selected in sorted(item.selected, key=lambda value: value.rank)
        ]
        expected = uniform_windows(
            parse_indices(row["downsample_idxs"]), requested_k=3, window_frames=16
        )
        if actual != expected:
            raise ValueError(f"FS0与历史uniform_windows漂移：{video_id}")


def _selected_window_rows(
    dataset: str,
    split: str,
    rows: pd.DataFrame,
    manifests: dict[str, dict[str, object]],
) -> tuple[list[dict], list[dict]]:
    """导出独立选择清单及每视频coverage诊断，不依赖dense评分。"""

    metadata = {f"{dataset}:{row['video_path']}": row for _, row in rows.iterrows()}
    selected_rows: list[dict] = []
    diagnostics: list[dict] = []
    for selector, mapping in manifests.items():
        for video_id, manifest in mapping.items():
            row = metadata[video_id]
            candidates = {item.candidate_id: item for item in manifest.candidates}
            chosen = sorted(manifest.selected, key=lambda value: value.rank)
            centers = np.asarray(
                [candidates[item.candidate_id].center_seconds for item in chosen],
                dtype=np.float64,
            )
            distances = [
                abs(float(left - right))
                for index, left in enumerate(centers)
                for right in centers[index + 1 :]
            ]
            union = {
                frame
                for item in chosen
                for frame in candidates[item.candidate_id].frame_indices
            }
            diagnostics.append({
                "selector": selector,
                "video_id": video_id,
                "dataset": dataset,
                "split": split,
                "subset": str(row["subset"]),
                "source_model": str(row["source_model"]),
                "candidate_windows": len(manifest.candidates),
                "effective_k": manifest.effective_k,
                "selected_center_span_seconds": (
                    float(centers.max() - centers.min()) if len(centers) else 0.0
                ),
                "selected_pairwise_distance_mean_seconds": (
                    float(np.mean(distances)) if distances else 0.0
                ),
                "selected_pairwise_distance_min_seconds": (
                    float(np.min(distances)) if distances else 0.0
                ),
                "selected_unique_dense_frames": len(union),
                "selection_reason_count": len({item.reason for item in chosen}),
            })
            for item in chosen:
                candidate = candidates[item.candidate_id]
                selected_rows.append({
                    "selector": selector,
                    "video_id": video_id,
                    "dataset": dataset,
                    "split": split,
                    "subset": str(row["subset"]),
                    "source_model": str(row["source_model"]),
                    "video_path": str(row["video_path"]),
                    "selection_rank": item.rank,
                    "selection_score": item.score,
                    "selection_reason": item.reason,
                    "candidate_id": candidate.candidate_id,
                    "start_seconds": candidate.start_seconds,
                    "end_seconds": candidate.end_seconds,
                    "center_seconds": candidate.center_seconds,
                    "frame_indices": json.dumps(list(candidate.frame_indices)),
                    **candidate.scores,
                })
    return selected_rows, diagnostics


def _score_split(
    *,
    dataset: str,
    split: str,
    rows: pd.DataFrame,
    manifests: dict[str, dict[str, object]],
    raw_root: Path,
    model: AlphaStallFeatureExtractor,
    global_parameters,
    local_parameters,
    device: str,
    chunk_videos: int,
    frame_batch_size: int,
    run_identity_sha256: str,
    report,
) -> tuple[pd.DataFrame, dict[str, object]]:
    adaptive = {name: manifests[name] for name in ADAPTIVE_SELECTORS}
    expected = [f"{dataset}:{path}" for path in rows["video_path"].astype(str)]
    manifest_ids = set(next(iter(adaptive.values())))
    if len(expected) != len(set(expected)) or set(expected) != manifest_ids:
        raise ValueError(
            f"{dataset}/{split} rows与WindowManifest身份不一致："
            f"rows={len(expected)}, manifest={len(manifest_ids)}"
        )
    ordered = rows.assign(video_id=expected).reset_index(drop=True)
    shard_dir = raw_root / dataset / split
    all_records = []
    total_dense_frames = 0
    resumed_shards = 0
    started = time.monotonic()
    for start in range(0, len(ordered), chunk_videos):
        stop = min(start + chunk_videos, len(ordered))
        shard_index = start // chunk_videos
        shard_path = shard_dir / f"shard-{shard_index:05d}.json"
        video_ids = ordered.iloc[start:stop]["video_id"].tolist()
        expected_records = sum(
            adaptive[selector][video_id].effective_k
            for video_id in video_ids
            for selector in ADAPTIVE_SELECTORS
        )
        if shard_path.is_file():
            payload = json.loads(shard_path.read_text(encoding="utf-8"))
            valid = (
                payload.get("schema_version") == RAW_SHARD_SCHEMA
                and payload.get("run_identity_sha256") == run_identity_sha256
                and payload.get("dataset") == dataset
                and payload.get("split") == split
                and payload.get("video_ids") == video_ids
                and len(payload.get("records", [])) == expected_records
            )
            if not valid:
                raise ValueError(f"raw score shard运行身份不匹配：{shard_path}")
            all_records.extend(payload["records"])
            total_dense_frames += int(payload["dense_unique_frames"])
            resumed_shards += 1
            report(dataset, split, stop, len(ordered), total_dense_frames, True)
            continue
        shard_records = []
        shard_dense_frames = 0
        for _, row in ordered.iloc[start:stop].iterrows():
            video_id = str(row["video_id"])
            video_manifests = {
                selector: mapping[video_id] for selector, mapping in adaptive.items()
            }
            requests = selected_window_requests(video_manifests)
            union = union_frame_indices(requests)
            if not requests or not union:
                raise ValueError(f"{video_id}没有可评分selected windows")
            frames = _decode(_source_path(str(row["video_path"])), union)
            output = model.frames_to_global_patch_embeddings(
                [frames], batch_size=frame_batch_size
            )[0]
            global_windows, patch_windows = request_feature_arrays(
                requests,
                extracted_frame_indices=union,
                global_features=output["global"],
                patch_features=output["patch"],
            )
            records = score_fixed_windows(
                requests,
                global_windows=global_windows,
                patch_windows=patch_windows,
                global_parameters=global_parameters,
                local_parameters=local_parameters,
                device=device,
            )
            per_selector: dict[str, list[dict]] = {}
            for record in records:
                per_selector.setdefault(str(record["selector"]), []).append(record)
            if set(per_selector) != set(ADAPTIVE_SELECTORS):
                raise ValueError(f"{video_id}评分结果缺少selector")
            for selector, selector_records in per_selector.items():
                selector_records.sort(key=lambda item: int(item["selection_rank"]))
                for window_id, record in enumerate(selector_records):
                    record.update({
                        "video_id": video_id,
                        "dataset": dataset,
                        "subset": str(row["subset"]),
                        "source_model": str(row["source_model"]),
                        "video_path": str(row["video_path"]),
                        "window_id": window_id,
                        "dense_unique_frames_video": len(union),
                    })
                    shard_records.append(record)
            shard_dense_frames += len(union)
        if len(shard_records) != expected_records:
            raise ValueError(f"{dataset}/{split} shard窗口数不符合WindowManifest")
        _atomic_json(shard_path, {
            "schema_version": RAW_SHARD_SCHEMA,
            "run_identity_sha256": run_identity_sha256,
            "dataset": dataset,
            "split": split,
            "video_ids": video_ids,
            "dense_unique_frames": shard_dense_frames,
            "records": shard_records,
        })
        all_records.extend(shard_records)
        total_dense_frames += shard_dense_frames
        report(dataset, split, stop, len(ordered), total_dense_frames, False)
    frame = pd.DataFrame(all_records)
    finite_columns = ["global_spatial", "global_t1", "patch_temporal_raw"]
    if frame.empty or not np.isfinite(
        frame[finite_columns].to_numpy(dtype=np.float64)
    ).all():
        raise ValueError(f"{dataset}/{split} dense评分为空或含非有限有效分量")
    return frame, {
        "videos": len(ordered),
        "dense_unique_frames": total_dense_frames,
        "seconds": time.monotonic() - started,
        "raw_shards": len(list(shard_dir.glob("shard-*.json"))),
        "resumed_shards": resumed_shards,
    }


def _calibrate_selector(
    calibration: pd.DataFrame, evaluation: pd.DataFrame, config: dict
) -> tuple[pd.DataFrame, pd.DataFrame]:
    calibrated_calibration, videos = _calibrate_and_aggregate(
        calibration, evaluation, config
    )
    reference = calibration[calibration["subset"].eq("real")]
    calibrated_evaluation = evaluation.copy()
    calibrated_evaluation["patch_temporal"] = _calibrate_component(
        calibrated_evaluation["patch_temporal_raw"], reference["patch_temporal_raw"]
    )
    calibrated_evaluation["global_score_window"] = _weighted(
        calibrated_evaluation,
        ["global_spatial", "global_t1"],
        [
            float(config["method"]["global"]["spatial_weight"]),
            float(config["method"]["global"]["temporal_weight"]),
        ],
    )
    calibrated_evaluation["local_score_window"] = calibrated_evaluation[
        "patch_temporal"
    ]
    windows = pd.concat([
        calibrated_calibration.assign(split="calibration"),
        calibrated_evaluation.assign(split="evaluation"),
    ], ignore_index=True)
    if not np.isfinite(videos["final_score"].to_numpy(dtype=np.float64)).all():
        raise ValueError("selector视频分数包含非有限值")
    return windows, videos


def _prepare(args: argparse.Namespace):
    if args.resume and args.overwrite:
        raise ValueError("--resume与--overwrite不能同时使用")
    if args.chunk_videos < 1:
        raise ValueError("--chunk-videos必须为正数")
    # dense batch=8是已与630GB cache验证到3e-6的数值合同。
    if args.frame_batch_size != 8:
        raise ValueError("CAES第一轮固定--frame-batch-size=8，避免baseline数值漂移")
    args.config = args.config.resolve()
    args.window_dir = args.window_dir.resolve()
    args.output_dir = args.output_dir.resolve()
    args.detector_reference_dir = args.detector_reference_dir.resolve()
    config = load_config(args.config)
    validate_config(config)
    source_dir = ROOT / "results/runs" / SOURCE_RUN
    source_manifest = json.loads(
        (source_dir / "run_manifest.json").read_text(encoding="utf-8")
    )
    source_config = load_config(source_dir / "resolved_config.yaml")
    if _detector_contract(config) != _detector_contract(source_config):
        raise ValueError("当前配置的冻结检测器合同与C0 source run不一致")
    window_manifest_path = args.window_dir / "run_manifest.json"
    window_run = json.loads(window_manifest_path.read_text(encoding="utf-8"))
    if window_run.get("schema_version") != "caes_window_selection_run_v1":
        raise ValueError("WindowManifest run schema不受支持")
    if window_run.get("config_sha256") != _sha256(args.config):
        raise ValueError("WindowManifest使用的配置文件与dense run不一致")
    if not window_run.get("coarse_contract_sha256"):
        raise ValueError("WindowManifest run缺少coarse contract")

    rows_by_dataset = {}
    manifests_by_split = {}
    local_references = {}
    manifest_hashes = {}
    dataset_fingerprints = {}
    manifest_root = ROOT / config["data"]["development_manifests"]
    for dataset in args.datasets:
        if dataset not in window_run.get("datasets", {}):
            raise ValueError(f"WindowManifest run不包含数据集：{dataset}")
        calibration_path = manifest_root / f"{dataset}_calibration.csv"
        evaluation_path = manifest_root / f"{dataset}_evaluation.csv"
        calibration_rows = load_manifest(str(calibration_path))
        evaluation_rows = load_manifest(str(evaluation_path))
        _ensure_disjoint(calibration_rows, evaluation_rows, dataset)
        calibration_rows = calibration_rows[
            calibration_rows["downsample_idxs"].map(
                lambda value: len(parse_indices(value)) >= 16
            )
        ].reset_index(drop=True)
        selected_calibration = _choose_calibration(
            calibration_rows,
            dataset,
            int(config["calibration"]["real_videos_per_dataset"]),
            int(config["calibration"]["seed"]),
        )
        for split in ("calibration", "evaluation"):
            manifests = _load_manifests(args.window_dir, window_run, dataset, split)
            manifests_by_split[(dataset, split)] = manifests
            for selector in SELECTORS:
                path = args.window_dir / "manifests" / dataset / f"{selector}_{split}.jsonl"
                manifest_hashes[f"{dataset}/{split}/{selector}"] = _sha256(path)
        evaluation_ids = set(manifests_by_split[(dataset, "evaluation")]["uniform"])
        evaluation_rows = evaluation_rows[
            evaluation_rows["video_path"].map(
                lambda value: f"{dataset}:{value}" in evaluation_ids
            )
        ].reset_index(drop=True)
        if len(evaluation_rows) != len(evaluation_ids):
            raise ValueError(f"{dataset} evaluation manifest身份无法完整解析")
        rows_by_dataset[dataset] = (selected_calibration, evaluation_rows)
        _verify_uniform_manifests(
            dataset, selected_calibration, manifests_by_split[(dataset, "calibration")]
        )
        _verify_uniform_manifests(
            dataset, evaluation_rows, manifests_by_split[(dataset, "evaluation")]
        )
        reference = load_frozen_local_reference(
            args.detector_reference_dir / f"{dataset}_local_d2.npz"
        )
        expected_calibration_ids = tuple(
            f"{dataset}:{path}" for path in selected_calibration["video_path"].astype(str)
        )
        if (
            reference.dataset != dataset
            or reference.source_run != SOURCE_RUN
            or reference.source_config_hash != source_manifest["config_hash"]
            or reference.calibration_ids != expected_calibration_ids
        ):
            raise ValueError(f"{dataset} frozen Local reference身份与本次run不一致")
        local_references[dataset] = reference
        dataset_fingerprints[dataset] = {
            "calibration_manifest_sha256": _sha256(calibration_path),
            "evaluation_manifest_sha256": _sha256(evaluation_path),
            "calibration_video_ids_sha256": _canonical_digest(expected_calibration_ids),
            "evaluation_video_ids_sha256": _canonical_digest(sorted(evaluation_ids)),
        }

    identity = {
        "schema_version": RUN_IDENTITY_SCHEMA,
        "config_hash": config_digest(config),
        "config_file_sha256": _sha256(args.config),
        "detector_contract": _detector_contract(config),
        "source_run": SOURCE_RUN,
        "source_config_hash": source_manifest["config_hash"],
        "window_run_manifest_sha256": _sha256(window_manifest_path),
        "coarse_contract_sha256": window_run["coarse_contract_sha256"],
        "datasets": list(args.datasets),
        "dataset_fingerprints": dataset_fingerprints,
        "window_manifest_sha256": manifest_hashes,
        "frozen_local_reference_sha256": {
            dataset: local_references[dataset].digest() for dataset in args.datasets
        },
        "selectors": list(SELECTORS),
        "chunk_videos": args.chunk_videos,
        "frame_batch_size": args.frame_batch_size,
    }
    identity_sha = _canonical_digest(identity)
    return (
        config, source_dir, source_manifest, window_run, rows_by_dataset,
        manifests_by_split, local_references, identity, identity_sha,
    )


def _execute(args: argparse.Namespace, prepared) -> None:
    (
        config, source_dir, source_manifest, _window_run, rows_by_dataset,
        manifests_by_split, local_references, identity, identity_sha,
    ) = prepared
    progress = {
        "status": "running",
        "phase": "initializing",
        "run_identity_sha256": identity_sha,
    }
    write_progress(args.output_dir, progress)

    def report(dataset, split, completed, total, dense_frames, resumed):
        progress.update({
            "phase": "dense_scoring",
            "dataset": dataset,
            "split": split,
            "completed_videos": completed,
            "total_videos": total,
            "dense_unique_frames": dense_frames,
            "resumed_shard": resumed,
            "updated_at": datetime.now(timezone.utc).isoformat(),
        })
        write_progress(args.output_dir, progress)
        mode = "恢复" if resumed else "评分"
        print(
            f"[{dataset}/{split}] {mode} {completed}/{total}，"
            f"unique frames={dense_frames}", flush=True,
        )

    global_parameters = _load_global_parameters(ROOT, config)
    model = AlphaStallFeatureExtractor(args.device)
    raw_root = args.output_dir / "raw_shards"
    all_windows, all_videos = [], []
    selected_rows, diagnostics = [], []
    run_metadata = {
        "datasets": {},
        "window_run": str(args.window_dir),
        "window_run_manifest_sha256": identity["window_run_manifest_sha256"],
    }
    source_windows_all = pd.read_csv(
        source_dir / "window_scores.csv", float_precision="round_trip"
    )
    source_videos_all = pd.read_csv(
        source_dir / "video_scores.csv", float_precision="round_trip"
    )
    fs0_windows, fs0_videos = [], []

    for dataset in args.datasets:
        selected_calibration, evaluation_rows = rows_by_dataset[dataset]
        calibration_manifests = manifests_by_split[(dataset, "calibration")]
        evaluation_manifests = manifests_by_split[(dataset, "evaluation")]
        for split, rows, manifests in (
            ("calibration", selected_calibration, calibration_manifests),
            ("evaluation", evaluation_rows, evaluation_manifests),
        ):
            selection, diagnostic = _selected_window_rows(dataset, split, rows, manifests)
            selected_rows.extend(selection)
            diagnostics.extend(diagnostic)

        local_reference = local_references[dataset]
        calibration_raw, calibration_meta = _score_split(
            dataset=dataset, split="calibration", rows=selected_calibration,
            manifests=calibration_manifests, raw_root=raw_root, model=model,
            global_parameters=global_parameters, local_parameters=local_reference.params,
            device=args.device, chunk_videos=args.chunk_videos,
            frame_batch_size=args.frame_batch_size,
            run_identity_sha256=identity_sha, report=report,
        )
        evaluation_raw, evaluation_meta = _score_split(
            dataset=dataset, split="evaluation", rows=evaluation_rows,
            manifests=evaluation_manifests, raw_root=raw_root, model=model,
            global_parameters=global_parameters, local_parameters=local_reference.params,
            device=args.device, chunk_videos=args.chunk_videos,
            frame_batch_size=args.frame_batch_size,
            run_identity_sha256=identity_sha, report=report,
        )
        for selector in ADAPTIVE_SELECTORS:
            calibration = calibration_raw[calibration_raw["selector"].eq(selector)].copy()
            evaluation = evaluation_raw[evaluation_raw["selector"].eq(selector)].copy()
            windows, videos = _calibrate_selector(calibration, evaluation, config)
            if "selector" not in windows:
                windows.insert(0, "selector", selector)
            videos.insert(0, "selector", selector)
            all_windows.append(windows)
            all_videos.append(videos)

        calibration_ids = {
            f"{dataset}:{path}" for path in selected_calibration["video_path"].astype(str)
        }
        evaluation_ids = {
            f"{dataset}:{path}" for path in evaluation_rows["video_path"].astype(str)
        }
        source_windows = source_windows_all[source_windows_all["dataset"].eq(dataset)]
        source_windows = source_windows[
            (source_windows["split"].eq("calibration") & source_windows["video_id"].isin(calibration_ids))
            | (source_windows["split"].eq("evaluation") & source_windows["video_id"].isin(evaluation_ids))
        ].copy()
        source_videos = source_videos_all[
            source_videos_all["dataset"].eq(dataset)
            & source_videos_all["video_id"].isin(evaluation_ids)
        ].copy()
        expected_fs0_windows = sum(
            item.effective_k for item in calibration_manifests["uniform"].values()
        ) + sum(item.effective_k for item in evaluation_manifests["uniform"].values())
        if (
            len(source_windows) != expected_fs0_windows
            or len(source_videos) != len(evaluation_ids)
            or source_videos["video_id"].duplicated().any()
        ):
            raise ValueError(f"{dataset} FS0 source分数与WindowManifest身份不一致")
        source_windows.insert(0, "selector", "uniform")
        source_videos.insert(0, "selector", "uniform")
        fs0_windows.append(source_windows)
        fs0_videos.append(source_videos)
        run_metadata["datasets"][dataset] = {
            "calibration": calibration_meta,
            "evaluation": evaluation_meta,
            "frozen_local_reference_sha256": local_reference.digest(),
            "evaluation_videos": len(evaluation_rows),
        }

    progress.update({"phase": "metrics", "updated_at": datetime.now(timezone.utc).isoformat()})
    write_progress(args.output_dir, progress)
    windows = pd.concat([*fs0_windows, *all_windows], ignore_index=True, sort=False)
    videos = pd.concat([*fs0_videos, *all_videos], ignore_index=True, sort=False)
    dataset_tables, generator_tables, pairwise_tables = [], [], []
    for selector, frame in videos.groupby("selector", sort=False):
        clean = frame.drop(columns="selector")
        dataset_table, generator_table = build_metric_tables(clean, selector)
        pairwise_table = build_pairwise_metric_table(
            clean, selector, int(config["metrics"]["pairwise_seed"])
        )
        for table in (dataset_table, generator_table, pairwise_table):
            table.insert(0, "selector", selector)
        dataset_tables.append(dataset_table)
        generator_tables.append(generator_table)
        pairwise_tables.append(pairwise_table)
    dataset_metrics = pd.concat(dataset_tables, ignore_index=True)
    generator_metrics = pd.concat(generator_tables, ignore_index=True)
    pairwise_metrics = pd.concat(pairwise_tables, ignore_index=True)
    bootstrap = paired_selector_bootstrap(
        videos,
        seed=int(config["metrics"]["pairwise_seed"]),
        iterations=int(config["metrics"]["bootstrap_iterations"]),
    )
    gate = evaluate_selector_gate(pairwise_metrics, bootstrap)
    outputs = {
        "selected_windows.csv": pd.DataFrame(selected_rows),
        "selection_diagnostics.csv": pd.DataFrame(diagnostics),
        "window_scores.csv": windows,
        "video_scores.csv": videos,
        "dataset_metrics.csv": dataset_metrics,
        "generator_metrics.csv": generator_metrics,
        "pairwise_metrics.csv": pairwise_metrics,
        "paired_bootstrap.csv": bootstrap,
        "gate_decision.csv": gate,
    }
    for name, frame in outputs.items():
        write_csv(args.output_dir, name, frame)
    score_hashes = {name: _sha256(args.output_dir / name) for name in outputs}
    _atomic_json(args.output_dir / "run_manifest.json", {
        "schema_version": "caes_dense_run_v2",
        "status": "completed",
        "completed_at": datetime.now(timezone.utc).isoformat(),
        "git_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip(),
        "run_identity_sha256": identity_sha,
        "config_hash": config_digest(config),
        "source_run": SOURCE_RUN,
        "source_config_hash": source_manifest["config_hash"],
        "selectors": list(SELECTORS),
        "score_artifact_sha256": score_hashes,
        **run_metadata,
    })
    progress.update({
        "status": "completed", "phase": "completed",
        "completed_at": datetime.now(timezone.utc).isoformat(),
    })
    write_progress(args.output_dir, progress)
    print(f"[完成] CAES Stage FS dense结果：{args.output_dir}", flush=True)


def main() -> None:
    args = parse_args()
    prepared = _prepare(args)
    identity = prepared[-2]
    identity_sha = prepared[-1]
    if args.output_dir.exists() and args.overwrite:
        shutil.rmtree(args.output_dir)
    if args.output_dir.exists() and not args.resume:
        raise FileExistsError(f"CAES dense结果目录已存在：{args.output_dir}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    identity_path = args.output_dir / "run_identity.json"
    expected_identity = {**identity, "identity_sha256": identity_sha}
    if args.resume:
        if not identity_path.is_file():
            raise FileNotFoundError("--resume要求已有run_identity.json")
        stored = json.loads(identity_path.read_text(encoding="utf-8"))
        if stored != expected_identity:
            raise ValueError("恢复运行的config/manifest/reference身份发生变化")
    else:
        _atomic_json(identity_path, expected_identity)
        dump_config(args.output_dir / "resolved_config.yaml", prepared[0])
        (args.output_dir / "command.txt").write_text(
            "命令：" + " ".join(shlex.quote(item) for item in [sys.executable, *sys.argv])
            + "\n开始时间（UTC）：" + datetime.now(timezone.utc).isoformat() + "\n",
            encoding="utf-8",
        )
    logs = args.output_dir / "logs"
    logs.mkdir(exist_ok=True)
    received_signal = None
    previous_handlers = {}

    def on_signal(signum, _frame):
        nonlocal received_signal
        received_signal = int(signum)
        raise RunTerminated(f"收到外部信号 {signal.Signals(signum).name}")

    for signum in (signal.SIGHUP, signal.SIGINT, signal.SIGTERM):
        previous_handlers[signum] = signal.getsignal(signum)
        signal.signal(signum, on_signal)
    with (logs / "run.log").open("a", encoding="utf-8", buffering=1) as log, \
         contextlib.redirect_stdout(_Tee(sys.stdout, log)), \
         contextlib.redirect_stderr(_Tee(sys.stderr, log)):
        try:
            print(f"[开始] CAES dense identity={identity_sha}", flush=True)
            _execute(args, prepared)
        except BaseException as error:
            failure = {
                "exception_type": type(error).__name__,
                "message": str(error),
                "received_signal": signal.Signals(received_signal).name if received_signal else None,
                "occurred_at_utc": datetime.now(timezone.utc).isoformat(),
                "traceback": traceback.format_exc(),
            }
            _atomic_json(args.output_dir / "failure.json", failure)
            write_progress(args.output_dir, {
                "status": "interrupted", "phase": "interrupted",
                "run_identity_sha256": identity_sha, "failure": failure,
            })
            print(f"[中断] {type(error).__name__}: {error}", flush=True)
            raise
        finally:
            for signum, handler in previous_handlers.items():
                signal.signal(signum, handler)


if __name__ == "__main__":
    main()
