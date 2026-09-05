#!/usr/bin/env python3
"""从1 FPS严格缓存一次生成development FS0-FS5 WindowManifest。"""

from __future__ import annotations

import argparse
import hashlib
import json
import shlex
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from config import config_digest, dump_config, load_config, validate_config
from data.cache_contract import prepare_feature_cache
from data.coarse_global_cache import (
    coarse_cache_path,
    expected_coarse_indices,
    load_coarse_entry,
)
from data.manifest import load_manifest
from data.sampling import parse_indices
from pipeline import _choose_calibration, _ensure_disjoint
from temporal_selection.candidates import generate_candidate_windows
from temporal_selection.manifest import write_window_manifests
from temporal_selection.reference import (
    attach_candidate_scores,
    fit_selector_reference,
    save_selector_reference,
    score_coarse_sequence,
)
from temporal_selection.selectors import select_windows


DEVELOPMENT_DATASETS = ("comgenvid", "videofeedback", "genvideo")
EXTERNAL_DATASETS = ("genvidbench",)
CONFIRMATION_DATASETS = ("vifbench",)
DATASETS = (*DEVELOPMENT_DATASETS, *EXTERNAL_DATASETS, *CONFIRMATION_DATASETS)
SELECTORS = (
    "uniform",
    "random",
    "feature_change",
    "real_anomaly",
    "real_anomaly_nms",
    "stratified_real_anomaly",
)
SELECTION_CONFIG = ROOT / "configs/benchmark.yaml"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/benchmark.yaml")
    parser.add_argument("--cache-dir", type=Path)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "results/caes/window_selection_seed17")
    parser.add_argument("--device", default="cuda:1")
    parser.add_argument(
        "--datasets", nargs="+", choices=DATASETS,
        default=list(DEVELOPMENT_DATASETS),
    )
    parser.add_argument(
        "--manifest-scope",
        choices=("development", "external", "confirmation"),
        default="development",
        help="选择开发集或外部集manifest；外部集当前仅支持genvidbench",
    )
    parser.add_argument(
        "--selectors", nargs="+", choices=SELECTORS, default=list(SELECTORS),
        help="只生成指定selector；外部确认可仅运行uniform与feature_change",
    )
    parser.add_argument("--limit-evaluation", type=int)
    parser.add_argument(
        "--calibration-real-count", type=int,
        help="覆盖real-only校准数量；ViF-Bench冻结协议固定为80",
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def _source_path(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def _with_temporal_selection(config: dict) -> tuple[dict, str]:
    """为早于CAES的外部run显式继承当前冻结的选择合同。"""

    if "temporal_selection" in config:
        return config, "input_config"
    defaults = load_config(SELECTION_CONFIG)
    merged = dict(config)
    merged["temporal_selection"] = defaults["temporal_selection"]
    runtime = dict(merged["runtime"])
    runtime["coarse_global_cache_dir"] = defaults["runtime"][
        "coarse_global_cache_dir"
    ]
    merged["runtime"] = runtime
    return merged, str(SELECTION_CONFIG.relative_to(ROOT))


def _manifest_paths(config: dict, dataset: str, scope: str) -> tuple[Path, Path]:
    """解析开发集或单一外部集manifest路径。"""

    if scope == "external":
        if dataset not in EXTERNAL_DATASETS:
            raise ValueError(f"外部manifest不支持数据集：{dataset}")
        root = ROOT / config["data"]["external_manifests"]
        return root / "calibration.csv", root / "evaluation.csv"
    if scope == "confirmation":
        if dataset not in CONFIRMATION_DATASETS:
            raise ValueError(f"确认集manifest不支持数据集：{dataset}")
        root = ROOT / config["data"]["confirmation_manifests"]
        return root / f"{dataset}_calibration.csv", root / f"{dataset}_evaluation.csv"
    if dataset not in DEVELOPMENT_DATASETS:
        raise ValueError(f"开发manifest不支持数据集：{dataset}")
    root = ROOT / config["data"]["development_manifests"]
    return root / f"{dataset}_calibration.csv", root / f"{dataset}_evaluation.csv"


def _load_payload(context, cache_root: Path, dataset: str, split: str, row: pd.Series):
    video_id = f"{dataset}:{row['video_path']}"
    positions, indices = expected_coarse_indices(row)
    path = coarse_cache_path(
        cache_root, dataset=dataset, split=split, video_id=video_id
    )
    payload = load_coarse_entry(
        context=context,
        path=path,
        source_video_path=_source_path(str(row["video_path"])),
        frame_indices=indices,
    )
    if payload["downsample_positions"] != positions:
        raise ValueError(f"coarse positions与manifest不一致：{video_id}")
    return video_id, payload


def main() -> None:
    args = parse_args()
    args.config = args.config.resolve()
    args.output_dir = args.output_dir.resolve()
    config, selection_config_source = _with_temporal_selection(load_config(args.config))
    if args.calibration_real_count is not None:
        if args.calibration_real_count < 2:
            raise ValueError("--calibration-real-count至少为2")
        config["calibration"]["real_videos_per_dataset"] = args.calibration_real_count
    validate_config(config)
    selection = config["temporal_selection"]
    cache_root = args.cache_dir or ROOT / config["runtime"]["coarse_global_cache_dir"]
    context = prepare_feature_cache(
        cache_root,
        expected_contract=None,
        policy="strict",
        create=False,
        required_cache_kind="global_embeddings",
    )
    stored_selection = context.contract["identity"]["extraction"]["frame_selection"]
    if stored_selection["base_fps"] != selection["dense_fps"] or stored_selection["coarse_fps"] != selection["coarse_fps"]:
        raise ValueError("coarse cache FPS与CAES配置不一致")
    if args.output_dir.exists() and args.overwrite:
        shutil.rmtree(args.output_dir)
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError(f"WindowManifest输出目录已存在：{args.output_dir}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    dump_config(args.output_dir / "resolved_config.yaml", config)
    (args.output_dir / "command.txt").write_text(
        "命令：" + " ".join(
            shlex.quote(item) for item in [sys.executable, *sys.argv]
        )
        + "\n开始时间（UTC）：" + datetime.now(timezone.utc).isoformat() + "\n",
        encoding="utf-8",
    )

    summary = {
        "schema_version": "caes_window_selection_run_v1",
        "status": "running",
        "started_at": datetime.now(timezone.utc).isoformat(),
        "git_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip(),
        "config_hash": config_digest(config),
        "config_sha256": hashlib.sha256(args.config.read_bytes()).hexdigest(),
        "coarse_contract_sha256": context.contract_sha256,
        "temporal_selection_source": selection_config_source,
        "temporal_selection_source_sha256": (
            hashlib.sha256(SELECTION_CONFIG.read_bytes()).hexdigest()
            if selection_config_source != "input_config"
            else hashlib.sha256(args.config.read_bytes()).hexdigest()
        ),
        "manifest_scope": args.manifest_scope,
        "selectors": {},
        "datasets": {},
    }
    (args.output_dir / "run_manifest.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    for dataset in args.datasets:
        calibration_path, evaluation_path = _manifest_paths(
            config, dataset, args.manifest_scope
        )
        calibration = load_manifest(str(calibration_path))
        evaluation = load_manifest(str(evaluation_path))
        _ensure_disjoint(calibration, evaluation, dataset)
        calibration = calibration[
            calibration["downsample_idxs"].map(lambda value: len(parse_indices(value)) >= 16)
        ].reset_index(drop=True)
        evaluation = evaluation[
            evaluation["downsample_idxs"].map(lambda value: len(parse_indices(value)) >= 16)
        ].reset_index(drop=True)
        selected_calibration = _choose_calibration(
            calibration,
            dataset,
            int(config["calibration"]["real_videos_per_dataset"]),
            int(config["calibration"]["seed"]),
        )
        if args.limit_evaluation is not None:
            evaluation = (
                evaluation.groupby(["subset", "source_model"], sort=True, group_keys=False)
                .head(args.limit_evaluation)
                .reset_index(drop=True)
            )

        calibration_payloads = [
            _load_payload(context, cache_root, dataset, "calibration", row)
            for _, row in selected_calibration.iterrows()
        ]
        reference = fit_selector_reference(
            [item[1]["global"].float().numpy() for item in calibration_payloads],
            [item[0] for item in calibration_payloads],
            coarse_contract_sha256=str(context.contract_sha256),
            device=args.device,
        )
        reference_path = args.output_dir / "references" / f"{dataset}.npz"
        reference_file_sha = save_selector_reference(reference_path, reference)
        reference_sha = reference.digest()

        dataset_summary = {
            "calibration_videos": len(selected_calibration),
            "evaluation_videos": len(evaluation),
            "reference_sha256": reference_sha,
            "reference_file_sha256": reference_file_sha,
            "splits": {},
        }
        for split, frame in (("calibration", selected_calibration), ("evaluation", evaluation)):
            manifests = {name: [] for name in args.selectors}
            signal_rows = []
            for row_index, row in frame.iterrows():
                video_id, payload = _load_payload(
                    context, cache_root, dataset, split, row
                )
                downsample = parse_indices(row["downsample_idxs"])
                candidates = generate_candidate_windows(
                    downsample,
                    base_fps=float(selection["dense_fps"]),
                    window_seconds=float(selection["window_seconds"]),
                    stride_seconds=float(selection["candidate_stride_seconds"]),
                )
                coarse_scores = score_coarse_sequence(
                    payload["global"].float().numpy(),
                    payload["downsample_positions"],
                    reference,
                )
                scored = attach_candidate_scores(
                    candidates,
                    coarse_scores,
                    window_frames=int(selection["frames_per_window"]),
                )
                for transition_id in range(len(coarse_scores["anomaly"])):
                    signal_rows.append({
                        "video_id": video_id,
                        "transition_id": transition_id,
                        "midpoint_position": float(coarse_scores["midpoint_position"][transition_id]),
                        "midpoint_seconds": float(coarse_scores["midpoint_position"][transition_id]) / float(selection["dense_fps"]),
                        "feature_change": float(coarse_scores["feature_change"][transition_id]),
                        "likelihood": float(coarse_scores["likelihood"][transition_id]),
                        "percentile": float(coarse_scores["percentile"][transition_id]),
                        "anomaly": float(coarse_scores["anomaly"][transition_id]),
                    })
                for selector_name in args.selectors:
                    manifests[selector_name].append(select_windows(
                        video_id=video_id,
                        duration_seconds=float(row["duration_seconds"]),
                        downsample_indices=downsample,
                        selector_name=selector_name,
                        requested_k=int(selection["k"]),
                        base_fps=float(selection["dense_fps"]),
                        window_seconds=float(selection["window_seconds"]),
                        stride_seconds=float(selection["candidate_stride_seconds"]),
                        seed=int(selection["seed"]),
                        nms_iou_threshold=float(selection["nms_iou_threshold"]),
                        scored_candidates=scored if selector_name != "uniform" else None,
                        selector_reference_sha256=(
                            reference_sha
                            if selector_name in {"real_anomaly", "real_anomaly_nms", "stratified_real_anomaly"}
                            else None
                        ),
                    ))
                if (row_index + 1) % max(1, len(frame) // 20) == 0 or row_index + 1 == len(frame):
                    print(f"[{dataset}/{split}] selector {row_index + 1}/{len(frame)}", flush=True)
            signal_path = args.output_dir / "signals" / dataset / f"{split}.csv.gz"
            signal_path.parent.mkdir(parents=True, exist_ok=True)
            pd.DataFrame(signal_rows).to_csv(signal_path, index=False, compression="gzip")
            split_summary = {
                "coarse_signals_sha256": hashlib.sha256(signal_path.read_bytes()).hexdigest(),
                "transitions": len(signal_rows),
                "selectors": {},
            }
            for selector_name, values in manifests.items():
                path = args.output_dir / "manifests" / dataset / f"{selector_name}_{split}.jsonl"
                digest = write_window_manifests(path, values)
                split_summary["selectors"][selector_name] = {
                    "path": path.relative_to(args.output_dir).as_posix(),
                    "sha256": digest,
                    "videos": len(values),
                    "selected_windows": sum(item.effective_k for item in values),
                }
            dataset_summary["splits"][split] = split_summary
        summary["datasets"][dataset] = dataset_summary
    for selector_name in args.selectors:
        summary["selectors"][selector_name] = {
            "seed": int(selection["seed"]),
            "k": int(selection["k"]),
            "window_seconds": float(selection["window_seconds"]),
            "candidate_stride_seconds": float(selection["candidate_stride_seconds"]),
            "nms_iou_threshold": float(selection["nms_iou_threshold"]),
            "calibration_mode": str(selection["calibration_mode"]),
        }
    summary["status"] = "completed"
    summary["completed_at"] = datetime.now(timezone.utc).isoformat()
    (args.output_dir / "run_manifest.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(f"[完成] WindowManifest已写入 {args.output_dir}", flush=True)


if __name__ == "__main__":
    main()
