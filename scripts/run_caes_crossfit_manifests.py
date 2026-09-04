#!/usr/bin/env python3
"""为Uniform/Feature-change/Real-anomaly生成5-fold OOF calibration manifests。"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from config import config_digest, dump_config, load_config, validate_config
from data.cache_contract import prepare_feature_cache
from data.coarse_global_cache import (
    coarse_cache_path,
    expected_coarse_indices,
    load_coarse_entry,
)
from data.manifest import load_manifest
from data.sampling import parse_indices
from pipeline import _choose_calibration
from temporal_selection.candidates import generate_candidate_windows
from temporal_selection.crossfit import (
    balanced_crossfit_assignments,
    validate_crossfit_partition,
)
from temporal_selection.manifest import (
    read_window_manifests,
    write_window_manifests,
)
from temporal_selection.reference import (
    attach_candidate_scores,
    fit_selector_reference,
    save_selector_reference,
    score_coarse_sequence,
)
from temporal_selection.selectors import select_windows


DATASETS = ("comgenvid", "videofeedback", "genvideo")
SELECTORS = ("uniform", "feature_change", "real_anomaly")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _digest_values(values: list[str]) -> str:
    return hashlib.sha256("\n".join(values).encode("utf-8")).hexdigest()


def _source_path(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def _load_payload(context, cache_root: Path, dataset: str, row: pd.Series):
    video_id = f"{dataset}:{row['video_path']}"
    positions, indices = expected_coarse_indices(row)
    path = coarse_cache_path(
        cache_root, dataset=dataset, split="calibration", video_id=video_id
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
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/benchmark.yaml")
    parser.add_argument(
        "--standard-window-dir", type=Path,
        default=ROOT / "results/caes/window_selection_seed17",
    )
    parser.add_argument(
        "--output-dir", type=Path,
        default=ROOT / "results/caes/crossfit5_seed17",
    )
    parser.add_argument("--cache-dir", type=Path)
    parser.add_argument("--device", default="cuda:1")
    parser.add_argument("--datasets", nargs="+", choices=DATASETS, default=list(DATASETS))
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    args.config = args.config.resolve()
    args.standard_window_dir = args.standard_window_dir.resolve()
    args.output_dir = args.output_dir.resolve()
    if args.output_dir.exists() and args.overwrite:
        shutil.rmtree(args.output_dir)
    if args.output_dir.exists():
        raise FileExistsError(f"crossfit输出目录已存在：{args.output_dir}")
    args.output_dir.mkdir(parents=True)

    config = load_config(args.config)
    validate_config(config)
    selection = config["temporal_selection"]
    folds = int(selection["crossfit_folds"])
    seed = int(selection["seed"])
    standard_run_path = args.standard_window_dir / "run_manifest.json"
    standard_run = json.loads(standard_run_path.read_text(encoding="utf-8"))
    if standard_run.get("status") != "completed":
        raise ValueError("standard WindowManifest run尚未完成")
    if standard_run.get("config_sha256") != _sha256(args.config):
        raise ValueError("crossfit配置与standard WindowManifest配置不一致")
    cache_root = args.cache_dir or ROOT / config["runtime"]["coarse_global_cache_dir"]
    context = prepare_feature_cache(
        cache_root,
        expected_contract=None,
        policy="strict",
        create=False,
        required_cache_kind="global_embeddings",
    )
    if context.contract_sha256 != standard_run["coarse_contract_sha256"]:
        raise ValueError("crossfit coarse cache与standard run合同不一致")
    dump_config(args.output_dir / "resolved_config.yaml", config)
    summary = {
        "schema_version": "caes_selector_crossfit_v1",
        "status": "running",
        "started_at": datetime.now(timezone.utc).isoformat(),
        "git_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip(),
        "config_hash": config_digest(config),
        "standard_window_run_sha256": _sha256(standard_run_path),
        "coarse_contract_sha256": context.contract_sha256,
        "folds": folds,
        "seed": seed,
        "selectors": list(SELECTORS),
        "datasets": {},
    }
    (args.output_dir / "run_manifest.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    manifest_root = ROOT / config["data"]["development_manifests"]
    for dataset in args.datasets:
        calibration = load_manifest(
            str(manifest_root / f"{dataset}_calibration.csv")
        )
        selected = _choose_calibration(
            calibration,
            dataset,
            int(config["calibration"]["real_videos_per_dataset"]),
            int(config["calibration"]["seed"]),
        )
        payloads = dict(
            _load_payload(context, cache_root, dataset, row)
            for _, row in selected.iterrows()
        )
        video_ids = [f"{dataset}:{path}" for path in selected["video_path"].astype(str)]
        assignments = balanced_crossfit_assignments(
            video_ids, folds=folds, seed=seed
        )
        validate_crossfit_partition(assignments, folds=folds)
        assignment_frame = pd.DataFrame({
            "video_id": video_ids,
            "fold": [assignments[video_id] for video_id in video_ids],
        })
        assignment_path = args.output_dir / "folds" / f"{dataset}.csv"
        assignment_path.parent.mkdir(parents=True, exist_ok=True)
        assignment_frame.to_csv(assignment_path, index=False)

        standard_values = {}
        standard_hashes = {}
        for selector in ("uniform", "feature_change"):
            path = (
                args.standard_window_dir / "manifests" / dataset
                / f"{selector}_calibration.jsonl"
            )
            values = read_window_manifests(path)
            mapping = {item.video_id: item for item in values}
            if set(mapping) != set(video_ids):
                raise ValueError(f"{dataset}/{selector} standard calibration身份漂移")
            standard_values[selector] = [mapping[video_id] for video_id in video_ids]
            standard_hashes[selector] = _sha256(path)

        fold_summary = []
        oof_real_anomaly = {}
        for fold in range(folds):
            heldout = [item for item in video_ids if assignments[item] == fold]
            training = [item for item in video_ids if assignments[item] != fold]
            if set(heldout) & set(training):
                raise ValueError(f"{dataset}/fold={fold}发生selector reference泄漏")
            reference = fit_selector_reference(
                [payloads[item]["global"].float().numpy() for item in training],
                training,
                coarse_contract_sha256=str(context.contract_sha256),
                device=args.device,
            )
            reference_path = (
                args.output_dir / "references" / dataset / f"fold_{fold}.npz"
            )
            reference_file_sha = save_selector_reference(reference_path, reference)
            for video_id in heldout:
                row = selected.iloc[video_ids.index(video_id)]
                payload = payloads[video_id]
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
                oof_real_anomaly[video_id] = select_windows(
                    video_id=video_id,
                    duration_seconds=float(row["duration_seconds"]),
                    downsample_indices=downsample,
                    selector_name="real_anomaly",
                    requested_k=int(selection["k"]),
                    base_fps=float(selection["dense_fps"]),
                    window_seconds=float(selection["window_seconds"]),
                    stride_seconds=float(selection["candidate_stride_seconds"]),
                    seed=seed,
                    nms_iou_threshold=float(selection["nms_iou_threshold"]),
                    scored_candidates=scored,
                    selector_reference_sha256=reference.digest(),
                )
            fold_summary.append({
                "fold": fold,
                "training_videos": len(training),
                "heldout_videos": len(heldout),
                "training_ids_sha256": _digest_values(training),
                "heldout_ids_sha256": _digest_values(heldout),
                "reference_sha256": reference.digest(),
                "reference_file_sha256": reference_file_sha,
            })
            print(
                f"[{dataset}] crossfit fold={fold}：train={len(training)}，"
                f"heldout={len(heldout)}",
                flush=True,
            )

        manifest_summary = {}
        for selector in ("uniform", "feature_change"):
            path = args.output_dir / "manifests" / dataset / f"{selector}_calibration.jsonl"
            digest = write_window_manifests(path, standard_values[selector])
            if digest != standard_hashes[selector]:
                raise ValueError(f"{dataset}/{selector} OOF控制未逐位复现standard")
            manifest_summary[selector] = {
                "sha256": digest,
                "equals_standard": True,
                "videos": len(video_ids),
            }
        ra_path = (
            args.output_dir / "manifests" / dataset
            / "real_anomaly_calibration.jsonl"
        )
        ra_values = [oof_real_anomaly[video_id] for video_id in video_ids]
        manifest_summary["real_anomaly"] = {
            "sha256": write_window_manifests(ra_path, ra_values),
            "equals_standard": False,
            "videos": len(ra_values),
            "selected_windows": sum(item.effective_k for item in ra_values),
        }
        summary["datasets"][dataset] = {
            "calibration_videos": len(video_ids),
            "fold_assignment_sha256": _sha256(assignment_path),
            "folds": fold_summary,
            "manifests": manifest_summary,
        }
        (args.output_dir / "run_manifest.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )

    summary["status"] = "completed"
    summary["completed_at"] = datetime.now(timezone.utc).isoformat()
    (args.output_dir / "run_manifest.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(f"[完成] 三selector crossfit manifests：{args.output_dir}")


if __name__ == "__main__":
    main()
