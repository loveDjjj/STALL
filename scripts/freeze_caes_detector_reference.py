#!/usr/bin/env python3
"""重拟合并验证FS0 Local D2参数，冻结为CAES共享检测参考。"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from config import load_config
from data.cache_contract import prepare_feature_cache
from data.manifest import load_manifest
from data.packed_cache import PackedCacheReader
from pipeline import (
    _choose_calibration,
    _fit_local_parameters,
    _load_cache_payload,
    _window_features,
)
from branches.local_branch import local_d2_features
from math_utils import StableGaussianParams, score_gaussian_aggregate_float64, stable_sorted
from temporal_selection.calibration import (
    FrozenLocalD2Reference,
    audit_reconstructed_scores,
    save_frozen_local_reference,
)


DEFAULT_SOURCE_RUN = "alpha_stall_full_d2_k3_no_spatial_refit"
DEVELOPMENT_DATASETS = ("comgenvid", "videofeedback", "genvideo")
EXTERNAL_DATASETS = ("genvidbench",)
DATASETS = (*DEVELOPMENT_DATASETS, *EXTERNAL_DATASETS)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda:1")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "precomputed/caes_detector")
    parser.add_argument("--datasets", nargs="+", choices=DATASETS, default=list(DEVELOPMENT_DATASETS))
    parser.add_argument("--source-run", default=DEFAULT_SOURCE_RUN)
    parser.add_argument(
        "--manifest-scope", choices=("development", "external"), default="development"
    )
    parser.add_argument(
        "--max-raw-abs-difference", type=float, default=5e-5,
        help="旧C0未保存参数时允许的CUDA重拟合raw绝对误差上限",
    )
    parser.add_argument(
        "--max-window-cdf-rank-steps", type=float, default=1.0,
        help="重拟合后允许的最大窗口CDF秩步数漂移",
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    source_dir = ROOT / "results/runs" / args.source_run
    source_manifest = json.loads((source_dir / "run_manifest.json").read_text(encoding="utf-8"))
    config = load_config(source_dir / "resolved_config.yaml")
    cache_root = ROOT / config["runtime"]["cache_dir"]
    context = prepare_feature_cache(
        cache_root, expected_contract=None, policy="strict", create=False,
        required_cache_kind="patch_embeddings",
    )
    reader = PackedCacheReader(cache_root)
    source_windows = pd.read_csv(
        source_dir / "window_scores.csv", float_precision="round_trip"
    )
    summaries = {}
    for dataset in args.datasets:
        output_path = args.output_dir / f"{dataset}_local_d2.npz"
        if output_path.exists() and not args.overwrite:
            raise FileExistsError(f"CAES detector reference已存在：{output_path}")
        if args.manifest_scope == "external":
            if dataset not in EXTERNAL_DATASETS:
                raise ValueError(f"外部manifest不支持数据集：{dataset}")
            calibration_path = ROOT / config["data"]["external_manifests"] / "calibration.csv"
        else:
            if dataset not in DEVELOPMENT_DATASETS:
                raise ValueError(f"开发manifest不支持数据集：{dataset}")
            calibration_path = (
                ROOT / config["data"]["development_manifests"]
                / f"{dataset}_calibration.csv"
            )
        calibration = load_manifest(str(calibration_path))
        selected = _choose_calibration(
            calibration,
            dataset,
            int(config["calibration"]["real_videos_per_dataset"]),
            int(config["calibration"]["seed"]),
        )
        features = []
        identities = []
        for _, row in selected.iterrows():
            payload = _load_cache_payload(
                ROOT, cache_root, row, context, reader
            )
            windows = _window_features(
                payload, row, int(config["sampling"]["num_windows"])
            )
            video_id = f"{dataset}:{row['video_path']}"
            for window_id, values in enumerate(windows):
                features.append(values)
                identities.append((video_id, window_id))
        fitted = _fit_local_parameters(features, config, args.device)["patch_temporal"]
        raw_parts = []
        batch_size = int(config["runtime"].get("score_batch_size", 16))
        for offset in range(0, len(features), batch_size):
            patch = torch.from_numpy(
                np.stack([item[1] for item in features[offset : offset + batch_size]])
            )
            raw, _ = score_gaussian_aggregate_float64(
                local_d2_features(patch), fitted, "mean",
                device=args.device, compute_percentile=False,
            )
            raw_parts.append(raw)
        raw = np.concatenate(raw_parts)
        comparison = pd.DataFrame(identities, columns=["video_id", "window_id"])
        comparison["refit_raw"] = raw
        expected = source_windows[
            source_windows["dataset"].eq(dataset)
            & source_windows["split"].eq("calibration")
        ][["video_id", "window_id", "patch_temporal_raw", "patch_temporal"]]
        comparison = comparison.merge(
            expected, on=["video_id", "window_id"], how="left", validate="one_to_one"
        )
        if comparison["patch_temporal_raw"].isna().any() or len(comparison) != len(expected):
            raise ValueError(f"{dataset} FS0 calibration窗口身份无法与source run对齐")
        audit = audit_reconstructed_scores(
            comparison["refit_raw"].to_numpy(dtype=np.float64),
            comparison["patch_temporal_raw"].to_numpy(dtype=np.float64),
            comparison["patch_temporal"].to_numpy(dtype=np.float64),
        )
        if audit["raw_max_abs_difference"] > args.max_raw_abs_difference:
            raise ValueError(
                f"{dataset} Local D2重拟合raw漂移超过门槛：{audit}"
            )
        if audit["window_cdf_max_rank_steps"] > args.max_window_cdf_rank_steps + 1e-9:
            raise ValueError(
                f"{dataset} Local D2重拟合CDF排序漂移超过门槛：{audit}"
            )
        params = StableGaussianParams(
            mean=fitted.mean,
            whitening=fitted.whitening,
            calibration_raw=stable_sorted(raw),
            shrinkage=fitted.shrinkage,
        )
        calibration_ids = tuple(
            f"{dataset}:{path}" for path in selected["video_path"].astype(str)
        )
        reference = FrozenLocalD2Reference(
            dataset=dataset,
            params=params,
            calibration_ids=calibration_ids,
            source_run=args.source_run,
            source_config_hash=str(source_manifest["config_hash"]),
        )
        file_sha = save_frozen_local_reference(output_path, reference)
        summaries[dataset] = {
            "windows": len(raw),
            "reconstruction_audit": audit,
            "reconstruction_thresholds": {
                "max_raw_abs_difference": args.max_raw_abs_difference,
                "max_window_cdf_rank_steps": args.max_window_cdf_rank_steps,
            },
            "reference_sha256": reference.digest(),
            "file_sha256": file_sha,
        }
        print(
            f"[{dataset}] 冻结完成，窗口={len(raw)}，"
            f"raw max={audit['raw_max_abs_difference']:.3e}，"
            f"CDF max={audit['window_cdf_max_rank_steps']:.1f} rank step",
            flush=True,
        )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "manifest.json").write_text(
        json.dumps({
            "schema_version": "caes_detector_reference_manifest_v1",
            "source_run": args.source_run,
            "source_config_hash": source_manifest["config_hash"],
            "source_cache_contract_sha256": context.contract_sha256,
            "datasets": summaries,
        }, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
