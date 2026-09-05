#!/usr/bin/env python3
"""仅用确认集 calibration real 拟合并冻结 Local D2 参考。"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from branches.local_branch import local_d2_features
from config import config_digest, load_config
from data.cache_contract import prepare_feature_cache
from data.manifest import load_manifest
from data.packed_cache import PackedCacheReader
from math_utils import StableGaussianParams, score_gaussian_aggregate_float64, stable_sorted
from pipeline import _fit_local_parameters, _load_cache_payload, _window_features
from temporal_selection.calibration import (
    FrozenLocalD2Reference,
    save_frozen_local_reference,
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default="vifbench", choices=("vifbench",))
    parser.add_argument("--config", type=Path, default=ROOT / "configs/benchmark.yaml")
    parser.add_argument(
        "--manifest",
        type=Path,
        default=ROOT / "data/manifests/confirmation/vifbench_calibration.csv",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "precomputed/caes_detector_confirmation",
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=ROOT / "cache/patch_embeddings_confirmation_vifbench",
    )
    parser.add_argument("--device", default="cuda:1")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    output = args.output_dir / f"{args.dataset}_local_d2.npz"
    if output.exists() and not args.overwrite:
        raise FileExistsError(f"确认集Local参考已存在：{output}")

    config = load_config(args.config)
    config["calibration"]["real_videos_per_dataset"] = 80
    config["method"]["fusion"]["global_weight"] = 0.5
    config["method"]["fusion"]["local_weight"] = 0.5
    rows = load_manifest(str(args.manifest))
    if len(rows) != 80 or not rows["subset"].eq("real").all():
        raise ValueError("ViF-Bench冻结协议要求恰好80条calibration real")

    cache_root = args.cache_dir
    context = prepare_feature_cache(
        cache_root,
        expected_contract=None,
        policy="strict",
        create=False,
        required_cache_kind="patch_embeddings",
    )
    reader = PackedCacheReader(cache_root)
    windows = []
    identities = []
    for _, row in rows.iterrows():
        payload = _load_cache_payload(ROOT, cache_root, row, context, reader)
        current = _window_features(payload, row, requested_k=3)
        video_id = f"{args.dataset}:{row['video_path']}"
        windows.extend(current)
        identities.extend((video_id, index) for index in range(len(current)))
    fitted = _fit_local_parameters(windows, config, args.device)["patch_temporal"]

    raw_parts = []
    for offset in range(0, len(windows), 48):
        patch = torch.from_numpy(
            np.stack([item[1] for item in windows[offset : offset + 48]])
        )
        raw, _ = score_gaussian_aggregate_float64(
            local_d2_features(patch),
            fitted,
            "mean",
            device=args.device,
            compute_percentile=False,
        )
        raw_parts.append(raw)
    raw = np.concatenate(raw_parts)
    params = StableGaussianParams(
        mean=fitted.mean,
        whitening=fitted.whitening,
        calibration_raw=stable_sorted(raw),
        shrinkage=fitted.shrinkage,
    )
    calibration_ids = tuple(
        f"{args.dataset}:{path}" for path in rows["video_path"].astype(str)
    )
    source_identity = "vifbench_confirmation_protocol_v1"
    reference = FrozenLocalD2Reference(
        dataset=args.dataset,
        params=params,
        calibration_ids=calibration_ids,
        source_run=source_identity,
        source_config_hash=config_digest(config),
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    file_sha = save_frozen_local_reference(output, reference)
    manifest = {
        "schema_version": "confirmation_local_reference_v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "dataset": args.dataset,
        "source_identity": source_identity,
        "config_hash": config_digest(config),
        "calibration_manifest_sha256": _sha256(args.manifest),
        "cache_contract_sha256": context.contract_sha256,
        "calibration_videos": len(rows),
        "calibration_windows": len(windows),
        "reference_sha256": reference.digest(),
        "file_sha256": file_sha,
    }
    (args.output_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
