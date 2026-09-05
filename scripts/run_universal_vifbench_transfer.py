#!/usr/bin/env python3
"""用不含 ViF-Bench 的 Universal-4 real bank 冻结评测 ViF-Bench。"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT / "src", ROOT / "scripts"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from config import config_digest, load_config
from cross_domain import evaluate_cross_domain_cell
from features import AlphaStallFeatureExtractor
from math_utils import StableGaussianParams
from pipeline import _load_global_parameters
from run_cross_domain_calibration import (
    ALL_DOMAINS,
    DOMAINS,
    DomainSpec,
    _load_rows_and_manifests,
    _score_split,
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _read_raw(directory: Path, domain: str, split: str) -> pd.DataFrame:
    records = []
    for path in sorted((directory / "raw_shards" / domain / split).glob("*.json")):
        records.extend(json.loads(path.read_text(encoding="utf-8"))["records"])
    if not records:
        raise FileNotFoundError(f"跨域raw结果为空：{domain}/{split}")
    return pd.DataFrame(records)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--cross-domain-dir",
        type=Path,
        default=ROOT / "results/runs/cross_domain_feature_k3_equal_fusion",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "results/runs/universal4_to_vifbench_feature_k3",
    )
    parser.add_argument("--device", default="cuda:1")
    parser.add_argument("--chunk-videos", type=int, default=64)
    parser.add_argument("--decode-workers", type=int, default=8)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    cross_manifest = json.loads(
        (args.cross_domain_dir / "run_manifest.json").read_text(encoding="utf-8")
    )
    if cross_manifest.get("status") != "completed":
        raise ValueError("四域跨校准矩阵尚未完成")
    if args.output_dir.exists() and args.overwrite:
        shutil.rmtree(args.output_dir)
    if args.output_dir.exists() and not args.resume:
        raise FileExistsError(f"Universal-4→ViF输出已存在：{args.output_dir}")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    universal_path = (
        args.cross_domain_dir / "universal_references/universal4_local_d2.npz"
    )
    with np.load(universal_path, allow_pickle=False) as data:
        universal = StableGaussianParams(
            mean=np.asarray(data["mean"], dtype=np.float64),
            whitening=np.asarray(data["whitening"], dtype=np.float64),
            calibration_raw=np.array([0.0, 1.0], dtype=np.float64),
        )
        source_domains = tuple(str(item) for item in data["domains"])
    if source_domains != ALL_DOMAINS:
        raise ValueError(f"Universal-4域身份异常：{source_domains}")

    config = load_config(ROOT / "configs/benchmark.yaml")
    config["method"]["fusion"]["global_weight"] = 0.5
    config["method"]["fusion"]["local_weight"] = 0.5
    vif = DomainSpec(
        "vifbench",
        ROOT / "data/manifests/confirmation/vifbench_calibration.csv",
        ROOT / "data/manifests/confirmation/vifbench_evaluation.csv",
        ROOT / "results/caes/confirmation_vifbench_seed17",
        ROOT / "precomputed/caes_detector_confirmation/vifbench_local_d2.npz",
        ROOT / "cache/patch_embeddings_confirmation_vifbench",
    )
    rows, manifests = _load_rows_and_manifests(vif, "evaluation")
    identity = {
        "schema_version": "universal4_to_vifbench_feature_k3_v1",
        "cross_domain_manifest_sha256": _sha256(
            args.cross_domain_dir / "run_manifest.json"
        ),
        "universal4_sha256": _sha256(universal_path),
        "vifbench_evaluation_manifest_sha256": _sha256(vif.evaluation),
        "vifbench_window_manifest_sha256": _sha256(
            vif.window_dir / "manifests/vifbench/feature_change_evaluation.jsonl"
        ),
        "config_hash": config_digest(config),
    }
    identity_sha = hashlib.sha256(
        json.dumps(identity, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    (args.output_dir / "run_identity.json").write_text(
        json.dumps({**identity, "identity_sha256": identity_sha}, indent=2) + "\n",
        encoding="utf-8",
    )

    model = AlphaStallFeatureExtractor(args.device)
    evaluation = _score_split(
        vif,
        "evaluation",
        rows,
        manifests,
        references={"universal4": universal},
        global_parameters=_load_global_parameters(ROOT, config),
        model=model,
        device=args.device,
        output_dir=args.output_dir,
        identity_sha=identity_sha,
        chunk_videos=args.chunk_videos,
        decode_workers=args.decode_workers,
    )
    evaluation["patch_temporal_raw"] = evaluation[
        "patch_temporal_raw__universal4"
    ]
    calibration = pd.concat([
        _read_raw(args.cross_domain_dir, domain, "calibration")
        for domain in source_domains
    ], ignore_index=True)
    calibration["patch_temporal_raw"] = calibration[
        "patch_temporal_raw__universal4"
    ]
    summary, points, generators = evaluate_cross_domain_cell(
        calibration,
        evaluation,
        config,
        calibration_bank="universal4",
        evaluation_domain="vifbench",
    )
    outputs = {
        "matrix_metrics.csv": summary,
        "operating_points.csv": points,
        "generator_metrics.csv": generators,
    }
    for name, frame in outputs.items():
        frame.to_csv(args.output_dir / name, index=False)
    manifest = {
        "status": "completed",
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "identity_sha256": identity_sha,
        "artifacts": {
            name: _sha256(args.output_dir / name) for name in outputs
        },
    }
    (args.output_dir / "run_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(summary.to_string(index=False), flush=True)
    print(points.to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
