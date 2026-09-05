#!/usr/bin/env python3
"""从已完成的跨域 raw shard 重建矩阵、操作点和实现哈希。"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT / "src", ROOT / "scripts"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from config import config_digest, load_config
from cross_domain import evaluate_cross_domain_cell
from run_cross_domain_calibration import (
    ALL_DOMAINS,
    UNIVERSAL_BANKS,
    _audit_diagonal,
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _read_split(directory: Path, domain: str, split: str) -> pd.DataFrame:
    records = []
    identity_sha = None
    for path in sorted((directory / "raw_shards" / domain / split).glob("*.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        current = payload.get("identity_sha256")
        if identity_sha is None:
            identity_sha = current
        elif current != identity_sha:
            raise ValueError(f"{domain}/{split} raw shard包含多个运行身份")
        records.extend(payload.get("records", []))
    if not records:
        raise FileNotFoundError(f"{domain}/{split}没有raw shard")
    return pd.DataFrame(records)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run-dir",
        type=Path,
        default=ROOT / "results/runs/cross_domain_feature_k3_equal_fusion",
    )
    args = parser.parse_args()
    identity_path = args.run_dir / "run_identity.json"
    identity = json.loads(identity_path.read_text(encoding="utf-8"))
    expected_identity = identity["identity_sha256"]
    raw = {}
    for domain in ALL_DOMAINS:
        for split in ("calibration", "evaluation"):
            frame = _read_split(args.run_dir, domain, split)
            shard_identity = json.loads(
                sorted((args.run_dir / "raw_shards" / domain / split).glob("*.json"))[0]
                .read_text(encoding="utf-8")
            )["identity_sha256"]
            if shard_identity != expected_identity:
                raise ValueError(f"{domain}/{split} raw身份与run_identity不一致")
            raw[(domain, split)] = frame

    config = load_config(ROOT / "configs/benchmark.yaml")
    config["method"]["fusion"]["global_weight"] = 0.5
    config["method"]["fusion"]["local_weight"] = 0.5
    bank_names = [*ALL_DOMAINS, *UNIVERSAL_BANKS]
    summaries, operating, generators = [], [], []
    for bank in bank_names:
        source_domains = (
            UNIVERSAL_BANKS[bank] if bank in UNIVERSAL_BANKS else (bank,)
        )
        calibration = pd.concat(
            [raw[(domain, "calibration")] for domain in source_domains],
            ignore_index=True,
        )
        calibration["patch_temporal_raw"] = calibration[
            f"patch_temporal_raw__{bank}"
        ]
        for target in ALL_DOMAINS:
            evaluation = raw[(target, "evaluation")].copy()
            evaluation["patch_temporal_raw"] = evaluation[
                f"patch_temporal_raw__{bank}"
            ]
            summary, points, per_generator = evaluate_cross_domain_cell(
                calibration,
                evaluation,
                config,
                calibration_bank=bank,
                evaluation_domain=target,
            )
            sources = "+".join(source_domains)
            for frame in (summary, points, per_generator):
                frame["calibration_source_domains"] = sources
            summaries.append(summary)
            operating.append(points)
            generators.append(per_generator)

    outputs = {
        "matrix_metrics.csv": pd.concat(summaries, ignore_index=True),
        "operating_points.csv": pd.concat(operating, ignore_index=True),
        "generator_metrics.csv": pd.concat(generators, ignore_index=True),
        "diagonal_raw_audit.csv": _audit_diagonal(raw),
    }
    for name, frame in outputs.items():
        frame.to_csv(args.run_dir / name, index=False)
    manifest = {
        "status": "completed",
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "raw_identity_sha256": expected_identity,
        "config_hash": config_digest(config),
        "implementation_sha256": {
            "finalize_cross_domain_calibration.py": _sha256(Path(__file__)),
            "cross_domain.py": _sha256(ROOT / "src/cross_domain.py"),
        },
        "artifacts": {
            name: _sha256(args.run_dir / name) for name in outputs
        },
    }
    (args.run_dir / "finalized_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(f"[完成] 跨域矩阵已从raw shard重建：{args.run_dir}", flush=True)


if __name__ == "__main__":
    main()
