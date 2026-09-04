#!/usr/bin/env python3
"""从既有CAES视频分数重建严格共享配对的指标、bootstrap与gate。"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from config import load_config
from temporal_selection.evaluation import (
    build_matched_selector_pairwise_table,
    evaluate_selector_gate,
    paired_selector_bootstrap,
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _atomic_csv(path: Path, frame: pd.DataFrame) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(temporary, index=False)
    temporary.replace(path)


def _atomic_json(path: Path, payload: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run-dir", type=Path,
        default=ROOT / "results/runs/caes_stage_fs",
    )
    args = parser.parse_args()
    run_dir = args.run_dir.resolve()
    manifest_path = run_dir / "run_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("status") != "completed":
        raise ValueError("只允许修正已完成CAES run的评测产物")
    video_path = run_dir / "video_scores.csv"
    expected_video_sha = manifest["score_artifact_sha256"]["video_scores.csv"]
    if _sha256(video_path) != expected_video_sha:
        raise ValueError("video_scores.csv与原run manifest哈希不一致")
    scores = pd.read_csv(video_path, float_precision="round_trip")
    config = load_config(run_dir / "resolved_config.yaml")
    seed = int(config["metrics"]["pairwise_seed"])
    iterations = int(config["metrics"]["bootstrap_iterations"])
    pairwise = build_matched_selector_pairwise_table(scores, seed=seed)
    bootstrap = paired_selector_bootstrap(
        scores, seed=seed, iterations=iterations
    )
    gate = evaluate_selector_gate(pairwise, bootstrap)

    targets = {
        "pairwise_metrics.csv": pairwise,
        "paired_bootstrap.csv": bootstrap,
        "gate_decision.csv": gate,
    }
    for name, frame in targets.items():
        destination = run_dir / name
        legacy = run_dir / name.replace(".csv", "_legacy_unmatched.csv")
        if destination.is_file() and not legacy.exists():
            shutil.copy2(destination, legacy)
        _atomic_csv(destination, frame)
        manifest["score_artifact_sha256"][name] = _sha256(destination)
    manifest["metric_pairing"] = {
        "mode": "shared_fs0_video_identity",
        "pair_identity_source": "uniform",
        "seed": seed,
        "correction_reason": (
            "旧表按selector分别抽样real，行序差异导致部分VideoFeedback配对身份不一致"
        ),
        "recomputed_at": datetime.now(timezone.utc).isoformat(),
        "implementation_sha256": _sha256(
            ROOT / "src/temporal_selection/evaluation.py"
        ),
    }
    _atomic_json(manifest_path, manifest)
    print(pairwise[pairwise["dataset"].eq("Macro-3")].to_string(index=False))
    print("\nGo/No-Go：")
    print(gate.to_string(index=False))


if __name__ == "__main__":
    main()
