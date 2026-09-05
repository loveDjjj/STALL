#!/usr/bin/env python3
"""汇总 GenVidBench 的 Uniform、Random 与 Feature-change 外部配对结果。"""

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
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from temporal_selection.evaluation import (  # noqa: E402
    build_matched_selector_pairwise_table,
    paired_selector_bootstrap,
)


RUNS = {
    "feature_change": "caes_external_genvidbench_feature_change",
    "random": "caes_external_genvidbench_random",
}


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load(selector: str, run_name: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    directory = ROOT / "results" / "runs" / run_name
    progress = json.loads((directory / "progress.json").read_text(encoding="utf-8"))
    if progress.get("status") != "completed":
        raise ValueError(f"外部selector run尚未完成：{run_name}")
    videos = pd.read_csv(directory / "video_scores.csv", float_precision="round_trip")
    selected = videos[videos["selector"].eq(selector)].copy()
    uniform = videos[videos["selector"].eq("uniform")].copy()
    if len(selected) != 600 or len(uniform) != 600:
        raise ValueError(f"{run_name}必须包含600条selector及600条Uniform分数")
    return selected, uniform


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "results/analysis/caes_external_genvidbench_selectors",
    )
    parser.add_argument("--iterations", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if args.output_dir.exists() and args.overwrite:
        shutil.rmtree(args.output_dir)
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError(f"分析输出已存在：{args.output_dir}")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    feature, feature_uniform = _load("feature_change", RUNS["feature_change"])
    random, random_uniform = _load("random", RUNS["random"])
    identity = ["video_id", "dataset", "subset", "source_model", "video_path"]
    # 保留原正式run的行顺序，使有限次bootstrap可逐值复现原run产物。
    left = feature_uniform.reset_index(drop=True)
    right = random_uniform.reset_index(drop=True)
    if not left[identity].equals(right[identity]):
        raise ValueError("两个外部run的Uniform视频身份不一致")
    score_columns = ["global_score", "local_score", "final_score"]
    differences = {
        column: float(np.max(np.abs(
            left[column].to_numpy(dtype=np.float64)
            - right[column].to_numpy(dtype=np.float64)
        )))
        for column in score_columns
    }
    if any(value != 0.0 for value in differences.values()):
        raise ValueError(f"两个外部run的Uniform分数发生漂移：{differences}")

    scores = pd.concat([left, random, feature], ignore_index=True)
    metrics = build_matched_selector_pairwise_table(scores, seed=args.seed)
    versus_uniform = paired_selector_bootstrap(
        scores, baseline="uniform", seed=args.seed, iterations=args.iterations
    )
    versus_random = paired_selector_bootstrap(
        pd.concat([random, feature], ignore_index=True),
        baseline="random",
        seed=args.seed,
        iterations=args.iterations,
    )
    bootstrap = pd.concat([versus_uniform, versus_random], ignore_index=True)
    metrics.to_csv(args.output_dir / "pairwise_metrics.csv", index=False)
    bootstrap.to_csv(args.output_dir / "paired_bootstrap.csv", index=False)
    manifest = {
        "status": "completed",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "runs": RUNS,
        "bootstrap_iterations": args.iterations,
        "bootstrap_seed": args.seed,
        "uniform_max_abs_difference": differences,
        "artifacts": {
            name: _sha256(args.output_dir / name)
            for name in ("pairwise_metrics.csv", "paired_bootstrap.csv")
        },
    }
    (args.output_dir / "analysis_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(metrics.to_string(index=False), flush=True)
    print(f"[完成] 三方法外部配对分析：{args.output_dir}", flush=True)


if __name__ == "__main__":
    main()
