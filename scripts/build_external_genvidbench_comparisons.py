#!/usr/bin/env python3
"""对齐 GenVidBench 外部 run，并生成 K3/K1、Final/Global 的配对置信区间。"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from evaluation.metrics import binary_metrics
from evaluation.tables import normalize_scores


KEY_COLUMNS = ("video_id", "dataset", "subset", "source_model", "video_path")
COMPARISONS = (
    (
        "final_k3_vs_final_k1",
        "alpha_stall_external_genvidbench",
        "alpha_stall_external_genvidbench_k1",
    ),
    (
        "final_k3_vs_global_k3",
        "alpha_stall_external_genvidbench",
        "alpha_stall_external_genvidbench_global_k3",
    ),
)


def _load_run(run_name: str) -> pd.DataFrame:
    directory = ROOT / "results" / "runs" / run_name
    progress = json.loads((directory / "progress.json").read_text(encoding="utf-8"))
    if progress.get("status") != "completed":
        raise ValueError(f"外部比较输入 run 尚未完成：{run_name}")
    return normalize_scores(
        pd.read_csv(directory / "video_scores.csv", float_precision="round_trip")
    )


def _align(candidate: pd.DataFrame, baseline: pd.DataFrame) -> pd.DataFrame:
    left = candidate[list(KEY_COLUMNS) + ["final_score"]].rename(
        columns={"final_score": "candidate_score"}
    )
    right = baseline[list(KEY_COLUMNS) + ["final_score"]].rename(
        columns={"final_score": "baseline_score"}
    )
    merged = left.merge(right, on=list(KEY_COLUMNS), how="inner", validate="one_to_one")
    if len(merged) != len(left) or len(merged) != len(right):
        raise ValueError(
            f"外部 run 视频身份不一致：candidate={len(left)}, "
            f"baseline={len(right)}, matched={len(merged)}"
        )
    if merged["dataset"].unique().tolist() != ["genvidbench"]:
        raise ValueError("外部比较只能包含 genvidbench")
    return merged


def _seed(seed: int, name: str) -> int:
    digest = hashlib.sha256(f"{seed}\0{name}".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "little") % (2**32)


def _compare(frame: pd.DataFrame, name: str, iterations: int, seed: int) -> list[dict]:
    candidate = binary_metrics(frame, "candidate_score")
    baseline = binary_metrics(frame, "baseline_score")
    metrics = (("auc", "auc"), ("ap_real", "real_positive_ap"))
    real_indices = np.flatnonzero(frame["subset"].eq("real").to_numpy())
    fake_indices = np.flatnonzero(frame["subset"].eq("annotated").to_numpy())
    rng = np.random.default_rng(_seed(seed, name))
    samples = {metric: np.empty(iterations, dtype=np.float64) for metric, _ in metrics}
    for iteration in range(iterations):
        # 分层抽样保持 300/300 类别比例，并对候选/基线复用同一视频抽样身份。
        indices = np.concatenate((
            rng.choice(real_indices, size=len(real_indices), replace=True),
            rng.choice(fake_indices, size=len(fake_indices), replace=True),
        ))
        sampled = frame.iloc[indices]
        candidate_sample = binary_metrics(sampled, "candidate_score")
        baseline_sample = binary_metrics(sampled, "baseline_score")
        for metric, key in metrics:
            samples[metric][iteration] = candidate_sample[key] - baseline_sample[key]
    rows = []
    for metric, key in metrics:
        values = samples[metric]
        rows.append({
            "comparison": name,
            "dataset": "genvidbench",
            "metric": metric,
            "candidate_value": candidate[key],
            "baseline_value": baseline[key],
            "delta": candidate[key] - baseline[key],
            "ci95_low": float(np.quantile(values, 0.025)),
            "ci95_high": float(np.quantile(values, 0.975)),
            "bootstrap_iterations": iterations,
            "bootstrap_seed": seed,
            "n_real": len(real_indices),
            "n_fake": len(fake_indices),
        })
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--iterations", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--overwrite", action="store_true")
    arguments = parser.parse_args()
    if arguments.iterations < 100:
        raise ValueError("bootstrap 次数至少为 100")
    output = ROOT / "results" / "runs" / "external_genvidbench_comparisons"
    if output.exists() and not arguments.overwrite:
        raise FileExistsError(f"外部比较目录已存在：{output}；重建时请传入 --overwrite")
    output.mkdir(parents=True, exist_ok=True)

    rows = []
    manifest = {
        "status": "completed",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "bootstrap_iterations": arguments.iterations,
        "bootstrap_seed": arguments.seed,
        "comparisons": [],
    }
    for name, candidate_name, baseline_name in COMPARISONS:
        aligned = _align(_load_run(candidate_name), _load_run(baseline_name))
        rows.extend(_compare(aligned, name, arguments.iterations, arguments.seed))
        manifest["comparisons"].append({
            "name": name,
            "candidate_run": candidate_name,
            "baseline_run": baseline_name,
            "aligned_video_count": len(aligned),
        })
        print(f"[完成] {name}：对齐 {len(aligned)} 条视频", flush=True)

    pd.DataFrame(rows).to_csv(output / "metric_deltas.csv", index=False)
    (output / "comparison_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(f"[完成] 已写入 {output}", flush=True)


if __name__ == "__main__":
    main()
