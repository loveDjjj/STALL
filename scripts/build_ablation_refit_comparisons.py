#!/usr/bin/env python3
"""从已完成的重拟合 run 构建 D1/D2、K1/K3 与 Spatial 的配对统计表。"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from config import load_config
from evaluation.metrics import binary_metrics
from evaluation.tables import _balanced_real_pair, normalize_scores


@dataclass(frozen=True)
class Comparison:
    """一个候选方法相对基线方法的固定对照。"""

    name: str
    candidate_run: str
    baseline_run: str


COMPARISONS = (
    Comparison(
        "local_d2_refit_vs_local_d1_refit",
        "alpha_stall_local_d2_refit",
        "alpha_stall_local_d1_refit",
    ),
    Comparison(
        "full_d2_k3_refit_vs_full_d1_refit",
        "alpha_stall_full_d2_k3_refit",
        "alpha_stall_full_d1_refit",
    ),
    Comparison(
        "full_d2_k3_refit_vs_full_k1_refit",
        "alpha_stall_full_d2_k3_refit",
        "alpha_stall_full_k1_refit",
    ),
    Comparison(
        "local_d2_refit_vs_local_spatial_only_refit",
        "alpha_stall_local_d2_refit",
        "alpha_stall_local_spatial_only_refit",
    ),
    Comparison(
        "local_spatial_d2_refit_vs_local_d2_refit",
        "alpha_stall_local_spatial_d2_refit",
        "alpha_stall_local_d2_refit",
    ),
    Comparison(
        "full_d2_k3_refit_vs_full_d2_k3_no_spatial_refit",
        "alpha_stall_full_d2_k3_refit",
        "alpha_stall_full_d2_k3_no_spatial_refit",
    ),
    Comparison(
        "full_d2_k3_no_spatial_refit_vs_global_only_k3_refit",
        "alpha_stall_full_d2_k3_no_spatial_refit",
        "alpha_stall_global_only_k3_refit",
    ),
    Comparison(
        "full_d2_k3_refit_vs_global_only_k3_refit",
        "alpha_stall_full_d2_k3_refit",
        "alpha_stall_global_only_k3_refit",
    ),
    Comparison(
        "full_d2_k3_no_spatial_refit_vs_full_d1_k3_no_spatial_refit",
        "alpha_stall_full_d2_k3_no_spatial_refit",
        "alpha_stall_full_d1_k3_no_spatial_refit",
    ),
    Comparison(
        "full_d2_k3_no_spatial_refit_vs_full_d2_k1_no_spatial_refit",
        "alpha_stall_full_d2_k3_no_spatial_refit",
        "alpha_stall_full_d2_k1_no_spatial_refit",
    ),
)

KEY_COLUMNS = ("video_id", "dataset", "subset", "source_model", "video_path")


def _run_dir(name: str) -> Path:
    path = ROOT / "results" / "runs" / name
    if not path.is_dir():
        raise FileNotFoundError(f"比较输入 run 不存在：{path}")
    return path


def _load_run(name: str) -> tuple[pd.DataFrame, dict]:
    """加载并校验一个已完成 run 的配置和逐视频分数。"""

    directory = _run_dir(name)
    progress = json.loads((directory / "progress.json").read_text(encoding="utf-8"))
    if progress.get("status") != "completed":
        raise ValueError(f"比较输入 run 尚未完成：{name}")
    scores_path = directory / "video_scores.csv"
    config_path = directory / "resolved_config.yaml"
    if not scores_path.is_file() or not config_path.is_file():
        raise FileNotFoundError(f"比较输入 run 缺少分数或配置：{name}")
    return normalize_scores(
        pd.read_csv(scores_path, float_precision="round_trip")
    ), load_config(config_path)


def _align(candidate: pd.DataFrame, baseline: pd.DataFrame, comparison: Comparison) -> pd.DataFrame:
    """按视频身份对齐两次评分，拒绝任何数据集、标签或生成器漂移。"""

    left = candidate[list(KEY_COLUMNS) + ["final_score"]].rename(
        columns={"final_score": "candidate_score"}
    )
    right = baseline[list(KEY_COLUMNS) + ["final_score"]].rename(
        columns={"final_score": "baseline_score"}
    )
    merged = left.merge(right, on=list(KEY_COLUMNS), how="inner", validate="one_to_one")
    if len(merged) != len(left) or len(merged) != len(right):
        raise ValueError(
            f"{comparison.name} 的候选与基线视频身份不一致："
            f"candidate={len(left)}, baseline={len(right)}, matched={len(merged)}"
        )
    return merged


def _pairwise_deltas(
    aligned: pd.DataFrame, comparison: Comparison, pairwise_seed: int
) -> pd.DataFrame:
    """按论文生成器配对口径计算候选减基线的 AUC/AP 点估计。"""

    rows: list[dict[str, object]] = []
    for dataset, dataset_frame in aligned.groupby("dataset", sort=True):
        real = dataset_frame[dataset_frame["subset"].eq("real")]
        per_generator = []
        for generator, fake in dataset_frame[dataset_frame["subset"].eq("annotated")].groupby(
            "source_model", sort=True
        ):
            pair = _balanced_real_pair(real, fake, pairwise_seed)
            candidate_metrics = binary_metrics(pair, "candidate_score")
            baseline_metrics = binary_metrics(pair, "baseline_score")
            per_generator.append((generator, candidate_metrics, baseline_metrics, len(pair) // 2))
        if not per_generator:
            raise ValueError(f"{comparison.name} 的 {dataset} 没有生成器配对")
        for metric, metric_key in (("auc", "auc"), ("ap_real", "real_positive_ap")):
            candidate_value = float(np.mean([item[1][metric_key] for item in per_generator]))
            baseline_value = float(np.mean([item[2][metric_key] for item in per_generator]))
            rows.append({
                "comparison": comparison.name,
                "scope": "generator_pairwise_dataset_macro",
                "dataset": dataset,
                "metric": metric,
                "candidate_run": comparison.candidate_run,
                "baseline_run": comparison.baseline_run,
                "candidate_value": candidate_value,
                "baseline_value": baseline_value,
                "delta": candidate_value - baseline_value,
                "n_generators": len(per_generator),
                "n_pairwise_videos": int(sum(2 * item[3] for item in per_generator)),
                "pairwise_seed": pairwise_seed,
            })
    table = pd.DataFrame(rows)
    macro_rows = []
    for metric, group in table.groupby("metric", sort=True):
        macro_rows.append({
            "comparison": comparison.name,
            "scope": "generator_pairwise_macro3",
            "dataset": "Macro-3",
            "metric": metric,
            "candidate_run": comparison.candidate_run,
            "baseline_run": comparison.baseline_run,
            "candidate_value": float(group["candidate_value"].mean()),
            "baseline_value": float(group["baseline_value"].mean()),
            "delta": float(group["delta"].mean()),
            "n_generators": int(group["n_generators"].sum()),
            "n_pairwise_videos": int(group["n_pairwise_videos"].sum()),
            "pairwise_seed": pairwise_seed,
        })
    return pd.concat([table, pd.DataFrame(macro_rows)], ignore_index=True)


def _bootstrap_seed(seed: int, *parts: str) -> int:
    """从全局种子生成稳定的比较/数据集级随机种子。"""

    digest = hashlib.sha256("\0".join((str(seed), *parts)).encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "little") % (2**32)


def _bootstrap_auc(
    aligned: pd.DataFrame,
    comparison: Comparison,
    pairwise_seed: int,
    bootstrap_seed: int,
    iterations: int,
) -> pd.DataFrame:
    """对主表同一生成器配对执行视频级 AUC 差异 bootstrap。"""

    per_dataset_pairs: dict[str, list[pd.DataFrame]] = {}
    point_deltas: dict[str, float] = {}
    for dataset, dataset_frame in aligned.groupby("dataset", sort=True):
        real = dataset_frame[dataset_frame["subset"].eq("real")]
        pairs = [
            _balanced_real_pair(real, fake, pairwise_seed)
            for _, fake in dataset_frame[dataset_frame["subset"].eq("annotated")].groupby(
                "source_model", sort=True
            )
        ]
        if not pairs:
            raise ValueError(f"{comparison.name} 的 {dataset} 没有可 bootstrap 的生成器配对")
        per_dataset_pairs[dataset] = pairs
        point_deltas[dataset] = float(np.mean([
            binary_metrics(pair, "candidate_score")["auc"]
            - binary_metrics(pair, "baseline_score")["auc"]
            for pair in pairs
        ]))

    samples_by_dataset: dict[str, np.ndarray] = {}
    for dataset, pairs in per_dataset_pairs.items():
        rng = np.random.default_rng(_bootstrap_seed(bootstrap_seed, comparison.name, dataset))
        samples = np.empty(iterations, dtype=np.float64)
        for index in range(iterations):
            deltas = []
            for pair in pairs:
                sampled = pair.iloc[rng.integers(0, len(pair), len(pair))]
                if sampled["subset"].nunique() == 2:
                    deltas.append(
                        binary_metrics(sampled, "candidate_score")["auc"]
                        - binary_metrics(sampled, "baseline_score")["auc"]
                    )
            if not deltas:
                raise ValueError(f"{comparison.name} 的 {dataset} bootstrap 未生成有效重采样")
            samples[index] = float(np.mean(deltas))
        samples_by_dataset[dataset] = samples

    rows = []
    for dataset, samples in samples_by_dataset.items():
        rows.append({
            "comparison": comparison.name,
            "scope": "generator_pairwise_dataset_macro",
            "dataset": dataset,
            "metric": "auc",
            "candidate_run": comparison.candidate_run,
            "baseline_run": comparison.baseline_run,
            "delta": point_deltas[dataset],
            "ci95_low": float(np.quantile(samples, 0.025)),
            "ci95_high": float(np.quantile(samples, 0.975)),
            "bootstrap_iterations": iterations,
            "bootstrap_seed": bootstrap_seed,
            "pairwise_seed": pairwise_seed,
        })
    macro_samples = np.stack(list(samples_by_dataset.values()), axis=0).mean(axis=0)
    rows.append({
        "comparison": comparison.name,
        "scope": "generator_pairwise_macro3",
        "dataset": "Macro-3",
        "metric": "auc",
        "candidate_run": comparison.candidate_run,
        "baseline_run": comparison.baseline_run,
        "delta": float(np.mean(list(point_deltas.values()))),
        "ci95_low": float(np.quantile(macro_samples, 0.025)),
        "ci95_high": float(np.quantile(macro_samples, 0.975)),
        "bootstrap_iterations": iterations,
        "bootstrap_seed": bootstrap_seed,
        "pairwise_seed": pairwise_seed,
    })
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="构建重拟合 D1/D2、K1/K3 与 Spatial 的跨运行配对比较，不重跑缓存或评分。"
    )
    parser.add_argument(
        "--output-run-name", default="ablation_refit_comparisons",
        help="结果写入 results/runs/<name>/，默认 ablation_refit_comparisons",
    )
    parser.add_argument("--iterations", type=int, default=1000, help="每个对照的 bootstrap 次数")
    parser.add_argument("--bootstrap-seed", type=int, default=17, help="bootstrap 随机种子")
    parser.add_argument("--overwrite", action="store_true", help="覆盖已有同名派生分析目录")
    arguments = parser.parse_args()
    if arguments.iterations < 100:
        raise ValueError("bootstrap 次数至少为 100")
    output_dir = ROOT / "results" / "runs" / arguments.output_run_name
    if output_dir.exists() and not arguments.overwrite:
        raise FileExistsError(f"派生分析目录已存在：{output_dir}；如需重建请传入 --overwrite")
    output_dir.mkdir(parents=True, exist_ok=True)

    metric_tables = []
    bootstrap_tables = []
    manifest = {
        "status": "completed",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "bootstrap_iterations": arguments.iterations,
        "bootstrap_seed": arguments.bootstrap_seed,
        "comparisons": [],
    }
    for comparison in COMPARISONS:
        candidate, candidate_config = _load_run(comparison.candidate_run)
        baseline, baseline_config = _load_run(comparison.baseline_run)
        pairwise_seed = int(candidate_config["metrics"]["pairwise_seed"])
        if pairwise_seed != int(baseline_config["metrics"]["pairwise_seed"]):
            raise ValueError(f"{comparison.name} 的 pairwise_seed 不一致")
        aligned = _align(candidate, baseline, comparison)
        metric_tables.append(_pairwise_deltas(aligned, comparison, pairwise_seed))
        bootstrap_tables.append(_bootstrap_auc(
            aligned, comparison, pairwise_seed, arguments.bootstrap_seed, arguments.iterations
        ))
        manifest["comparisons"].append({
            "name": comparison.name,
            "candidate_run": comparison.candidate_run,
            "baseline_run": comparison.baseline_run,
            "aligned_video_count": len(aligned),
            "pairwise_seed": pairwise_seed,
        })
        print(f"[完成] {comparison.name}：对齐 {len(aligned)} 条视频", flush=True)

    pd.concat(metric_tables, ignore_index=True).to_csv(
        output_dir / "pairwise_metric_deltas.csv", index=False
    )
    pd.concat(bootstrap_tables, ignore_index=True).to_csv(
        output_dir / "bootstrap_auc_deltas.csv", index=False
    )
    (output_dir / "comparison_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(f"[完成] 已写入 {output_dir}", flush=True)


if __name__ == "__main__":
    main()
