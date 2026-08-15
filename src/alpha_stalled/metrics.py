"""
STALL 评测使用的独立指标工具。

从私有 videoDetection repo 中抽出并整理为自包含实现：
  - ScoreDirection enum
  - Score dataclass
  - predictor_scalar2metrics
  - build_results_table  (逐生成器 AUC/AP，使用 pairwise balanced 比较)
  - print_results        (格式化表格输出)

Alpha-STALLED 正式实验另外统一使用：
  - pairwise_frames      (每个生成器与真实来源均衡样本配对)
  - metric_tables        (生成器、数据集及 Macro-3 AUC/AP)
  - paired_bootstrap     (固定生成器配对内的配对 cluster bootstrap)
"""

from __future__ import annotations

import hashlib
import logging
from dataclasses import dataclass
from enum import Enum
from typing import Dict

import numpy as np
import pandas as pd
from tabulate import tabulate

logger = logging.getLogger(__name__)


def _to_numpy(x) -> np.ndarray:
    if isinstance(x, np.ndarray):
        return x
    if isinstance(x, list):
        return np.array(x)
    import torch
    if isinstance(x, torch.Tensor):
        return x.cpu().numpy()
    raise TypeError(f"无法将 {type(x)} 转换为 numpy array")


# ─────────────────────────────────────────────────────────────────────────────
# 分数方向
# ─────────────────────────────────────────────────────────────────────────────

class ScoreDirection(Enum):
    """分数越高表示视频更像真实还是更像生成。"""
    HIGHER_IS_REAL = 1
    HIGHER_IS_FAKE = 0


@dataclass
class Score:
    """分数数组及其方向。"""
    value: np.ndarray
    direction: ScoreDirection = ScoreDirection.HIGHER_IS_REAL


# ─────────────────────────────────────────────────────────────────────────────
# 核心指标计算
# ─────────────────────────────────────────────────────────────────────────────

def predictor_scalar2metrics(predictor_scalar, labels, threshold=None):
    """从分数和二分类标签计算 AUC/AP，并可选计算 F1/Accuracy。

    Args:
        predictor_scalar: 一维检测器分数数组。
        labels:           一维二分类数组；1 表示真实，0 表示 annotated/fake。
        threshold:        若提供，同时计算 F1 和 Accuracy。

    返回：
        包含 "AUC"、"AP" 的 dict，并可选包含 "F1_score"、"Accuracy"。
    """
    from sklearn.metrics import (
        roc_auc_score,
        average_precision_score,
        f1_score,
        accuracy_score,
    )

    predictor_scalar = _to_numpy(predictor_scalar)
    labels = _to_numpy(labels)

    assert predictor_scalar.ndim == 1
    assert labels.ndim == 1
    assert predictor_scalar.shape == labels.shape
    assert np.all(np.isin(labels, [0, 1])), "Labels 必须为二值（0 或 1）"

    metrics = {
        "AUC": roc_auc_score(labels, predictor_scalar),
        "AP": average_precision_score(labels, predictor_scalar),
    }
    if threshold is not None:
        preds = (predictor_scalar > threshold).astype(int)
        metrics["F1_score"] = f1_score(labels, preds)
        metrics["Accuracy"] = accuracy_score(labels, preds)

    return metrics


# ─────────────────────────────────────────────────────────────────────────────
# 数据集平衡辅助函数
# ─────────────────────────────────────────────────────────────────────────────

def _sample_proportional(df: pd.DataFrame, target_size: int, group_key: str, seed: int) -> pd.DataFrame:
    if len(df) <= target_size:
        return df
    group_counts = df[group_key].value_counts()
    target_per_group = (group_counts / len(df) * target_size).round().astype(int)
    diff = target_size - target_per_group.sum()
    if diff != 0:
        for i in range(abs(diff)):
            g = target_per_group.sort_values(ascending=False).index[i % len(target_per_group)]
            target_per_group[g] += 1 if diff > 0 else -1
    parts = [
        df[df[group_key] == g].sample(min(n, len(df[df[group_key] == g])), random_state=seed)
        for g, n in target_per_group.items() if n > 0
    ]
    return pd.concat(parts, ignore_index=True).sample(frac=1, random_state=seed)


def _balance_datasets(real_df: pd.DataFrame, annotated_df: pd.DataFrame, seed: int) -> pd.DataFrame:
    if len(annotated_df) >= len(real_df):
        sampled = _sample_proportional(annotated_df, len(real_df), "source_model", seed)
        return pd.concat([real_df, sampled], ignore_index=True)
    else:
        sampled = _sample_proportional(real_df, len(annotated_df), "source_model", seed)
        return pd.concat([sampled, annotated_df], ignore_index=True)


def _sample_balanced_real(real_df: pd.DataFrame, n: int, seed: int) -> pd.DataFrame:
    # 从每个真实来源均匀采样：每个 source_model 使用相同 quota（n // num_sources）。
    # 整数除法保证总数始终 <= n，因此最后的 .sample() 只负责打乱。
    sources = real_df["source_model"].unique()
    n_per = max(1, n // len(sources))
    parts = [
        g.sample(min(len(g), n_per), random_state=seed)
        for _, g in real_df.groupby("source_model")
    ]
    result = pd.concat(parts, ignore_index=True)
    return result.sample(n=min(n, len(result)), random_state=seed)


# ─────────────────────────────────────────────────────────────────────────────
# 分组指标计算
# ─────────────────────────────────────────────────────────────────────────────

def _compute_metrics(data: pd.DataFrame, score_names: list, score_directions: dict) -> dict:
    counts = data["subset"].value_counts()
    metrics: dict = {
        "n_real": counts.get("real", 0),
        "n_annotated": counts.get("annotated", 0),
        "n_total": len(data),
    }
    for name in score_names:
        if name not in data.columns:
            continue
        direction = score_directions.get(name, ScoreDirection.HIGHER_IS_REAL)
        if direction == ScoreDirection.HIGHER_IS_REAL:
            labels = (data["subset"] == "real").astype(np.uint8).values
        else:
            labels = (data["subset"] == "annotated").astype(np.uint8).values
        m = predictor_scalar2metrics(data[name].to_numpy(), labels)
        for k, v in m.items():
            metrics[f"{name} {k}"] = float(v)
    return metrics


# ─────────────────────────────────────────────────────────────────────────────
# Pairwise comparisons
# ─────────────────────────────────────────────────────────────────────────────

def _run_pairwise(df: pd.DataFrame, score_names: list, score_directions: dict, seed: int) -> dict:
    real_df = df[df["subset"] == "real"]
    results = {}
    for model, group in df[df["subset"] == "annotated"].groupby("source_model"):
        sampled_real = _sample_balanced_real(real_df, len(group), seed)
        group_trimmed = group.head(len(sampled_real))
        comparison = pd.concat([group_trimmed, sampled_real], ignore_index=True)
        results[model] = _compute_metrics(comparison, score_names, score_directions)
    return results


def _calculate_averages(pairwise: dict) -> dict:
    if not pairwise:
        return {}
    keys = next(iter(pairwise.values())).keys()
    return {k: float(np.mean([r[k] for r in pairwise.values()])) for k in keys}


# ─────────────────────────────────────────────────────────────────────────────
# AUC-flipped diagnostic row
# ─────────────────────────────────────────────────────────────────────────────

def _auc_flipped_row(results_df: pd.DataFrame) -> pd.DataFrame | None:
    auc_cols = [c for c in results_df.columns if c.endswith(" AUC")]
    if not auc_cols:
        return None
    indiv = results_df[~results_df["Generative Model"].isin(["All", "Average"])]
    if indiv.empty:
        return None
    row: dict = {"Generative Model": "AUC Flipped?"}
    for col in auc_cols:
        scores = indiv[col].values
        consistent = np.all(scores <= 0.5) or np.all(scores >= 0.5)
        row[col] = not consistent
    for col in results_df.columns:
        if col not in row:
            row[col] = np.nan if (col.endswith(" AP") or col.startswith("n_")) else False
    return pd.DataFrame([row])


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────

def build_results_table(
    df: pd.DataFrame,
    score_directions: dict,
    seed: int = 42,
    skip_global_compare: bool = False,
    verbose: bool = True,
) -> pd.DataFrame:
    """构建逐生成器 AUC/AP 结果表。

    Args:
        df:                   包含 subset、source_model 和若干分数列的 DataFrame。
                              ``subset`` 必须为 "real" 或 "annotated"。
        score_directions:     分数列名到 ScoreDirection 的映射。
        seed:                 平衡采样随机种子。
        skip_global_compare:  跳过 "All" 全局比较行。
        verbose:              打印逐模型样本数。

    返回：
        DataFrame；行包含各生成模型和 "Average"，列包含
        "<score> AUC"、"<score> AP"、n_real、n_annotated、n_total。
    """
    score_names = list(score_directions.keys())
    real_count = len(df[df["subset"] == "real"])
    annotated_count = len(df[df["subset"] == "annotated"])

    if real_count == 0 or annotated_count == 0:
        raise ValueError("数据集必须同时包含 'real' 和 'annotated' 行。")

    results: dict = {}

    if not skip_global_compare:
        balanced = _balance_datasets(
            df[df["subset"] == "real"], df[df["subset"] == "annotated"], seed
        )
        results["All"] = _compute_metrics(balanced, score_names, score_directions)

    pairwise = _run_pairwise(df, score_names, score_directions, seed)
    if verbose:
        for model, r in pairwise.items():
            print(f"  {model}: {r['n_annotated']} fake vs {r['n_real']} real")
    results.update(pairwise)

    if len(pairwise) > 1:
        results["Average"] = _calculate_averages(pairwise)

    res_df = pd.DataFrame.from_dict(results, orient="index").reset_index()
    res_df.rename(columns={"index": "Generative Model"}, inplace=True)

    return res_df


def get_results_df(inf_df, scores_d: Dict[str, Score]) -> pd.DataFrame:
    """eval 脚本使用的便捷包装。

    Args:
        inf_df:   包含 subset、source_model 列的 DataFrame（或 HuggingFace Dataset）。
        scores_d: 分数名到 Score(value, direction) 的映射。

    返回：
        build_results_table 返回的结果 DataFrame。
    """
    try:
        import datasets
        if isinstance(inf_df, datasets.Dataset):
            inf_df = inf_df.select_columns(["subset", "source_model"]).to_pandas()
    except ImportError:
        pass

    scores_val = {k: v.value for k, v in scores_d.items()}
    directions = {k: v.direction for k, v in scores_d.items()}
    scores_df = pd.DataFrame(scores_val)
    combined = pd.concat([inf_df.reset_index(drop=True), scores_df], axis=1)
    return build_results_table(combined, directions, skip_global_compare=True, verbose=False)


def print_results(df: pd.DataFrame, include_counts: bool = True, auc_only: bool = False):
    """向 stdout 打印格式化结果表。

    Args:
        df:             build_results_table / get_results_df 的输出。
        include_counts: 包含 n_real / n_annotated 列（不包含 n_total）。
        auc_only:       只显示 AUC 列（隐藏 AP）。
    """
    count_cols = ["n_real", "n_annotated", "n_total"]
    display_count_cols = ["n_real", "n_annotated"]
    _count_headers = {"n_real": "#real", "n_annotated": "#fake"}
    main_df = df[df["Generative Model"] != "AUC Flipped?"]

    metric_cols = [
        c for c in main_df.columns
        if c not in count_cols + ["Generative Model"]
        and (not auc_only or c.endswith(" AUC"))
    ]

    ordered = ["Generative Model"]
    if include_counts:
        ordered += [c for c in display_count_cols if c in main_df.columns]
    ordered += sorted(metric_cols)

    # Strip score name prefix for display: "final_score AUC" -> "AUC", "final_score AP" -> "AP"
    def _short_header(col):
        if col in _count_headers:
            return _count_headers[col]
        for suffix in (" AUC", " AP"):
            if col.endswith(suffix):
                return suffix.strip()
        return col

    display_headers = [_short_header(c) for c in ordered]

    floatfmt = []
    colalign = []
    for col in ordered:
        if col == "Generative Model":
            floatfmt.append("")
            colalign.append("left")
        elif col in count_cols:
            floatfmt.append(".0f")
            colalign.append("center")
        else:
            floatfmt.append(".3f")
            colalign.append("center")

    display_df = main_df[ordered].copy()
    for col in display_count_cols:
        if col in display_df.columns:
            display_df[col] = display_df[col].astype(int)

    table_str = tabulate(
        display_df,
        headers=display_headers,
        tablefmt="github",
        floatfmt=floatfmt,
        colalign=colalign,
        showindex=False,
    )
    GAP = max(60, len(table_str.splitlines()[0]))
    print("\n" + "=" * GAP)
    print("RESULTS".center(GAP))
    print("=" * GAP)
    print(table_str)
    print("=" * GAP)


# Public name for new code; the underscored alias remains for historical callers.
sample_balanced_real = _sample_balanced_real


def pairwise_frames(df: pd.DataFrame, seed: int) -> dict[str, pd.DataFrame]:
    """Pair every generated source with an equal-source-quota real sample."""

    real = df[df["subset"] == "real"]
    frames: dict[str, pd.DataFrame] = {}
    generated = df[df["subset"] == "annotated"]
    for generator, fake in generated.groupby("source_model", sort=True):
        sampled_real = sample_balanced_real(real, len(fake), seed)
        fake = fake.head(len(sampled_real))
        frames[str(generator)] = pd.concat([sampled_real, fake], ignore_index=True)
    return frames


def auc_ap(frame: pd.DataFrame, score: str) -> tuple[float, float]:
    """Compute real-positive AUC/AP for one already-balanced pair frame."""

    label = frame["subset"].eq("real").astype(np.uint8).to_numpy()
    values = frame[score].to_numpy(dtype=np.float64)
    result = predictor_scalar2metrics(values, label)
    return float(result["AUC"]), float(result["AP"])


def metric_tables(
    per_video: pd.DataFrame,
    seed: int,
    *,
    score_columns: list[str] | tuple[str, ...],
    config_names: dict[str, str] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build generator, dataset, and Macro-3 pairwise-balanced metrics."""

    names = config_names or {column: column for column in score_columns}
    generator_rows: list[dict] = []
    for dataset, dataset_df in per_video.groupby("dataset", sort=False):
        for generator, pair in pairwise_frames(dataset_df, seed).items():
            counts = pair["subset"].value_counts()
            for config in score_columns:
                auc, ap = auc_ap(pair, config)
                generator_rows.append(
                    {
                        "dataset": dataset,
                        "generator": generator,
                        "config": config,
                        "config_name": names[config],
                        "n_real": int(counts.get("real", 0)),
                        "n_fake": int(counts.get("annotated", 0)),
                        "auc": auc,
                        "ap": ap,
                    }
                )
    generator_metrics = pd.DataFrame(generator_rows)
    dataset_metrics = (
        generator_metrics.groupby(
            ["dataset", "config", "config_name"], as_index=False
        ).agg(
            n_generators=("generator", "nunique"),
            auc=("auc", "mean"),
            ap=("ap", "mean"),
        )
    )
    macro = (
        dataset_metrics.groupby(["config", "config_name"], as_index=False).agg(
            n_generators=("n_generators", "sum"),
            auc=("auc", "mean"),
            ap=("ap", "mean"),
        )
    )
    macro.insert(0, "dataset", "Macro-3")
    dataset_metrics = pd.concat([dataset_metrics, macro], ignore_index=True)
    return dataset_metrics, generator_metrics


def _stable_bootstrap_seed(seed: int, *parts: str) -> int:
    digest = hashlib.sha256("\0".join((str(seed), *parts)).encode("utf-8")).digest()
    return int.from_bytes(digest[:4], "little")


def paired_bootstrap(
    per_video: pd.DataFrame,
    seed: int,
    iterations: int,
    comparisons: tuple[tuple[str, str, str], ...],
) -> pd.DataFrame:
    """Paired cluster bootstrap of generator-macro AUC/AP deltas."""

    rows: list[dict] = []
    for dataset, dataset_df in per_video.groupby("dataset", sort=False):
        pairs = pairwise_frames(dataset_df, seed)
        for new, base, comparison in comparisons:
            point_auc: list[float] = []
            point_ap: list[float] = []
            for pair in pairs.values():
                new_auc, new_ap = auc_ap(pair, new)
                base_auc, base_ap = auc_ap(pair, base)
                point_auc.append(new_auc - base_auc)
                point_ap.append(new_ap - base_ap)

            rng = np.random.default_rng(
                _stable_bootstrap_seed(seed, str(dataset), comparison)
            )
            boot_auc = np.empty(iterations, dtype=np.float64)
            boot_ap = np.empty(iterations, dtype=np.float64)
            split_pairs = []
            for pair in pairs.values():
                real = pair[pair["subset"] == "real"].reset_index(drop=True)
                fake = pair[pair["subset"] == "annotated"].reset_index(drop=True)
                split_pairs.append((real, fake))
            for iteration in range(iterations):
                auc_deltas: list[float] = []
                ap_deltas: list[float] = []
                for real, fake in split_pairs:
                    real_idx = rng.integers(0, len(real), len(real))
                    fake_idx = rng.integers(0, len(fake), len(fake))
                    sample = pd.concat(
                        [real.iloc[real_idx], fake.iloc[fake_idx]], ignore_index=True
                    )
                    new_auc, new_ap = auc_ap(sample, new)
                    base_auc, base_ap = auc_ap(sample, base)
                    auc_deltas.append(new_auc - base_auc)
                    ap_deltas.append(new_ap - base_ap)
                boot_auc[iteration] = np.mean(auc_deltas)
                boot_ap[iteration] = np.mean(ap_deltas)
            for metric, point, samples in (
                ("auc", float(np.mean(point_auc)), boot_auc),
                ("ap", float(np.mean(point_ap)), boot_ap),
            ):
                rows.append(
                    {
                        "dataset": dataset,
                        "scope": "generator_macro",
                        "comparison": comparison,
                        "new_config": new,
                        "base_config": base,
                        "metric": metric,
                        "delta": point,
                        "ci95_low": float(np.quantile(samples, 0.025)),
                        "ci95_high": float(np.quantile(samples, 0.975)),
                        "bootstrap_iterations": iterations,
                    }
                )
    return pd.DataFrame(rows)


def stable_seed(seed: int, *parts: str) -> int:
    """Build the historical 64-bit deterministic seed used by cluster audits."""

    digest = hashlib.sha256("\0".join((str(seed), *parts)).encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "little")


def binary_metrics(
    frame: pd.DataFrame, score_column: str = "S"
) -> dict[str, float]:
    """Compute the extended locked-U0 binary diagnostics for a score column."""

    from sklearn.metrics import average_precision_score, auc, roc_curve

    real_label = frame["subset"].eq("real").to_numpy(dtype=np.uint8)
    if len(real_label) == 0 or real_label.min() == real_label.max():
        raise ValueError("metrics require both real and fake videos")
    realness = frame[score_column].to_numpy(dtype=np.float64)
    fake_label = 1 - real_label
    anomaly = 1.0 - realness
    fpr, tpr, _ = roc_curve(fake_label, anomaly)
    fnr = 1.0 - tpr
    index = int(np.argmin(np.abs(fpr - fnr)))
    predicted_real = realness >= 0.5
    real_recall = float(predicted_real[real_label == 1].mean())
    fake_recall = float((~predicted_real[real_label == 0]).mean())

    def tpr_at(target: float) -> float:
        eligible = tpr[fpr <= target]
        return float(eligible.max()) if len(eligible) else 0.0

    return {
        "auc": float(auc(fpr, tpr)),
        "fake_positive_ap": float(average_precision_score(fake_label, anomaly)),
        "real_positive_ap": float(average_precision_score(real_label, realness)),
        "balanced_accuracy_at_0p5": 0.5 * (real_recall + fake_recall),
        "fake_tpr_at_1pct_real_fpr": tpr_at(0.01),
        "fake_tpr_at_5pct_real_fpr": tpr_at(0.05),
        "eer": float(0.5 * (fpr[index] + fnr[index])),
    }


def repeat_by_count(frame: pd.DataFrame, counts: dict[str, int]) -> pd.DataFrame:
    """Repeat video rows according to cluster-bootstrap multiplicities."""

    pieces = [
        frame.loc[[index] * counts[str(video_id)]]
        for index, video_id in zip(frame.index, frame["video_id"])
        if counts.get(str(video_id), 0)
    ]
    if not pieces:
        raise ValueError("cluster bootstrap produced an empty class")
    return pd.concat(pieces, ignore_index=True)


def macro_cluster_bootstrap(
    per_video: pd.DataFrame,
    candidates: list[str],
    seed: int,
    iterations: int,
    base_config: str = "MW0",
) -> pd.DataFrame:
    """Bootstrap paired generator deltas and reduce them to Macro-3."""

    dataset_pairs: list[list[pd.DataFrame]] = []
    for _, dataset_frame in per_video.groupby("dataset", sort=False):
        pairs = list(pairwise_frames(dataset_frame, seed).values())
        if pairs:
            dataset_pairs.append(pairs)
    if not dataset_pairs:
        raise ValueError("macro bootstrap requires at least one dataset with generator pairs")
    rows: list[dict] = []
    for candidate in candidates:
        point = {"auc": [], "ap": []}
        split_datasets: list[list[tuple[pd.DataFrame, pd.DataFrame]]] = []
        for pairs in dataset_pairs:
            dataset_point = {"auc": [], "ap": []}
            split_pairs: list[tuple[pd.DataFrame, pd.DataFrame]] = []
            for pair in pairs:
                new_auc, new_ap = auc_ap(pair, candidate)
                old_auc, old_ap = auc_ap(pair, base_config)
                dataset_point["auc"].append(new_auc - old_auc)
                dataset_point["ap"].append(new_ap - old_ap)
                split_pairs.append(
                    (
                        pair[pair["subset"] == "real"].reset_index(drop=True),
                        pair[pair["subset"] == "annotated"].reset_index(drop=True),
                    )
                )
            point["auc"].append(float(np.mean(dataset_point["auc"])))
            point["ap"].append(float(np.mean(dataset_point["ap"])))
            split_datasets.append(split_pairs)
        rng = np.random.default_rng(
            _stable_bootstrap_seed(seed, "Macro-3", candidate)
        )
        samples = {
            "auc": np.empty(iterations, dtype=np.float64),
            "ap": np.empty(iterations, dtype=np.float64),
        }
        for iteration in range(iterations):
            macro_deltas = {"auc": [], "ap": []}
            for split_pairs in split_datasets:
                dataset_deltas = {"auc": [], "ap": []}
                for real, fake in split_pairs:
                    sampled = pd.concat(
                        [
                            real.iloc[rng.integers(0, len(real), len(real))],
                            fake.iloc[rng.integers(0, len(fake), len(fake))],
                        ],
                        ignore_index=True,
                    )
                    new_auc, new_ap = auc_ap(sampled, candidate)
                    old_auc, old_ap = auc_ap(sampled, base_config)
                    dataset_deltas["auc"].append(new_auc - old_auc)
                    dataset_deltas["ap"].append(new_ap - old_ap)
                macro_deltas["auc"].append(float(np.mean(dataset_deltas["auc"])))
                macro_deltas["ap"].append(float(np.mean(dataset_deltas["ap"])))
            samples["auc"][iteration] = np.mean(macro_deltas["auc"])
            samples["ap"][iteration] = np.mean(macro_deltas["ap"])
        for metric in ("auc", "ap"):
            rows.append(
                {
                    "dataset": "Macro-3",
                    "comparison": f"{candidate}-{base_config}",
                    "new_config": candidate,
                    "base_config": base_config,
                    "metric": metric,
                    "delta": float(np.mean(point[metric])),
                    "ci95_low": float(np.quantile(samples[metric], 0.025)),
                    "ci95_high": float(np.quantile(samples[metric], 0.975)),
                    "bootstrap_iterations": iterations,
                }
            )
    return pd.DataFrame(rows)


__all__ = [
    "Score",
    "ScoreDirection",
    "build_results_table",
    "auc_ap",
    "binary_metrics",
    "get_results_df",
    "macro_cluster_bootstrap",
    "metric_tables",
    "paired_bootstrap",
    "pairwise_frames",
    "predictor_scalar2metrics",
    "print_results",
    "repeat_by_count",
    "sample_balanced_real",
    "stable_seed",
]
