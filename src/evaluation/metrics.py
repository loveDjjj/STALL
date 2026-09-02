"""Alpha STALL 结果表和方法比较所需的最小评测工具。"""

from __future__ import annotations

import hashlib

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score, roc_curve


def binary_metrics(frame: pd.DataFrame, score_column: str = "final_score") -> dict[str, float]:
    """计算真实视频为正类时的 AUC、AP 和固定 FPR 检出率。"""

    real = frame["subset"].eq("real").to_numpy(dtype=np.uint8)
    if len(real) == 0 or real.min() == real.max():
        raise ValueError("指标计算同时需要真实和生成视频")
    scores = frame[score_column].to_numpy(dtype=np.float64)
    fake = 1 - real
    anomaly = 1.0 - scores
    fpr, tpr, _ = roc_curve(fake, anomaly)

    def tpr_at(limit: float) -> float:
        eligible = tpr[fpr <= limit]
        return float(eligible.max()) if len(eligible) else 0.0

    def fpr_at(target_tpr: float) -> float:
        eligible = fpr[tpr >= target_tpr]
        return float(eligible.min()) if len(eligible) else 1.0

    return {
        "auc": float(roc_auc_score(real, scores)),
        "real_positive_ap": float(average_precision_score(real, scores)),
        "fake_tpr_at_0_1pct_real_fpr": tpr_at(0.001),
        "fake_tpr_at_1pct_real_fpr": tpr_at(0.01),
        "real_fpr_at_95pct_fake_tpr": fpr_at(0.95),
    }


def _seed(seed: int, *parts: str) -> int:
    digest = hashlib.sha256("\0".join((str(seed), *parts)).encode("utf-8")).digest()
    # Pandas 的 random_state 仍使用 MT19937，种子必须落在无符号 32 位范围内。
    return int.from_bytes(digest[:8], "little") % (2**32)


def _generator_pairs(frame: pd.DataFrame, seed: int) -> list[pd.DataFrame]:
    real = frame[frame["subset"].eq("real")]
    pairs: list[pd.DataFrame] = []
    for generator, fake in frame[frame["subset"].eq("annotated")].groupby("source_model", sort=True):
        count = min(len(real), len(fake))
        if count == 0:
            continue
        sampled_real = real.sample(n=count, random_state=_seed(seed, str(generator)))
        pairs.append(pd.concat([sampled_real, fake.head(count)], ignore_index=True))
    return pairs


def paired_bootstrap(
    per_video: pd.DataFrame,
    *,
    seed: int,
    iterations: int,
    comparisons: tuple[tuple[str, str, str], ...],
) -> pd.DataFrame:
    """按数据集、生成器配对执行视频级 bootstrap 差异估计。"""

    rows: list[dict[str, object]] = []
    for dataset, dataset_frame in per_video.groupby("dataset", sort=True):
        pairs = _generator_pairs(dataset_frame, seed)
        if not pairs:
            continue
        for candidate, baseline, label in comparisons:
            point = [
                binary_metrics(pair, candidate)["auc"] - binary_metrics(pair, baseline)["auc"]
                for pair in pairs
            ]
            rng = np.random.default_rng(_seed(seed, str(dataset), label))
            samples = np.empty(iterations, dtype=np.float64)
            for index in range(iterations):
                deltas = []
                for pair in pairs:
                    sampled = pair.iloc[rng.integers(0, len(pair), len(pair))]
                    if sampled["subset"].nunique() < 2:
                        continue
                    deltas.append(
                        binary_metrics(sampled, candidate)["auc"]
                        - binary_metrics(sampled, baseline)["auc"]
                    )
                samples[index] = np.mean(deltas) if deltas else np.nan
            finite = samples[np.isfinite(samples)]
            if not len(finite):
                raise ValueError("bootstrap 未生成有效重采样")
            rows.append({
                "dataset": dataset,
                "comparison": label,
                "metric": "auc",
                "delta": float(np.mean(point)),
                "ci95_low": float(np.quantile(finite, 0.025)),
                "ci95_high": float(np.quantile(finite, 0.975)),
                "bootstrap_iterations": iterations,
            })
    return pd.DataFrame(rows)
