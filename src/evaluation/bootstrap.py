"""基于视频样本的成对 bootstrap 统计入口。"""

from __future__ import annotations

import pandas as pd

from .metrics import paired_bootstrap


def compare_methods(
    scores: pd.DataFrame,
    *,
    candidate_column: str,
    baseline_column: str,
    seed: int,
    iterations: int,
) -> pd.DataFrame:
    """比较同一批视频上的两个分数字段，返回数据集级置信区间。"""

    return paired_bootstrap(
        scores,
        seed=seed,
        iterations=iterations,
        comparisons=((candidate_column, baseline_column, f"{candidate_column}_vs_{baseline_column}"),),
    )


__all__ = ["compare_methods"]
