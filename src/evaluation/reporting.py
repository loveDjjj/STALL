"""从标准运行结果收集论文表所需的指标。"""

from __future__ import annotations

from pathlib import Path

import pandas as pd


def collect_dataset_metrics(run_directories: list[Path]) -> pd.DataFrame:
    """合并多个 run 的数据集级指标，并拒绝缺失的标准产物。"""

    frames: list[pd.DataFrame] = []
    for directory in run_directories:
        path = directory / "dataset_metrics.csv"
        if not path.is_file():
            raise FileNotFoundError(f"缺少数据集级指标：{path}")
        frames.append(pd.read_csv(path))
    if not frames:
        raise ValueError("至少需要一个已完成 run 才能汇总报表")
    return pd.concat(frames, ignore_index=True)


__all__ = ["collect_dataset_metrics"]
