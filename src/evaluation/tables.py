"""将规范化逐视频分数转换为数据集级和生成器级指标表。"""

from __future__ import annotations

import pandas as pd

from .metrics import binary_metrics


REQUIRED_SCORE_COLUMNS = {"video_id", "dataset", "subset", "source_model", "final_score"}


def normalize_scores(frame: pd.DataFrame) -> pd.DataFrame:
    missing = REQUIRED_SCORE_COLUMNS.difference(frame.columns)
    if missing:
        raise ValueError(f"逐视频分数 CSV 缺少字段：{sorted(missing)}")
    result = frame.copy()
    result["video_id"] = result["video_id"].astype(str)
    if result["video_id"].duplicated().any():
        raise ValueError("逐视频分数 CSV 含有重复 video_id")
    if not set(result["subset"].unique()).issubset({"real", "annotated"}):
        raise ValueError("subset 字段只能使用 real 或 annotated")
    result["final_score"] = pd.to_numeric(result["final_score"], errors="raise")
    return result


def build_metric_tables(scores: pd.DataFrame, run_name: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    dataset_rows: list[dict] = []
    generator_rows: list[dict] = []
    for dataset, dataset_frame in scores.groupby("dataset", sort=True):
        metrics = binary_metrics(dataset_frame, "final_score")
        dataset_rows.append(
            {
                "run_name": run_name,
                "dataset": dataset,
                "auc": metrics["auc"],
                "ap_real": metrics["real_positive_ap"],
                "tpr_at_1pct_fpr": metrics["fake_tpr_at_1pct_real_fpr"],
                "n_real": int(dataset_frame["subset"].eq("real").sum()),
                "n_fake": int(dataset_frame["subset"].eq("annotated").sum()),
                "n_generators": int(dataset_frame.loc[dataset_frame["subset"].eq("annotated"), "source_model"].nunique()),
            }
        )
        real = dataset_frame[dataset_frame["subset"].eq("real")]
        for generator, fake in dataset_frame[dataset_frame["subset"].eq("annotated")].groupby("source_model", sort=True):
            paired = pd.concat([real, fake], ignore_index=True)
            values = binary_metrics(paired, "final_score")
            generator_rows.append(
                {
                    "run_name": run_name,
                    "dataset": dataset,
                    "generator": generator,
                    "auc": values["auc"],
                    "ap_real": values["real_positive_ap"],
                    "tpr_at_1pct_fpr": values["fake_tpr_at_1pct_real_fpr"],
                    "n_real": len(real),
                    "n_fake": len(fake),
                }
            )
    return pd.DataFrame(dataset_rows), pd.DataFrame(generator_rows)
