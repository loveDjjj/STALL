"""将规范化逐视频分数转换为诊断表与论文口径指标表。"""

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
                "tpr_at_0_1pct_fpr": metrics["fake_tpr_at_0_1pct_real_fpr"],
                "tpr_at_1pct_fpr": metrics["fake_tpr_at_1pct_real_fpr"],
                "fpr_at_95pct_tpr": metrics["real_fpr_at_95pct_fake_tpr"],
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
                    "tpr_at_0_1pct_fpr": values["fake_tpr_at_0_1pct_real_fpr"],
                    "tpr_at_1pct_fpr": values["fake_tpr_at_1pct_real_fpr"],
                    "fpr_at_95pct_tpr": values["real_fpr_at_95pct_fake_tpr"],
                    "n_real": len(real),
                    "n_fake": len(fake),
                }
            )
    return pd.DataFrame(dataset_rows), pd.DataFrame(generator_rows)


def _balanced_real_pair(
    real: pd.DataFrame, fake: pd.DataFrame, seed: int
) -> pd.DataFrame:
    """按锁定论文协议构建一个生成器的真实/生成平衡配对。

    真实视频有多个来源时，先为每个来源分配相同配额，再以固定种子打乱；
    这避免真实来源比例随生成器规模而改变。该规则与历史 U0 主表一致。
    """

    target = min(len(real), len(fake))
    if target == 0:
        raise ValueError("生成器配对同时需要真实与生成视频")
    groups = [
        group.sample(n=min(len(group), max(1, target // real["source_model"].nunique())), random_state=seed)
        for _, group in real.groupby("source_model", sort=True)
    ]
    sampled_real = pd.concat(groups, ignore_index=True).sample(
        n=min(target, sum(len(group) for group in groups)), random_state=seed
    )
    return pd.concat([sampled_real, fake.head(len(sampled_real))], ignore_index=True)


def build_pairwise_metric_table(
    scores: pd.DataFrame, run_name: str, pairwise_seed: int
) -> pd.DataFrame:
    """生成论文主表使用的生成器配对宏平均 AUC 与 real-positive AP。"""

    rows: list[dict] = []
    for dataset, dataset_frame in scores.groupby("dataset", sort=True):
        real = dataset_frame[dataset_frame["subset"].eq("real")]
        generator_rows = []
        for _, fake in dataset_frame[dataset_frame["subset"].eq("annotated")].groupby("source_model", sort=True):
            pair = _balanced_real_pair(real, fake, pairwise_seed)
            generator_rows.append((pair, binary_metrics(pair, "final_score")))
        if not generator_rows:
            raise ValueError(f"{dataset} 没有可用于论文配对指标的生成器")
        rows.append(
            {
                "run_name": run_name,
                "scope": "generator_pairwise_dataset_macro",
                "dataset": dataset,
                "auc": float(sum(item[1]["auc"] for item in generator_rows) / len(generator_rows)),
                "ap_real": float(sum(item[1]["real_positive_ap"] for item in generator_rows) / len(generator_rows)),
                "n_generators": len(generator_rows),
                "n_pairwise_real": int(sum(len(item[0][item[0]["subset"].eq("real")]) for item in generator_rows)),
                "n_pairwise_fake": int(sum(len(item[0][item[0]["subset"].eq("annotated")]) for item in generator_rows)),
                "pairwise_seed": pairwise_seed,
            }
        )
    table = pd.DataFrame(rows)
    macro = {
        "run_name": run_name,
        "scope": "generator_pairwise_macro3",
        "dataset": "Macro-3",
        "auc": float(table["auc"].mean()),
        "ap_real": float(table["ap_real"].mean()),
        "n_generators": int(table["n_generators"].sum()),
        "n_pairwise_real": int(table["n_pairwise_real"].sum()),
        "n_pairwise_fake": int(table["n_pairwise_fake"].sum()),
        "pairwise_seed": pairwise_seed,
    }
    return pd.concat([table, pd.DataFrame([macro])], ignore_index=True)
