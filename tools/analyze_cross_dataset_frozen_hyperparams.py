"""基于已有结果审计 cross-dataset frozen hyperparameter 泛化性。

该脚本不重新评测模型，只读取已经落盘的 alpha/beta/region/aggregation
敏感性结果，回答一个期刊审稿更关心的问题：如果超参数不是在目标数据集上
oracle 选择，而是在其他数据集上选择后冻结，性能会损失多少。

选择协议：

1. transfer matrix：用 source dataset 上平均 AUC 最优的超参数，套到 target dataset；
2. leave-one-dataset-out：对每个 target，在其余两个数据集上取平均 AUC 最优超参数，
   再报告 target 上的 frozen performance 和 target oracle gap。

所有 gap 定义为 frozen - target_oracle；因此负值表示冻结配置低于目标 oracle。
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


DATASETS = ("comgenvid", "videofeedback", "genvideo")
DISPLAY = {
    "comgenvid": "ComGenVid",
    "videofeedback": "VideoFeedback",
    "genvideo": "GenVideo",
}


def _read_alpha(root: Path) -> pd.DataFrame:
    frames = []
    for dataset in DATASETS:
        path = root / f"results/paper_sweeps/{dataset}_alpha/{dataset}_same_grid_second_order_summary.csv"
        df = pd.read_csv(path)
        frames.append(df[["dataset", "alpha", "avg_auc", "avg_ap", "n_rows"]].copy())
    return pd.concat(frames, ignore_index=True).assign(kind="alpha")


def _read_beta(root: Path) -> pd.DataFrame:
    path = root / "results/paper_sensitivity/beta_sensitivity_summary.csv"
    df = pd.read_csv(path)
    df = df[df["mode"] == "fused_alpha0p60"].copy()
    return df[["dataset", "beta", "avg_auc", "avg_ap", "n_rows"]].assign(kind="beta")


def _read_region(root: Path) -> pd.DataFrame:
    path = root / "results/journal_experiments/region_sensitivity/region_sensitivity_summary.csv"
    df = pd.read_csv(path)
    return df[["dataset", "patch_region_size", "avg_auc", "avg_ap"]].assign(kind="region")


def _read_aggregation(root: Path) -> pd.DataFrame:
    """读取每个数据集主 region 下 mean / bottom-k 聚合结果。

    ComGenVid 的 bottom-k 加密网格在 bottomk_sensitivity 目录；
    GenVideo 与 VideoFeedback 的 bottom-k 对照在 aggregation_sensitivity 目录。
    mean baseline 均从 region_sensitivity 的各数据集最优 region 中读取。
    """

    region = _read_region(root)
    best_regions = region.loc[region.groupby("dataset")["avg_auc"].idxmax()]
    rows = []
    for row in best_regions.itertuples(index=False):
        rows.append(
            {
                "dataset": row.dataset,
                "aggregation_choice": "mean",
                "patch_region_size": int(row.patch_region_size),
                "avg_auc": float(row.avg_auc),
                "avg_ap": float(row.avg_ap),
            }
        )

    for rel in (
        "results/journal_experiments/bottomk_sensitivity/bottomk_sensitivity_summary.csv",
        "results/journal_experiments/aggregation_sensitivity/aggregation_sensitivity_summary.csv",
    ):
        path = root / rel
        if not path.exists():
            continue
        df = pd.read_csv(path)
        df = df[df["aggregation"] == "bottomk_mean"].copy()
        df = df[df["bottomk_ratio"].isin([0.20, 0.50])]
        for row in df.itertuples(index=False):
            rows.append(
                {
                    "dataset": row.dataset,
                    "aggregation_choice": f"bottomk_{float(row.bottomk_ratio):.2f}",
                    "patch_region_size": int(row.patch_region_size),
                    "avg_auc": float(row.avg_auc),
                    "avg_ap": float(row.avg_ap),
                }
            )

    out = pd.DataFrame(rows)
    common = set.intersection(
        *[set(group["aggregation_choice"]) for _, group in out.groupby("dataset")]
    )
    return (
        out[out["aggregation_choice"].isin(common)]
        .drop_duplicates(["dataset", "aggregation_choice"], keep="first")
        .sort_values(["dataset", "aggregation_choice"])
        .reset_index(drop=True)
        .assign(kind="aggregation")
    )


def _transfer_matrix(df: pd.DataFrame, param_col: str) -> pd.DataFrame:
    rows = []
    for source in DATASETS:
        source_df = df[df["dataset"] == source]
        source_best = source_df.loc[source_df["avg_auc"].idxmax()]
        selected = source_best[param_col]
        source_oracle_auc = float(source_best["avg_auc"])
        for target in DATASETS:
            target_df = df[df["dataset"] == target]
            target_oracle = target_df.loc[target_df["avg_auc"].idxmax()]
            target_match = target_df[target_df[param_col] == selected]
            if target_match.empty:
                continue
            frozen = target_match.iloc[0]
            rows.append(
                {
                    "source_dataset": source,
                    "target_dataset": target,
                    "selected_param": selected,
                    "source_oracle_auc": source_oracle_auc,
                    "target_frozen_auc": float(frozen["avg_auc"]),
                    "target_frozen_ap": float(frozen["avg_ap"]),
                    "target_oracle_param": target_oracle[param_col],
                    "target_oracle_auc": float(target_oracle["avg_auc"]),
                    "target_oracle_ap": float(target_oracle["avg_ap"]),
                    "auc_gap_vs_target_oracle": float(frozen["avg_auc"] - target_oracle["avg_auc"]),
                    "ap_gap_vs_target_oracle": float(frozen["avg_ap"] - target_oracle["avg_ap"]),
                }
            )
    return pd.DataFrame(rows)


def _leave_one_out(df: pd.DataFrame, param_col: str) -> pd.DataFrame:
    rows = []
    params = sorted(df[param_col].unique())
    for target in DATASETS:
        train = [d for d in DATASETS if d != target]
        train_df = df[df["dataset"].isin(train)]
        train_mean = (
            train_df.groupby(param_col, as_index=False)
            .agg(train_mean_auc=("avg_auc", "mean"), train_mean_ap=("avg_ap", "mean"))
            .sort_values(["train_mean_auc", "train_mean_ap"], ascending=False)
        )
        # 只允许所有训练数据集都有结果的参数，避免不完整网格误导选择。
        valid_params = []
        for param in params:
            covered = set(train_df[train_df[param_col] == param]["dataset"])
            if covered == set(train):
                valid_params.append(param)
        train_mean = train_mean[train_mean[param_col].isin(valid_params)]
        selected = train_mean.iloc[0][param_col]
        target_df = df[df["dataset"] == target]
        target_frozen = target_df[target_df[param_col] == selected].iloc[0]
        target_oracle = target_df.loc[target_df["avg_auc"].idxmax()]
        rows.append(
            {
                "target_dataset": target,
                "train_datasets": "+".join(train),
                "selected_param": selected,
                "train_mean_auc": float(train_mean.iloc[0]["train_mean_auc"]),
                "target_frozen_auc": float(target_frozen["avg_auc"]),
                "target_frozen_ap": float(target_frozen["avg_ap"]),
                "target_oracle_param": target_oracle[param_col],
                "target_oracle_auc": float(target_oracle["avg_auc"]),
                "target_oracle_ap": float(target_oracle["avg_ap"]),
                "auc_gap_vs_target_oracle": float(target_frozen["avg_auc"] - target_oracle["avg_auc"]),
                "ap_gap_vs_target_oracle": float(target_frozen["avg_ap"] - target_oracle["avg_ap"]),
            }
        )
    return pd.DataFrame(rows)


def _fmt_param(value: object) -> str:
    if isinstance(value, float):
        if value.is_integer():
            return str(int(value))
        return f"{value:.2f}"
    return str(value)


def _write_markdown(results: dict[str, tuple[pd.DataFrame, pd.DataFrame]], out_path: Path) -> None:
    lines = [
        "# Cross-dataset frozen hyperparameter 分析",
        "",
        "本分析只复用已有敏感性结果，不重新提取特征、不重新评测视频。",
        "目标是区分目标数据集 oracle sweep 与跨数据集冻结配置的泛化性。",
        "",
        "选择准则统一为平均 AUC：transfer matrix 使用 source dataset 上的最优参数；",
        "leave-one-dataset-out 使用另外两个数据集平均 AUC 最优的参数，再报告目标数据集性能。",
        "",
    ]
    for name, (_, loo) in results.items():
        lines.extend(
            [
                f"## {name}",
                "",
                "| held-out target | train datasets | frozen param | frozen AUC/AP | target oracle param | oracle AUC/AP | ΔAUC | ΔAP |",
                "|---|---|---:|---:|---:|---:|---:|---:|",
            ]
        )
        for row in loo.itertuples(index=False):
            lines.append(
                f"| {DISPLAY[row.target_dataset]} | {row.train_datasets} | {_fmt_param(row.selected_param)} | "
                f"{row.target_frozen_auc:.4f} / {row.target_frozen_ap:.4f} | "
                f"{_fmt_param(row.target_oracle_param)} | {row.target_oracle_auc:.4f} / {row.target_oracle_ap:.4f} | "
                f"{row.auc_gap_vs_target_oracle:+.4f} | {row.ap_gap_vs_target_oracle:+.4f} |"
            )
        mean_abs_gap = loo["auc_gap_vs_target_oracle"].abs().mean()
        worst_gap = loo["auc_gap_vs_target_oracle"].min()
        lines.extend(
            [
                "",
                f"- 平均 |ΔAUC| = {mean_abs_gap:.4f}；最差 ΔAUC = {worst_gap:+.4f}。",
                "",
            ]
        )

    lines.extend(
        [
            "## 论文写作建议",
            "",
            "- alpha 的 frozen gap 直接反映全局 STALL 与 patch 二阶时序证据的融合比例是否可跨数据集迁移。",
            "- beta 的 frozen gap 用于说明 patch 内部空间证据只提供辅助作用，局部二阶时序仍是主贡献。",
            "- region 的 frozen gap 通常会更大，因为它对应局部时序证据的空间支持域，受视频分辨率、运动尺度和生成器类型影响。",
            "- aggregation 的 frozen gap 用于说明 mean 与 bottom-k 不是普适优劣关系；应将其写为数据集局部异常分布的边界分析，而不是事后重选默认配置。",
            "",
            "结论上，手稿应报告 oracle sweep 作为敏感性上界，同时给出 leave-one-dataset-out frozen 结果作为泛化性证据；",
            "默认配置不应声称是所有数据集的全局最优，而应描述为在不使用测试批次 rank、标签或生成器来源的前提下，将全局校准与局部二阶时序证据稳定结合的固定推理规则。",
        ]
    )
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _plot_loo(results: dict[str, tuple[pd.DataFrame, pd.DataFrame]], out_stem: Path) -> None:
    rows = []
    for name, (_, loo) in results.items():
        for row in loo.itertuples(index=False):
            rows.append(
                {
                    "hyperparam": name,
                    "target": DISPLAY[row.target_dataset],
                    "auc_gap": row.auc_gap_vs_target_oracle,
                }
            )
    df = pd.DataFrame(rows)
    if df.empty:
        return
    fig, ax = plt.subplots(figsize=(7.0, 3.5))
    labels = []
    values = []
    colors = []
    palette = {
        "alpha": "#0072B2",
        "beta": "#009E73",
        "region": "#D55E00",
        "aggregation": "#CC79A7",
    }
    for _, row in df.iterrows():
        labels.append(f"{row['hyperparam']}\n{row['target']}")
        values.append(row["auc_gap"])
        colors.append(palette.get(row["hyperparam"], "#4D4D4D"))
    ax.bar(range(len(values)), values, color=colors)
    ax.axhline(0.0, color="#4D4D4D", linewidth=0.8)
    ax.set_ylabel("AUC gap vs target oracle")
    ax.set_title("Leave-one-dataset-out frozen hyperparameter gap")
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    out_stem.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_stem.with_suffix(".svg"))
    fig.savefig(out_stem.with_suffix(".png"), dpi=220)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path(__file__).resolve().parents[1]
        / "results/journal_experiments/cross_dataset_frozen_hyperparams",
    )
    args = parser.parse_args()
    root = args.root.resolve()
    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    sources = {
        "alpha": (_read_alpha(root), "alpha"),
        "beta": (_read_beta(root), "beta"),
        "region": (_read_region(root), "patch_region_size"),
        "aggregation": (_read_aggregation(root), "aggregation_choice"),
    }
    results: dict[str, tuple[pd.DataFrame, pd.DataFrame]] = {}
    for name, (df, param_col) in sources.items():
        transfer = _transfer_matrix(df, param_col)
        loo = _leave_one_out(df, param_col)
        transfer.to_csv(out_dir / f"{name}_transfer_matrix.csv", index=False)
        loo.to_csv(out_dir / f"{name}_leave_one_out.csv", index=False)
        results[name] = (transfer, loo)

    _write_markdown(results, out_dir / "cross_dataset_frozen_hyperparams.md")
    _plot_loo(results, root / "results/paper_figures/cross_dataset_frozen_hyperparams")
    print(f"saved -> {out_dir}")


if __name__ == "__main__":
    main()
