#!/usr/bin/env python3
"""汇总期刊补充实跑实验的 metrics CSV。"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


COLORS = {
    "auc": "#0072B2",
    "ap": "#D55E00",
}


def _avg_from_metrics(path: Path) -> tuple[float, float]:
    metrics = pd.read_csv(path)
    avg = metrics[metrics["Generative Model"] == "Average"]
    if avg.empty:
        raise ValueError(f"{path} 缺少 Average 行")
    row = avg.iloc[0]
    auc_col = next(c for c in metrics.columns if c.endswith(" AUC"))
    ap_col = next(c for c in metrics.columns if c.endswith(" AP"))
    return float(row[auc_col]), float(row[ap_col])


def _rel(path: Path, root: Path) -> str:
    try:
        return str(path.resolve().relative_to(root.resolve()))
    except ValueError:
        return str(path)


def _save_metric_curve(
    df: pd.DataFrame,
    x_col: str,
    title: str,
    xlabel: str,
    output_stem: Path,
    group_col: str | None = None,
) -> None:
    """Save a compact AUC/AP sensitivity curve as SVG and PNG."""
    if df.empty:
        return
    output_stem.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(5.2, 3.4))
    if group_col is None:
        plot_groups = [("", df.sort_values(x_col))]
    else:
        plot_groups = [(str(key), group.sort_values(x_col)) for key, group in df.groupby(group_col)]

    for label, group in plot_groups:
        suffix = f" ({label})" if label else ""
        ax.plot(group[x_col], group["avg_auc"], marker="o", color=COLORS["auc"], label=f"AUC{suffix}")
        ax.plot(group[x_col], group["avg_ap"], marker="s", color=COLORS["ap"], label=f"AP{suffix}")
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Average score")
    ax.grid(True, alpha=0.25, linewidth=0.8)
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(output_stem.with_suffix(".svg"))
    fig.savefig(output_stem.with_suffix(".png"), dpi=200)
    plt.close(fig)


def summarize_bottomk(root: Path) -> pd.DataFrame:
    out_dir = root / "results/journal_experiments/bottomk_sensitivity"
    rows: list[dict] = []
    pattern = re.compile(r"(?P<dataset>.+)_region(?P<region>\d+)_bottomk(?P<tag>\d+p\d+)_metrics\.csv")
    for path in sorted(out_dir.glob("*_metrics.csv")):
        match = pattern.fullmatch(path.name)
        if not match:
            continue
        auc, ap = _avg_from_metrics(path)
        bottomk = float(match.group("tag").replace("p", "."))
        score_csv = path.with_name(path.name.replace("_metrics.csv", "_patch.csv"))
        rows.append(
            {
                "dataset": match.group("dataset"),
                "patch_region_size": int(match.group("region")),
                "aggregation": "bottomk_mean",
                "bottomk_ratio": bottomk,
                "score_csv": _rel(score_csv, root),
                "metrics_csv": _rel(path, root),
                "avg_auc": auc,
                "avg_ap": ap,
                "status": "journal_full_eval",
            }
        )

    release_metrics = root / "results/paper_tables/comgenvid_patch_only_metrics.csv"
    if release_metrics.exists():
        auc, ap = _avg_from_metrics(release_metrics)
        rows.append(
            {
                "dataset": "comgenvid",
                "patch_region_size": 3,
                "aggregation": "bottomk_mean",
                "bottomk_ratio": 0.20,
                "score_csv": "results/paper_scores/comgenvid_patch_second_order.csv",
                "metrics_csv": _rel(release_metrics, root),
                "avg_auc": auc,
                "avg_ap": ap,
                "status": "release_baseline",
            }
        )

    df = pd.DataFrame(rows)
    if df.empty:
        return df
    df = (
        df.drop_duplicates(["dataset", "patch_region_size", "aggregation", "bottomk_ratio"], keep="first")
        .sort_values(["dataset", "patch_region_size", "bottomk_ratio"])
        .reset_index(drop=True)
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_dir / "bottomk_sensitivity_summary.csv", index=False)
    _save_metric_curve(
        df,
        x_col="bottomk_ratio",
        title="ComGenVid bottom-k sensitivity",
        xlabel="Bottom-k ratio",
        output_stem=root / "results/paper_figures/comgenvid_bottomk_sensitivity",
    )
    lines = [
        "# Bottom-k 敏感性实跑进展",
        "",
        "本表只汇总已经完成的全量 patch eval。未完成的 bottom-k 点不填补、不插值。",
        "",
        "| dataset | region | aggregation | bottom-k | 平均 AUC | 平均 AP | 状态 |",
        "|---|---:|---|---:|---:|---:|---|",
    ]
    for row in df.itertuples(index=False):
        lines.append(
            f"| {row.dataset} | {row.patch_region_size} | {row.aggregation} | "
            f"{row.bottomk_ratio:.2f} | {row.avg_auc:.4f} | {row.avg_ap:.4f} | {row.status} |"
        )
    lines.append("")
    comgenvid = df[(df["dataset"] == "comgenvid") & (df["patch_region_size"] == 3)]
    if not comgenvid.empty:
        best_auc = comgenvid.loc[comgenvid["avg_auc"].idxmax()]
        best_ap = comgenvid.loc[comgenvid["avg_ap"].idxmax()]
        lines.append(
            "初步结论：ComGenVid region=3 下，当前已完成 bottom-k 点中，"
            f"AUC 最优为 bottomk={best_auc['bottomk_ratio']:.2f} "
            f"({best_auc['avg_auc']:.4f})，AP 最优为 bottomk={best_ap['bottomk_ratio']:.2f} "
            f"({best_ap['avg_ap']:.4f})。该结果用于敏感性分析，不改变 release 默认配置。"
        )
    (out_dir / "bottomk_sensitivity_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return df


def summarize_region(root: Path) -> pd.DataFrame:
    out_dir = root / "results/journal_experiments/region_sensitivity"
    rows: list[dict] = []
    pattern = re.compile(r"(?P<dataset>.+)_region(?P<region>\d+)_(?P<aggregation>mean)_metrics\.csv")
    for path in sorted(out_dir.glob("*_metrics.csv")):
        match = pattern.fullmatch(path.name)
        if not match:
            continue
        auc, ap = _avg_from_metrics(path)
        rows.append(
            {
                "dataset": match.group("dataset"),
                "patch_region_size": int(match.group("region")),
                "aggregation": match.group("aggregation"),
                "metrics_csv": _rel(path, root),
                "avg_auc": auc,
                "avg_ap": ap,
            }
        )
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    df = df.sort_values(["dataset", "patch_region_size"]).reset_index(drop=True)
    out_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_dir / "region_sensitivity_summary.csv", index=False)
    for dataset, group in df.groupby("dataset"):
        _save_metric_curve(
            group,
            x_col="patch_region_size",
            title=f"{dataset} region-size sensitivity",
            xlabel="Patch region size",
            output_stem=root / f"results/paper_figures/{dataset}_region_sensitivity",
        )
    lines = [
        "# Region size 敏感性实跑进展",
        "",
        "| dataset | region | aggregation | 平均 AUC | 平均 AP |",
        "|---|---:|---|---:|---:|",
    ]
    for row in df.itertuples(index=False):
        lines.append(
            f"| {row.dataset} | {row.patch_region_size} | {row.aggregation} | "
            f"{row.avg_auc:.4f} | {row.avg_ap:.4f} |"
        )
    (out_dir / "region_sensitivity_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return df


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--kind", choices=["all", "bottomk", "region"], default="all")
    args = parser.parse_args()
    root = args.root.resolve()
    if args.kind in {"all", "bottomk"}:
        bottomk = summarize_bottomk(root)
        print(f"bottomk rows: {len(bottomk)}")
    if args.kind in {"all", "region"}:
        region = summarize_region(root)
        print(f"region rows: {len(region)}")


if __name__ == "__main__":
    main()
