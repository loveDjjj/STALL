"""汇总 ComGenVid patch temporal derivative order 对照实验。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]


CONFIGS = [
    {
        "order": 2,
        "mode": "same_grid_second_order",
        "score_csv": "results/paper_scores/comgenvid_patch_second_order.csv",
        "metrics_csv": "results/paper_tables/comgenvid_patch_only_metrics.csv",
        "params": "precomputed/patch_params_comgenvid_real_second_order_region3_bottomk0p20_v2.npz",
        "role": "current Alpha-STALLED patch temporal branch",
    },
    {
        "order": 3,
        "mode": "same_grid_third_order",
        "score_csv": "results/journal_experiments/temporal_derivative_order/comgenvid_patch_third_order.csv",
        "metrics_csv": "results/journal_experiments/temporal_derivative_order/comgenvid_patch_third_order_metrics.csv",
        "params": "precomputed/patch_params_comgenvid_real_same_grid_third_order_region3_bottomk0p20_v2.npz",
        "role": "higher-order derivative control",
    },
    {
        "order": 4,
        "mode": "same_grid_fourth_order",
        "score_csv": "results/journal_experiments/temporal_derivative_order/comgenvid_patch_fourth_order.csv",
        "metrics_csv": "results/journal_experiments/temporal_derivative_order/comgenvid_patch_fourth_order_metrics.csv",
        "params": "precomputed/patch_params_comgenvid_real_same_grid_fourth_order_region3_bottomk0p20_v2.npz",
        "role": "higher-order derivative control",
    },
]


def _average_metrics(path: Path) -> tuple[float, float, int, int]:
    df = pd.read_csv(path)
    row = df[df["Generative Model"] == "Average"].iloc[0]
    return (
        float(row["patch_final_score AUC"]),
        float(row["patch_final_score AP"]),
        int(row["n_real"]),
        int(row["n_annotated"]),
    )


def _load_config(path: Path) -> dict:
    data = np.load(path, allow_pickle=True)
    value = data["aggregation_config"]
    if isinstance(value, np.ndarray):
        value = value.item()
    return json.loads(str(value))


def build_summary(root: Path, out_dir: Path) -> pd.DataFrame:
    rows = []
    baseline_auc = baseline_ap = None
    for config in CONFIGS:
        auc, ap, n_real, n_fake = _average_metrics(root / config["metrics_csv"])
        if config["order"] == 2:
            baseline_auc, baseline_ap = auc, ap
        agg_config = _load_config(root / config["params"])
        rows.append(
            {
                "dataset": "ComGenVid",
                "order": config["order"],
                "patch_temp_mode": config["mode"],
                "role": config["role"],
                "n_real": n_real,
                "n_generated": n_fake,
                "average_auc": auc,
                "average_ap": ap,
                "delta_auc_vs_order2": auc - baseline_auc if baseline_auc is not None else 0.0,
                "delta_ap_vs_order2": ap - baseline_ap if baseline_ap is not None else 0.0,
                "aggregation": agg_config.get("mode"),
                "bottomk_ratio": float(agg_config.get("bottomk_ratio")),
                "patch_region_size": int(agg_config.get("patch_region_size")),
                "score_csv": config["score_csv"],
                "metrics_csv": config["metrics_csv"],
                "params": config["params"],
            }
        )
    return pd.DataFrame(rows)


def write_markdown(summary: pd.DataFrame, out_dir: Path) -> None:
    lines = [
        "# ComGenVid temporal derivative order 对照",
        "",
        "本实验只在 ComGenVid 上补充 D=3/D=4 高阶同网格 patch temporal derivative 对照，用于回应“二阶局部时序证据是否随意选择”的审稿风险。",
        "实验不铺开三数据集；它沿用当前 ComGenVid patch 主线设置：2 秒 compact cache、真实视频校准、patch region=3、bottom-k=0.20、pairwise balanced AUC/AP。",
        "",
        "## 结果",
        "",
        "| order | patch mode | 平均 AUC | 平均 AP | ΔAUC vs D=2 | ΔAP vs D=2 | 角色 |",
        "|---:|---|---:|---:|---:|---:|---|",
    ]
    for row in summary.itertuples(index=False):
        lines.append(
            f"| {row.order} | `{row.patch_temp_mode}` | {row.average_auc:.4f} | {row.average_ap:.4f} | "
            f"{row.delta_auc_vs_order2:+.4f} | {row.delta_ap_vs_order2:+.4f} | {row.role} |"
        )
    best = summary.sort_values("average_auc", ascending=False).iloc[0]
    lines.extend(
        [
            "",
            "## 结论",
            "",
            f"- 当前主线 D=2 的平均 AUC/AP 为 {summary.iloc[0].average_auc:.4f} / {summary.iloc[0].average_ap:.4f}。",
            f"- D=3 和 D=4 均低于 D=2；其中 D=4 为 {summary.iloc[2].average_auc:.4f} / {summary.iloc[2].average_ap:.4f}，相对 D=2 的 ΔAUC={summary.iloc[2].delta_auc_vs_order2:+.4f}。",
            "- 结果支持保留二阶局部时序证据作为 Alpha-STALLED patch 分支主线：三阶/四阶差分会强调更高频的局部变化，但在当前 16 帧、region=3、bottom-k=0.20 设定下没有带来收益。",
            "- 该实验是代表性机制对照，不应写成三数据集鲁棒性结论；若审稿人要求完整 temporal derivative sweep，再扩展到 VideoFeedback/GenVideo。",
            "",
            "## 重建命令",
            "",
            "```bash",
            "conda run --no-capture-output -n stall python src/create_patch_params.py \\",
            "  --csv cache/indexes/comgenvid.csv \\",
            "  --patch-emb-cache cache/patch_embeddings/comgenvid \\",
            "  --output precomputed/patch_params_comgenvid_real_same_grid_third_order_region3_bottomk0p20_v2.npz \\",
            "  --duration 2 --compact --real-only --max-patches-for-fit 300000 \\",
            "  --aggregation bottomk_mean --bottomk-ratio 0.20 \\",
            "  --patch-temp-mode same_grid_third_order --patch-region-size 3",
            "",
            "conda run --no-capture-output -n stall python src/create_patch_params.py \\",
            "  --csv cache/indexes/comgenvid.csv \\",
            "  --patch-emb-cache cache/patch_embeddings/comgenvid \\",
            "  --output precomputed/patch_params_comgenvid_real_same_grid_fourth_order_region3_bottomk0p20_v2.npz \\",
            "  --duration 2 --compact --real-only --max-patches-for-fit 300000 \\",
            "  --aggregation bottomk_mean --bottomk-ratio 0.20 \\",
            "  --patch-temp-mode same_grid_fourth_order --patch-region-size 3",
            "```",
            "",
            "机器可读文件：",
            "",
            "- `comgenvid_temporal_derivative_order_summary.csv`",
            "- `comgenvid_patch_third_order.csv` / `comgenvid_patch_third_order_metrics.csv`",
            "- `comgenvid_patch_fourth_order.csv` / `comgenvid_patch_fourth_order_metrics.csv`",
            "- `comgenvid_temporal_derivative_order.svg` / `.png`",
        ]
    )
    (out_dir / "comgenvid_temporal_derivative_order.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def plot_summary(summary: pd.DataFrame, out_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(6.2, 3.4), constrained_layout=True)
    x = np.arange(len(summary))
    width = 0.34
    ax.bar(x - width / 2, summary["average_auc"], width, label="AUC", color="#4C78A8")
    ax.bar(x + width / 2, summary["average_ap"], width, label="AP", color="#F58518")
    ax.set_xticks(x, [f"D={int(v)}" for v in summary["order"]])
    ax.set_ylim(0.84, 0.94)
    ax.set_ylabel("ComGenVid average")
    ax.set_title("Patch temporal derivative order control")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(frameon=False)
    for idx, row in enumerate(summary.itertuples(index=False)):
        ax.text(idx - width / 2, row.average_auc + 0.002, f"{row.average_auc:.3f}", ha="center", va="bottom", fontsize=8)
        ax.text(idx + width / 2, row.average_ap + 0.002, f"{row.average_ap:.3f}", ha="center", va="bottom", fontsize=8)
    fig.savefig(out_dir / "comgenvid_temporal_derivative_order.svg")
    fig.savefig(out_dir / "comgenvid_temporal_derivative_order.png", dpi=240)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=ROOT / "results/journal_experiments/temporal_derivative_order",
    )
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    summary = build_summary(args.root, args.out_dir)
    summary.to_csv(args.out_dir / "comgenvid_temporal_derivative_order_summary.csv", index=False)
    write_markdown(summary, args.out_dir)
    plot_summary(summary, args.out_dir)
    print(summary[["order", "average_auc", "average_ap", "delta_auc_vs_order2", "delta_ap_vs_order2"]].to_string(index=False))
    print(f"saved -> {args.out_dir}")


if __name__ == "__main__":
    main()
