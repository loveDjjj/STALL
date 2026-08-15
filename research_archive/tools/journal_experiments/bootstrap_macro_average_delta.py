"""对生成器宏平均指标做 paired bootstrap ΔAUC/ΔAP 置信区间。

已有 ``paired_bootstrap_delta_summary.csv`` 是逐生成器口径。本脚本进一步在
每次 bootstrap 中先计算每个生成器的 balanced real/fake AUC/AP，再对生成器
取宏平均，得到与论文 Average 行一致的总体差异不确定性。

该脚本只读取 ``results/paper_scores``，不重新运行模型或提取特征。
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score


DATASETS = ("comgenvid", "videofeedback", "genvideo")
KEY_COLUMNS = ["subset", "source_model", "filename"]
COMPARISONS = (
    ("alpha_minus_global", "alpha_score", "global_score"),
    ("alpha_minus_patch", "alpha_score", "patch_score"),
    ("patch_minus_global", "patch_score", "global_score"),
)
DISPLAY = {
    "comgenvid": "ComGenVid",
    "videofeedback": "VideoFeedback",
    "genvideo": "GenVideo",
}


def _read_score(root: Path, dataset: str, kind: str) -> pd.DataFrame:
    return pd.read_csv(root / f"results/paper_scores/{dataset}_{kind}.csv")


def _aligned_method_scores(root: Path, dataset: str) -> pd.DataFrame:
    global_df = _read_score(root, dataset, "global")[KEY_COLUMNS + ["final_score"]].rename(
        columns={"final_score": "global_score"}
    )
    patch_df = _read_score(root, dataset, "patch_second_order")[KEY_COLUMNS + ["patch_final_score"]].rename(
        columns={"patch_final_score": "patch_score"}
    )
    alpha_df = _read_score(root, dataset, "alpha_stalled")[KEY_COLUMNS + ["final_score"]].rename(
        columns={"final_score": "alpha_score"}
    )
    return global_df.merge(patch_df, on=KEY_COLUMNS, validate="one_to_one").merge(
        alpha_df, on=KEY_COLUMNS, validate="one_to_one"
    )


def _sample_balanced_real_with_replacement(real_df: pd.DataFrame, n: int, rng: np.random.Generator) -> pd.DataFrame:
    """尽量按真实视频来源均衡采样，与主 metrics 的 balanced pairwise 口径一致。"""

    groups = list(real_df.groupby("source_model"))
    if not groups:
        raise ValueError("缺少真实视频样本")
    base = n // len(groups)
    remainder = n % len(groups)
    pieces = []
    for i, (_, group) in enumerate(groups):
        take = base + (1 if i < remainder else 0)
        pieces.append(group.iloc[rng.integers(0, len(group), size=take)])
    return pd.concat(pieces, ignore_index=True)


def _metrics_for_sample(sample: pd.DataFrame, score_col: str) -> tuple[float, float]:
    labels = (sample["subset"] == "real").astype(np.uint8).to_numpy()
    scores = sample[score_col].astype(float).to_numpy()
    return float(roc_auc_score(labels, scores)), float(average_precision_score(labels, scores))


def bootstrap_dataset(df: pd.DataFrame, n_boot: int, seed: int) -> pd.DataFrame:
    real_df = df[df["subset"] == "real"].reset_index(drop=True)
    source_models = sorted(df[df["subset"] != "real"]["source_model"].unique())
    rng = np.random.default_rng(seed)
    store = {name: {"auc": [], "ap": []} for name, _, _ in COMPARISONS}

    for _ in range(n_boot):
        per_model_metrics: dict[str, dict[str, list[float]]] = {
            "global_score": {"auc": [], "ap": []},
            "patch_score": {"auc": [], "ap": []},
            "alpha_score": {"auc": [], "ap": []},
        }
        for source_model in source_models:
            fake_df = df[(df["subset"] != "real") & (df["source_model"] == source_model)].reset_index(drop=True)
            n = len(fake_df)
            fake_sample = fake_df.iloc[rng.integers(0, len(fake_df), size=n)]
            real_sample = _sample_balanced_real_with_replacement(real_df, n, rng)
            sample = pd.concat([real_sample, fake_sample], ignore_index=True)
            for score_col in per_model_metrics:
                auc, ap = _metrics_for_sample(sample, score_col)
                per_model_metrics[score_col]["auc"].append(auc)
                per_model_metrics[score_col]["ap"].append(ap)

        macro = {
            score_col: (
                float(np.mean(values["auc"])),
                float(np.mean(values["ap"])),
            )
            for score_col, values in per_model_metrics.items()
        }
        for name, score_a, score_b in COMPARISONS:
            store[name]["auc"].append(macro[score_a][0] - macro[score_b][0])
            store[name]["ap"].append(macro[score_a][1] - macro[score_b][1])

    rows = []
    for name, values in store.items():
        auc = np.asarray(values["auc"], dtype=float)
        ap = np.asarray(values["ap"], dtype=float)
        rows.append(
            {
                "comparison": name,
                "n_boot": n_boot,
                "delta_auc_mean": float(auc.mean()),
                "delta_auc_ci_low": float(np.quantile(auc, 0.025)),
                "delta_auc_ci_high": float(np.quantile(auc, 0.975)),
                "delta_ap_mean": float(ap.mean()),
                "delta_ap_ci_low": float(np.quantile(ap, 0.025)),
                "delta_ap_ci_high": float(np.quantile(ap, 0.975)),
            }
        )
    return pd.DataFrame(rows)


def build_summary(root: Path, n_boot: int, seed: int) -> pd.DataFrame:
    frames = []
    for i, dataset in enumerate(DATASETS):
        df = _aligned_method_scores(root, dataset)
        summary = bootstrap_dataset(df, n_boot=n_boot, seed=seed + i * 1009)
        summary.insert(0, "dataset", dataset)
        frames.append(summary)
    return pd.concat(frames, ignore_index=True)


def write_markdown(summary: pd.DataFrame, out_path: Path) -> None:
    lines = [
        "# 生成器宏平均 paired bootstrap ΔAUC/ΔAP",
        "",
        "本分析只读取 `results/paper_scores/`，不重新运行模型。",
        "每个 bootstrap 轮次内，先对每个生成器分别做 balanced real/fake 重采样并计算 AUC/AP，",
        "再对生成器取宏平均，最后报告方法差值的 95% CI。",
        "",
        "| dataset | comparison | ΔAUC mean | ΔAUC 95% CI | ΔAP mean | ΔAP 95% CI | 判定 |",
        "|---|---|---:|---:|---:|---:|---|",
    ]
    for row in summary.itertuples(index=False):
        if row.delta_auc_ci_low > 0:
            decision = "AUC CI > 0"
        elif row.delta_auc_ci_high < 0:
            decision = "AUC CI < 0"
        else:
            decision = "AUC CI crosses 0"
        lines.append(
            f"| {DISPLAY[row.dataset]} | {row.comparison} | {row.delta_auc_mean:+.4f} | "
            f"[{row.delta_auc_ci_low:+.4f}, {row.delta_auc_ci_high:+.4f}] | "
            f"{row.delta_ap_mean:+.4f} | [{row.delta_ap_ci_low:+.4f}, {row.delta_ap_ci_high:+.4f}] | {decision} |"
        )

    alpha = summary[summary["comparison"] == "alpha_minus_global"]
    lines.extend(
        [
            "",
            "## 关键结论",
            "",
        ]
    )
    for row in alpha.itertuples(index=False):
        if row.delta_auc_ci_low > 0:
            conclusion = "宏平均 AUC 提升在 95% CI 下为正。"
        elif row.delta_auc_ci_high < 0:
            conclusion = "宏平均 AUC 在 95% CI 下下降。"
        else:
            conclusion = "宏平均 AUC CI 跨 0，应避免写成显著总体提升。"
        lines.append(
            f"- {DISPLAY[row.dataset]}: Alpha-STALLED 相对 global-only ΔAUC={row.delta_auc_mean:+.4f} "
            f"({row.delta_auc_ci_low:+.4f}, {row.delta_auc_ci_high:+.4f})，{conclusion}"
        )
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def plot_alpha_delta(summary: pd.DataFrame, out_stem: Path) -> None:
    plot_df = summary[summary["comparison"] == "alpha_minus_global"].copy()
    if plot_df.empty:
        return
    fig, ax = plt.subplots(figsize=(5.6, 3.2))
    x = np.arange(len(plot_df))
    means = plot_df["delta_auc_mean"].to_numpy()
    low = plot_df["delta_auc_ci_low"].to_numpy()
    high = plot_df["delta_auc_ci_high"].to_numpy()
    yerr = np.vstack([means - low, high - means])
    ax.errorbar(x, means, yerr=yerr, fmt="o", color="#0072B2", capsize=4)
    ax.axhline(0.0, color="#4D4D4D", linewidth=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels([DISPLAY[d] for d in plot_df["dataset"]])
    ax.set_ylabel("Macro-average ΔAUC vs global-only")
    ax.set_title("Macro-average paired bootstrap: Alpha-STALLED gains")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    out_stem.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_stem.with_suffix(".svg"))
    fig.savefig(out_stem.with_suffix(".png"), dpi=220)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[3])
    parser.add_argument("--n-boot", type=int, default=500)
    parser.add_argument("--seed", type=int, default=20260720)
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path(__file__).resolve().parents[3]
        / "results/journal_experiments/macro_average_bootstrap",
    )
    args = parser.parse_args()
    if args.n_boot < 10:
        raise ValueError("--n-boot 应至少为 10")
    root = args.root.resolve()
    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = build_summary(root, n_boot=args.n_boot, seed=args.seed)
    summary.to_csv(out_dir / "macro_average_paired_bootstrap_delta.csv", index=False)
    write_markdown(summary, out_dir / "macro_average_paired_bootstrap_delta.md")
    plot_alpha_delta(summary, root / "results/paper_figures/macro_average_bootstrap_alpha_delta")
    print(f"saved -> {out_dir}")


if __name__ == "__main__":
    main()
