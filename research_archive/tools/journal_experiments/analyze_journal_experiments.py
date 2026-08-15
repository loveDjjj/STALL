#!/usr/bin/env python3
"""生成期刊版补充实验分析和可视化资产。

本脚本只读取 ``results/paper_scores/`` 和已经生成的 alpha sweep，不重新提取
DINOv3 特征，也不访问原始视频。目标是基于现有基础消融继续补齐：

1. patch 内部空间/时序权重 beta 敏感性；
2. 固定 alpha 下的 beta 敏感性；
3. 逐生成器增益/退化，暴露 failure modes；
4. 真实/生成分数分布；
5. 可直接写入期刊手稿的中文分析报告。
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

REPO_ROOT = Path(__file__).resolve().parents[3]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from alpha_stalled.metrics import ScoreDirection, build_results_table


DATASETS = ("comgenvid", "videofeedback", "genvideo")
DISPLAY_NAMES = {
    "comgenvid": "ComGenVid",
    "videofeedback": "VideoFeedback",
    "genvideo": "GenVideo",
}
KEY_COLUMNS = ["subset", "source_model", "filename"]
PALETTE = {
    "auc": "#0072B2",
    "ap": "#D55E00",
    "global": "#4D4D4D",
    "patch": "#009E73",
    "alpha": "#CC79A7",
    "real": "#0072B2",
    "annotated": "#D55E00",
}


@dataclass(frozen=True)
class Paths:
    root: Path
    sensitivity_dir: Path
    figure_dir: Path


def _set_plot_style() -> None:
    plt.rcParams.update(
        {
            "figure.dpi": 140,
            "savefig.dpi": 300,
            "font.size": 8,
            "axes.titlesize": 9,
            "axes.labelsize": 8,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
            "legend.fontsize": 7,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.linewidth": 0.8,
            "lines.linewidth": 1.6,
            "svg.fonttype": "none",
        }
    )


def _metric_table(df: pd.DataFrame, score_col: str, seed: int) -> pd.DataFrame:
    metrics = build_results_table(
        df[["subset", "source_model", score_col]].copy(),
        {score_col: ScoreDirection.HIGHER_IS_REAL},
        seed=seed,
        skip_global_compare=True,
        verbose=False,
    )
    return metrics.rename(
        columns={
            "Generative Model": "source_model",
            f"{score_col} AUC": "auc",
            f"{score_col} AP": "ap",
            "n_annotated": "n_fake",
        }
    )


def _avg_row(metrics: pd.DataFrame) -> pd.Series:
    avg = metrics[metrics["source_model"] == "Average"]
    if avg.empty:
        raise ValueError("指标表缺少 Average 行")
    return avg.iloc[0]


def _read_score(root: Path, dataset: str, kind: str) -> pd.DataFrame:
    return pd.read_csv(root / f"results/paper_scores/{dataset}_{kind}.csv")


def _read_metrics(root: Path, dataset: str, kind: str) -> pd.DataFrame:
    return pd.read_csv(root / f"results/paper_tables/{dataset}_{kind}_metrics.csv")


def _ensure_dirs(paths: Paths) -> None:
    paths.sensitivity_dir.mkdir(parents=True, exist_ok=True)
    paths.figure_dir.mkdir(parents=True, exist_ok=True)


def run_beta_sweep(paths: Paths, betas: list[float], alpha: float, seed: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    """扫描 patch_score = beta * spatial + (1 - beta) * temporal。"""
    summary_rows: list[dict] = []
    per_model_tables: list[pd.DataFrame] = []

    for dataset in DATASETS:
        patch = _read_score(paths.root, dataset, "patch_second_order")
        global_df = _read_score(paths.root, dataset, "global")

        required_patch = KEY_COLUMNS + ["patch_spat_percentile", "patch_temp_percentile"]
        missing = [c for c in required_patch if c not in patch.columns]
        if missing:
            raise ValueError(f"{dataset} patch CSV 缺少列: {missing}")
        missing = [c for c in KEY_COLUMNS + ["final_score"] if c not in global_df.columns]
        if missing:
            raise ValueError(f"{dataset} global CSV 缺少列: {missing}")

        merged = patch[required_patch].merge(
            global_df[KEY_COLUMNS + ["final_score"]].rename(columns={"final_score": "global_score"}),
            on=KEY_COLUMNS,
            how="inner",
            validate="one_to_one",
        )
        if len(merged) != len(patch) or len(merged) != len(global_df):
            print(
                f"警告: {dataset} beta sweep 只使用交集 {len(merged)} 行；"
                f"patch={len(patch)} global={len(global_df)}"
            )

        for beta in betas:
            scored = merged.copy()
            scored["patch_beta_score"] = (
                beta * scored["patch_spat_percentile"]
                + (1.0 - beta) * scored["patch_temp_percentile"]
            )
            scored["fused_beta_score"] = (
                alpha * scored["global_score"] + (1.0 - alpha) * scored["patch_beta_score"]
            )

            for mode, score_col in (
                ("patch_only", "patch_beta_score"),
                ("fused_alpha0p60", "fused_beta_score"),
            ):
                metrics = _metric_table(scored, score_col, seed)
                metrics.insert(0, "mode", mode)
                metrics.insert(0, "beta", beta)
                metrics.insert(0, "dataset", dataset)
                per_model_tables.append(metrics)
                avg = _avg_row(metrics)
                summary_rows.append(
                    {
                        "dataset": dataset,
                        "mode": mode,
                        "beta": beta,
                        "alpha": alpha if mode == "fused_alpha0p60" else np.nan,
                        "avg_auc": float(avg["auc"]),
                        "avg_ap": float(avg["ap"]),
                        "n_rows": len(scored),
                    }
                )

    summary = pd.DataFrame(summary_rows)
    per_model = pd.concat(per_model_tables, ignore_index=True)
    summary.to_csv(paths.sensitivity_dir / "beta_sensitivity_summary.csv", index=False)
    per_model.to_csv(paths.sensitivity_dir / "beta_sensitivity_per_model.csv", index=False)
    return summary, per_model


def build_component_delta(paths: Paths) -> pd.DataFrame:
    rows: list[dict] = []
    for dataset in DATASETS:
        global_m = _read_metrics(paths.root, dataset, "global_only")
        patch_m = _read_metrics(paths.root, dataset, "patch_only")
        alpha_m = _read_metrics(paths.root, dataset, "alpha_stalled")
        global_m = global_m.rename(
            columns={
                "final_score AUC": "global_auc",
                "final_score AP": "global_ap",
            }
        )
        patch_m = patch_m.rename(
            columns={
                "patch_final_score AUC": "patch_auc",
                "patch_final_score AP": "patch_ap",
            }
        )
        alpha_m = alpha_m.rename(
            columns={
                "final_score AUC": "alpha_auc",
                "final_score AP": "alpha_ap",
            }
        )
        merged = global_m.merge(patch_m, on="Generative Model").merge(alpha_m, on="Generative Model")
        for _, row in merged.iterrows():
            model = row["Generative Model"]
            if model in {"All", "Average", "AUC Flipped?"}:
                continue
            global_auc = row["global_auc"]
            patch_auc = row["patch_auc"]
            alpha_auc = row["alpha_auc"]
            global_ap = row["global_ap"]
            patch_ap = row["patch_ap"]
            alpha_ap = row["alpha_ap"]
            rows.append(
                {
                    "dataset": dataset,
                    "source_model": model,
                    "global_auc": global_auc,
                    "patch_auc": patch_auc,
                    "alpha_auc": alpha_auc,
                    "global_ap": global_ap,
                    "patch_ap": patch_ap,
                    "alpha_ap": alpha_ap,
                    "patch_minus_global_auc": patch_auc - global_auc,
                    "alpha_minus_global_auc": alpha_auc - global_auc,
                    "alpha_minus_patch_auc": alpha_auc - patch_auc,
                    "patch_minus_global_ap": patch_ap - global_ap,
                    "alpha_minus_global_ap": alpha_ap - global_ap,
                    "alpha_minus_patch_ap": alpha_ap - patch_ap,
                }
            )
    out = pd.DataFrame(rows)
    out.to_csv(paths.sensitivity_dir / "component_per_generator_delta.csv", index=False)
    return out


def build_score_distribution_summary(paths: Paths) -> pd.DataFrame:
    rows: list[dict] = []
    for dataset in DATASETS:
        alpha_scores = _read_score(paths.root, dataset, "alpha_stalled")
        for (subset, source_model), group in alpha_scores.groupby(["subset", "source_model"]):
            values = group["final_score"].astype(float)
            rows.append(
                {
                    "dataset": dataset,
                    "subset": subset,
                    "source_model": source_model,
                    "n": len(group),
                    "mean": values.mean(),
                    "std": values.std(ddof=1),
                    "q05": values.quantile(0.05),
                    "q25": values.quantile(0.25),
                    "median": values.quantile(0.50),
                    "q75": values.quantile(0.75),
                    "q95": values.quantile(0.95),
                }
            )
    out = pd.DataFrame(rows)
    out.to_csv(paths.sensitivity_dir / "alpha_score_distribution_summary.csv", index=False)
    return out


def _sample_balanced_real_with_replacement(real_df: pd.DataFrame, n: int, rng: np.random.Generator) -> pd.DataFrame:
    sources = list(real_df["source_model"].unique())
    base = n // len(sources)
    remainder = n - base * len(sources)
    parts = []
    for idx, source in enumerate(sources):
        quota = base + (1 if idx < remainder else 0)
        if quota <= 0:
            continue
        group = real_df[real_df["source_model"] == source]
        take = rng.integers(0, len(group), size=quota)
        parts.append(group.iloc[take])
    return pd.concat(parts, ignore_index=True)


def _bootstrap_metric_ci(
    df: pd.DataFrame,
    score_col: str,
    source_model: str,
    n_boot: int,
    seed: int,
) -> dict[str, float]:
    real_df = df[df["subset"] == "real"].reset_index(drop=True)
    fake_df = df[(df["subset"] != "real") & (df["source_model"] == source_model)].reset_index(drop=True)
    if real_df.empty or fake_df.empty:
        raise ValueError(f"{source_model} 缺少 real 或 fake 样本，无法 bootstrap")

    rng = np.random.default_rng(seed)
    aucs: list[float] = []
    aps: list[float] = []
    n = len(fake_df)
    for _ in range(n_boot):
        fake_take = rng.integers(0, len(fake_df), size=n)
        fake_sample = fake_df.iloc[fake_take]
        real_sample = _sample_balanced_real_with_replacement(real_df, n, rng)
        scores = pd.concat([real_sample[score_col], fake_sample[score_col]], ignore_index=True).astype(float)
        labels = np.r_[np.ones(len(real_sample), dtype=np.uint8), np.zeros(len(fake_sample), dtype=np.uint8)]
        aucs.append(float(roc_auc_score(labels, scores)))
        aps.append(float(average_precision_score(labels, scores)))

    auc_arr = np.array(aucs)
    ap_arr = np.array(aps)
    return {
        "auc_mean": float(auc_arr.mean()),
        "auc_ci_low": float(np.quantile(auc_arr, 0.025)),
        "auc_ci_high": float(np.quantile(auc_arr, 0.975)),
        "ap_mean": float(ap_arr.mean()),
        "ap_ci_low": float(np.quantile(ap_arr, 0.025)),
        "ap_ci_high": float(np.quantile(ap_arr, 0.975)),
    }


def build_bootstrap_ci(paths: Paths, n_boot: int, seed: int) -> pd.DataFrame:
    """用已有逐视频分数估计逐生成器 AUC/AP 95% bootstrap CI。"""
    rows: list[dict] = []
    method_specs = [
        ("global_only", "global", "final_score"),
        ("patch_only", "patch_second_order", "patch_final_score"),
        ("alpha_stalled", "alpha_stalled", "final_score"),
    ]
    for dataset in DATASETS:
        for method, score_kind, score_col in method_specs:
            df = _read_score(paths.root, dataset, score_kind)
            for source_model in sorted(df[df["subset"] != "real"]["source_model"].unique()):
                ci = _bootstrap_metric_ci(
                    df=df,
                    score_col=score_col,
                    source_model=source_model,
                    n_boot=n_boot,
                    seed=seed,
                )
                rows.append(
                    {
                        "dataset": dataset,
                        "method": method,
                        "source_model": source_model,
                        "n_boot": n_boot,
                        **ci,
                    }
                )
    out = pd.DataFrame(rows)
    out.to_csv(paths.sensitivity_dir / "bootstrap_ci_summary.csv", index=False)
    return out


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
    merged = global_df.merge(patch_df, on=KEY_COLUMNS, validate="one_to_one").merge(
        alpha_df,
        on=KEY_COLUMNS,
        validate="one_to_one",
    )
    return merged


def build_paired_bootstrap_delta(paths: Paths, n_boot: int, seed: int) -> pd.DataFrame:
    """同一次 resampling 下估计方法间 ΔAUC/ΔAP 的 paired bootstrap CI。"""
    rows: list[dict] = []
    comparisons = [
        ("alpha_minus_global", "alpha_score", "global_score"),
        ("alpha_minus_patch", "alpha_score", "patch_score"),
        ("patch_minus_global", "patch_score", "global_score"),
    ]
    for dataset in DATASETS:
        df = _aligned_method_scores(paths.root, dataset)
        real_df = df[df["subset"] == "real"].reset_index(drop=True)
        rng = np.random.default_rng(seed)
        for source_model in sorted(df[df["subset"] != "real"]["source_model"].unique()):
            fake_df = df[(df["subset"] != "real") & (df["source_model"] == source_model)].reset_index(drop=True)
            n = len(fake_df)
            delta_store = {
                name: {"auc": [], "ap": []}
                for name, _, _ in comparisons
            }
            for _ in range(n_boot):
                fake_sample = fake_df.iloc[rng.integers(0, len(fake_df), size=n)]
                real_sample = _sample_balanced_real_with_replacement(real_df, n, rng)
                sample = pd.concat([real_sample, fake_sample], ignore_index=True)
                labels = (sample["subset"] == "real").astype(np.uint8).to_numpy()
                method_metrics: dict[str, tuple[float, float]] = {}
                for score_col in ("global_score", "patch_score", "alpha_score"):
                    scores = sample[score_col].astype(float).to_numpy()
                    method_metrics[score_col] = (
                        float(roc_auc_score(labels, scores)),
                        float(average_precision_score(labels, scores)),
                    )
                for name, score_a, score_b in comparisons:
                    delta_store[name]["auc"].append(method_metrics[score_a][0] - method_metrics[score_b][0])
                    delta_store[name]["ap"].append(method_metrics[score_a][1] - method_metrics[score_b][1])

            for name in delta_store:
                auc = np.array(delta_store[name]["auc"])
                ap = np.array(delta_store[name]["ap"])
                rows.append(
                    {
                        "dataset": dataset,
                        "source_model": source_model,
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
    out = pd.DataFrame(rows)
    out.to_csv(paths.sensitivity_dir / "paired_bootstrap_delta_summary.csv", index=False)
    return out


def build_failure_candidate_table(paths: Paths, top_k: int = 20) -> pd.DataFrame:
    """列出最值得做案例可视化的负迁移/冲突样本候选。"""
    rows: list[pd.DataFrame] = []
    for dataset in DATASETS:
        df = _aligned_method_scores(paths.root, dataset)
        df["alpha_minus_global_score"] = df["alpha_score"] - df["global_score"]
        df["patch_minus_global_score"] = df["patch_score"] - df["global_score"]
        df["alpha_minus_patch_score"] = df["alpha_score"] - df["patch_score"]

        generated = df[df["subset"] != "real"].copy()
        real = df[df["subset"] == "real"].copy()
        categories = [
            (
                "generated_score_increased_by_alpha",
                generated.sort_values("alpha_minus_global_score", ascending=False).head(top_k),
                "生成视频被 Alpha-STALLED 打得更像真实，可能削弱检测。",
            ),
            (
                "real_score_decreased_by_alpha",
                real.sort_values("alpha_minus_global_score", ascending=True).head(top_k),
                "真实视频被 Alpha-STALLED 打得更像生成，可能削弱真实召回。",
            ),
            (
                "generated_patch_global_conflict",
                generated.sort_values("patch_minus_global_score", ascending=False).head(top_k),
                "patch 分支显著高于 global，适合检查局部证据是否误导融合。",
            ),
            (
                "generated_alpha_still_high",
                generated.sort_values("alpha_score", ascending=False).head(top_k),
                "Alpha-STALLED 仍最难识别的生成视频。",
            ),
            (
                "real_alpha_low",
                real.sort_values("alpha_score", ascending=True).head(top_k),
                "Alpha-STALLED 最容易误伤的真实视频。",
            ),
        ]
        for category, table, note in categories:
            out = table.copy()
            out.insert(0, "analysis_note", note)
            out.insert(0, "category", category)
            out.insert(0, "dataset", dataset)
            rows.append(out)
    result = pd.concat(rows, ignore_index=True)
    keep = [
        "dataset",
        "category",
        "analysis_note",
        "subset",
        "source_model",
        "filename",
        "global_score",
        "patch_score",
        "alpha_score",
        "alpha_minus_global_score",
        "patch_minus_global_score",
        "alpha_minus_patch_score",
    ]
    result = result[keep]
    result.to_csv(paths.sensitivity_dir / "failure_case_candidates.csv", index=False)
    return result


def plot_alpha_sensitivity(paths: Paths) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.2), sharey=True)
    for ax, dataset in zip(axes, DATASETS):
        summary = pd.read_csv(
            paths.root
            / f"results/paper_sweeps/{dataset}_alpha/{dataset}_same_grid_second_order_summary.csv"
        )
        ax.plot(summary["alpha"], summary["avg_auc"], color=PALETTE["auc"], label="AUC")
        ax.plot(summary["alpha"], summary["avg_ap"], color=PALETTE["ap"], linestyle="--", label="AP")
        ax.axvline(0.60, color="#777777", linewidth=0.9, linestyle=":")
        best = summary.sort_values(["avg_auc", "avg_ap"], ascending=False).iloc[0]
        ax.scatter([best["alpha"]], [best["avg_auc"]], color=PALETTE["auc"], s=16, zorder=3)
        ax.set_title(DISPLAY_NAMES[dataset])
        ax.set_xlabel("global fusion weight α")
        ax.grid(axis="y", alpha=0.18)
    axes[0].set_ylabel("pairwise balanced score")
    axes[0].legend(frameon=False, loc="lower right")
    fig.tight_layout(w_pad=1.0)
    _save_figure(fig, paths.figure_dir / "alpha_sensitivity_curves")


def plot_beta_sensitivity(paths: Paths, beta_summary: pd.DataFrame) -> None:
    for mode, title, stem in (
        ("patch_only", "Patch-only β sensitivity", "beta_sensitivity_patch_only"),
        ("fused_alpha0p60", "Frozen-α fusion β sensitivity", "beta_sensitivity_fused_alpha0p60"),
    ):
        fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.2), sharey=True)
        for ax, dataset in zip(axes, DATASETS):
            sub = beta_summary[(beta_summary["dataset"] == dataset) & (beta_summary["mode"] == mode)]
            ax.plot(sub["beta"], sub["avg_auc"], color=PALETTE["auc"], label="AUC")
            ax.plot(sub["beta"], sub["avg_ap"], color=PALETTE["ap"], linestyle="--", label="AP")
            best = sub.sort_values(["avg_auc", "avg_ap"], ascending=False).iloc[0]
            ax.scatter([best["beta"]], [best["avg_auc"]], color=PALETTE["auc"], s=16, zorder=3)
            ax.set_title(DISPLAY_NAMES[dataset])
            ax.set_xlabel("spatial weight β")
            ax.grid(axis="y", alpha=0.18)
        axes[0].set_ylabel("pairwise balanced score")
        axes[0].legend(frameon=False, loc="lower right")
        fig.suptitle(title, y=1.03, fontsize=9)
        fig.tight_layout(w_pad=1.0)
        _save_figure(fig, paths.figure_dir / stem)


def plot_component_delta(paths: Paths, delta: pd.DataFrame) -> None:
    plot_df = delta.copy()
    plot_df["label"] = plot_df["dataset"].map(DISPLAY_NAMES) + " / " + plot_df["source_model"]
    cols = ["patch_minus_global_auc", "alpha_minus_global_auc", "alpha_minus_patch_auc"]
    data = plot_df[cols].to_numpy()
    fig_height = max(3.0, 0.18 * len(plot_df))
    fig, ax = plt.subplots(figsize=(5.4, fig_height))
    vmax = max(0.02, float(np.nanmax(np.abs(data))))
    im = ax.imshow(data, aspect="auto", cmap="RdBu_r", vmin=-vmax, vmax=vmax)
    ax.set_yticks(np.arange(len(plot_df)))
    ax.set_yticklabels(plot_df["label"])
    ax.set_xticks(np.arange(len(cols)))
    ax.set_xticklabels(["Patch−Global", "Alpha−Global", "Alpha−Patch"], rotation=25, ha="right")
    ax.set_title("Per-generator AUC deltas")
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            value = data[i, j]
            ax.text(j, i, f"{value:+.3f}", ha="center", va="center", fontsize=6)
    cbar = fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02)
    cbar.set_label("ΔAUC")
    fig.tight_layout()
    _save_figure(fig, paths.figure_dir / "per_generator_auc_delta_heatmap")


def plot_score_distributions(paths: Paths) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.2), sharey=False)
    bins = np.linspace(0, 1, 31)
    for ax, dataset in zip(axes, DATASETS):
        scores = _read_score(paths.root, dataset, "alpha_stalled")
        real = scores[scores["subset"] == "real"]["final_score"].astype(float)
        fake = scores[scores["subset"] != "real"]["final_score"].astype(float)
        ax.hist(real, bins=bins, density=True, histtype="step", color=PALETTE["real"], label="real")
        ax.hist(fake, bins=bins, density=True, histtype="stepfilled", alpha=0.22, color=PALETTE["annotated"], label="generated")
        ax.set_title(DISPLAY_NAMES[dataset])
        ax.set_xlabel("Alpha-STALLED score")
        ax.grid(axis="y", alpha=0.18)
    axes[0].set_ylabel("density")
    axes[0].legend(frameon=False, loc="upper left")
    fig.tight_layout(w_pad=1.0)
    _save_figure(fig, paths.figure_dir / "alpha_score_distribution_panel")


def plot_bootstrap_delta_ci(paths: Paths, paired_delta: pd.DataFrame) -> None:
    """画 paired bootstrap 下 Alpha-STALLED 相对 global-only 的 AUC 差异。"""
    plot_df = paired_delta[paired_delta["comparison"] == "alpha_minus_global"].copy()
    plot_df["label"] = plot_df["dataset"].map(DISPLAY_NAMES) + " / " + plot_df["source_model"]
    plot_df = plot_df.sort_values(["dataset", "delta_auc_mean"], ascending=[True, True])
    y = np.arange(len(plot_df))
    colors = [PALETTE["alpha"] if v >= 0 else "#999999" for v in plot_df["delta_auc_mean"]]
    fig_height = max(3.0, 0.18 * len(plot_df))
    fig, ax = plt.subplots(figsize=(5.6, fig_height))
    xerr = np.vstack(
        [
            plot_df["delta_auc_mean"] - plot_df["delta_auc_ci_low"],
            plot_df["delta_auc_ci_high"] - plot_df["delta_auc_mean"],
        ]
    )
    ax.barh(y, plot_df["delta_auc_mean"], color=colors, alpha=0.78)
    ax.errorbar(plot_df["delta_auc_mean"], y, xerr=xerr, fmt="none", ecolor="#333333", elinewidth=0.7, capsize=1.5)
    ax.axvline(0, color="#333333", linewidth=0.8)
    ax.set_yticks(y)
    ax.set_yticklabels(plot_df["label"])
    ax.set_xlabel("ΔAUC relative to global-only")
    ax.set_title("Paired bootstrap uncertainty of Alpha-STALLED gains")
    ax.grid(axis="x", alpha=0.18)
    fig.tight_layout()
    _save_figure(fig, paths.figure_dir / "bootstrap_alpha_minus_global_auc_ci")


def _save_figure(fig: plt.Figure, stem: Path) -> None:
    fig.savefig(stem.with_suffix(".svg"), bbox_inches="tight")
    fig.savefig(stem.with_suffix(".png"), bbox_inches="tight")
    plt.close(fig)


def write_report(
    paths: Paths,
    beta_summary: pd.DataFrame,
    delta: pd.DataFrame,
    ci: pd.DataFrame,
    paired_delta: pd.DataFrame,
    failure_cases: pd.DataFrame,
) -> None:
    lines: list[str] = [
        "# 期刊版补充实验与可视化分析",
        "",
        "本报告基于现有 `results/paper_scores/` 和 `results/paper_sweeps/` 生成，",
        "不重新提取 DINOv3 特征，不使用测试批次 rank、生成器标签路由或真实/生成标签参与推理。",
        "",
        "## 已有基础消融是否足够",
        "",
        "当前基础版消融已经覆盖：",
        "",
        "- 三数据集 global-only、patch-only、Alpha-STALLED 组件消融；",
        "- ComGenVid 上 patch 空间、lag-1、multi-lag、motion-hard、motion-soft、同网格二阶时序消融；",
        "- 三数据集 global/patch 融合权重 alpha sweep。",
        "",
        "因此，不建议重复做同类基础消融。期刊版更应该补强以下证据：",
        "",
        "1. 超参数边界：alpha、patch 内部 beta、patch region、bottom-k、temporal run length；",
        "2. 稳定性：逐生成器增益/退化、跨数据集 best-alpha 漂移、随机采样置信区间；",
        "3. 失败模式：短视频覆盖缺口、patch 分支相对 global 的负迁移来源；",
        "4. 可解释可视化：分数分布、每生成器 delta heatmap、超参数曲线、代表性 patch anomaly map。",
        "",
        "## 本次新增的无重算特征分析",
        "",
    ]

    for mode in ("patch_only", "fused_alpha0p60"):
        title = "patch-only beta sweep" if mode == "patch_only" else "fixed-alpha fusion beta sweep"
        lines.extend([f"### {title}", "", "| 数据集 | Best beta | Best AUC / AP | beta=0.10 AUC / AP |", "|---|---:|---:|---:|"])
        for dataset in DATASETS:
            sub = beta_summary[(beta_summary["dataset"] == dataset) & (beta_summary["mode"] == mode)]
            best = sub.sort_values(["avg_auc", "avg_ap"], ascending=False).iloc[0]
            beta010 = sub.iloc[(sub["beta"] - 0.10).abs().argsort().iloc[0]]
            lines.append(
                f"| {DISPLAY_NAMES[dataset]} | {best.beta:.2f} | "
                f"{best.avg_auc:.4f} / {best.avg_ap:.4f} | "
                f"{beta010.avg_auc:.4f} / {beta010.avg_ap:.4f} |"
            )
        lines.append("")

    lines.extend(
        [
            "## 逐生成器增益/退化结论",
            "",
            "下表列出 Alpha-STALLED 相对 global-only 的 AUC 变化范围。大于 0 表示融合提升，",
            "小于 0 表示 patch 分支对该生成器产生负迁移。",
            "",
            "| 数据集 | 最小 ΔAUC | 最大 ΔAUC | 平均 ΔAUC | 负迁移生成器数 |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for dataset in DATASETS:
        sub = delta[delta["dataset"] == dataset]
        lines.append(
            f"| {DISPLAY_NAMES[dataset]} | "
            f"{sub['alpha_minus_global_auc'].min():+.4f} | "
            f"{sub['alpha_minus_global_auc'].max():+.4f} | "
            f"{sub['alpha_minus_global_auc'].mean():+.4f} | "
            f"{int((sub['alpha_minus_global_auc'] < 0).sum())} |"
        )

    lines.extend(
        [
            "",
            "## Bootstrap 置信区间",
            "",
            "本次新增 `n_boot` 次逐生成器 bootstrap，分别对 global-only、patch-only 和",
            "Alpha-STALLED 的 AUC/AP 估计 95% CI；同时使用同一次重采样计算 paired ΔAUC/ΔAP CI。",
            "该分析不改变主指标，只用于判断提升是否稳定。",
            "",
            "| 数据集 | Alpha-STALLED AUC 95% CI 中位宽度 | Global-only AUC 95% CI 中位宽度 |",
            "|---|---:|---:|",
        ]
    )
    for dataset in DATASETS:
        sub = ci[ci["dataset"] == dataset].copy()
        alpha_width = sub[sub["method"] == "alpha_stalled"].eval("auc_ci_high - auc_ci_low").median()
        global_width = sub[sub["method"] == "global_only"].eval("auc_ci_high - auc_ci_low").median()
        lines.append(f"| {DISPLAY_NAMES[dataset]} | {alpha_width:.4f} | {global_width:.4f} |")

    lines.extend(
        [
            "",
            "Paired bootstrap 下，Alpha-STALLED 相对 global-only 的 ΔAUC 概况：",
            "",
            "| 数据集 | ΔAUC 均值范围 | 95% CI 完全大于 0 的生成器数 | 95% CI 完全小于 0 的生成器数 |",
            "|---|---:|---:|---:|",
        ]
    )
    for dataset in DATASETS:
        sub = paired_delta[
            (paired_delta["dataset"] == dataset)
            & (paired_delta["comparison"] == "alpha_minus_global")
        ]
        lines.append(
            f"| {DISPLAY_NAMES[dataset]} | "
            f"{sub['delta_auc_mean'].min():+.4f} to {sub['delta_auc_mean'].max():+.4f} | "
            f"{int((sub['delta_auc_ci_low'] > 0).sum())} | "
            f"{int((sub['delta_auc_ci_high'] < 0).sum())} |"
        )

    lines.extend(
        [
            "",
            "## 失败样本候选",
            "",
            "已生成 `failure_case_candidates.csv`，用于后续做代表性视频或 patch anomaly map。",
            "这些候选只用于事后审计，不参与模型推理。",
            "",
            "| 类别 | 样本数 | 用途 |",
            "|---|---:|---|",
        ]
    )
    for category, group in failure_cases.groupby("category"):
        note = group["analysis_note"].iloc[0]
        lines.append(f"| `{category}` | {len(group)} | {note} |")

    lines.extend(
        [
            "",
            "## 生成的图和表",
            "",
            "- `results/paper_sensitivity/beta_sensitivity_summary.csv`：beta 平均指标曲线；",
            "- `results/paper_sensitivity/beta_sensitivity_per_model.csv`：beta 逐生成器指标；",
            "- `results/paper_sensitivity/component_per_generator_delta.csv`：global、patch、Alpha-STALLED 逐生成器差值；",
            "- `results/paper_sensitivity/alpha_score_distribution_summary.csv`：Alpha-STALLED 分数分布统计；",
            "- `results/paper_sensitivity/bootstrap_ci_summary.csv`：逐生成器 AUC/AP bootstrap 置信区间；",
            "- `results/paper_sensitivity/paired_bootstrap_delta_summary.csv`：方法差异的 paired bootstrap ΔAUC/ΔAP 置信区间；",
            "- `results/paper_sensitivity/failure_case_candidates.csv`：后续案例可视化和失败样本审计候选；",
            "- `results/paper_figures/alpha_sensitivity_curves.svg`：alpha 敏感性曲线；",
            "- `results/paper_figures/beta_sensitivity_patch_only.svg`：patch-only beta 敏感性；",
            "- `results/paper_figures/beta_sensitivity_fused_alpha0p60.svg`：固定 alpha 下 beta 敏感性；",
            "- `results/paper_figures/per_generator_auc_delta_heatmap.svg`：逐生成器 AUC delta heatmap；",
            "- `results/paper_figures/alpha_score_distribution_panel.svg`：真实/生成分数分布；",
            "- `results/paper_figures/bootstrap_alpha_minus_global_auc_ci.svg`：Alpha-STALLED 相对 global-only 的 paired bootstrap AUC 增益区间。",
            "",
            "## 仍建议补充的实跑实验",
            "",
            "优先级 P0：",
            "",
            "1. **patch region size 敏感性**：region=1/2/3，在三个数据集上统一重算 patch params 与 patch score。当前配置在不同数据集使用不同 region，期刊审稿会追问是否调参过度。",
            "2. **aggregation 敏感性**：mean vs bottom-k mean，bottom-k ratio 建议 0.05/0.10/0.20/0.30/0.50。ComGenVid 已有较多迹象，但 VideoFeedback/GenVideo 需要对应证据。",
            "3. **patch 可解释案例图**：基于 `failure_case_candidates.csv` 选取真实/生成代表视频，回到 patch cache 或原视频绘制 patch anomaly map。",
            "",
            "优先级 P1：",
            "",
            "4. **duration/window 敏感性**：1s/2s/3s/4s，尤其解释 HotShot/MoonValley/Hotshot-XL 的短视频边界。",
            "5. **cross-dataset frozen hyperparameter**：用一个数据集选出的 alpha/beta/region，在其他数据集冻结评测，区分 oracle sweep 和可泛化配置。",
            "6. **runtime 和存储开销**：global-only、patch cache prefill、patch-only eval、fusion 的时间和 cache 规模。",
            "",
            "优先级 P2：",
            "",
            "7. **paired bootstrap 扩展到生成器平均指标**：当前已输出逐生成器 paired ΔAUC/ΔAP；如果手稿需要一个总体显著性结论，可进一步对生成器平均指标做 paired bootstrap。",
            "8. **失败样本人工审计**：从候选表检查是否来自低运动、短时长、压缩伪影或真实视频域偏移。",
            "",
            "## 图表规范",
            "",
            "本次图遵循期刊/Nature-leaning 的基础规范：优先 SVG 矢量图，PNG 作为预览；",
            "使用色盲友好配色；每个面板只回答一个问题；图题和轴标签直接说明变量含义；",
            "敏感性曲线明确标出冻结超参数位置，避免把 oracle sweep 误写成主方法选择协议。",
        ]
    )
    (paths.sensitivity_dir / "journal_experiment_gap_analysis.md").write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=REPO_ROOT)
    parser.add_argument("--alpha", type=float, default=0.60)
    parser.add_argument("--beta-grid", default="0:1:0.05")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--bootstrap", type=int, default=500)
    args = parser.parse_args()

    root = args.root.resolve()
    paths = Paths(
        root=root,
        sensitivity_dir=root / "results/paper_sensitivity",
        figure_dir=root / "results/paper_figures",
    )
    _ensure_dirs(paths)
    _set_plot_style()

    start, stop, step = [float(x) for x in args.beta_grid.split(":")]
    count = int(round((stop - start) / step)) + 1
    betas = [round(start + i * step, 10) for i in range(count)]

    beta_summary, _ = run_beta_sweep(paths, betas, alpha=args.alpha, seed=args.seed)
    delta = build_component_delta(paths)
    ci = build_bootstrap_ci(paths, n_boot=args.bootstrap, seed=args.seed)
    paired_delta = build_paired_bootstrap_delta(paths, n_boot=args.bootstrap, seed=args.seed)
    failure_cases = build_failure_candidate_table(paths)
    build_score_distribution_summary(paths)
    plot_alpha_sensitivity(paths)
    plot_beta_sensitivity(paths, beta_summary)
    plot_component_delta(paths, delta)
    plot_score_distributions(paths)
    plot_bootstrap_delta_ci(paths, paired_delta)
    write_report(paths, beta_summary, delta, ci, paired_delta, failure_cases)

    print(f"已生成敏感性分析: {paths.sensitivity_dir}")
    print(f"已生成图表: {paths.figure_dir}")


if __name__ == "__main__":
    main()
