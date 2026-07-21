"""Build data-dense manuscript tables and figures for the IEEE Alpha-STALLED paper.

The script is intentionally read-only with respect to experiment outputs. It
turns already committed CSV evidence under results/ into LaTeX tables and
matplotlib figures under paper/ieee_alpha_stalled/.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
PAPER = ROOT / "paper" / "ieee_alpha_stalled"
FIG_DIR = PAPER / "figures" / "results"
TAB_DIR = PAPER / "tables"
NOTE_DIR = PAPER / "notes"

DATASET_DISPLAY = {
    "comgenvid": "ComGenVid",
    "videofeedback": "VideoFeedback",
    "genvideo": "GenVideo",
}

METHOD_DISPLAY = {
    "global_auc": "Global",
    "patch_auc": "Patch",
    "alpha_auc": "Alpha",
}

BLUE = "#2F6BBA"
RED = "#C43C39"
ORANGE = "#E6862A"
GREEN = "#4E9A51"
GRAY = "#6C6C6C"


def _ensure_dirs() -> None:
    for p in [FIG_DIR, TAB_DIR, NOTE_DIR]:
        p.mkdir(parents=True, exist_ok=True)


def _latex_escape(value: object) -> str:
    text = str(value)
    return (
        text.replace("\\", "\\textbackslash{}")
        .replace("&", "\\&")
        .replace("%", "\\%")
        .replace("$", "\\$")
        .replace("#", "\\#")
        .replace("_", "\\_")
        .replace("{", "\\{")
        .replace("}", "\\}")
        .replace("~", "\\textasciitilde{}")
        .replace("^", "\\textasciicircum{}")
    )


def _fmt(x: float, digits: int = 4) -> str:
    return f"{float(x):.{digits}f}"


def _fmt_delta(x: float, digits: int = 4) -> str:
    return f"{float(x):+.{digits}f}"


def _average_metrics(path: Path) -> tuple[float, float, int, int]:
    df = pd.read_csv(path)
    row = df[df["Generative Model"] == "Average"].iloc[0]
    auc_col = [c for c in df.columns if c.endswith(" AUC")][0]
    ap_col = [c for c in df.columns if c.endswith(" AP")][0]
    return float(row[auc_col]), float(row[ap_col]), int(row["n_real"]), int(row["n_annotated"])


def _write_text(path: Path, text: str) -> None:
    path.write_text(text.rstrip() + "\n", encoding="utf-8")
    print(f"saved {path.relative_to(ROOT)}")


def build_patch_temporal_table() -> pd.DataFrame:
    rows = [
        {
            "group": "静态对照",
            "variant": "Spatial-only",
            "mode": "patch spatial percentile",
            "source": ROOT / "results/paper_tables/comgenvid_patch_spatial_metrics.csv",
        },
        {
            "group": "一阶对照",
            "variant": "D=1 lag-1",
            "mode": "same_grid_lag1",
            "source": ROOT / "results/paper_tables/comgenvid_patch_lag1_metrics.csv",
        },
        {
            "group": "一阶对照",
            "variant": "D=1 multi-lag",
            "mode": "same_grid_multilag",
            "source": ROOT / "results/paper_tables/comgenvid_patch_multilag_metrics.csv",
        },
        {
            "group": "匹配扰动",
            "variant": "Motion-hard",
            "mode": "motion hard control",
            "source": ROOT / "results/paper_tables/comgenvid_patch_motionhard_metrics.csv",
        },
        {
            "group": "匹配扰动",
            "variant": "Motion-soft",
            "mode": "motion soft control",
            "source": ROOT / "results/paper_tables/comgenvid_patch_motionsoft_metrics.csv",
        },
        {
            "group": "二阶主线",
            "variant": "D=2 same-grid",
            "mode": "same_grid_second_order",
            "source": ROOT / "results/paper_tables/comgenvid_patch_only_metrics.csv",
        },
        {
            "group": "高阶对照",
            "variant": "D=3 same-grid",
            "mode": "same_grid_third_order",
            "source": ROOT / "results/journal_experiments/temporal_derivative_order/comgenvid_patch_third_order_metrics.csv",
        },
        {
            "group": "高阶对照",
            "variant": "D=4 same-grid",
            "mode": "same_grid_fourth_order",
            "source": ROOT / "results/journal_experiments/temporal_derivative_order/comgenvid_patch_fourth_order_metrics.csv",
        },
    ]
    out_rows = []
    for row in rows:
        auc, ap, n_real, n_fake = _average_metrics(row["source"])
        out_rows.append({**row, "auc": auc, "ap": ap, "n_real": n_real, "n_fake": n_fake})
    df = pd.DataFrame(out_rows)
    baseline = df.loc[df["variant"] == "D=2 same-grid"].iloc[0]
    df["delta_auc_vs_d2"] = df["auc"] - baseline["auc"]
    df["delta_ap_vs_d2"] = df["ap"] - baseline["ap"]

    lines = [
        r"\begin{table*}[!t]",
        r"  \centering",
        r"  \caption{ComGenVid 局部 patch 证据的完整时序定义消融。所有行均使用真实视频百分位校准和 pairwise balanced AUC/AP；D=1 lag-1 补足一阶差分对照，D=2 same-grid 为当前主线。}",
        r"  \label{tab:patch_temporal_comprehensive}",
        r"  \resizebox{\textwidth}{!}{%",
        r"  \begin{tabular}{lllcccc}",
        r"    \toprule",
        r"    证据组 & 变体 & 局部时序定义 & AUC & AP & $\Delta$AUC vs D=2 & $\Delta$AP vs D=2 \\",
        r"    \midrule",
    ]
    for r in df.itertuples(index=False):
        variant = r"\textbf{D=2 same-grid}" if r.variant == "D=2 same-grid" else _latex_escape(r.variant)
        lines.append(
            "    "
            + " & ".join(
                [
                    _latex_escape(r.group),
                    variant,
                    _latex_escape(r.mode),
                    _fmt(r.auc),
                    _fmt(r.ap),
                    _fmt_delta(r.delta_auc_vs_d2),
                    _fmt_delta(r.delta_ap_vs_d2),
                ]
            )
            + r" \\"
        )
    lines += [
        r"    \bottomrule",
        r"  \end{tabular}%",
        r"  }",
        r"\end{table*}",
    ]
    _write_text(TAB_DIR / "patch_temporal_comprehensive.tex", "\n".join(lines))
    df.to_csv(TAB_DIR / "patch_temporal_comprehensive.csv", index=False)
    return df


def build_per_generator_component_table() -> pd.DataFrame:
    df = pd.read_csv(ROOT / "results/paper_sensitivity/component_per_generator_delta.csv")
    df["dataset_display"] = df["dataset"].map(DATASET_DISPLAY)
    lines = [
        r"\begin{table*}[!t]",
        r"  \centering",
        r"  \caption{按生成器展开的 global-only、patch-only 与 \method 组件结果。表中同时报告 AUC/AP 和 \method 相对两个分支的 AUC 差值，用于显示融合收益与负迁移边界。}",
        r"  \label{tab:per_generator_component_matrix}",
        r"  \resizebox{\textwidth}{!}{%",
        r"  \begin{tabular}{llcccccccc}",
        r"    \toprule",
        r"    数据集 & 生成器 & Global AUC & Patch AUC & \method AUC & $\Delta_{\alpha-G}$ AUC & $\Delta_{\alpha-P}$ AUC & Global AP & Patch AP & \method AP \\",
        r"    \midrule",
    ]
    for dataset, group in df.groupby("dataset", sort=False):
        for i, r in enumerate(group.itertuples(index=False)):
            ds = DATASET_DISPLAY[dataset] if i == 0 else ""
            lines.append(
                "    "
                + " & ".join(
                    [
                        _latex_escape(ds),
                        _latex_escape(r.source_model),
                        _fmt(r.global_auc),
                        _fmt(r.patch_auc),
                        _fmt(r.alpha_auc),
                        _fmt_delta(r.alpha_minus_global_auc),
                        _fmt_delta(r.alpha_minus_patch_auc),
                        _fmt(r.global_ap),
                        _fmt(r.patch_ap),
                        _fmt(r.alpha_ap),
                    ]
                )
                + r" \\"
            )
        lines.append(r"    \midrule")
    lines[-1] = r"    \bottomrule"
    lines += [
        r"  \end{tabular}%",
        r"  }",
        r"\end{table*}",
    ]
    _write_text(TAB_DIR / "per_generator_component_matrix.tex", "\n".join(lines))
    return df


def build_bootstrap_table() -> pd.DataFrame:
    df = pd.read_csv(ROOT / "results/paper_sensitivity/paired_bootstrap_delta_summary.csv")
    df = df[df["comparison"].isin(["alpha_minus_global", "alpha_minus_patch"])].copy()
    wide = df.pivot(index=["dataset", "source_model"], columns="comparison")
    rows = []
    for idx in wide.index:
        dataset, source = idx
        rows.append(
            {
                "dataset": dataset,
                "source_model": source,
                "ag_mean": wide.loc[idx, ("delta_auc_mean", "alpha_minus_global")],
                "ag_low": wide.loc[idx, ("delta_auc_ci_low", "alpha_minus_global")],
                "ag_high": wide.loc[idx, ("delta_auc_ci_high", "alpha_minus_global")],
                "ap_mean": wide.loc[idx, ("delta_auc_mean", "alpha_minus_patch")],
                "ap_low": wide.loc[idx, ("delta_auc_ci_low", "alpha_minus_patch")],
                "ap_high": wide.loc[idx, ("delta_auc_ci_high", "alpha_minus_patch")],
            }
        )
    out = pd.DataFrame(rows)
    lines = [
        r"\begin{table*}[!t]",
        r"  \centering",
        r"  \caption{生成器级 paired bootstrap AUC 差值区间。$\Delta_{\alpha-G}$ 检验 \method 相对 global-only 的收益；$\Delta_{\alpha-P}$ 检验融合是否超过 patch-only。}",
        r"  \label{tab:per_generator_bootstrap_delta}",
        r"  \resizebox{\textwidth}{!}{%",
        r"  \begin{tabular}{llcccccc}",
        r"    \toprule",
        r"    数据集 & 生成器 & $\Delta_{\alpha-G}$ mean & 95\% CI & 结论 & $\Delta_{\alpha-P}$ mean & 95\% CI & 结论 \\",
        r"    \midrule",
    ]
    for dataset, group in out.groupby("dataset", sort=False):
        for i, r in enumerate(group.itertuples(index=False)):
            ds = DATASET_DISPLAY[dataset] if i == 0 else ""
            ag_result = "正向" if r.ag_low > 0 else ("负向" if r.ag_high < 0 else "跨零")
            ap_result = "正向" if r.ap_low > 0 else ("负向" if r.ap_high < 0 else "跨零")
            lines.append(
                "    "
                + " & ".join(
                    [
                        _latex_escape(ds),
                        _latex_escape(r.source_model),
                        _fmt_delta(r.ag_mean),
                        f"[{_fmt_delta(r.ag_low)}, {_fmt_delta(r.ag_high)}]",
                        _latex_escape(ag_result),
                        _fmt_delta(r.ap_mean),
                        f"[{_fmt_delta(r.ap_low)}, {_fmt_delta(r.ap_high)}]",
                        _latex_escape(ap_result),
                    ]
                )
                + r" \\"
            )
        lines.append(r"    \midrule")
    lines[-1] = r"    \bottomrule"
    lines += [
        r"  \end{tabular}%",
        r"  }",
        r"\end{table*}",
    ]
    _write_text(TAB_DIR / "per_generator_bootstrap_delta.tex", "\n".join(lines))
    return out


def build_hyperparameter_table() -> pd.DataFrame:
    region = pd.read_csv(ROOT / "results/journal_experiments/region_sensitivity/region_sensitivity_summary.csv")
    region["family"] = "region"
    region["setting"] = "region=" + region["patch_region_size"].astype(int).astype(str) + ", mean"

    aggregation = pd.read_csv(ROOT / "results/journal_experiments/aggregation_sensitivity/aggregation_sensitivity_summary.csv")
    aggregation["family"] = "aggregation"
    aggregation["setting"] = aggregation.apply(
        lambda r: (
            f"region={int(r.patch_region_size)}, {r.aggregation}"
            if r.aggregation == "mean"
            else f"region={int(r.patch_region_size)}, bottom-k={r.bottomk_ratio:.2f}"
        ),
        axis=1,
    )

    bottomk = pd.read_csv(ROOT / "results/journal_experiments/bottomk_sensitivity/bottomk_sensitivity_summary.csv")
    bottomk["family"] = "bottom-k"
    bottomk["setting"] = bottomk.apply(
        lambda r: f"region={int(r.patch_region_size)}, bottom-k={r.bottomk_ratio:.2f}",
        axis=1,
    )

    cols = ["family", "dataset", "setting", "avg_auc", "avg_ap"]
    out = pd.concat([region[cols], aggregation[cols], bottomk[cols]], ignore_index=True)
    out["dataset_display"] = out["dataset"].map(DATASET_DISPLAY)
    out["oracle_auc_by_family_dataset"] = out.groupby(["family", "dataset"])["avg_auc"].transform("max")
    out["delta_auc_vs_family_best"] = out["avg_auc"] - out["oracle_auc_by_family_dataset"]

    lines = [
        r"\begin{table*}[!t]",
        r"  \centering",
        r"  \caption{局部尺度和聚合方式敏感性。表中 $\Delta$AUC 相对同一数据集、同一超参数族内最佳设置计算，用于区分稳定设计和数据集相关边界。}",
        r"  \label{tab:hyperparameter_sensitivity_matrix}",
        r"  \resizebox{\textwidth}{!}{%",
        r"  \begin{tabular}{lllccc}",
        r"    \toprule",
        r"    超参数族 & 数据集 & 设置 & AUC & AP & $\Delta$AUC vs family best \\",
        r"    \midrule",
    ]
    for family, group in out.groupby("family", sort=False):
        for dataset, sub in group.groupby("dataset", sort=False):
            for i, r in enumerate(sub.itertuples(index=False)):
                fam = family if (i == 0 and dataset == group["dataset"].iloc[0]) else ""
                ds = DATASET_DISPLAY[dataset] if i == 0 else ""
                lines.append(
                    "    "
                    + " & ".join(
                        [
                            _latex_escape(fam),
                            _latex_escape(ds),
                            _latex_escape(r.setting),
                            _fmt(r.avg_auc),
                            _fmt(r.avg_ap),
                            _fmt_delta(r.delta_auc_vs_family_best),
                        ]
                    )
                    + r" \\"
                )
            lines.append(r"    \midrule")
    lines[-1] = r"    \bottomrule"
    lines += [
        r"  \end{tabular}%",
        r"  }",
        r"\end{table*}",
    ]
    _write_text(TAB_DIR / "hyperparameter_sensitivity_matrix.tex", "\n".join(lines))
    return out


def build_runtime_boundary_table() -> None:
    duration = pd.read_csv(ROOT / "results/journal_experiments/duration_window_representative/comgenvid_duration_window_comparison.csv")
    runtime = pd.read_csv(ROOT / "results/journal_experiments/video_stage_runtime_benchmark/video_stage_runtime_benchmark.csv")
    storage = pd.read_csv(ROOT / "results/journal_experiments/runtime_storage_audit/storage_audit.csv")
    global_gib = storage[storage["name"].str.startswith("global_embedding_cache")]["size_gib"].sum()
    patch_gib = storage[storage["name"].str.startswith("compact_patch_cache")]["size_gib"].sum()
    lines = [
        r"\begin{table}[!t]",
        r"  \centering",
        r"  \caption{时间窗口和计算/存储边界。1 秒结果为 ComGenVid patch-only 代表性实验；运行时间为 ComGenVid debug-2 clean-cache 阶段计时。}",
        r"  \label{tab:runtime_window_boundary}",
        r"  \begin{tabular}{llcc}",
        r"    \toprule",
        r"    类型 & 设置 & AUC/时间 & AP/存储 \\",
        r"    \midrule",
    ]
    duration_names = {
        1: "region=3, bottom-k=0.20",
        2: "main 2s setting",
    }
    for r in duration.itertuples(index=False):
        lines.append(
            "    "
            + " & ".join(
                [
                    f"{int(r.duration_sec)}s patch",
                    _latex_escape(duration_names.get(int(r.duration_sec), r.setting)),
                    _fmt(r.auc),
                    _fmt(r.ap),
                ]
            )
            + r" \\"
        )
    lines.append(r"    \midrule")
    for r in runtime.itertuples(index=False):
        short = {
            "global_compact_embedding_and_scoring": "global score",
            "patch_compact_cache_prefill": "patch prefill",
            "patch_cached_scoring": "patch score",
        }.get(r.stage, r.stage)
        lines.append(
            "    "
            + " & ".join(
                [
                    "runtime",
                    _latex_escape(short),
                    f"{float(r.elapsed_sec):.2f}s",
                    "--",
                ]
            )
            + r" \\"
        )
    lines.append(r"    \midrule")
    lines.append(f"    storage & global cache & -- & {global_gib:.2f} GiB \\\\")
    lines.append(f"    storage & patch compact cache & -- & {patch_gib:.2f} GiB \\\\")
    lines += [
        r"    \bottomrule",
        r"  \end{tabular}",
        r"\end{table}",
    ]
    _write_text(TAB_DIR / "runtime_window_boundary.tex", "\n".join(lines))


def plot_global_patch_joint() -> None:
    fig, axes = plt.subplots(1, 3, figsize=(10.8, 3.25), sharex=True, sharey=True, constrained_layout=True)
    for ax, dataset in zip(axes, ["comgenvid", "videofeedback", "genvideo"]):
        path = ROOT / f"results/paper_scores/{dataset}_alpha_stalled.csv"
        df = pd.read_csv(path)
        df = df.copy()
        df["label"] = np.where(df["subset"].eq("real"), "real", "generated")
        plot_df = pd.concat(
            [
                df[df["label"] == "real"].sample(min(1800, (df["label"] == "real").sum()), random_state=7),
                df[df["label"] == "generated"].sample(min(1800, (df["label"] == "generated").sum()), random_state=11),
            ]
        )
        real = plot_df[plot_df["label"] == "real"]
        fake = plot_df[plot_df["label"] == "generated"]
        ax.scatter(fake["global_score"], fake["patch_score"], s=8, c=RED, alpha=0.20, edgecolors="none", label="generated")
        ax.scatter(real["global_score"], real["patch_score"], s=8, c=BLUE, alpha=0.22, edgecolors="none", label="real")
        ax.axvline(0.5, color="#BBBBBB", lw=0.8, ls="--")
        ax.axhline(0.5, color="#BBBBBB", lw=0.8, ls="--")
        corr = df[["global_score", "patch_score"]].corr().iloc[0, 1]
        real_mean = df[df["label"] == "real"][["global_score", "patch_score"]].mean()
        fake_mean = df[df["label"] == "generated"][["global_score", "patch_score"]].mean()
        ax.set_title(DATASET_DISPLAY[dataset])
        ax.text(
            0.03,
            0.97,
            f"r={corr:.2f}\nreal=({real_mean.global_score:.2f},{real_mean.patch_score:.2f})\nfake=({fake_mean.global_score:.2f},{fake_mean.patch_score:.2f})",
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=7.5,
            bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="#CCCCCC", alpha=0.9),
        )
        ax.grid(alpha=0.18)
    axes[0].set_ylabel("Patch second-order percentile score")
    for ax in axes:
        ax.set_xlabel("Global STALL percentile score")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
    axes[-1].legend(frameon=False, loc="lower right", fontsize=8)
    fig.savefig(FIG_DIR / "global_patch_likelihood_joint_panel.pdf")
    fig.savefig(FIG_DIR / "global_patch_likelihood_joint_panel.png", dpi=260)
    fig.savefig(FIG_DIR / "global_patch_likelihood_joint_panel.svg")
    plt.close(fig)
    print("saved figures/results/global_patch_likelihood_joint_panel.*")


def plot_patch_temporal_landscape(df: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(10.0, 3.35), constrained_layout=True)
    order = df.sort_values("auc", ascending=True)
    colors = [GREEN if v == "D=2 same-grid" else GRAY for v in order["variant"]]
    axes[0].barh(order["variant"], order["auc"], color=colors)
    axes[0].set_xlim(0.78, 0.94)
    axes[0].set_xlabel("AUC")
    axes[0].set_title("Patch evidence variants")
    for y, val in enumerate(order["auc"]):
        axes[0].text(val + 0.002, y, f"{val:.3f}", va="center", fontsize=8)

    tmp = df.copy()
    tmp["variant_short"] = tmp["variant"].str.replace(" same-grid", "", regex=False)
    x = np.arange(len(tmp))
    axes[1].plot(x, tmp["delta_auc_vs_d2"], marker="o", color=ORANGE, label="ΔAUC")
    axes[1].plot(x, tmp["delta_ap_vs_d2"], marker="s", color=BLUE, label="ΔAP")
    axes[1].axhline(0, color="#333333", lw=0.8)
    axes[1].set_xticks(x, tmp["variant_short"], rotation=35, ha="right")
    axes[1].set_ylabel("Delta vs D=2")
    axes[1].set_title("Loss relative to the second-order branch")
    axes[1].grid(axis="y", alpha=0.25)
    axes[1].legend(frameon=False)
    fig.savefig(FIG_DIR / "patch_temporal_design_landscape.pdf")
    fig.savefig(FIG_DIR / "patch_temporal_design_landscape.png", dpi=260)
    fig.savefig(FIG_DIR / "patch_temporal_design_landscape.svg")
    plt.close(fig)
    print("saved figures/results/patch_temporal_design_landscape.*")


def plot_component_matrix(df: pd.DataFrame) -> None:
    df = df.copy()
    df["row"] = df["dataset"].map(DATASET_DISPLAY) + " / " + df["source_model"]
    mat = df.set_index("row")[["global_auc", "patch_auc", "alpha_auc"]]
    fig, axes = plt.subplots(1, 2, figsize=(10.8, 7.6), gridspec_kw={"width_ratios": [1.0, 1.05]}, constrained_layout=True)
    im = axes[0].imshow(mat.values, aspect="auto", cmap="YlGnBu", vmin=0.60, vmax=0.98)
    axes[0].set_xticks(range(3), ["Global", "Patch", "Alpha"])
    axes[0].set_yticks(range(len(mat)), mat.index, fontsize=7)
    axes[0].set_title("Per-generator AUC")
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            axes[0].text(j, i, f"{mat.iloc[i, j]:.3f}", ha="center", va="center", fontsize=6.5, color="#111111")
    fig.colorbar(im, ax=axes[0], fraction=0.046, pad=0.02)

    delta = df.set_index("row")[["alpha_minus_global_auc", "alpha_minus_patch_auc"]]
    y = np.arange(len(delta))
    axes[1].barh(y - 0.18, delta["alpha_minus_global_auc"], height=0.32, color=BLUE, label="Alpha - Global")
    axes[1].barh(y + 0.18, delta["alpha_minus_patch_auc"], height=0.32, color=ORANGE, label="Alpha - Patch")
    axes[1].axvline(0, color="#333333", lw=0.8)
    axes[1].set_yticks(y, delta.index, fontsize=7)
    axes[1].invert_yaxis()
    axes[1].set_xlabel("ΔAUC")
    axes[1].set_title("Branch complementarity and negative transfer")
    axes[1].grid(axis="x", alpha=0.25)
    axes[1].legend(frameon=False, fontsize=8)
    fig.savefig(FIG_DIR / "component_per_generator_matrix.pdf")
    fig.savefig(FIG_DIR / "component_per_generator_matrix.png", dpi=260)
    fig.savefig(FIG_DIR / "component_per_generator_matrix.svg")
    plt.close(fig)
    print("saved figures/results/component_per_generator_matrix.*")


def plot_hyperparameter_grid(hyper: pd.DataFrame) -> None:
    region = hyper[hyper["family"] == "region"].copy()
    aggregation = hyper[hyper["family"] == "aggregation"].copy()
    bottomk = hyper[hyper["family"] == "bottom-k"].copy()
    frozen_paths = {
        "region": ROOT / "results/journal_experiments/cross_dataset_frozen_hyperparams/region_leave_one_out.csv",
        "aggregation": ROOT / "results/journal_experiments/cross_dataset_frozen_hyperparams/aggregation_leave_one_out.csv",
        "alpha": ROOT / "results/journal_experiments/cross_dataset_frozen_hyperparams/alpha_leave_one_out.csv",
        "beta": ROOT / "results/journal_experiments/cross_dataset_frozen_hyperparams/beta_leave_one_out.csv",
    }
    frozen = []
    for name, path in frozen_paths.items():
        if path.exists():
            x = pd.read_csv(path)
            frozen.append({"family": name, "mean_abs_gap": x["auc_gap_vs_target_oracle"].abs().mean()})
    frozen = pd.DataFrame(frozen)

    fig, axes = plt.subplots(2, 2, figsize=(10.2, 6.2), constrained_layout=True)
    ax = axes[0, 0]
    for dataset, g in region.groupby("dataset", sort=False):
        x = g["setting"].str.extract(r"region=(\d+)")[0].astype(int)
        ax.plot(x, g["avg_auc"], marker="o", label=DATASET_DISPLAY[dataset])
    ax.set_title("Region-size sensitivity")
    ax.set_xlabel("patch region size")
    ax.set_ylabel("AUC")
    ax.grid(alpha=0.25)
    ax.legend(frameon=False, fontsize=8)

    ax = axes[0, 1]
    labels = aggregation["dataset"].map(DATASET_DISPLAY) + "\n" + aggregation["setting"].str.replace(", ", "\n", regex=False)
    ax.bar(range(len(aggregation)), aggregation["avg_auc"], color=GREEN)
    ax.set_xticks(range(len(aggregation)), labels, rotation=0, fontsize=7)
    ax.set_ylim(0.72, 0.86)
    ax.set_ylabel("AUC")
    ax.set_title("Aggregation sensitivity")
    ax.grid(axis="y", alpha=0.25)

    ax = axes[1, 0]
    bx = bottomk["setting"].str.extract(r"bottom-k=(.*)")[0].astype(float)
    ax.plot(bx, bottomk["avg_auc"], marker="o", color=ORANGE, label="AUC")
    ax.plot(bx, bottomk["avg_ap"], marker="s", color=BLUE, label="AP")
    ax.set_title("ComGenVid bottom-k ratio")
    ax.set_xlabel("bottom-k ratio")
    ax.set_ylabel("metric")
    ax.grid(alpha=0.25)
    ax.legend(frameon=False)

    ax = axes[1, 1]
    ax.bar(frozen["family"], frozen["mean_abs_gap"], color=[GREEN, GRAY, BLUE, ORANGE][: len(frozen)])
    ax.set_title("Leave-one-dataset-out oracle gap")
    ax.set_ylabel("mean |AUC gap|")
    ax.grid(axis="y", alpha=0.25)
    for i, r in enumerate(frozen.itertuples(index=False)):
        ax.text(i, r.mean_abs_gap + 0.0008, f"{r.mean_abs_gap:.3f}", ha="center", va="bottom", fontsize=8)

    fig.savefig(FIG_DIR / "hyperparameter_sensitivity_grid.pdf")
    fig.savefig(FIG_DIR / "hyperparameter_sensitivity_grid.png", dpi=260)
    fig.savefig(FIG_DIR / "hyperparameter_sensitivity_grid.svg")
    plt.close(fig)
    print("saved figures/results/hyperparameter_sensitivity_grid.*")


def write_global_order_note() -> None:
    text = """# 全局 temporal derivative order 是否应补实验

## 当前代码状态

- patch 分支在 `src/patch_matching.py` 和 `src/eval_patch_fast.py` 中显式支持 `same_grid_lag1`、`same_grid_second_order`、`same_grid_third_order` 和 `same_grid_fourth_order`。
- 全局分支当前通过 `src/create_params.py` 和 `src/eval.py` 使用原 STALL 式全局 embedding 校准参数；release 中没有命令行参数可以直接把全局 temporal likelihood 切换到 D=2/D=3/D=4。
- 因此，全局不同阶数不是现有 score CSV 的重汇总问题，而是需要重定义全局时序特征、重建真实视频校准参数并重跑三数据集 score 的新实验。

## 写作建议

当前主文不应伪造全局 D sweep，也不应把 patch D=2 的机制结论外推到 global 分支。合理表述是：

1. 原 STALL global-only 作为固定主干和主要 baseline 保留；
2. 本文的 temporal derivative order 消融限定在新增 patch 分支，因为创新点是局部同网格二阶证据；
3. 若审稿人要求验证“全局高阶导数是否同样有效”，应作为后续 P2 实验：在 `create_params.py` 中增加全局 order 参数，重建 real-calibration `.npz`，再对 ComGenVid、VideoFeedback 和 GenVideo 统一重跑。

## 最小实验计划

1. 在 `src/create_params.py` 增加 global temporal order 参数，默认保持 D=1 以兼容原 STALL。
2. 在 `STALL._scores_from_embs` 对应全局 temporal likelihood 处接收相同 order。
3. 对 VATEX/目标真实校准集分别构建 D=1/2/3 参数。
4. 三数据集重跑 global-only，并与当前 patch D=2 和 Alpha-STALLED 对齐。

当前阶段结论：可以分析其必要性，但不建议在主文中声称已有 global D=2/D=3/D=4 结果。
"""
    _write_text(NOTE_DIR / "global_derivative_order_gap_analysis.md", text)


def update_notes() -> None:
    text = """# Rich evidence upgrade

本轮优化按 algorithmic paper 的实验链条补强图表：主张必须绑定比较、消融或边界证据。

## 新增主文表格

- `tables/per_generator_component_matrix.tex`：20 个生成器级 global/patch/Alpha AUC/AP 和 delta。
- `tables/per_generator_bootstrap_delta.tex`：20 个生成器级 paired bootstrap AUC 差值区间。
- `tables/patch_temporal_comprehensive.tex`：ComGenVid patch spatial、D=1、multi-lag、motion-hard/soft、D=2/3/4 完整局部证据对照。
- `tables/hyperparameter_sensitivity_matrix.tex`：region、aggregation、bottom-k 多数据集敏感性大表。
- `tables/runtime_window_boundary.tex`：时间窗口、运行时间和存储成本边界。

## 新增主文图

- `figures/results/global_patch_likelihood_joint_panel.*`：仿 STALL Fig. 1 的全局--局部联合分数散点图，但纵轴使用本文新增 patch second-order 分支。
- `figures/results/component_per_generator_matrix.*`：生成器级 AUC 热图和分支 delta 条形图。
- `figures/results/patch_temporal_design_landscape.*`：局部证据变体排序和相对 D=2 损失。
- `figures/results/hyperparameter_sensitivity_grid.*`：region、aggregation、bottom-k 和 frozen hyperparameter gap 的四面板图。

## 边界

- D=1 已由 `comgenvid_patch_lag1_metrics.csv` 纳入，不是新缺口。
- 全局不同阶数目前不是已有 release 的 CSV 级实验；已记录到 `notes/global_derivative_order_gap_analysis.md`，不在主文伪造结果。
"""
    _write_text(NOTE_DIR / "rich_evidence_upgrade.md", text)


def main() -> None:
    _ensure_dirs()
    temporal = build_patch_temporal_table()
    component = build_per_generator_component_table()
    build_bootstrap_table()
    hyper = build_hyperparameter_table()
    build_runtime_boundary_table()
    plot_global_patch_joint()
    plot_patch_temporal_landscape(temporal)
    plot_component_matrix(component)
    plot_hyperparameter_grid(hyper)
    write_global_order_note()
    update_notes()


if __name__ == "__main__":
    main()
