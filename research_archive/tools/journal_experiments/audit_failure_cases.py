"""生成 Alpha-STALLED 失败/边界样本的可复现审计表。

该脚本将已有 failure candidates、逐生成器 paired bootstrap、宏平均 paired
bootstrap 和 index 元数据合并，形成一个不依赖人工观看视频的 failure audit。
它不参与模型推理或调参，只用于论文中的 failure-mode / limitation 说明，以及
后续人工挑选视频关键帧。
"""

from __future__ import annotations

import argparse
from pathlib import Path
from urllib.parse import unquote

import matplotlib.pyplot as plt
import pandas as pd


DATASETS = ("comgenvid", "videofeedback", "genvideo")
DISPLAY = {
    "comgenvid": "ComGenVid",
    "videofeedback": "VideoFeedback",
    "genvideo": "GenVideo",
}

RISK_LABELS = {
    "generated_score_increased_by_alpha": "generated_false_real_risk",
    "generated_patch_global_conflict": "patch_global_conflict",
    "generated_alpha_still_high": "residual_generated_hard_case",
    "real_score_decreased_by_alpha": "real_recall_degradation_risk",
    "real_alpha_low": "real_false_positive_risk",
}


def _read_index_metadata(root: Path, dataset: str) -> pd.DataFrame:
    index = pd.read_csv(root / f"cache/indexes/{dataset}.csv")
    rows = []
    for _, row in index.iterrows():
        source_model = str(row["source_model"])
        subset = str(row["subset"])
        video_path = str(row["video_path"])
        marker = f"/{subset}/{source_model}/"
        if marker in video_path:
            rel_filename = video_path.split(marker, 1)[1]
        else:
            rel_filename = Path(video_path).name
        rows.append(
            {
                "dataset": dataset,
                "subset": subset,
                "source_model": source_model,
                "filename_decoded": rel_filename,
                "filename_basename": Path(video_path).name,
                "video_path": video_path,
                "fps": float(row["fps"]),
                "duration_seconds": float(row["duration_seconds"]),
                "num_frames": int(row["num_frames"]),
                "has_2s_window": pd.notna(row["2_sec_idxs"]),
            }
        )
    return pd.DataFrame(rows)


def _merge_index(candidates: pd.DataFrame, root: Path) -> pd.DataFrame:
    meta = pd.concat([_read_index_metadata(root, dataset) for dataset in DATASETS], ignore_index=True)
    out = candidates.copy()
    out["filename_decoded"] = out["filename"].map(lambda x: unquote(str(x)))
    out = out.merge(
        meta,
        on=["dataset", "subset", "source_model", "filename_decoded"],
        how="left",
        validate="many_to_one",
    )
    missing = out["video_path"].isna()
    if missing.any():
        fallback = meta.rename(columns={"filename_basename": "filename"})
        fill = out[missing].drop(columns=["video_path", "fps", "duration_seconds", "num_frames", "has_2s_window"])
        fill = fill.merge(
            fallback[
                [
                    "dataset",
                    "subset",
                    "source_model",
                    "filename",
                    "video_path",
                    "fps",
                    "duration_seconds",
                    "num_frames",
                    "has_2s_window",
                ]
            ],
            on=["dataset", "subset", "source_model", "filename"],
            how="left",
        )
        out.loc[missing, ["video_path", "fps", "duration_seconds", "num_frames", "has_2s_window"]] = fill[
            ["video_path", "fps", "duration_seconds", "num_frames", "has_2s_window"]
        ].to_numpy()
    return out


def _generator_ci_class(row: pd.Series) -> str:
    if pd.isna(row.get("delta_auc_ci_low")):
        return "no_generator_ci"
    if row["delta_auc_ci_high"] < 0:
        return "stable_negative_transfer"
    if row["delta_auc_ci_low"] > 0:
        return "stable_positive_transfer"
    return "boundary_ci_crosses_zero"


def build_audit(root: Path) -> pd.DataFrame:
    candidates = pd.read_csv(root / "results/paper_sensitivity/failure_case_candidates.csv")
    paired = pd.read_csv(root / "results/paper_sensitivity/paired_bootstrap_delta_summary.csv")
    paired = paired[paired["comparison"] == "alpha_minus_global"][
        [
            "dataset",
            "source_model",
            "delta_auc_mean",
            "delta_auc_ci_low",
            "delta_auc_ci_high",
            "delta_ap_mean",
            "delta_ap_ci_low",
            "delta_ap_ci_high",
        ]
    ].rename(
        columns={
            "delta_auc_mean": "generator_delta_auc_mean",
            "delta_auc_ci_low": "generator_delta_auc_ci_low",
            "delta_auc_ci_high": "generator_delta_auc_ci_high",
            "delta_ap_mean": "generator_delta_ap_mean",
            "delta_ap_ci_low": "generator_delta_ap_ci_low",
            "delta_ap_ci_high": "generator_delta_ap_ci_high",
        }
    )
    macro = pd.read_csv(
        root / "results/journal_experiments/macro_average_bootstrap/macro_average_paired_bootstrap_delta.csv"
    )
    macro = macro[macro["comparison"] == "alpha_minus_global"][
        ["dataset", "delta_auc_mean", "delta_auc_ci_low", "delta_auc_ci_high"]
    ].rename(
        columns={
            "delta_auc_mean": "macro_delta_auc_mean",
            "delta_auc_ci_low": "macro_delta_auc_ci_low",
            "delta_auc_ci_high": "macro_delta_auc_ci_high",
        }
    )

    audit = _merge_index(candidates, root)
    audit = audit.merge(paired, on=["dataset", "source_model"], how="left")
    audit = audit.merge(macro, on="dataset", how="left")
    audit["risk_type"] = audit["category"].map(RISK_LABELS)
    audit["generator_ci_class"] = audit.apply(
        lambda row: (
            "stable_negative_transfer"
            if pd.notna(row["generator_delta_auc_ci_high"]) and row["generator_delta_auc_ci_high"] < 0
            else "stable_positive_transfer"
            if pd.notna(row["generator_delta_auc_ci_low"]) and row["generator_delta_auc_ci_low"] > 0
            else "boundary_ci_crosses_zero"
            if pd.notna(row["generator_delta_auc_ci_low"])
            else "no_generator_ci"
        ),
        axis=1,
    )
    audit["score_conflict_strength"] = audit["patch_minus_global_score"].abs()
    audit["alpha_shift_strength"] = audit["alpha_minus_global_score"].abs()
    audit["paper_priority"] = audit.apply(_priority, axis=1)
    audit["recommended_use"] = audit.apply(_recommended_use, axis=1)

    ordered = [
        "dataset",
        "paper_priority",
        "risk_type",
        "category",
        "subset",
        "source_model",
        "filename",
        "video_path",
        "duration_seconds",
        "fps",
        "num_frames",
        "has_2s_window",
        "global_score",
        "patch_score",
        "alpha_score",
        "alpha_minus_global_score",
        "patch_minus_global_score",
        "alpha_minus_patch_score",
        "generator_ci_class",
        "generator_delta_auc_mean",
        "generator_delta_auc_ci_low",
        "generator_delta_auc_ci_high",
        "macro_delta_auc_mean",
        "macro_delta_auc_ci_low",
        "macro_delta_auc_ci_high",
        "recommended_use",
    ]
    audit = audit.sort_values(
        ["paper_priority", "dataset", "risk_type", "alpha_shift_strength"],
        ascending=[True, True, True, False],
    )
    return audit[ordered]


def _priority(row: pd.Series) -> str:
    if row["generator_ci_class"] == "stable_negative_transfer" and row["subset"] != "real":
        return "P0"
    if row["category"] in {"real_alpha_low", "real_score_decreased_by_alpha"} and row["alpha_score"] < 0.02:
        return "P0"
    if row["generator_ci_class"] == "boundary_ci_crosses_zero":
        return "P1"
    if row["score_conflict_strength"] >= 0.25:
        return "P1"
    return "P2"


def _recommended_use(row: pd.Series) -> str:
    if row["paper_priority"] == "P0" and row["subset"] != "real":
        return "failure figure: stable generator-level negative transfer or hard generated case"
    if row["paper_priority"] == "P0" and row["subset"] == "real":
        return "failure figure: real-video false-positive / recall boundary"
    if row["risk_type"] == "patch_global_conflict":
        return "mechanism audit: patch-global disagreement"
    if row["risk_type"] == "residual_generated_hard_case":
        return "limitation audit: generated video remains high-scoring"
    return "supplementary audit candidate"


def summarize(audit: pd.DataFrame) -> pd.DataFrame:
    return (
        audit.groupby(["dataset", "paper_priority", "risk_type", "generator_ci_class"], dropna=False)
        .size()
        .reset_index(name="n_cases")
        .sort_values(["dataset", "paper_priority", "risk_type", "generator_ci_class"])
    )


def write_markdown(audit: pd.DataFrame, summary: pd.DataFrame, out_path: Path) -> None:
    lines = [
        "# Failure / boundary case 审计",
        "",
        "本审计合并 failure candidate、逐生成器 paired bootstrap、生成器宏平均 bootstrap 和 index 元数据。",
        "它不观看视频、不重新提取特征、不参与调参，只用于选择论文 failure-mode 案例和限制性分析。",
        "",
        "## 审计口径",
        "",
        "- `P0`：优先进入正文或主补充图的案例；包括稳定负迁移生成器的生成样本，以及极低分真实样本。",
        "- `P1`：适合进入补充材料的边界案例；包括 CI 跨 0 的生成器或强 patch/global 冲突。",
        "- `P2`：保留为扩展审计，不建议占用正文版面。",
        "",
        "## 数量汇总",
        "",
        "| dataset | priority | risk type | generator CI class | cases |",
        "|---|---|---|---|---:|",
    ]
    for row in summary.itertuples(index=False):
        lines.append(
            f"| {DISPLAY[row.dataset]} | {row.paper_priority} | {row.risk_type} | "
            f"{row.generator_ci_class} | {row.n_cases} |"
        )

    top_parts = []
    p0 = audit[audit["paper_priority"] == "P0"].copy()
    for (_, risk_type), group in p0.groupby(["dataset", "risk_type"]):
        if risk_type in {"generated_false_real_risk", "patch_global_conflict"}:
            group = group.sort_values("alpha_minus_global_score", ascending=False)
        elif risk_type == "residual_generated_hard_case":
            group = group.sort_values("alpha_score", ascending=False)
        else:
            group = group.sort_values("alpha_score", ascending=True)
        top_parts.append(group.head(4))
    top = (
        pd.concat(top_parts, ignore_index=True)
        .drop_duplicates(["dataset", "subset", "source_model", "filename"], keep="first")
        .sort_values(["dataset", "risk_type", "source_model", "filename"])
    )
    lines.extend(
        [
            "",
            "## P0 推荐案例",
            "",
            "| dataset | source | file | risk | global | patch | alpha | gen ΔAUC CI | recommended use |",
            "|---|---|---|---|---:|---:|---:|---:|---|",
        ]
    )
    for row in top.itertuples(index=False):
        ci = (
            "n/a"
            if pd.isna(row.generator_delta_auc_ci_low)
            else f"[{row.generator_delta_auc_ci_low:+.4f}, {row.generator_delta_auc_ci_high:+.4f}]"
        )
        lines.append(
            f"| {DISPLAY[row.dataset]} | {row.subset}/{row.source_model} | `{row.filename}` | {row.risk_type} | "
            f"{row.global_score:.4f} | {row.patch_score:.4f} | {row.alpha_score:.4f} | {ci} | {row.recommended_use} |"
        )

    lines.extend(
        [
            "",
            "## 写作建议",
            "",
            "- 正文可以用宏平均 paired bootstrap 说明三数据集总体提升均稳定为正。",
            "- failure-mode 小节不要否认 VideoFeedback 内部负迁移；应明确 Text2Video-Zero / VideoCrafter2 是稳定负迁移来源。",
            "- patch-global conflict 案例适合展示局部二阶时序证据可能过强，从而把生成视频推向真实侧。",
            "- 真实视频低分案例适合写成 real-domain / low-motion / compression boundary，下一步若要更强证据需人工观看关键帧。",
        ]
    )
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def plot_priority_counts(summary: pd.DataFrame, out_stem: Path) -> None:
    counts = summary.groupby(["dataset", "paper_priority"])["n_cases"].sum().unstack(fill_value=0)
    for col in ("P0", "P1", "P2"):
        if col not in counts.columns:
            counts[col] = 0
    counts = counts.loc[list(DATASETS), ["P0", "P1", "P2"]]
    fig, ax = plt.subplots(figsize=(5.6, 3.2))
    bottom = None
    colors = {"P0": "#D55E00", "P1": "#E69F00", "P2": "#999999"}
    x = range(len(counts))
    for col in ["P0", "P1", "P2"]:
        ax.bar(x, counts[col], bottom=bottom, label=col, color=colors[col])
        bottom = counts[col] if bottom is None else bottom + counts[col]
    ax.set_xticks(list(x))
    ax.set_xticklabels([DISPLAY[d] for d in counts.index])
    ax.set_ylabel("Number of audited candidates")
    ax.set_title("Failure audit priority distribution")
    ax.legend(frameon=False)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    out_stem.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_stem.with_suffix(".svg"))
    fig.savefig(out_stem.with_suffix(".png"), dpi=220)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[3])
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path(__file__).resolve().parents[3]
        / "results/journal_experiments/failure_case_audit",
    )
    args = parser.parse_args()
    root = args.root.resolve()
    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    audit = build_audit(root)
    summary = summarize(audit)
    audit.to_csv(out_dir / "failure_case_audit.csv", index=False)
    summary.to_csv(out_dir / "failure_case_audit_summary.csv", index=False)
    write_markdown(audit, summary, out_dir / "failure_case_audit.md")
    plot_priority_counts(summary, root / "results/paper_figures/failure_case_audit_priority")
    print(f"saved -> {out_dir}")


if __name__ == "__main__":
    main()
