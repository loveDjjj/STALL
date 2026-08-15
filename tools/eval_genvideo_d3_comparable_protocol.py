#!/usr/bin/env python3
"""GenVideo 上接近 D3 评测口径的 CSV-stage 对照。

本脚本不重新抽帧、不重建 embedding，只读取已经落盘的 GenVideo 三分支分数：

1. original STALL / global likelihood；
2. local patch likelihood；
3. D3-style global second-order volatility。

它输出两类结果：

- ``project_pairwise``：项目内统一的 pairwise-balanced 逐生成器宏平均 AUC/AP；
- ``head1000_unbalanced``：接近 D3 官方 ``eval.py`` 的每个生成器
  ``real.head(1000) + fake.head(1000)`` 直接 AUC/AP；
- ``head1000_balanced``：每个生成器 head1000 后再 real/fake 取相同数量。

注意：这仍不是 D3 官方复现。D3 主结果使用 XCLIP-B/16、D3 自己的视频转帧、
中心裁边和 resize 预处理。本脚本用于把“指标与样本上限口径”这一因素从
encoder/预处理因素中拆出来。
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from alpha_stalled.metrics import ScoreDirection, build_results_table


KEY_COLUMNS = ["subset", "source_model", "filename"]


def _read_component(path: Path, score_col: str, out_col: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    missing = [c for c in KEY_COLUMNS + [score_col] if c not in df.columns]
    if missing:
        raise ValueError(f"{path} 缺少列: {missing}")
    out = df[KEY_COLUMNS + [score_col]].copy()
    for col in KEY_COLUMNS:
        out[col] = out[col].astype(str)
    out[score_col] = pd.to_numeric(out[score_col], errors="coerce")
    return out.rename(columns={score_col: out_col})


def load_merged_scores(
    global_csv: Path,
    patch_csv: Path,
    volatility_csv: Path,
    global_score_col: str,
    patch_score_col: str,
    volatility_score_col: str,
) -> pd.DataFrame:
    merged = (
        _read_component(global_csv, global_score_col, "global_score")
        .merge(
            _read_component(patch_csv, patch_score_col, "patch_score"),
            on=KEY_COLUMNS,
            how="inner",
            validate="one_to_one",
        )
        .merge(
            _read_component(volatility_csv, volatility_score_col, "volatility_score"),
            on=KEY_COLUMNS,
            how="inner",
            validate="one_to_one",
        )
        .replace([np.inf, -np.inf], np.nan)
        .dropna(subset=["global_score", "patch_score", "volatility_score"])
    )
    if merged.empty:
        raise ValueError("global/patch/volatility 三分支没有有效交集")
    return merged


def add_fixed_scores(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["global_only"] = out["global_score"]
    out["patch_only"] = out["patch_score"]
    out["volatility_only"] = out["volatility_score"]
    out["global_volatility_0p55_0p45"] = 0.55 * out["global_score"] + 0.45 * out["volatility_score"]
    out["global_volatility_0p45_0p55"] = 0.45 * out["global_score"] + 0.55 * out["volatility_score"]
    out["three_branch_oracle_auc"] = (
        0.30 * out["global_score"] + 0.35 * out["patch_score"] + 0.35 * out["volatility_score"]
    )
    out["three_branch_oracle_ap"] = (
        0.25 * out["global_score"] + 0.30 * out["patch_score"] + 0.45 * out["volatility_score"]
    )
    return out


def load_logo_weights(logo_csv: Path | None) -> dict[str, tuple[float, float, float]] | None:
    if logo_csv is None or not logo_csv.exists():
        return None
    logo = pd.read_csv(logo_csv)
    required = {"heldout", "w_global", "w_patch", "w_volatility"}
    missing = required.difference(logo.columns)
    if missing:
        raise ValueError(f"{logo_csv} 缺少列: {sorted(missing)}")
    return {
        str(r.heldout): (float(r.w_global), float(r.w_patch), float(r.w_volatility))
        for r in logo.itertuples(index=False)
    }


def _score_with_weights(data: pd.DataFrame, weights: tuple[float, float, float]) -> np.ndarray:
    wg, wp, wv = weights
    return (
        wg * data["global_score"].to_numpy(dtype=np.float64)
        + wp * data["patch_score"].to_numpy(dtype=np.float64)
        + wv * data["volatility_score"].to_numpy(dtype=np.float64)
    )


def project_pairwise_metrics(df: pd.DataFrame, method_cols: list[str], seed: int) -> pd.DataFrame:
    frames = []
    for method in method_cols:
        metrics = build_results_table(
            df[["subset", "source_model", method]].rename(columns={method: "final_score"}),
            {"final_score": ScoreDirection.HIGHER_IS_REAL},
            seed=seed,
            skip_global_compare=True,
            verbose=False,
        )
        metrics.insert(0, "method", method)
        metrics.insert(0, "protocol", "project_pairwise")
        frames.append(metrics)
    return pd.concat(frames, ignore_index=True)


def project_logo_metrics(
    df: pd.DataFrame,
    logo_weights: dict[str, tuple[float, float, float]],
    seed: int,
) -> pd.DataFrame:
    """用每个 heldout generator 的 LOGO 权重计算 pairwise-balanced 指标。

    不能把 LOGO 写成普通逐行 score 列，因为真实视频行的 ``source_model`` 是
    MSR-VTT；同一个真实视频在不同 heldout generator 比较中应使用不同权重。
    """
    real_all = df[df["subset"] == "real"].copy()
    fake_all = df[df["subset"] == "annotated"].copy()
    rows = []
    for model, fake_group in fake_all.groupby("source_model", sort=True):
        weights = logo_weights.get(str(model))
        if weights is None:
            continue
        real = real_all.sample(n=min(len(real_all), len(fake_group)), random_state=seed)
        fake = fake_group.head(len(real))
        data = pd.concat([real, fake], ignore_index=True).copy()
        data["three_branch_logo"] = _score_with_weights(data, weights)
        auc, ap = _direct_metrics(data, "three_branch_logo")
        rows.append(
            {
                "protocol": "project_pairwise",
                "method": "three_branch_logo",
                "Generative Model": model,
                "n_real": int((data["subset"] == "real").sum()),
                "n_annotated": int((data["subset"] == "annotated").sum()),
                "n_total": int(len(data)),
                "final_score AUC": auc,
                "final_score AP": ap,
            }
        )
    out = pd.DataFrame(rows)
    avg = {
        "protocol": "project_pairwise",
        "method": "three_branch_logo",
        "Generative Model": "Average",
        "n_real": float(out["n_real"].mean()),
        "n_annotated": float(out["n_annotated"].mean()),
        "n_total": float(out["n_total"].mean()),
        "final_score AUC": float(out["final_score AUC"].mean()),
        "final_score AP": float(out["final_score AP"].mean()),
    }
    return pd.concat([out, pd.DataFrame([avg])], ignore_index=True)


def _direct_metrics(data: pd.DataFrame, score_col: str) -> tuple[float, float]:
    labels = (data["subset"] == "real").astype(np.uint8).to_numpy()
    scores = data[score_col].to_numpy(dtype=np.float64)
    return float(roc_auc_score(labels, scores)), float(average_precision_score(labels, scores))


def direct_head_metrics(
    df: pd.DataFrame,
    method_cols: list[str],
    *,
    cap: int,
    balanced: bool,
) -> pd.DataFrame:
    real_all = df[df["subset"] == "real"].copy()
    fake_all = df[df["subset"] == "annotated"].copy()
    rows = []
    for method in method_cols:
        for model, fake_group in fake_all.groupby("source_model", sort=True):
            fake = fake_group.head(cap)
            real = real_all.head(cap)
            if balanced:
                n = min(len(real), len(fake))
                real = real.head(n)
                fake = fake.head(n)
            data = pd.concat([real, fake], ignore_index=True)
            if data["subset"].nunique() != 2:
                continue
            auc, ap = _direct_metrics(data, method)
            rows.append(
                {
                    "protocol": f"head{cap}_{'balanced' if balanced else 'unbalanced'}",
                    "method": method,
                    "Generative Model": model,
                    "n_real": int((data["subset"] == "real").sum()),
                    "n_annotated": int((data["subset"] == "annotated").sum()),
                    "n_total": int(len(data)),
                    "final_score AUC": auc,
                    "final_score AP": ap,
                }
            )
    out = pd.DataFrame(rows)
    avg_rows = []
    for (protocol, method), group in out.groupby(["protocol", "method"], sort=False):
        avg_rows.append(
            {
                "protocol": protocol,
                "method": method,
                "Generative Model": "Average",
                "n_real": float(group["n_real"].mean()),
                "n_annotated": float(group["n_annotated"].mean()),
                "n_total": float(group["n_total"].mean()),
                "final_score AUC": float(group["final_score AUC"].mean()),
                "final_score AP": float(group["final_score AP"].mean()),
            }
        )
    return pd.concat([out, pd.DataFrame(avg_rows)], ignore_index=True)


def direct_head_logo_metrics(
    df: pd.DataFrame,
    logo_weights: dict[str, tuple[float, float, float]],
    *,
    cap: int,
    balanced: bool,
) -> pd.DataFrame:
    real_all = df[df["subset"] == "real"].copy()
    fake_all = df[df["subset"] == "annotated"].copy()
    rows = []
    for model, fake_group in fake_all.groupby("source_model", sort=True):
        weights = logo_weights.get(str(model))
        if weights is None:
            continue
        fake = fake_group.head(cap)
        real = real_all.head(cap)
        if balanced:
            n = min(len(real), len(fake))
            real = real.head(n)
            fake = fake.head(n)
        data = pd.concat([real, fake], ignore_index=True).copy()
        data["three_branch_logo"] = _score_with_weights(data, weights)
        auc, ap = _direct_metrics(data, "three_branch_logo")
        rows.append(
            {
                "protocol": f"head{cap}_{'balanced' if balanced else 'unbalanced'}",
                "method": "three_branch_logo",
                "Generative Model": model,
                "n_real": int((data["subset"] == "real").sum()),
                "n_annotated": int((data["subset"] == "annotated").sum()),
                "n_total": int(len(data)),
                "final_score AUC": auc,
                "final_score AP": ap,
            }
        )
    out = pd.DataFrame(rows)
    avg = {
        "protocol": f"head{cap}_{'balanced' if balanced else 'unbalanced'}",
        "method": "three_branch_logo",
        "Generative Model": "Average",
        "n_real": float(out["n_real"].mean()),
        "n_annotated": float(out["n_annotated"].mean()),
        "n_total": float(out["n_total"].mean()),
        "final_score AUC": float(out["final_score AUC"].mean()),
        "final_score AP": float(out["final_score AP"].mean()),
    }
    return pd.concat([out, pd.DataFrame([avg])], ignore_index=True)


def make_summary(metrics: pd.DataFrame, d3_paper_genvideo_map: float) -> pd.DataFrame:
    avg = metrics[metrics["Generative Model"] == "Average"].copy()
    avg = avg.rename(
        columns={
            "final_score AUC": "mean_auc",
            "final_score AP": "mean_ap",
        }
    )
    avg["d3_paper_genvideo_map"] = float(d3_paper_genvideo_map)
    avg["ap_gap_to_d3_paper"] = avg["mean_ap"] - float(d3_paper_genvideo_map)
    return avg[
        [
            "protocol",
            "method",
            "n_real",
            "n_annotated",
            "mean_auc",
            "mean_ap",
            "d3_paper_genvideo_map",
            "ap_gap_to_d3_paper",
        ]
    ].sort_values(["protocol", "mean_ap"], ascending=[True, False])


def write_markdown(
    path: Path,
    summary: pd.DataFrame,
    method_metrics: pd.DataFrame,
    merged: pd.DataFrame,
    d3_paper_genvideo_map: float,
    cap: int,
    logo_used: bool,
) -> None:
    lines = [
        "# GenVideo D3 可比口径诊断",
        "",
        "本目录只做 CSV-stage 诊断，不重新抽视频、不重新提取 embedding。目标是拆分 D3 与当前 Alpha-STALLED 之间的差距来源：",
        "",
        "1. 指标与样本上限口径；",
        "2. encoder / 预处理 / 抽帧协议；",
        "3. global second-order 与 local patch likelihood 的互补性。",
        "",
        "## 输入覆盖",
        "",
        f"- 三分支交集行数：`{len(merged)}`。",
        f"- real 行数：`{int((merged['subset'] == 'real').sum())}`。",
        f"- fake 行数：`{int((merged['subset'] == 'annotated').sum())}`。",
        f"- D3 官方代码近似口径：每个生成器 `real.head({cap}) + fake.head({cap})` 直接计算 AP/AUC。",
        f"- D3 论文 GenVideo mAP 参考值：`{d3_paper_genvideo_map:.4f}`。",
        f"- 是否使用三分支 LOGO 权重：`{logo_used}`。",
        "",
        "## 平均结果",
        "",
        "| protocol | method | mean AUC | mean AP | AP gap vs D3 paper |",
        "|---|---|---:|---:|---:|",
    ]
    for row in summary.itertuples(index=False):
        lines.append(
            f"| {row.protocol} | {row.method} | {row.mean_auc:.4f} | {row.mean_ap:.4f} | {row.ap_gap_to_d3_paper:+.4f} |"
        )
    lines.extend(
        [
            "",
            "## 解释",
            "",
            "- `head1000_unbalanced` 最接近 D3 官方 `eval.py` 的 CSV 采样方式，但仍沿用本项目已有 DINOv3/STALLED 分数，不等于 D3 官方 XCLIP-B/16 复现。",
            "- `head1000_balanced` 用于排除 Sora、WildScrape 等短源在 unbalanced real=1000 设置下造成的 AP 稀释或放大。",
            "- 如果 `head1000_*` 相比 `project_pairwise` 明显变化，说明样本上限和直接 AP 口径能解释部分差距；剩余差距主要应继续从 D3-exact 抽帧/裁边/XCLIP 特征复现中验证。",
            "- 当前三分支结果若持续高于 global+volatility，说明 local patch likelihood 不是 D3-style global second-order 的重复项，而是互补证据。",
            "",
            "## 输出文件",
            "",
            "- `genvideo_d3_comparable_method_metrics.csv`：逐生成器和 Average 指标。",
            "- `genvideo_d3_comparable_summary.csv`：每个 protocol/method 的平均结果。",
            "- 本 markdown：协议说明和结果解读。",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="GenVideo D3 可比口径 CSV-stage 诊断。")
    base = REPO_ROOT / "results/journal_experiments/global_second_order_volatility/genvideo"
    parser.add_argument("--global-csv", type=Path, default=base / "genvideo_holdout_global_2s_fallback_1s.csv")
    parser.add_argument(
        "--patch-csv",
        type=Path,
        default=base / "fallback_patch/genvideo_holdout_patch_real500_2s_fallback_1s.csv",
    )
    parser.add_argument(
        "--volatility-csv",
        type=Path,
        default=base / "genvideo_holdout_d3style_dinov3_l2_2s_fallback_1s_scores.csv",
    )
    parser.add_argument(
        "--logo-csv",
        type=Path,
        default=base / "fallback_patch/three_branch/genvideo_fallback_global_patch500_volatility_l2_logo.csv",
    )
    parser.add_argument("--global-score-col", default="final_score")
    parser.add_argument("--patch-score-col", default="patch_final_score")
    parser.add_argument("--volatility-score-col", default="global_second_order_volatility_score")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=base / "d3_comparable_protocol",
    )
    parser.add_argument("--head-cap", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--d3-paper-genvideo-map", type=float, default=0.9846)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    merged = load_merged_scores(
        args.global_csv,
        args.patch_csv,
        args.volatility_csv,
        args.global_score_col,
        args.patch_score_col,
        args.volatility_score_col,
    )
    scored = add_fixed_scores(merged)
    logo_weights = load_logo_weights(args.logo_csv)
    logo_used = logo_weights is not None
    method_cols = [
        "global_only",
        "patch_only",
        "volatility_only",
        "global_volatility_0p55_0p45",
        "global_volatility_0p45_0p55",
        "three_branch_oracle_auc",
        "three_branch_oracle_ap",
    ]
    metric_frames = [
        project_pairwise_metrics(scored, method_cols, args.seed),
        direct_head_metrics(scored, method_cols, cap=args.head_cap, balanced=False),
        direct_head_metrics(scored, method_cols, cap=args.head_cap, balanced=True),
    ]
    if logo_weights is not None:
        metric_frames.extend(
            [
                project_logo_metrics(scored, logo_weights, args.seed),
                direct_head_logo_metrics(scored, logo_weights, cap=args.head_cap, balanced=False),
                direct_head_logo_metrics(scored, logo_weights, cap=args.head_cap, balanced=True),
            ]
        )
    metrics = pd.concat(metric_frames, ignore_index=True)
    summary = make_summary(metrics, args.d3_paper_genvideo_map)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = args.output_dir / "genvideo_d3_comparable_method_metrics.csv"
    summary_path = args.output_dir / "genvideo_d3_comparable_summary.csv"
    md_path = args.output_dir / "genvideo_d3_comparable_protocol.md"
    metrics.to_csv(metrics_path, index=False)
    summary.to_csv(summary_path, index=False)
    write_markdown(md_path, summary, metrics, scored, args.d3_paper_genvideo_map, args.head_cap, logo_used)

    print(f"merged rows={len(scored)} fixed_methods={len(method_cols)} logo_used={logo_used}")
    print(f"已保存逐生成器指标: {metrics_path}")
    print(f"已保存平均摘要: {summary_path}")
    print(f"已保存协议说明: {md_path}")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
