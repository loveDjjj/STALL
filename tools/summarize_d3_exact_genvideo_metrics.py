#!/usr/bin/env python3
"""汇总 GenVideo D3-exact metrics，并与当前三分支结果对齐比较。"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PROTOCOL_DIR = (
    REPO_ROOT
    / "results/journal_experiments/global_second_order_volatility/genvideo/"
    / "d3_comparable_protocol"
)
DEFAULT_EVAL_DIR = DEFAULT_PROTOCOL_DIR / "d3_exact_protocol/d3_exact_eval"
DEFAULT_CURRENT_METRICS = DEFAULT_PROTOCOL_DIR / "genvideo_d3_comparable_method_metrics.csv"


def infer_source_from_fake_csv(path_value: str) -> str:
    stem = Path(path_value).stem
    return stem.replace("_head1000", "")


def load_d3_exact_metrics(eval_dir: Path, pattern: str) -> pd.DataFrame:
    frames = []
    for path in sorted(eval_dir.glob(pattern)):
        df = pd.read_csv(path)
        if df.empty:
            continue
        row = df.iloc[0].copy()
        row["metrics_path"] = path.name
        row["source_model"] = infer_source_from_fake_csv(str(row.get("fake_csv", path.name)))
        frames.append(row.to_frame().T)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def load_d3_exact_scores(eval_dir: Path, pattern: str) -> pd.DataFrame:
    frames = []
    for path in sorted(eval_dir.glob(pattern)):
        df = pd.read_csv(path)
        if df.empty:
            continue
        df["evaluation_source_model"] = path.name.replace("_head1000_XCLIP-16_l2_scores.csv", "")
        frames.append(df)
    if not frames:
        return pd.DataFrame()
    scores = pd.concat(frames, ignore_index=True)
    key = ["content_path", "label", "source_model", "filename"]
    score_column = "d3_second_order_std"
    real = scores[scores["label"].astype(int) == 0]
    if real.groupby(key, dropna=False)[score_column].nunique().gt(1).any():
        raise ValueError("shared official real videos have inconsistent scores across pairs")
    scores = scores.drop_duplicates(key, keep="first").copy()
    scores["content_path"] = scores["content_path"].map(
        lambda value: str(Path(str(value)).relative_to(REPO_ROOT))
        if Path(str(value)).is_absolute() and Path(str(value)).is_relative_to(REPO_ROOT)
        else str(value)
    )
    return scores.drop(columns=["evaluation_source_model", "row_index"], errors="ignore")


def load_current_reference(path: Path, protocol: str, method: str) -> pd.DataFrame:
    if not path.is_file():
        return pd.DataFrame()
    df = pd.read_csv(path)
    sub = df[
        (df["protocol"] == protocol)
        & (df["method"] == method)
        & (~df["Generative Model"].isin(["Average", "All", "AUC Flipped?"]))
    ].copy()
    return sub.rename(
        columns={
            "Generative Model": "source_model",
            "final_score AUC": f"{method}_auc",
            "final_score AP": f"{method}_ap",
        }
    )[["source_model", f"{method}_auc", f"{method}_ap"]]


def summarize(d3: pd.DataFrame, current: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    if d3.empty:
        return d3, pd.DataFrame()
    keep = [
        "source_model",
        "encoder",
        "loss",
        "n_total",
        "n_ok",
        "n_failed",
        "d3_official_real_ap",
        "real_auc",
        "fake_ap_with_neg_score",
        "fake_auc_with_neg_score",
        "metrics_path",
    ]
    available = [c for c in keep if c in d3.columns]
    per = d3[available].copy()
    if not current.empty:
        per = per.merge(current, on="source_model", how="left")
        if "three_branch_logo_ap" in per.columns:
            per["d3_real_ap_minus_three_branch_logo_ap"] = (
                pd.to_numeric(per["d3_official_real_ap"], errors="coerce")
                - pd.to_numeric(per["three_branch_logo_ap"], errors="coerce")
            )
    numeric_cols = [
        c
        for c in [
            "d3_official_real_ap",
            "real_auc",
            "fake_ap_with_neg_score",
            "fake_auc_with_neg_score",
            "three_branch_logo_auc",
            "three_branch_logo_ap",
            "d3_real_ap_minus_three_branch_logo_ap",
        ]
        if c in per.columns
    ]
    macro = {c: float(pd.to_numeric(per[c], errors="coerce").mean()) for c in numeric_cols}
    macro["n_models"] = int(len(per))
    macro_df = pd.DataFrame([macro])
    return per, macro_df


def write_markdown(path: Path, per: pd.DataFrame, macro: pd.DataFrame, pattern: str) -> None:
    lines = [
        "# GenVideo D3-exact metrics 汇总",
        "",
        f"输入 pattern：`{pattern}`。",
        "",
    ]
    if per.empty:
        lines.extend(
            [
                "当前未找到 D3-exact metrics CSV。",
                "",
                "这通常表示还没有安装依赖、抽帧或执行 `tools/run_d3_exact_genvideo_batch.py --execute`。",
            ]
        )
    else:
        row = macro.iloc[0]
        lines.extend(
            [
                "## Macro summary",
                "",
                f"- models: `{int(row.n_models)}`",
                f"- D3 real AP/mAP: `{row.get('d3_official_real_ap', float('nan')):.4f}`",
                f"- D3 real AUC: `{row.get('real_auc', float('nan')):.4f}`",
            ]
        )
        if "three_branch_logo_ap" in macro.columns:
            lines.append(f"- current three-branch LOGO AP: `{row.three_branch_logo_ap:.4f}`")
        if "d3_real_ap_minus_three_branch_logo_ap" in macro.columns:
            lines.append(
                f"- D3 AP - current three-branch LOGO AP: `{row.d3_real_ap_minus_three_branch_logo_ap:+.4f}`"
            )
        has_current = "three_branch_logo_ap" in per.columns
        lines.extend(["", "## Per-generator", ""])
        if has_current:
            lines.extend(
                [
                    "| source | D3 AP | D3 AUC | current LOGO AP | ΔAP |",
                    "|---|---:|---:|---:|---:|",
                ]
            )
        else:
            lines.extend(["| source | D3 AP | D3 AUC |", "|---|---:|---:|"])
        for r in per.itertuples(index=False):
            d3_ap = getattr(r, "d3_official_real_ap", float("nan"))
            d3_auc = getattr(r, "real_auc", float("nan"))
            if has_current:
                cur_ap = getattr(r, "three_branch_logo_ap", float("nan"))
                delta = getattr(r, "d3_real_ap_minus_three_branch_logo_ap", float("nan"))
                lines.append(
                    f"| {r.source_model} | {d3_ap:.4f} | {d3_auc:.4f} | {cur_ap:.4f} | {delta:+.4f} |"
                )
            else:
                lines.append(f"| {r.source_model} | {d3_ap:.4f} | {d3_auc:.4f} |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="汇总 GenVideo D3-exact metrics。")
    parser.add_argument("--eval-dir", type=Path, default=DEFAULT_EVAL_DIR)
    parser.add_argument("--pattern", default="*_XCLIP-16_l2_metrics.csv")
    parser.add_argument("--score-pattern", default="*_XCLIP-16_l2_scores.csv")
    parser.add_argument("--current-metrics", type=Path, default=DEFAULT_CURRENT_METRICS)
    parser.add_argument("--current-protocol", default="head1000_unbalanced")
    parser.add_argument("--current-method", default="three_branch_logo")
    parser.add_argument("--output-per-generator-csv", type=Path)
    parser.add_argument("--output-macro-csv", type=Path)
    parser.add_argument("--output-md", type=Path)
    parser.add_argument("--output-per-video-csv", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.eval_dir.mkdir(parents=True, exist_ok=True)
    if args.output_per_generator_csv is None:
        args.output_per_generator_csv = args.eval_dir / "d3_exact_genvideo_per_generator_summary.csv"
    if args.output_macro_csv is None:
        args.output_macro_csv = args.eval_dir / "d3_exact_genvideo_macro_summary.csv"
    if args.output_md is None:
        args.output_md = args.eval_dir / "d3_exact_genvideo_summary.md"

    d3 = load_d3_exact_metrics(args.eval_dir, args.pattern)
    scores = load_d3_exact_scores(args.eval_dir, args.score_pattern)
    current = load_current_reference(args.current_metrics, args.current_protocol, args.current_method)
    per, macro = summarize(d3, current)
    per.to_csv(args.output_per_generator_csv, index=False)
    macro.to_csv(args.output_macro_csv, index=False)
    write_markdown(args.output_md, per, macro, args.pattern)
    if args.output_per_video_csv is not None:
        scores.to_csv(args.output_per_video_csv, index=False)
    print(f"d3 metrics found={len(d3)}")
    print(f"per_generator={args.output_per_generator_csv}")
    print(f"macro={args.output_macro_csv}")
    print(f"markdown={args.output_md}")
    if args.output_per_video_csv is not None:
        print(f"per_video={args.output_per_video_csv} rows={len(scores)}")
    if not macro.empty:
        print(macro.to_string(index=False))


if __name__ == "__main__":
    main()
