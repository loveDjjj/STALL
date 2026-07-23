#!/usr/bin/env python3
"""在同一批 D3 smoke 样本上比较 D3-style 二阶分数与既有 STALL/Alpha-STALLED 分数。

用途：

1. 读取 `eval_d3_exact_from_frames.py` 生成的 per-video scores；
2. 按 `subset/source_model/filename` 合并已有三分支分数；
3. 在每个 fake source 的同一组 real/fake 视频上计算 AP/AUC；
4. 输出 per-source、macro 和 markdown 说明。

该脚本用于协议诊断，不替代 D3-XCLIP 全量复现。默认输入通常是 20 real +
20 fake/source 的 smoke test，因此结果只能说明样本对齐后的趋势。
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_EVAL_DIR = (
    REPO_ROOT
    / "results/journal_experiments/global_second_order_volatility/genvideo/"
    / "d3_comparable_protocol/d3_exact_protocol/d3_exact_eval"
)
DEFAULT_THREE_BRANCH = (
    REPO_ROOT
    / "results/journal_experiments/global_second_order_volatility/genvideo/"
    / "fallback_patch/three_branch/genvideo_fallback_global_patch500_volatility_l2_merged_scores.csv"
)
DEFAULT_VOLATILITY_RAW = (
    REPO_ROOT
    / "results/journal_experiments/global_second_order_volatility/genvideo/"
    / "genvideo_holdout_d3style_dinov3_l2_2s_fallback_1s_scores.csv"
)


METHOD_WEIGHTS: dict[str, tuple[float, float, float] | None] = {
    "d3_raw": None,
    "existing_volatility_raw": None,
    "global_only": (1.0, 0.0, 0.0),
    "patch_only": (0.0, 1.0, 0.0),
    "volatility_only": (0.0, 0.0, 1.0),
    "global_vol_0p55_0p45": (0.55, 0.0, 0.45),
    "global_vol_0p45_0p55": (0.45, 0.0, 0.55),
    "three_default_0p30_0p35_0p35": (0.30, 0.35, 0.35),
    "three_vol_patch_heavy_0p20_0p35_0p45": (0.20, 0.35, 0.45),
}


def infer_source(path: Path) -> str:
    return path.name.replace("_head1000_DINOv3-L-local_l2_scores.csv", "")


def load_d3_scores(eval_dir: Path, pattern: str) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for path in sorted(eval_dir.glob(pattern)):
        df = pd.read_csv(path)
        if df.empty:
            continue
        source = infer_source(path)
        df["eval_source_model"] = source
        df["subset"] = np.where(df["label"].astype(int).eq(0), "real", "annotated")
        # real rows keep source_model=MSR-VTT; fake rows use the evaluated generator.
        df["source_model"] = np.where(df["subset"].eq("real"), "MSR-VTT", source)
        frames.append(df)
    if not frames:
        raise FileNotFoundError(f"{eval_dir} 中没有匹配 {pattern} 的 D3 score CSV")
    out = pd.concat(frames, ignore_index=True)
    out["filename"] = out["filename"].astype(str)
    return out


def load_three_branch(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    required = {"subset", "source_model", "filename", "global_score", "patch_score", "volatility_score"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"{path} 缺少列: {sorted(missing)}")
    for col in ["subset", "source_model", "filename"]:
        df[col] = df[col].astype(str)
    for col in ["global_score", "patch_score", "volatility_score"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df[list(required)]


def load_existing_volatility_raw(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    required = {"subset", "source_model", "filename", "global_second_order_volatility_raw"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"{path} 缺少列: {sorted(missing)}")
    for col in ["subset", "source_model", "filename"]:
        df[col] = df[col].astype(str)
    df["existing_volatility_raw"] = pd.to_numeric(
        df["global_second_order_volatility_raw"], errors="coerce"
    )
    return df[["subset", "source_model", "filename", "existing_volatility_raw"]]


def add_method_scores(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["d3_raw"] = pd.to_numeric(out["d3_second_order_std"], errors="coerce")
    out["existing_volatility_raw"] = pd.to_numeric(out["existing_volatility_raw"], errors="coerce")
    for name, weights in METHOD_WEIGHTS.items():
        if weights is None:
            continue
        wg, wp, wv = weights
        out[name] = wg * out["global_score"] + wp * out["patch_score"] + wv * out["volatility_score"]
    return out


def score_metrics(df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, float | str | int]] = []
    for source, sub in df.groupby("eval_source_model", sort=True):
        y_real = 1 - sub["label"].astype(int).to_numpy()
        for method in METHOD_WEIGHTS:
            valid = sub[["label", method]].replace([np.inf, -np.inf], np.nan).dropna()
            if valid["label"].nunique() != 2:
                continue
            y_real_valid = 1 - valid["label"].astype(int).to_numpy()
            score = valid[method].to_numpy(dtype=np.float64)
            rows.append(
                {
                    "source_model": source,
                    "method": method,
                    "n_total": int(len(sub)),
                    "n_valid": int(len(valid)),
                    "n_real": int((valid["label"].astype(int) == 0).sum()),
                    "n_fake": int((valid["label"].astype(int) == 1).sum()),
                    "ap": float(average_precision_score(y_real_valid, score)),
                    "auc": float(roc_auc_score(y_real_valid, score)),
                }
            )
        del y_real
    return pd.DataFrame(rows)


def macro_summary(per_source: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for method, sub in per_source.groupby("method", sort=False):
        rows.append(
            {
                "method": method,
                "n_sources": int(sub["source_model"].nunique()),
                "mean_ap": float(sub["ap"].mean()),
                "mean_auc": float(sub["auc"].mean()),
                "min_ap": float(sub["ap"].min()),
                "min_ap_source": str(sub.loc[sub["ap"].idxmin(), "source_model"]),
            }
        )
    out = pd.DataFrame(rows)
    order = {name: i for i, name in enumerate(METHOD_WEIGHTS)}
    out["_order"] = out["method"].map(order)
    return out.sort_values(["mean_ap", "_order"], ascending=[False, True]).drop(columns=["_order"])


def write_markdown(path: Path, per_source: pd.DataFrame, macro: pd.DataFrame, pattern: str) -> None:
    size_ref = per_source[per_source["method"] == "d3_raw"].copy()
    if not size_ref.empty:
        total_rows = int(size_ref["n_valid"].sum())
        real_min = int(size_ref["n_real"].min())
        real_max = int(size_ref["n_real"].max())
        fake_min = int(size_ref["n_fake"].min())
        fake_max = int(size_ref["n_fake"].max())
        size_text = (
            f"合计 `{total_rows}` 个 source-pair rows；每源 real 为 `{real_min}`–`{real_max}`，"
            f"fake 为 `{fake_min}`–`{fake_max}`。"
        )
    else:
        size_text = "具体样本数见 per-source CSV。"
    lines = [
        "# D3 smoke 与既有 Alpha-STALLED 分数的同样本对齐比较",
        "",
        f"- D3 score pattern: `{pattern}`",
        f"- 比较对象：每个 GenVideo fake source 的同一批 D3 rows；{size_text}",
        "- 指标方向：所有方法均按“分数越高越真实”计算 real AP/AUC。",
        "- 解释边界：这是 smoke/protocol 诊断，不是 D3-XCLIP 全量复现，也不是论文主结果。",
        "",
        "## Macro summary",
        "",
        "| method | mean AP | mean AUC | min AP | min-AP source |",
        "|---|---:|---:|---:|---|",
    ]
    for r in macro.itertuples(index=False):
        lines.append(
            f"| `{r.method}` | {r.mean_ap:.4f} | {r.mean_auc:.4f} | {r.min_ap:.4f} | {r.min_ap_source} |"
        )

    key_methods = [
        "d3_raw",
        "existing_volatility_raw",
        "volatility_only",
        "global_vol_0p45_0p55",
        "three_default_0p30_0p35_0p35",
        "three_vol_patch_heavy_0p20_0p35_0p45",
    ]
    pivot = per_source[per_source["method"].isin(key_methods)].pivot(
        index="source_model", columns="method", values="ap"
    )
    lines.extend(
        [
            "",
            "## Per-source AP",
            "",
            "| source | D3 raw | existing vol raw | existing vol percentile | global+vol | three default | vol+patch-heavy |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for source, row in pivot.sort_index().iterrows():
        lines.append(
            "| "
            + f"{source} | "
            + f"{row.get('d3_raw', np.nan):.4f} | "
            + f"{row.get('existing_volatility_raw', np.nan):.4f} | "
            + f"{row.get('volatility_only', np.nan):.4f} | "
            + f"{row.get('global_vol_0p45_0p55', np.nan):.4f} | "
            + f"{row.get('three_default_0p30_0p35_0p35', np.nan):.4f} | "
            + f"{row.get('three_vol_patch_heavy_0p20_0p35_0p45', np.nan):.4f} |"
        )

    lines.extend(
        [
            "",
            "## 直接结论",
            "",
            "- 如果 D3 raw 在同一样本上仍明显低于三分支，说明 DINOv3-L + D3 二阶统计本身不是 D3 论文高分的充分条件。",
            "- 如果 global+vol 或三分支在同一样本上明显强于 D3 raw，说明当前差距更应优先从 XCLIP encoder 和 D3 全量官方协议验证，而不是继续调融合权重。",
            "- 如果某些源在 D3 raw 上显著强于三分支，可作为下一轮引入 XCLIP/D3-style 分支的优先目标源。",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="同样本比较 D3 smoke 与既有三分支分数。")
    parser.add_argument("--eval-dir", type=Path, default=DEFAULT_EVAL_DIR)
    parser.add_argument("--pattern", default="*_head1000_DINOv3-L-local_l2_scores.csv")
    parser.add_argument("--three-branch-csv", type=Path, default=DEFAULT_THREE_BRANCH)
    parser.add_argument("--volatility-raw-csv", type=Path, default=DEFAULT_VOLATILITY_RAW)
    parser.add_argument(
        "--output-prefix",
        type=Path,
        default=DEFAULT_EVAL_DIR / "d3_exact_DINOv3-L-local_l2_smoke20_aligned_existing_score_comparison",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    d3 = load_d3_scores(args.eval_dir, args.pattern)
    three = load_three_branch(args.three_branch_csv)
    volatility_raw = load_existing_volatility_raw(args.volatility_raw_csv)
    merged = d3.merge(
        three,
        on=["subset", "source_model", "filename"],
        how="left",
        validate="many_to_one",
    )
    merged = merged.merge(
        volatility_raw,
        on=["subset", "source_model", "filename"],
        how="left",
        validate="many_to_one",
    )
    missing = merged[["global_score", "patch_score", "volatility_score"]].isna().any(axis=1)
    if missing.any():
        missing_rows = merged.loc[missing, ["eval_source_model", "subset", "source_model", "filename"]].head(10)
        raise ValueError(f"有 {int(missing.sum())} 行无法合并三分支分数，示例：\n{missing_rows}")

    scored = add_method_scores(merged)
    per = score_metrics(scored)
    macro = macro_summary(per)

    args.output_prefix.parent.mkdir(parents=True, exist_ok=True)
    scored.to_csv(args.output_prefix.with_suffix(".scores.csv"), index=False)
    per.to_csv(args.output_prefix.with_suffix(".per_source.csv"), index=False)
    macro.to_csv(args.output_prefix.with_suffix(".macro.csv"), index=False)
    write_markdown(args.output_prefix.with_suffix(".md"), per, macro, args.pattern)

    print(f"merged_rows={len(scored)} sources={scored['eval_source_model'].nunique()}")
    print(f"scores={args.output_prefix.with_suffix('.scores.csv')}")
    print(f"per_source={args.output_prefix.with_suffix('.per_source.csv')}")
    print(f"macro={args.output_prefix.with_suffix('.macro.csv')}")
    print(f"markdown={args.output_prefix.with_suffix('.md')}")
    print(macro.to_string(index=False))


if __name__ == "__main__":
    main()
