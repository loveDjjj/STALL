#!/usr/bin/env python3
"""融合全局 STALL 与 patch STALL 分数，并计算论文指标。

当全局和 patch 逐视频 score CSV 已生成后，本文件是 Alpha-STALLED 的轻量主入口。
该路径刻意不使用测试批次 rank、依赖 subset 的路由或来源特定 gate。

分数方向：分数越高，视频越接近真实视频。
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from metrics import ScoreDirection, build_results_table


KEY_COLUMNS = ["subset", "source_model", "filename"]


def _read_scores(path: Path, score_col: str, prefix: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    missing = [c for c in KEY_COLUMNS + [score_col] if c not in df.columns]
    if missing:
        raise ValueError(f"{path} 缺少列: {missing}")
    out = df[KEY_COLUMNS + [score_col]].copy()
    for col in KEY_COLUMNS:
        out[col] = out[col].astype(str)
    return out.rename(columns={score_col: f"{prefix}_score"})


def fuse_scores(
    global_csv: Path,
    patch_csv: Path,
    alpha: float,
    global_score_col: str,
    patch_score_col: str,
) -> pd.DataFrame:
    if not 0.0 <= alpha <= 1.0:
        raise ValueError(f"alpha 必须位于 [0,1]，当前为 {alpha}")

    global_df = _read_scores(global_csv, global_score_col, "global")
    patch_df = _read_scores(patch_csv, patch_score_col, "patch")
    merged = global_df.merge(patch_df, on=KEY_COLUMNS, how="inner", validate="one_to_one")
    if len(merged) != len(global_df) or len(merged) != len(patch_df):
        raise ValueError(
            f"Global/patch CSV 交集不一致: merged={len(merged)}, "
            f"global={len(global_df)}, patch={len(patch_df)}"
        )
    merged["alpha"] = float(alpha)
    merged["final_score"] = alpha * merged["global_score"] + (1.0 - alpha) * merged["patch_score"]
    return merged


def compute_metrics(df: pd.DataFrame, seed: int) -> pd.DataFrame:
    return build_results_table(
        df[["subset", "source_model", "final_score"]],
        {"final_score": ScoreDirection.HIGHER_IS_REAL},
        seed=seed,
        skip_global_compare=True,
        verbose=False,
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="为 Alpha-STALLED 融合全局 STALL 与 patch STALL 分数。"
    )
    parser.add_argument("--global-csv", type=Path, required=True)
    parser.add_argument("--patch-csv", type=Path, required=True)
    parser.add_argument("--output-csv", type=Path, required=True)
    parser.add_argument("--metrics-csv", type=Path, required=True)
    parser.add_argument("--alpha", type=float, required=True)
    parser.add_argument("--global-score-col", default="final_score")
    parser.add_argument("--patch-score-col", default="patch_final_score")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    fused = fuse_scores(
        args.global_csv,
        args.patch_csv,
        alpha=args.alpha,
        global_score_col=args.global_score_col,
        patch_score_col=args.patch_score_col,
    )
    metrics = compute_metrics(fused, seed=args.seed)

    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    args.metrics_csv.parent.mkdir(parents=True, exist_ok=True)
    fused.to_csv(args.output_csv, index=False)
    metrics.to_csv(args.metrics_csv, index=False)

    avg = metrics[metrics["Generative Model"] == "Average"]
    if not avg.empty:
        row = avg.iloc[0]
        print(
            f"Alpha-STALLED alpha={args.alpha:.3f}: "
            f"AUC={row['final_score AUC']:.4f} AP={row['final_score AP']:.4f}"
        )
    print(f"已保存融合分数: {args.output_csv}")
    print(f"已保存指标: {args.metrics_csv}")


if __name__ == "__main__":
    main()
