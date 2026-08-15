#!/usr/bin/env python3
"""按论文指标协议评估已有逐视频 score CSV。

Alpha-STALLED 手稿使用 pairwise real-vs-generator 比较，并对真实视频做平衡采样；
具体实现位于 ``src/metrics.py``。本工具让组件消融和局部时序消融复用同一协议，
避免在公开路径中保留一次性分析脚本。

期望 CSV 列：
  - subset: "real" or "annotated"
  - source_model: 真实来源或生成器名称
  - ``--score-col`` 指定的分数列

默认分数方向为“越高越接近真实视频”。
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from alpha_stalled.score_csv import evaluate_score_csv


def main() -> None:
    parser = argparse.ArgumentParser(
        description="使用 Alpha-STALLED 论文指标评估一个 score CSV。"
    )
    parser.add_argument("--csv", type=Path, required=True)
    parser.add_argument("--score-col", required=True)
    parser.add_argument("--output-csv", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--higher-is", choices=["real", "fake"], default="real")
    parser.add_argument("--include-global-compare", action="store_true")
    args = parser.parse_args()

    metrics = evaluate_score_csv(
        args.csv,
        args.score_col,
        seed=args.seed,
        higher_is=args.higher_is,
        skip_global_compare=not args.include_global_compare,
    )
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    metrics.to_csv(args.output_csv, index=False)

    avg = metrics[metrics["Generative Model"] == "Average"]
    if not avg.empty:
        row = avg.iloc[0]
        print(
            f"{args.csv} [{args.score_col}]: "
            f"AUC={row[f'{args.score_col} AUC']:.4f} "
            f"AP={row[f'{args.score_col} AP']:.4f}"
        )
    print(f"已保存指标: {args.output_csv}")


if __name__ == "__main__":
    main()
