#!/usr/bin/env python3
"""从 per-source metrics CSV 构建数据集级汇总。"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--metrics-csv", type=Path, nargs="+", required=True)
    parser.add_argument("--output-csv", type=Path, required=True)
    args = parser.parse_args()

    rows = []
    for path in args.metrics_csv:
        df = pd.read_csv(path)
        missing = [c for c in ["dataset", "source_model", "auc", "ap", "n_fake"] if c not in df.columns]
        if missing:
            raise ValueError(f"{path} 缺少列: {missing}")
        avg = df[df["source_model"] == "Average"]
        sources = df[df["source_model"] != "Average"]
        if len(avg) != 1:
            raise ValueError(f"{path} 需要恰好一个 Average 行，实际为 {len(avg)}")
        avg_row = avg.iloc[0]
        rows.append(
            {
                "dataset": avg_row["dataset"],
                "avg_auc": float(avg_row["auc"]),
                "avg_ap": float(avg_row["ap"]),
                "min_source_auc": float(sources["auc"].min()),
                "min_source_ap": float(sources["ap"].min()),
                "n_sources": int(len(sources)),
                "n_fake": int(sources["n_fake"].sum()),
            }
        )
    out = pd.DataFrame(rows)
    out.loc[len(out)] = {
        "dataset": "Mean",
        "avg_auc": float(out["avg_auc"].mean()),
        "avg_ap": float(out["avg_ap"].mean()),
        "min_source_auc": float(out["min_source_auc"].min()),
        "min_source_ap": float(out["min_source_ap"].min()),
        "n_sources": int(out["n_sources"].sum()),
        "n_fake": int(out["n_fake"].sum()),
    }
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.output_csv, index=False)
    print(out.to_string(index=False, float_format=lambda x: f"{x:.6f}"))
    print(f"已保存汇总 -> {args.output_csv}")


if __name__ == "__main__":
    main()
