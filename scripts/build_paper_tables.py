#!/usr/bin/env python3
"""从已完成 run 的逐视频分数增量生成论文配对宏平均指标表。"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

import pandas as pd

from config import load_config
from evaluation.tables import build_pairwise_metric_table, normalize_scores


def main() -> None:
    parser = argparse.ArgumentParser(
        description="从 video_scores.csv 重建论文配对宏平均指标，不重跑特征或评分。"
    )
    parser.add_argument("--run-name", required=True, help="results/runs 下已完成运行的名称")
    arguments = parser.parse_args()
    run_dir = ROOT / "results" / "runs" / arguments.run_name
    config_path = run_dir / "resolved_config.yaml"
    scores_path = run_dir / "video_scores.csv"
    if not config_path.is_file() or not scores_path.is_file():
        raise FileNotFoundError("run 必须包含 resolved_config.yaml 与 video_scores.csv")
    config = load_config(config_path)
    # round_trip 保留运行时 float64 分数，避免 CDF 并列附近的末位变化影响排序指标。
    scores = normalize_scores(pd.read_csv(scores_path, float_precision="round_trip"))
    # 早于本脚本的已完成 run 尚未保存该字段；历史 U0 论文协议固定使用 42。
    pairwise_seed = int(config["metrics"].get("pairwise_seed", 42))
    table = build_pairwise_metric_table(
        scores, arguments.run_name, pairwise_seed
    )
    destination = run_dir / "pairwise_metrics.csv"
    table.to_csv(destination, index=False)
    print(f"[完成] 已写入 {destination}")


if __name__ == "__main__":
    main()
