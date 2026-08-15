#!/usr/bin/env python3
"""Summarize the ComGenVid 1s representative duration/window experiment."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[3]


def _read_average(metrics_csv: Path, score_col: str) -> tuple[float, float]:
    df = pd.read_csv(metrics_csv)
    avg = df[df["Generative Model"] == "Average"]
    if avg.empty:
        raise ValueError(f"{metrics_csv} 缺少 Average 行")
    row = avg.iloc[0]
    return float(row[f"{score_col} AUC"]), float(row[f"{score_col} AP"])


def _read_prefill(prefill_csv: Path) -> dict[str, str]:
    if not prefill_csv.exists():
        return {}
    row = pd.read_csv(prefill_csv).iloc[0].to_dict()
    return {k: str(v) for k, v in row.items()}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="生成 ComGenVid 1s 代表性 duration/window 实验摘要。"
    )
    parser.add_argument(
        "--one-sec-metrics",
        type=Path,
        default=REPO_ROOT
        / Path(
            "results/journal_experiments/duration_window_representative/"
            "comgenvid_1s_metrics_full.csv"
        ),
    )
    parser.add_argument(
        "--two-sec-metrics",
        type=Path,
        default=REPO_ROOT / "results/paper_tables/comgenvid_patch_only_metrics.csv",
    )
    parser.add_argument(
        "--prefill-summary",
        type=Path,
        default=REPO_ROOT
        / Path(
            "results/journal_experiments/duration_window_representative/"
            "comgenvid_1s_prefill_full.csv"
        ),
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=REPO_ROOT
        / Path(
            "results/journal_experiments/duration_window_representative/"
            "comgenvid_duration_window_comparison.csv"
        ),
    )
    parser.add_argument(
        "--output-md",
        type=Path,
        default=REPO_ROOT
        / Path(
            "results/journal_experiments/duration_window_representative/"
            "comgenvid_duration_window_representative.md"
        ),
    )
    args = parser.parse_args()

    one_auc, one_ap = _read_average(args.one_sec_metrics, "patch_final_score")
    two_auc, two_ap = _read_average(args.two_sec_metrics, "patch_final_score")

    rows = [
        {
            "dataset": "ComGenVid",
            "branch": "patch-only",
            "duration_sec": 1,
            "setting": "region3_bottomk0p20_same_grid_second_order",
            "auc": one_auc,
            "ap": one_ap,
            "delta_auc_vs_2s": one_auc - two_auc,
            "delta_ap_vs_2s": one_ap - two_ap,
            "source": str(args.one_sec_metrics),
        },
        {
            "dataset": "ComGenVid",
            "branch": "patch-only",
            "duration_sec": 2,
            "setting": "main_release_region3_bottomk0p20_same_grid_second_order",
            "auc": two_auc,
            "ap": two_ap,
            "delta_auc_vs_2s": 0.0,
            "delta_ap_vs_2s": 0.0,
            "source": str(args.two_sec_metrics),
        },
    ]
    comparison = pd.DataFrame(rows)

    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    comparison.to_csv(args.output_csv, index=False)

    prefill = _read_prefill(args.prefill_summary)
    written = prefill.get("written", "NA")
    misses = prefill.get("misses_before", "NA")

    md = [
        "# ComGenVid 1s 代表性 duration/window 实验",
        "",
        "## 目的",
        "",
        (
            "参考 STALL 论文附录中对视频时长和采样窗口的稳定性分析，"
            "本实验在当前 Alpha-STALLED 公开资产上补充一个真实运行的 "
            "ComGenVid 1s 代表性窗口实验。实验只改变 patch 分支的时间窗口："
            "主实验为 2s，本补充实验为 1s；patch 时序特征、区域聚合和 "
            "bottom-k 设置保持为主实验配置。"
        ),
        "",
        "## 设置",
        "",
        "- 数据集：ComGenVid",
        "- 分支：patch-only；不报告 Alpha-STALLED 融合分数，因为当前资产没有匹配的 1s global score。",
        "- 1s 设置：same-grid second-order temporal feature, region size 3, bottomk_mean, bottom-k ratio 0.20, patch spatial/temporal weight 0.10/0.90。",
        "- 2s 对照：`results/paper_tables/comgenvid_patch_only_metrics.csv`。",
        f"- 1s cache prefill：misses_before={misses}, written={written}。",
        "",
        "## 结果",
        "",
        comparison[
            [
                "duration_sec",
                "auc",
                "ap",
                "delta_auc_vs_2s",
                "delta_ap_vs_2s",
                "source",
            ]
        ].to_markdown(index=False, floatfmt=".6f"),
        "",
        "## 解读",
        "",
    ]

    if one_auc >= two_auc - 0.01:
        md.append(
            (
                "1s 窗口下 patch-only 性能与 2s 主实验保持在同一水平，说明当前局部二阶时序证据"
                "并不严格依赖较长采样窗口；这与参考 STALL 论文中短窗口下方法仍保持可用的分析方向一致。"
            )
        )
    else:
        md.append(
            (
                "1s 窗口相对 2s 主实验存在可见下降，说明局部二阶时序证据从更完整的短时动态中获益；"
                "论文中应将 2s 作为默认窗口，并把 1s 结果作为短视频可用性边界而非主结果。"
            )
        )

    md.extend(
        [
            "",
            "## 文件",
            "",
            f"- 1s scores: `{args.one_sec_metrics.parent / 'comgenvid_1s_patch_full.csv'}`",
            f"- 1s metrics: `{args.one_sec_metrics}`",
            f"- comparison CSV: `{args.output_csv}`",
        ]
    )

    args.output_md.write_text("\n".join(md) + "\n", encoding="utf-8")
    print(f"已保存: {args.output_csv}")
    print(f"已保存: {args.output_md}")


if __name__ == "__main__":
    main()
