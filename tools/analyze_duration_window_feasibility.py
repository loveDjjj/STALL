"""审计 duration/window 敏感性实验的现有可行性。

该脚本只读取 index CSV 与已有 compact patch cache，不重新提取 DINOv3 特征。
目标是回答：

1. 每个数据集在 1/2/3/4 秒窗口上有多少索引可用；
2. 对应 compact patch cache 覆盖多少；
3. 是否满足真实视频校准与生成视频评测的最低条件；
4. 若不能直接实跑 duration/window 敏感性，缺口在哪里。
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


DATASETS = ("comgenvid", "videofeedback", "genvideo")
DURATIONS = (1, 2, 3, 4)


def _cache_count(cache_root: Path, subset: str, source_model: str, duration: int) -> int:
    root = cache_root / subset / source_model
    if not root.exists():
        return 0
    return sum(1 for _ in root.glob(f"*_{duration}s.pt"))


def build_audit(root: Path) -> pd.DataFrame:
    rows: list[dict] = []
    for dataset in DATASETS:
        index_path = root / f"cache/indexes/{dataset}.csv"
        cache_root = root / f"cache/patch_embeddings/{dataset}"
        if not index_path.exists():
            continue
        df = pd.read_csv(index_path)
        for duration in DURATIONS:
            window_col = f"{duration}_sec_idxs"
            if window_col not in df.columns:
                continue
            eligible = df[df[window_col].notna()].copy()
            for (subset, source_model), group in eligible.groupby(["subset", "source_model"]):
                indexed = int(len(group))
                cached = _cache_count(cache_root, str(subset), str(source_model), duration)
                rows.append(
                    {
                        "dataset": dataset,
                        "duration_sec": duration,
                        "subset": subset,
                        "source_model": source_model,
                        "indexed_videos": indexed,
                        "compact_patch_cache": cached,
                        "cache_coverage": cached / indexed if indexed else 0.0,
                    }
                )
    return pd.DataFrame(rows).sort_values(["dataset", "duration_sec", "subset", "source_model"]).reset_index(drop=True)


def summarize(audit: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict] = []
    grouped = {
        key: group
        for key, group in audit.groupby(["dataset", "duration_sec"])
    }
    for dataset in DATASETS:
        for duration in DURATIONS:
            group = grouped.get((dataset, duration), pd.DataFrame(columns=audit.columns))
            if group.empty:
                rows.append(
                    {
                        "dataset": dataset,
                        "duration_sec": duration,
                        "real_indexed": 0,
                        "fake_indexed": 0,
                        "real_cached": 0,
                        "fake_cached": 0,
                        "real_sources": 0,
                        "fake_sources": 0,
                        "direct_patch_eval_ready": False,
                        "note": _note(dataset, duration, 0, 0, 0, 0),
                    }
                )
                continue
            real_indexed = int(group.loc[group["subset"] == "real", "indexed_videos"].sum())
            fake_indexed = int(group.loc[group["subset"] != "real", "indexed_videos"].sum())
            real_cached = int(group.loc[group["subset"] == "real", "compact_patch_cache"].sum())
            fake_cached = int(group.loc[group["subset"] != "real", "compact_patch_cache"].sum())
            real_sources = int(group.loc[group["subset"] == "real", "source_model"].nunique())
            fake_sources = int(group.loc[group["subset"] != "real", "source_model"].nunique())
            can_calibrate = real_cached > 0
            can_evaluate = fake_cached > 0
            direct_patch_eval_ready = can_calibrate and can_evaluate
            rows.append(
                {
                    "dataset": dataset,
                    "duration_sec": duration,
                    "real_indexed": real_indexed,
                    "fake_indexed": fake_indexed,
                    "real_cached": real_cached,
                    "fake_cached": fake_cached,
                    "real_sources": real_sources,
                    "fake_sources": fake_sources,
                    "direct_patch_eval_ready": direct_patch_eval_ready,
                    "note": _note(dataset, duration, real_indexed, fake_indexed, real_cached, fake_cached),
                }
            )
    return pd.DataFrame(rows).sort_values(["dataset", "duration_sec"]).reset_index(drop=True)


def _note(
    dataset: str,
    duration: int,
    real_indexed: int,
    fake_indexed: int,
    real_cached: int,
    fake_cached: int,
) -> str:
    if real_cached > 0 and fake_cached > 0:
        return "已有真实与生成 compact patch cache，可直接创建该 duration 的 patch params 并评测。"
    if real_indexed == 0 and fake_indexed == 0:
        return "index 无该时长窗口；需重新定义窗口或更换数据。"
    if real_cached == 0 and fake_cached > 0:
        return "只有生成 cache，缺真实 cache，不能做真实视频校准。"
    if real_cached > 0 and fake_cached == 0:
        return "只有真实 cache，缺生成 cache，不能完成检测评测。"
    return "index 有窗口但缺 compact patch cache；需先 prefill 对应 duration cache。"


def write_markdown(summary: pd.DataFrame, audit: pd.DataFrame, out_path: Path) -> None:
    lines = [
        "# Duration / window 敏感性可行性审计",
        "",
        "本审计只读取 index CSV 与已有 compact patch cache，不重新提取 DINOv3 特征。",
        "结论用于决定是否值得立即实跑 1s/2s/3s/4s duration 敏感性实验。",
        "",
        "## 数据集级结论",
        "",
        "| dataset | duration | real indexed | fake indexed | real cache | fake cache | 可直接 patch eval | 说明 |",
        "|---|---:|---:|---:|---:|---:|---|---|",
    ]
    for row in summary.itertuples(index=False):
        ready = "yes" if row.direct_patch_eval_ready else "no"
        lines.append(
            f"| {row.dataset} | {row.duration_sec}s | {row.real_indexed} | {row.fake_indexed} | "
            f"{row.real_cached} | {row.fake_cached} | {ready} | {row.note} |"
        )

    lines.extend(
        [
            "",
            "## 关键判断",
            "",
            "- 当前三数据集只有 2s compact patch cache 达到主实验可用覆盖，因此主线 2s 结果是唯一已经完整落地的 duration。",
            "- VideoFeedback 虽有 1s index 全覆盖，但已有 1s patch cache 只覆盖 Hotshot-XL 生成视频，缺真实视频 cache，不能进行真实视频校准。",
            "- GenVideo 与 ComGenVid 的 index 支持 1s/3s/4s 窗口，但 compact patch cache 目前只存在 2s，因此直接实跑 duration sweep 会触发大规模 cache prefill。",
            "- 若需要投期刊前补 duration/window 敏感性，建议先只补一个代表数据集的 1s/2s 对照，并把 3s/4s 作为计算预算较高的扩展实验。",
            "",
            "## VideoFeedback 1s cache 覆盖明细",
            "",
            "| subset/source | indexed 1s | cached 1s | coverage |",
            "|---|---:|---:|---:|",
        ]
    )
    vf1 = audit[(audit["dataset"] == "videofeedback") & (audit["duration_sec"] == 1)].copy()
    for row in vf1.sort_values(["subset", "source_model"]).itertuples(index=False):
        lines.append(
            f"| {row.subset}/{row.source_model} | {row.indexed_videos} | "
            f"{row.compact_patch_cache} | {row.cache_coverage:.3f} |"
        )

    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "results/journal_experiments/duration_window_feasibility",
    )
    args = parser.parse_args()
    root = args.root.resolve()
    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    audit = build_audit(root)
    summary = summarize(audit)
    audit.to_csv(out_dir / "duration_window_cache_audit.csv", index=False)
    summary.to_csv(out_dir / "duration_window_summary.csv", index=False)
    write_markdown(summary, audit, out_dir / "duration_window_feasibility.md")
    print(f"rows: audit={len(audit)}, summary={len(summary)}")
    print(f"saved -> {out_dir}")


if __name__ == "__main__":
    main()
