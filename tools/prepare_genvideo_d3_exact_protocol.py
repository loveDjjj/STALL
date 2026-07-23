#!/usr/bin/env python3
"""准备 GenVideo 的 D3-exact 复现实验协议文件。

本脚本不抽帧、不运行 XCLIP，只基于当前 GenVideo enriched index 生成：

1. D3 官方 `eval.py` 可读取的 real/fake CSV；
2. 每个视频计划写入的 frame folder；
3. 按 D3 官方 `video2frame.py` 逻辑计算的 3s@8fps 抽帧计划；
4. 覆盖和样本上限审计 Markdown。

D3 官方 CSV 关键列：

- `content_path`：某个视频抽帧后的文件夹；
- `label`：real=0，fake=1；
- `type_id`：可读标签。

注意：官方 `video2frame.py` 的随机起点取决于遍历顺序。本脚本固定按
`subset/source_model/filename` 排序后用 seed=42 生成 `start_time`，保证本项目内可复现；
它是 D3-exact 输入协议的本地可复现版本，而不是逐字节复刻官方 glob 顺序。
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path
import random
import re

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = REPO_ROOT.parent
FAKE_MODELS = [
    "Crafter",
    "Gen2",
    "HotShot",
    "Lavie",
    "ModelScope",
    "MoonValley",
    "MorphStudio",
    "Show_1",
    "Sora",
    "WildScrape",
]


def safe_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value.strip())


def resolve_video_path(value: str) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    candidates = [
        PROJECT_ROOT / path,
        REPO_ROOT / path,
        Path.cwd() / path,
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()
    return (PROJECT_ROOT / path).resolve()


def frame_dir_for(row: pd.Series, frames_root: Path) -> Path:
    subset = safe_name(str(row["subset"]))
    source = safe_name(str(row["source_model"]))
    stem = safe_name(Path(str(row["video_path"])).stem)
    return frames_root / subset / source / stem


def d3_start_time(duration: float, rng: random.Random) -> int:
    if duration <= 3:
        return 0
    return int(math.floor(rng.uniform(0, duration - 3)))


def expected_extracted_frames(duration: float, start_time: int, frame_rate: int) -> int:
    available = max(0.0, min(3.0, duration - float(start_time)))
    return int(math.floor(available * frame_rate + 1e-9))


def expected_d3_read_frames(extracted_frames: int) -> int:
    if extracted_frames < 8:
        return 0
    return 8 if extracted_frames < 16 else 16


def load_index(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    required = {"video_path", "subset", "source_model", "fps", "duration_seconds", "num_frames"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"{path} 缺少列: {sorted(missing)}")
    df = df.copy()
    df["filename"] = df["video_path"].map(lambda x: Path(str(x)).name)
    df = df.sort_values(["subset", "source_model", "filename"]).reset_index(drop=True)
    return df


def build_plan(df: pd.DataFrame, frames_root: Path, seed: int, frame_rate: int) -> pd.DataFrame:
    rng = random.Random(seed)
    rows = []
    for row in df.itertuples(index=False):
        row_s = pd.Series(row._asdict())
        video_path_abs = resolve_video_path(str(row.video_path))
        frame_dir = frame_dir_for(row_s, frames_root)
        duration = float(row.duration_seconds)
        start = d3_start_time(duration, rng)
        extracted = expected_extracted_frames(duration, start, frame_rate)
        read_frames = expected_d3_read_frames(extracted)
        label = 0 if str(row.subset) == "real" else 1
        rows.append(
            {
                "subset": str(row.subset),
                "source_model": str(row.source_model),
                "filename": str(row.filename),
                "video_path": str(row.video_path),
                "video_path_abs": str(video_path_abs),
                "content_path": str(frame_dir),
                "label": label,
                "type_id": "Real Video" if label == 0 else "AI Video",
                "fps": float(row.fps),
                "duration_seconds": duration,
                "num_frames": int(row.num_frames),
                "d3_start_time": start,
                "d3_extract_duration": 3,
                "d3_frame_rate": frame_rate,
                "expected_extracted_frames": extracted,
                "expected_d3_read_frames": read_frames,
                "d3_eval_eligible": read_frames >= 8,
                "video_exists": video_path_abs.exists(),
            }
        )
    return pd.DataFrame(rows)


def write_d3_csvs(plan: pd.DataFrame, out_dir: Path, cap: int) -> pd.DataFrame:
    csv_dir = out_dir / "csv"
    csv_dir.mkdir(parents=True, exist_ok=True)

    eligible = plan[plan["d3_eval_eligible"] & plan["video_exists"]].copy()
    real = eligible[eligible["subset"] == "real"].head(cap).copy()
    d3_cols = [
        "content_path",
        "label",
        "type_id",
        "video_path_abs",
        "source_model",
        "filename",
        "d3_start_time",
        "d3_extract_duration",
        "d3_frame_rate",
        "expected_extracted_frames",
        "expected_d3_read_frames",
    ]
    real_csv = csv_dir / "real_MSRVTT_head1000.csv"
    real[d3_cols].to_csv(real_csv, index=False)

    rows = [
        {
            "csv_role": "real",
            "source_model": "MSR-VTT",
            "csv_path": str(real_csv),
            "rows": len(real),
            "eligible_total": int((eligible["subset"] == "real").sum()),
        }
    ]
    for model in FAKE_MODELS:
        fake_all = eligible[(eligible["subset"] == "annotated") & (eligible["source_model"] == model)].copy()
        fake = fake_all.head(cap)
        fake_csv = csv_dir / f"{safe_name(model)}_head1000.csv"
        fake[d3_cols].to_csv(fake_csv, index=False)
        rows.append(
            {
                "csv_role": "fake",
                "source_model": model,
                "csv_path": str(fake_csv),
                "rows": len(fake),
                "eligible_total": len(fake_all),
            }
        )
    return pd.DataFrame(rows)


def write_ffmpeg_runlist(plan: pd.DataFrame, out_dir: Path) -> Path:
    runlist = plan[plan["d3_eval_eligible"] & plan["video_exists"]].copy()
    path = out_dir / "d3_frame_extraction_runlist.csv"
    runlist[
        [
            "subset",
            "source_model",
            "filename",
            "video_path_abs",
            "content_path",
            "d3_start_time",
            "d3_extract_duration",
            "d3_frame_rate",
            "expected_extracted_frames",
            "expected_d3_read_frames",
        ]
    ].to_csv(path, index=False)
    return path


def summarize(plan: pd.DataFrame, csv_manifest: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    source_summary = (
        plan.groupby(["subset", "source_model"], as_index=False)
        .agg(
            videos=("filename", "count"),
            video_exists=("video_exists", "sum"),
            d3_eval_eligible=("d3_eval_eligible", "sum"),
            min_duration=("duration_seconds", "min"),
            median_duration=("duration_seconds", "median"),
            min_expected_read_frames=("expected_d3_read_frames", "min"),
            median_expected_read_frames=("expected_d3_read_frames", "median"),
        )
        .sort_values(["subset", "source_model"])
    )
    return source_summary, csv_manifest


def write_markdown(
    path: Path,
    source_summary: pd.DataFrame,
    csv_manifest: pd.DataFrame,
    out_dir: Path,
    frames_root: Path,
    cap: int,
    seed: int,
) -> None:
    lines = [
        "# GenVideo D3-exact CSV 协议准备",
        "",
        "本目录准备 D3 官方 `eval.py` 所需的 CSV 层输入，但不抽帧、不运行 XCLIP。",
        "",
        "## 协议",
        "",
        f"- 输入上限：每个 real/fake CSV 最多 `head({cap})`。",
        f"- D3 抽帧计划：3s、8fps，模型实际读取最多 16 帧。",
        f"- 本地随机起点：按 `subset/source_model/filename` 排序后使用 seed `{seed}`。",
        f"- frames root：`{frames_root}`。",
        "",
        "## CSV manifest",
        "",
        "| role | source | rows | eligible total | csv |",
        "|---|---|---:|---:|---|",
    ]
    for row in csv_manifest.itertuples(index=False):
        lines.append(
            f"| {row.csv_role} | {row.source_model} | {int(row.rows)} | {int(row.eligible_total)} | `{Path(row.csv_path).name}` |"
        )
    lines.extend(
        [
            "",
            "## Source coverage",
            "",
            "| subset | source | videos | exists | D3 eligible | min/median duration | min/median read frames |",
            "|---|---|---:|---:|---:|---:|---:|",
        ]
    )
    for row in source_summary.itertuples(index=False):
        lines.append(
            f"| {row.subset} | {row.source_model} | {int(row.videos)} | {int(row.video_exists)} | "
            f"{int(row.d3_eval_eligible)} | {row.min_duration:.2f}/{row.median_duration:.2f} | "
            f"{row.min_expected_read_frames:.0f}/{row.median_expected_read_frames:.0f} |"
        )
    lines.extend(
        [
            "",
            "## 后续执行边界",
            "",
            "当前只生成 CSV 和 runlist。真正运行 D3-exact 仍需要：",
            "",
            "1. 安装 `transformers`、`albumentations`、`moviepy`；",
            "2. 下载 `microsoft/xclip-base-patch16`；",
            "3. 根据 `d3_frame_extraction_runlist.csv` 抽帧到 `content_path`；",
            "4. 对每个 fake CSV 调用 D3 官方 `eval.py`。",
            "",
            "建议先抽取少量 real/fake 做 smoke test，确认 label 方向和 AP 正类后再跑全量 10 个生成器。",
            "",
            "## 输出",
            "",
            f"- `d3_exact_video_plan.csv`：全部视频的抽帧计划。",
            f"- `d3_frame_extraction_runlist.csv`：可执行抽帧 runlist。",
            f"- `csv/*.csv`：D3 `eval.py` 风格 real/fake CSV。",
            f"- `d3_exact_source_summary.csv`：每个来源的覆盖审计。",
            f"- `d3_exact_csv_manifest.csv`：CSV 行数和路径清单。",
            "",
            f"输出目录：`{out_dir}`。",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    base = REPO_ROOT / "results/journal_experiments/global_second_order_volatility/genvideo/d3_comparable_protocol/d3_exact_protocol"
    parser = argparse.ArgumentParser(description="准备 GenVideo D3-exact CSV 协议文件。")
    parser.add_argument(
        "--index-csv",
        type=Path,
        default=REPO_ROOT / "cache/indexes/genvideo_eval_holdout_real7984_all_fake.csv",
    )
    parser.add_argument("--output-dir", type=Path, default=base)
    parser.add_argument("--frames-root", type=Path, default=PROJECT_ROOT / "cache/d3_frames/genvideo")
    parser.add_argument("--cap", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--frame-rate", type=int, default=8)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    plan = build_plan(load_index(args.index_csv), args.frames_root, args.seed, args.frame_rate)
    csv_manifest = write_d3_csvs(plan, args.output_dir, args.cap)
    runlist_path = write_ffmpeg_runlist(plan, args.output_dir)
    source_summary, csv_manifest = summarize(plan, csv_manifest)

    plan.to_csv(args.output_dir / "d3_exact_video_plan.csv", index=False)
    source_summary.to_csv(args.output_dir / "d3_exact_source_summary.csv", index=False)
    csv_manifest.to_csv(args.output_dir / "d3_exact_csv_manifest.csv", index=False)
    write_markdown(
        args.output_dir / "d3_exact_protocol.md",
        source_summary,
        csv_manifest,
        args.output_dir,
        args.frames_root,
        args.cap,
        args.seed,
    )

    print(f"plan rows={len(plan)}")
    print(f"runlist={runlist_path}")
    print(f"csv manifest rows={len(csv_manifest)}")
    print(csv_manifest.to_string(index=False))


if __name__ == "__main__":
    main()
