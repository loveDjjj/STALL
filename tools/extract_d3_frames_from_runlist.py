#!/usr/bin/env python3
"""按 D3-exact runlist 抽取 3s@8fps JPEG 帧。

默认 dry-run，只检查输入并输出将要执行的任务；必须显式传入 ``--execute``
才会创建 frame 目录并调用 ffmpeg。该脚本用于
``prepare_genvideo_d3_exact_protocol.py`` 生成的
``d3_frame_extraction_runlist.csv``。
"""

from __future__ import annotations

import argparse
from pathlib import Path
import subprocess

import pandas as pd
from tqdm import tqdm


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RUNLIST = (
    REPO_ROOT
    / "results/journal_experiments/global_second_order_volatility/genvideo/"
    / "d3_comparable_protocol/d3_exact_protocol/d3_frame_extraction_runlist.csv"
)


def frame_count(path: Path) -> int:
    if not path.exists():
        return 0
    return sum(1 for p in path.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png"})


def ffmpeg_command(row: pd.Series) -> list[str]:
    output_dir = Path(str(row["content_path"]))
    return [
        "ffmpeg",
        "-loglevel",
        "quiet",
        "-y",
        "-ss",
        str(int(row["d3_start_time"])),
        "-t",
        str(int(row["d3_extract_duration"])),
        "-i",
        str(row["video_path_abs"]),
        "-vf",
        f"fps={int(row['d3_frame_rate'])}",
        str(output_dir / "%d.jpg"),
    ]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="按 D3 runlist 抽取 JPEG 帧。")
    parser.add_argument("--runlist", type=Path, default=DEFAULT_RUNLIST)
    parser.add_argument("--output-summary-csv", type=Path)
    parser.add_argument("--limit", type=int, default=None, help="只处理前 N 条，用于 smoke test。")
    parser.add_argument("--subset", help="可选：只处理指定 subset，例如 real 或 annotated。")
    parser.add_argument("--source-model", help="可选：只处理指定 source_model。")
    parser.add_argument("--execute", action="store_true", help="实际执行 ffmpeg；默认只 dry-run。")
    parser.add_argument("--overwrite", action="store_true", help="已存在帧目录时重新抽取。")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    df = pd.read_csv(args.runlist)
    required = {
        "video_path_abs",
        "content_path",
        "d3_start_time",
        "d3_extract_duration",
        "d3_frame_rate",
        "expected_d3_read_frames",
    }
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"{args.runlist} 缺少列: {sorted(missing)}")
    if args.subset is not None:
        if "subset" not in df.columns:
            raise ValueError("--subset 需要 runlist 中包含 subset 列")
        df = df[df["subset"].astype(str) == str(args.subset)].copy()
    if args.source_model is not None:
        if "source_model" not in df.columns:
            raise ValueError("--source-model 需要 runlist 中包含 source_model 列")
        df = df[df["source_model"].astype(str) == str(args.source_model)].copy()
    if args.limit is not None:
        df = df.head(args.limit).copy()

    rows = []
    for row in tqdm(list(df.itertuples(index=False)), desc="D3 frame extraction", unit="video"):
        item = pd.Series(row._asdict())
        video = Path(str(item["video_path_abs"]))
        out_dir = Path(str(item["content_path"]))
        before = frame_count(out_dir)
        status = "dry_run"
        return_code = None
        stderr_tail = ""
        if not video.exists():
            status = "missing_video"
        elif before >= int(item["expected_d3_read_frames"]) and not args.overwrite:
            status = "exists"
        elif args.execute:
            out_dir.mkdir(parents=True, exist_ok=True)
            if args.overwrite:
                for frame in out_dir.glob("*.jpg"):
                    frame.unlink()
            proc = subprocess.run(ffmpeg_command(item), capture_output=True, text=True)
            return_code = int(proc.returncode)
            stderr_tail = (proc.stderr or "")[-500:]
            status = "ok" if proc.returncode == 0 else "ffmpeg_failed"
        rows.append(
            {
                "video_path_abs": str(video),
                "content_path": str(out_dir),
                "expected_d3_read_frames": int(item["expected_d3_read_frames"]),
                "frames_before": before,
                "frames_after": frame_count(out_dir),
                "status": status,
                "return_code": return_code,
                "stderr_tail": stderr_tail,
                "command": " ".join(ffmpeg_command(item)),
            }
        )

    summary = pd.DataFrame(rows)
    if args.output_summary_csv is None:
        suffix_parts = ["execute" if args.execute else "dryrun"]
        if args.subset is not None:
            suffix_parts.append(str(args.subset))
        if args.source_model is not None:
            suffix_parts.append(str(args.source_model))
        suffix = "_".join(suffix_parts)
        args.output_summary_csv = args.runlist.with_name(f"d3_frame_extraction_{suffix}_summary.csv")
    summary.to_csv(args.output_summary_csv, index=False)
    print(f"execute={args.execute} rows={len(summary)}")
    print(summary["status"].value_counts(dropna=False).to_string())
    print(f"summary={args.output_summary_csv}")
    if not args.execute:
        print("dry-run only; add --execute to write frames")


if __name__ == "__main__":
    main()
