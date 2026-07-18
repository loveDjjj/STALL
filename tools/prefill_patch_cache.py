#!/usr/bin/env python3
"""预填充 patch embedding cache 的 CLI 包装。"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

def _is_missing_window(value: object) -> bool:
    if value is None:
        return True
    if isinstance(value, float) and pd.isna(value):
        return True
    text = str(value)
    return text == "" or text.lower() == "nan" or text == "[]"


def _cache_path(cache_root: Path, subset: str, source_model: str, stem: str, duration: int, compact: bool) -> Path:
    if compact:
        return cache_root / subset / source_model / f"{stem}_{duration}s.pt"
    return cache_root / subset / source_model / f"{stem}.pt"


def _count_misses(csv_path: str, cache_root: str, duration: int, compact: bool, debug_n: int | None) -> int:
    df = pd.read_csv(csv_path)
    if debug_n is not None:
        df = (
            df.groupby(["subset", "source_model"], group_keys=False)
            .apply(lambda g: g.head(debug_n))
            .reset_index(drop=True)
        )
    window_col = f"{duration}_sec_idxs"
    cache_root_path = Path(cache_root)
    misses = 0
    for _, row in df.iterrows():
        if compact and _is_missing_window(row.get(window_col)):
            continue
        stem = Path(str(row["video_path"])).stem
        path = _cache_path(
            cache_root_path,
            str(row["subset"]),
            str(row["source_model"]),
            stem,
            duration,
            compact,
        )
        misses += int(not path.exists())
    return misses


def main() -> None:
    parser = argparse.ArgumentParser(description="为一个 CSV 预填充 PatchSTALL patch embedding cache。")
    parser.add_argument("--csv", required=True)
    parser.add_argument("--patch-emb-cache", required=True)
    parser.add_argument("--duration", type=int, default=2)
    parser.add_argument("--compact", action="store_true")
    parser.add_argument("--debug-n", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--video-batch", type=int, default=4)
    parser.add_argument("--frame-batch", type=int, default=32)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--execute", action="store_true", help="实际加载 DINO 并写入缺失的 cache 文件。")
    parser.add_argument("--output-summary-csv", type=Path, required=True)
    args = parser.parse_args()

    misses = _count_misses(
        args.csv,
        args.patch_emb_cache,
        duration=args.duration,
        compact=args.compact,
        debug_n=args.debug_n,
    )
    written = 0
    status = "DRY_RUN"
    if args.execute and misses:
        from dataset_utils_patch import prefill_patch_emb_cache  # noqa: WPS433
        from stall_patch import PatchSTALL  # noqa: WPS433

        model = PatchSTALL(device=args.device, data_dict=None, load_dino=True)
        for _ in tqdm(
            prefill_patch_emb_cache(
                args.csv,
                args.patch_emb_cache,
                model=model,
                batch_size=args.frame_batch,
                duration_sec=args.duration,
                debug_n=args.debug_n,
                compact=args.compact,
                num_workers=args.num_workers,
                video_batch=args.video_batch,
            ),
            total=misses,
            desc="Patch cache",
            unit=" video",
            dynamic_ncols=True,
        ):
            written += 1
        status = "EXECUTED"
    elif args.execute:
        status = "EXECUTED_NO_MISSES"

    summary = pd.DataFrame(
        [
            {
                "csv": args.csv,
                "patch_emb_cache": args.patch_emb_cache,
                "duration": args.duration,
                "compact": args.compact,
                "debug_n": args.debug_n,
                "num_workers": args.num_workers,
                "video_batch": args.video_batch,
                "frame_batch": args.frame_batch,
                "device": args.device,
                "execute": args.execute,
                "status": status,
                "misses_before": int(misses),
                "written": int(written),
            }
        ]
    )
    args.output_summary_csv.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(args.output_summary_csv, index=False)
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
