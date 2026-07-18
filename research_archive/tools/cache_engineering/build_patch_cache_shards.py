#!/usr/bin/env python3
"""Pack per-video Patch-STall embedding caches into sequential shards.

This tool does not change the original cache unless --remove-source is set.
It creates:

    <shard-root>/<dataset>/shard_00000.pt
    <shard-root>/<dataset>/shard_00001.pt
    <shard-root>/<dataset>/index.csv

Each shard stores lists of tensors to support variable shapes if needed, but
compact 2s Patch-STall caches should normally have fixed [T, P, D].
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import pandas as pd
import torch
from tqdm import tqdm


def _is_missing(value) -> bool:
    return value is None or (isinstance(value, float) and pd.isna(value)) or str(value) == "nan"


def _cache_path(cache_root: Path, subset: str, source_model: str, video_path: str, duration: int, compact: bool) -> Path:
    stem = Path(video_path).stem
    suffix = f"_{duration}s.pt" if compact else ".pt"
    return cache_root / subset / source_model / f"{stem}{suffix}"


def _filter_df(df: pd.DataFrame, duration: int, debug: int | None, max_total: int | None, shuffle: bool, seed: int) -> pd.DataFrame:
    window_col = f"{duration}_sec_idxs"
    if window_col in df.columns:
        df = df[~df[window_col].map(_is_missing)].copy()
    if shuffle:
        df = df.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    if debug:
        df = (
            df.groupby(["subset", "source_model"], group_keys=False)
            .apply(lambda g: g.head(debug))
            .reset_index(drop=True)
        )
    if max_total:
        df = df.head(max_total).reset_index(drop=True)
    return df


def _pack_records(records: list[dict]) -> dict:
    return {
        "global": [r["global"] for r in records],
        "patch": [r["patch"] for r in records],
        "grid_size": [r["grid_size"] for r in records],
        "frame_indices": [r["frame_indices"] for r in records],
        "metadata": [r["metadata"] for r in records],
    }


def build_shards(args: argparse.Namespace) -> None:
    csv_path = Path(args.csv)
    cache_root = Path(args.patch_cache)
    shard_root = Path(args.shard_root) / args.dataset
    shard_root.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(csv_path)
    df = _filter_df(df, args.duration, args.debug, args.max_total, args.shuffle, args.seed)

    index_rows = []
    records = []
    shard_id = 0
    read_times = []
    written = 0
    missing = 0

    for _, row in tqdm(df.iterrows(), total=len(df), desc="Build shards"):
        source_path = _cache_path(
            cache_root,
            str(row["subset"]),
            str(row["source_model"]),
            str(row["video_path"]),
            args.duration,
            args.compact,
        )
        if not source_path.exists():
            missing += 1
            continue

        t0 = time.perf_counter()
        payload = torch.load(source_path, map_location="cpu", weights_only=True)
        read_times.append(time.perf_counter() - t0)

        metadata = {
            "subset": row["subset"],
            "source_model": row["source_model"],
            "filename": Path(str(row["video_path"])).name,
            "video_path": row["video_path"],
            "source_cache_path": str(source_path),
        }
        records.append(
            {
                "global": payload["global"],
                "patch": payload["patch"],
                "grid_size": payload["grid_size"],
                "frame_indices": payload["frame_indices"],
                "metadata": metadata,
            }
        )

        if len(records) >= args.shard_size:
            shard_path = shard_root / f"shard_{shard_id:05d}.pt"
            torch.save(_pack_records(records), shard_path)
            for offset, rec in enumerate(records):
                index_rows.append({**rec["metadata"], "shard": shard_path.name, "offset": offset})
            if args.remove_source:
                for rec in records:
                    Path(rec["metadata"]["source_cache_path"]).unlink(missing_ok=True)
            written += len(records)
            records = []
            shard_id += 1

    if records:
        shard_path = shard_root / f"shard_{shard_id:05d}.pt"
        torch.save(_pack_records(records), shard_path)
        for offset, rec in enumerate(records):
            index_rows.append({**rec["metadata"], "shard": shard_path.name, "offset": offset})
        if args.remove_source:
            for rec in records:
                Path(rec["metadata"]["source_cache_path"]).unlink(missing_ok=True)
        written += len(records)

    index = pd.DataFrame(index_rows)
    index.to_csv(shard_root / "index.csv", index=False)
    stats = {
        "csv": str(csv_path),
        "patch_cache": str(cache_root),
        "shard_root": str(shard_root),
        "duration": args.duration,
        "compact": args.compact,
        "shard_size": args.shard_size,
        "written": written,
        "missing": missing,
        "read_time_mean_sec": float(sum(read_times) / max(len(read_times), 1)),
        "read_time_max_sec": float(max(read_times) if read_times else 0.0),
    }
    (shard_root / "stats.json").write_text(json.dumps(stats, indent=2), encoding="utf-8")
    print(json.dumps(stats, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", required=True)
    parser.add_argument("--patch-cache", required=True)
    parser.add_argument("--shard-root", required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--duration", type=int, default=2)
    parser.add_argument("--compact", action="store_true")
    parser.add_argument("--shard-size", type=int, default=256)
    parser.add_argument("--debug", type=int, default=None)
    parser.add_argument("--max-total", type=int, default=None)
    parser.add_argument("--shuffle", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--remove-source", action="store_true")
    args = parser.parse_args()
    build_shards(args)


if __name__ == "__main__":
    main()
