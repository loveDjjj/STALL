#!/usr/bin/env python3
"""Benchmark sequential shard reads for Patch-STall cache shards."""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import pandas as pd
import torch
from tqdm import tqdm


def benchmark(args: argparse.Namespace) -> None:
    shard_root = Path(args.shard_root) / args.dataset
    index_path = shard_root / "index.csv"
    index = pd.read_csv(index_path)
    if args.max_total:
        index = index.head(args.max_total)

    count = 0
    load_time = 0.0
    access_time = 0.0
    for shard_name, group in tqdm(index.groupby("shard", sort=False), desc="Read shards"):
        shard_path = shard_root / shard_name
        t0 = time.perf_counter()
        payload = torch.load(shard_path, map_location="cpu", weights_only=True)
        load_time += time.perf_counter() - t0
        t1 = time.perf_counter()
        for offset in group["offset"].tolist():
            _ = payload["global"][offset].shape
            _ = payload["patch"][offset].shape
            count += 1
        access_time += time.perf_counter() - t1

    total = load_time + access_time
    print(f"videos={count}")
    print(f"load_time_sec={load_time:.4f}")
    print(f"access_time_sec={access_time:.4f}")
    print(f"total_time_sec={total:.4f}")
    print(f"videos_per_sec={count / max(total, 1e-9):.2f}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--shard-root", required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--max-total", type=int, default=None)
    args = parser.parse_args()
    benchmark(args)


if __name__ == "__main__":
    main()
