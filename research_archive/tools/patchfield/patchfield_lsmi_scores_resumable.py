#!/usr/bin/env python3
"""Resumable LSMI-only PatchField scorer.

This keeps the same LSMI feature definition as ``patchfield_lsmi_scores.py`` but
flushes partial rows to disk, so large datasets can resume after slow IO or an
interrupted run.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import torch
from tqdm import tqdm

from patchfield_lsmi_scores import _lsmi_features
from patchfield_split_scores import (
    _cache_path,
    _is_missing,
    _parse_fprs,
    _real_anomaly_rank,
    _source_metrics,
)


KEY_COLUMNS = ["subset", "source_model", "filename"]


def _key(row: pd.Series) -> tuple[str, str, str]:
    return (
        str(row["subset"]),
        str(row["source_model"]),
        Path(str(row["video_path"])).name,
    )


def _write_rows(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame(rows)
    frame.to_csv(path, mode="a", index=False, header=not path.exists())


def _load_done(path: Path) -> set[tuple[str, str, str]]:
    if not path.exists():
        return set()
    done = pd.read_csv(path, usecols=KEY_COLUMNS)
    return set(done.itertuples(index=False, name=None))


def _finalize(partial_csv: Path, output_csv: Path, output_summary_csv: Path, fprs: list[float]) -> tuple[pd.DataFrame, pd.DataFrame]:
    scores = pd.read_csv(partial_csv).drop_duplicates(KEY_COLUMNS, keep="last").reset_index(drop=True)
    if len(scores) == 0:
        raise ValueError("No patch cache rows were scored")
    real_mask = scores["subset"].astype(str).str.lower() == "real"
    scores["lsmi_anomaly_rank"] = _real_anomaly_rank(scores["lsmi_anomaly_raw"], real_mask)
    scores["score_lsmi_real"] = -scores["lsmi_anomaly_rank"]
    scores["score_lsmi_anti_real"] = scores["lsmi_anomaly_rank"]

    metrics = []
    for score_col in ("score_lsmi_real", "score_lsmi_anti_real"):
        m = _source_metrics(scores, score_col, fprs)
        m.insert(0, "score_col", score_col)
        metrics.append(m)
    summary = pd.concat(metrics, ignore_index=True)

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    output_summary_csv.parent.mkdir(parents=True, exist_ok=True)
    scores.to_csv(output_csv, index=False)
    summary.to_csv(output_summary_csv, index=False)
    return scores, summary


def run(args: argparse.Namespace) -> tuple[pd.DataFrame, pd.DataFrame]:
    df = pd.read_csv(args.csv)
    if args.shuffle:
        df = df.sample(frac=1.0, random_state=args.seed).reset_index(drop=True)
    if args.debug is not None:
        df = (
            df.groupby(["subset", "source_model"], group_keys=False)
            .apply(lambda g: g.head(args.debug))
            .reset_index(drop=True)
        )
    if args.max_total is not None:
        df = df.head(args.max_total).reset_index(drop=True)

    partial = Path(args.partial_csv)
    done = _load_done(partial) if args.resume else set()
    cache_root = Path(args.patch_cache)
    rows: list[dict[str, object]] = []
    missing = 0
    skipped_done = 0

    try:
        for _, row in tqdm(df.iterrows(), total=len(df), desc="PF-LSMI-resume"):
            row_key = _key(row)
            if row_key in done:
                skipped_done += 1
                continue
            if _is_missing(row.get(f"{args.duration}_sec_idxs")):
                missing += 1
                continue
            path = _cache_path(cache_root, str(row["subset"]), str(row["source_model"]), str(row["video_path"]), args.duration)
            if not path.exists():
                missing += 1
                continue
            payload = torch.load(path, map_location="cpu", weights_only=True)
            feats = _lsmi_features(payload["patch"].numpy(), payload.get("grid_size", None))
            feats.update({"subset": row_key[0], "source_model": row_key[1], "filename": row_key[2]})
            rows.append(feats)
            done.add(row_key)
            if len(rows) >= args.flush_every:
                _write_rows(partial, rows)
                rows.clear()
        _write_rows(partial, rows)
    finally:
        _write_rows(partial, rows)

    print(f"partial={partial} done={len(done)} skipped_done={skipped_done} missing={missing}")
    scores, summary = _finalize(partial, Path(args.output_csv), Path(args.output_summary_csv), args.fprs)
    avg = summary[summary["source_model"] == "Average"]
    print(avg.to_string(index=False, float_format=lambda x: f"{x:.6f}"))
    return scores, summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", required=True)
    parser.add_argument("--patch-cache", required=True)
    parser.add_argument("--output-csv", required=True)
    parser.add_argument("--output-summary-csv", required=True)
    parser.add_argument("--partial-csv", required=True)
    parser.add_argument("--duration", type=int, default=2)
    parser.add_argument("--debug", type=int, default=None)
    parser.add_argument("--max-total", type=int, default=None)
    parser.add_argument("--shuffle", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--flush-every", type=int, default=250)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--fprs", type=_parse_fprs, default=_parse_fprs("0.001,0.005,0.01"))
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
