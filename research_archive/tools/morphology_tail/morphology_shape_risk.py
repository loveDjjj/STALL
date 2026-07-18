#!/usr/bin/env python3
"""Build direction-agnostic morphology shape-risk scores.

The input is a CSV produced by patch_anomaly_morphology_scores.py. That tool
already adds real-calibrated percentile columns for each feature. This script
turns selected percentile columns into a single realness score where values
near the real calibration center are more real-like and either tail is risky.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def _center_realness(pct_high: np.ndarray) -> np.ndarray:
    pct_high = np.asarray(pct_high, dtype=np.float64)
    return 1.0 - 2.0 * np.abs(pct_high - 0.5)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-csv", type=Path, required=True)
    parser.add_argument("--output-csv", type=Path, required=True)
    parser.add_argument(
        "--features",
        default="frame_entropy_r0p2,max_frame_mass_r0p2,active_frame_frac_r0p2,patch_entropy_r0p2",
        help="Comma-separated base morphology features with *_pct_high columns.",
    )
    args = parser.parse_args()

    df = pd.read_csv(args.input_csv)
    features = [x.strip() for x in args.features.split(",") if x.strip()]
    realness_cols = []
    for feature in features:
        pct_col = f"{feature}_pct_high"
        if pct_col not in df.columns:
            raise ValueError(f"{args.input_csv} missing {pct_col}")
        out_col = f"{feature}_center_realness"
        df[out_col] = _center_realness(df[pct_col].to_numpy(dtype=np.float64)).astype(np.float32)
        realness_cols.append(out_col)

    df["morph_shape_center_realness"] = df[realness_cols].mean(axis=1).astype(np.float32)
    df["morph_temporal_center_realness"] = df[
        [c for c in realness_cols if c.startswith(("frame_entropy", "max_frame_mass", "active_frame_frac"))]
    ].mean(axis=1).astype(np.float32)
    df["morph_entropy_center_realness"] = df[
        [c for c in realness_cols if c.startswith(("frame_entropy", "patch_entropy"))]
    ].mean(axis=1).astype(np.float32)

    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.output_csv, index=False)
    print(f"Saved morphology shape risk -> {args.output_csv}")
    print(df[["morph_shape_center_realness", "morph_temporal_center_realness", "morph_entropy_center_realness"]].describe().to_string())


if __name__ == "__main__":
    main()
