#!/usr/bin/env python3
"""Apply the universal PatchField add-on to frozen alpha=0.60 final scores."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from patchfield_reliability_correction_sweep import KEY_COLUMNS, _read, _real_rank
from patchfield_stability_gate_sweep import FROZEN_EXTRA_COLUMNS, _correct, _features, _gate


def _soft_strength(pf_over_final: np.ndarray) -> np.ndarray:
    low = 0.50
    high = 0.65
    low_weight = 0.0025
    high_weight = 0.010
    ramp = np.clip((pf_over_final - low) / (high - low), 0.0, 1.0)
    active = (pf_over_final >= low).astype(float)
    return active * (low_weight + ramp * (high_weight - low_weight))


def run(args: argparse.Namespace) -> pd.DataFrame:
    frozen = _read(args.frozen_final_scores_csv, FROZEN_EXTRA_COLUMNS).rename(
        columns={"final_score": "frozen_final_score"}
    )
    patch = _read(args.patchfield_scores_csv, [args.patchfield_score_col])
    df = frozen.merge(patch, on=KEY_COLUMNS, how="inner", validate="one_to_one")
    if len(df) == 0:
        raise ValueError("No overlap between frozen scores and PatchField scores")

    real_mask = df["subset"].str.lower() == "real"
    df["patchfield_rank"] = _real_rank(df[args.patchfield_score_col], real_mask)
    feat = _features(df)
    final = feat["final"]
    pf = feat["pf"]

    benchmark_gate = _gate(feat, "pf_over_final_ge", 0.70)
    service_gate = _gate(feat, "pf_over_final_ge", 0.50)
    soft_strength = _soft_strength(feat["pf_over_final"])
    df["universal_pf_benchmark_score"] = _correct(final, pf, benchmark_gate, 0.015, "convex")
    df["universal_pf_service_score"] = _correct(final, pf, service_gate, 0.005, "convex")
    df["universal_pf_soft_score"] = np.clip((1.0 - soft_strength) * final + soft_strength * pf, 0.0, 1.0)
    df["universal_pf_benchmark_gate"] = benchmark_gate
    df["universal_pf_service_gate"] = service_gate
    df["universal_pf_soft_strength"] = soft_strength

    out_cols = [
        *KEY_COLUMNS,
        "frozen_final_score",
        "patchfield_rank",
        "global_rank",
        "raw_rank",
        "persistence_rank",
        "sample_gate",
        "use_split",
        "universal_pf_benchmark_gate",
        "universal_pf_benchmark_score",
        "universal_pf_service_gate",
        "universal_pf_service_score",
        "universal_pf_soft_strength",
        "universal_pf_soft_score",
    ]
    return df[out_cols].copy()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--frozen-final-scores-csv", type=Path, required=True)
    parser.add_argument("--patchfield-scores-csv", type=Path, required=True)
    parser.add_argument("--patchfield-score-col", required=True)
    parser.add_argument("--output-csv", type=Path, required=True)
    args = parser.parse_args()

    out = run(args)
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.output_csv, index=False)
    print(out.head(5).to_string(index=False, float_format=lambda x: f"{x:.6f}"))
    print(f"rows={len(out)}")
    print(f"benchmark_gate_mean={float(out['universal_pf_benchmark_gate'].mean()):.6f}")
    print(f"service_gate_mean={float(out['universal_pf_service_gate'].mean()):.6f}")
    print(f"soft_strength_mean={float(out['universal_pf_soft_strength'].mean()):.6f}")
    print(f"Saved -> {args.output_csv}")


if __name__ == "__main__":
    main()
