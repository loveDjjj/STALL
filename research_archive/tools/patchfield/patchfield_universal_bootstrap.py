#!/usr/bin/env python3
"""Bootstrap stability for universal PatchField final-score corrections."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

from patchfield_reliability_correction_sweep import KEY_COLUMNS, _read, _real_rank
from patchfield_service_gate_sweep import _format_float, _parse_floats
from patchfield_stability_gate_sweep import FROZEN_EXTRA_COLUMNS, _correct, _features, _gate


def _load_scores(args: argparse.Namespace) -> pd.DataFrame:
    frozen = _read(args.frozen_final_scores_csv, FROZEN_EXTRA_COLUMNS).rename(
        columns={"final_score": "frozen_final_score"}
    )
    patch = _read(args.patchfield_scores_csv, [args.patchfield_score_col])
    df = frozen.merge(patch, on=KEY_COLUMNS, how="inner", validate="one_to_one")
    if len(df) == 0:
        raise ValueError("No overlap between frozen scores and PatchField scores")
    real_mask = df["subset"].str.lower() == "real"
    df["patchfield_rank"] = _real_rank(df[args.patchfield_score_col], real_mask)
    return df


def _add_candidate_scores(df: pd.DataFrame) -> pd.DataFrame:
    feat = _features(df)
    final = feat["final"]
    pf = feat["pf"]
    out = df.copy()
    for name, threshold, weight in [
        ("benchmark_u070_w0015", 0.70, 0.015),
        ("service_u050_w0005", 0.50, 0.005),
    ]:
        gate = _gate(feat, "pf_over_final_ge", threshold)
        out[name] = _correct(final, pf, gate, weight, "convex")
    return out


def _resample_group(group: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    idx = rng.integers(0, len(group), size=len(group))
    return group.iloc[idx]


def _resample_dataset(df: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    parts = []
    real = df[df["subset"].str.lower() == "real"]
    parts.append(_resample_group(real, rng))
    fake = df[df["subset"].str.lower() != "real"]
    for _source, group in fake.groupby("source_model", sort=True):
        parts.append(_resample_group(group, rng))
    return pd.concat(parts, ignore_index=True)


def _auc_ap(real_scores: np.ndarray, fake_scores: np.ndarray) -> tuple[float, float]:
    y = np.r_[np.ones(len(real_scores), dtype=int), np.zeros(len(fake_scores), dtype=int)]
    s = np.r_[real_scores, fake_scores]
    return float(roc_auc_score(y, s)), float(average_precision_score(y, s))


def _metrics_once(sample: pd.DataFrame, candidate_col: str, fprs: list[float]) -> dict[str, float]:
    real = sample[sample["subset"].str.lower() == "real"]
    fake = sample[sample["subset"].str.lower() != "real"]
    real_base = real["frozen_final_score"].to_numpy(float)
    real_candidate = real[candidate_col].to_numpy(float)

    auc_deltas = []
    ap_deltas = []
    service_deltas = {fpr: [] for fpr in fprs}
    for _source, group in fake.groupby("source_model", sort=True):
        fake_base = group["frozen_final_score"].to_numpy(float)
        fake_candidate = group[candidate_col].to_numpy(float)
        base_auc, base_ap = _auc_ap(real_base, fake_base)
        cand_auc, cand_ap = _auc_ap(real_candidate, fake_candidate)
        auc_deltas.append(cand_auc - base_auc)
        ap_deltas.append(cand_ap - base_ap)

        for fpr in fprs:
            base_threshold = float(np.quantile(real_base, fpr, method="higher"))
            candidate_threshold = float(np.quantile(real_candidate, fpr, method="higher"))
            base_recall = float(np.mean(fake_base <= base_threshold))
            candidate_recall = float(np.mean(fake_candidate <= candidate_threshold))
            service_deltas[fpr].append(candidate_recall - base_recall)

    out = {
        "avg_delta_auc": float(np.mean(auc_deltas)),
        "avg_delta_ap": float(np.mean(ap_deltas)),
        "min_delta_auc": float(np.min(auc_deltas)),
        "min_delta_ap": float(np.min(ap_deltas)),
    }
    for fpr, values in service_deltas.items():
        tag = _format_float(fpr)
        out[f"avg_delta_fake_recall_fpr{tag}"] = float(np.mean(values))
        out[f"min_delta_fake_recall_fpr{tag}"] = float(np.min(values))
    return out


def _summarize_bootstrap(rows: pd.DataFrame, fprs: list[float]) -> pd.DataFrame:
    metric_cols = [
        "avg_delta_auc",
        "avg_delta_ap",
        "min_delta_auc",
        "min_delta_ap",
        *[f"avg_delta_fake_recall_fpr{_format_float(fpr)}" for fpr in fprs],
        *[f"min_delta_fake_recall_fpr{_format_float(fpr)}" for fpr in fprs],
    ]
    out_rows = []
    for (dataset, candidate), group in rows.groupby(["dataset", "candidate"], sort=True):
        row = {"dataset": dataset, "candidate": candidate, "n_bootstrap": len(group)}
        for col in metric_cols:
            values = group[col].to_numpy(float)
            row[f"{col}_mean"] = float(np.mean(values))
            row[f"{col}_p05"] = float(np.quantile(values, 0.05))
            row[f"{col}_p50"] = float(np.quantile(values, 0.50))
            row[f"{col}_p95"] = float(np.quantile(values, 0.95))
            row[f"{col}_prob_positive"] = float(np.mean(values > 0.0))
        out_rows.append(row)
    return pd.DataFrame(out_rows)


def run(args: argparse.Namespace) -> tuple[pd.DataFrame, pd.DataFrame]:
    scored = _add_candidate_scores(_load_scores(args))
    rng = np.random.default_rng(args.seed)
    fprs = _parse_floats(args.fprs)
    rows = []
    candidate_cols = ["benchmark_u070_w0015", "service_u050_w0005"]
    for boot_idx in range(args.n_bootstrap):
        sample = _resample_dataset(scored, rng)
        for candidate_col in candidate_cols:
            row = {
                "dataset": args.dataset,
                "bootstrap_idx": boot_idx,
                "candidate": candidate_col,
            }
            row.update(_metrics_once(sample, candidate_col, fprs))
            rows.append(row)
    raw = pd.DataFrame(rows)
    summary = _summarize_bootstrap(raw, fprs)
    return raw, summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--frozen-final-scores-csv", type=Path, required=True)
    parser.add_argument("--patchfield-scores-csv", type=Path, required=True)
    parser.add_argument("--patchfield-score-col", required=True)
    parser.add_argument("--n-bootstrap", type=int, default=500)
    parser.add_argument("--seed", type=int, default=20260709)
    parser.add_argument("--fprs", default="0.001,0.005,0.01")
    parser.add_argument("--output-raw-csv", type=Path, required=True)
    parser.add_argument("--output-summary-csv", type=Path, required=True)
    args = parser.parse_args()

    raw, summary = run(args)
    args.output_raw_csv.parent.mkdir(parents=True, exist_ok=True)
    raw.to_csv(args.output_raw_csv, index=False)
    summary.to_csv(args.output_summary_csv, index=False)
    print(summary.to_string(index=False, float_format=lambda x: f"{x:.6f}"))
    print(f"Saved raw -> {args.output_raw_csv}")
    print(f"Saved summary -> {args.output_summary_csv}")


if __name__ == "__main__":
    main()
