#!/usr/bin/env python3
"""Training-free adaptive fusion for global STALL and patch STALL scores."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score


KEY_COLUMNS = ["subset", "source_model", "filename"]


def _read_global(path: Path, score_col: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    missing = [c for c in KEY_COLUMNS + [score_col] if c not in df.columns]
    if missing:
        raise ValueError(f"{path} missing columns: {missing}")
    out = df[KEY_COLUMNS + [score_col]].copy()
    out = out.rename(columns={score_col: "global_score"})
    for col in KEY_COLUMNS:
        out[col] = out[col].astype(str)
    return out


def _read_patch(path: Path, score_col: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    required = KEY_COLUMNS + [score_col, "patch_spat_percentile", "patch_temp_percentile"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"{path} missing columns: {missing}")
    cols = required + [c for c in ["patch_final_score"] if c in df.columns]
    out = df[cols].copy()
    out = out.rename(columns={score_col: "patch_score"})
    for col in KEY_COLUMNS:
        out[col] = out[col].astype(str)
    return out


def _metrics(df: pd.DataFrame, score_col: str = "final_score") -> pd.DataFrame:
    real = df[df["subset"].str.lower() == "real"]
    fake = df[df["subset"].str.lower() != "real"]
    rows: list[dict[str, object]] = []
    for model, fake_group in fake.groupby("source_model", sort=True):
        y_true = np.concatenate([np.ones(len(real)), np.zeros(len(fake_group))])
        y_score = np.concatenate(
            [
                real[score_col].to_numpy(np.float64),
                fake_group[score_col].to_numpy(np.float64),
            ]
        )
        rows.append(
            {
                "source_model": model,
                "n_real": len(real),
                "n_fake": len(fake_group),
                "auc": roc_auc_score(y_true, y_score),
                "ap": average_precision_score(y_true, y_score),
            }
        )
    out = pd.DataFrame(rows)
    if out.empty:
        raise ValueError("No fake rows found")
    out.loc[len(out)] = {
        "source_model": "Average",
        "n_real": int(round(out["n_real"].mean())),
        "n_fake": int(round(out["n_fake"].mean())),
        "auc": out["auc"].mean(),
        "ap": out["ap"].mean(),
    }
    return out


def _clip_alpha(alpha: np.ndarray, low: float, high: float) -> np.ndarray:
    return np.clip(alpha, low, high)


def _compute_alpha(
    df: pd.DataFrame,
    strategy: str,
    base_alpha: float,
    min_alpha: float,
    max_alpha: float,
    patch_boost: float,
    global_boost: float,
    disagreement_threshold: float,
    strong_patch_threshold: float,
    weak_patch_threshold: float,
    global_conf_threshold: float,
    low_score_threshold: float,
) -> np.ndarray:
    g = df["global_score"].to_numpy(np.float64)
    p = df["patch_score"].to_numpy(np.float64)
    ps = df["patch_spat_percentile"].to_numpy(np.float64)
    pt = df["patch_temp_percentile"].to_numpy(np.float64)
    patch_anomaly = np.minimum(ps, pt)
    disagreement = np.abs(g - p)
    global_conf = np.abs(g - 0.5)

    alpha = np.full(len(df), base_alpha, dtype=np.float64)

    if strategy in {"fixed", "none"}:
        return _clip_alpha(alpha, min_alpha, max_alpha)

    if strategy in {"patch_anomaly", "hybrid", "conservative_hybrid", "aggressive_hybrid"}:
        strong_patch = patch_anomaly < strong_patch_threshold
        weak_patch = patch_anomaly > weak_patch_threshold
        alpha[strong_patch] -= patch_boost
        alpha[weak_patch] += global_boost * 0.5

    if strategy in {"disagreement", "hybrid", "conservative_hybrid", "aggressive_hybrid"}:
        disagree = disagreement > disagreement_threshold
        # When the two branches conflict, prefer the branch that is farther away from 0.5.
        patch_more_conf = np.abs(p - 0.5) > np.abs(g - 0.5)
        alpha[disagree & patch_more_conf] -= patch_boost
        alpha[disagree & ~patch_more_conf] += global_boost

    if strategy in {"global_conf", "hybrid", "conservative_hybrid"}:
        confident_global = global_conf > global_conf_threshold
        alpha[confident_global] += global_boost

    if strategy in {"low_score_patch_rescue", "hybrid", "aggressive_hybrid"}:
        both_low = (g < low_score_threshold) & (p < low_score_threshold)
        patch_much_lower = p + 0.15 < g
        alpha[both_low | patch_much_lower] -= patch_boost * 0.5

    return _clip_alpha(alpha, min_alpha, max_alpha)


def _run_config(df: pd.DataFrame, args: argparse.Namespace, strategy: str, base_alpha: float) -> tuple[pd.DataFrame, pd.DataFrame]:
    alpha = _compute_alpha(
        df,
        strategy=strategy,
        base_alpha=base_alpha,
        min_alpha=args.min_alpha,
        max_alpha=args.max_alpha,
        patch_boost=args.patch_boost,
        global_boost=args.global_boost,
        disagreement_threshold=args.disagreement_threshold,
        strong_patch_threshold=args.strong_patch_threshold,
        weak_patch_threshold=args.weak_patch_threshold,
        global_conf_threshold=args.global_conf_threshold,
        low_score_threshold=args.low_score_threshold,
    )
    scored = df.copy()
    scored["alpha"] = alpha
    scored["final_score"] = alpha * scored["global_score"] + (1.0 - alpha) * scored["patch_score"]
    metrics = _metrics(scored)
    return scored, metrics


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--tag", required=True)
    parser.add_argument("--global-csv", type=Path, required=True)
    parser.add_argument("--patch-csv", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--global-score-col", default="final_score")
    parser.add_argument("--patch-score-col", default="final_score")
    parser.add_argument("--strategies", default="fixed,patch_anomaly,disagreement,global_conf,hybrid,conservative_hybrid,aggressive_hybrid,low_score_patch_rescue")
    parser.add_argument("--base-alphas", default="0.55,0.60,0.65,0.70,0.75,0.80")
    parser.add_argument("--min-alpha", type=float, default=0.20)
    parser.add_argument("--max-alpha", type=float, default=0.90)
    parser.add_argument("--patch-boost", type=float, default=0.15)
    parser.add_argument("--global-boost", type=float, default=0.15)
    parser.add_argument("--disagreement-threshold", type=float, default=0.25)
    parser.add_argument("--strong-patch-threshold", type=float, default=0.25)
    parser.add_argument("--weak-patch-threshold", type=float, default=0.75)
    parser.add_argument("--global-conf-threshold", type=float, default=0.30)
    parser.add_argument("--low-score-threshold", type=float, default=0.35)
    parser.add_argument("--save-best-fused-csv", action="store_true")
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    global_df = _read_global(args.global_csv, args.global_score_col)
    patch_df = _read_patch(args.patch_csv, args.patch_score_col)
    df = global_df.merge(patch_df, on=KEY_COLUMNS, how="inner", validate="one_to_one")
    if len(df) != len(global_df) or len(df) != len(patch_df):
        print(f"WARNING: merged={len(df)} global={len(global_df)} patch={len(patch_df)}")

    strategies = [x.strip() for x in args.strategies.split(",") if x.strip()]
    base_alphas = [float(x) for x in args.base_alphas.split(",") if x.strip()]
    summary_rows: list[dict[str, object]] = []
    metric_frames: list[pd.DataFrame] = []
    best_scored: pd.DataFrame | None = None
    best_key: tuple[float, float] = (-1.0, -1.0)

    for strategy in strategies:
        for base_alpha in base_alphas:
            scored, metrics = _run_config(df, args, strategy, base_alpha)
            avg = metrics[metrics["source_model"] == "Average"].iloc[0]
            alpha_values = scored["alpha"].to_numpy(np.float64)
            row = {
                "dataset": args.dataset,
                "tag": args.tag,
                "strategy": strategy,
                "base_alpha": base_alpha,
                "avg_auc": avg["auc"],
                "avg_ap": avg["ap"],
                "alpha_mean": alpha_values.mean(),
                "alpha_std": alpha_values.std(),
                "alpha_min": alpha_values.min(),
                "alpha_max": alpha_values.max(),
                "patch_weight_mean": (1.0 - alpha_values).mean(),
                "n_rows": len(scored),
            }
            summary_rows.append(row)
            metrics = metrics.copy()
            metrics.insert(0, "base_alpha", base_alpha)
            metrics.insert(0, "strategy", strategy)
            metric_frames.append(metrics)
            key = (float(avg["auc"]), float(avg["ap"]))
            if key > best_key:
                best_key = key
                best_scored = scored

    summary = pd.DataFrame(summary_rows).sort_values(["avg_auc", "avg_ap"], ascending=False)
    per_model = pd.concat(metric_frames, ignore_index=True)
    prefix = f"{args.dataset}_{args.tag}"
    summary_path = args.output_dir / f"{prefix}_adaptive_summary.csv"
    per_model_path = args.output_dir / f"{prefix}_adaptive_per_model.csv"
    summary.to_csv(summary_path, index=False)
    per_model.to_csv(per_model_path, index=False)
    if args.save_best_fused_csv and best_scored is not None:
        best_scored.to_csv(args.output_dir / f"{prefix}_adaptive_best_fused.csv", index=False)

    best = summary.iloc[0]
    print(
        f"{args.dataset}/{args.tag}: best strategy={best.strategy} base_alpha={best.base_alpha:.2f} "
        f"AUC={best.avg_auc:.4f} AP={best.avg_ap:.4f} alpha_mean={best.alpha_mean:.3f}"
    )
    print(f"  summary: {summary_path}")
    print(f"  per_model: {per_model_path}")


if __name__ == "__main__":
    main()
