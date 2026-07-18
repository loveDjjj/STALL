#!/usr/bin/env python3
"""Reliability-aware fusion for global STALL and patch STALL scores.

The fusion is training-free with respect to fake labels. Reliability is
estimated from the real subset score distribution only: scores far from the
real calibration center are treated as high-confidence branch evidence.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score


KEY_COLUMNS = ["subset", "source_model", "filename"]


def _parse_floats(value: str) -> list[float]:
    if ":" in value:
        start, stop, step = [float(x) for x in value.split(":")]
        count = int(round((stop - start) / step)) + 1
        return [round(start + i * step, 10) for i in range(count)]
    return [float(x) for x in value.split(",") if x.strip()]


def _read_scores(path: Path, score_col: str, out_col: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    missing = [c for c in KEY_COLUMNS + [score_col] if c not in df.columns]
    if missing:
        raise ValueError(f"{path} missing columns: {missing}")
    out = df[KEY_COLUMNS + [score_col]].copy()
    out = out.rename(columns={score_col: out_col})
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
                real[score_col].to_numpy(dtype=np.float64),
                fake_group[score_col].to_numpy(dtype=np.float64),
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
    result = pd.DataFrame(rows)
    if result.empty:
        raise ValueError("No fake rows found; cannot compute metrics")
    result.loc[len(result)] = {
        "source_model": "Average",
        "n_real": int(round(result["n_real"].mean())),
        "n_fake": int(round(result["n_fake"].mean())),
        "auc": result["auc"].mean(),
        "ap": result["ap"].mean(),
    }
    return result


def _empirical_percentile(values: np.ndarray, real_values: np.ndarray) -> np.ndarray:
    real_sorted = np.sort(real_values.astype(np.float64))
    if len(real_sorted) == 0:
        raise ValueError("No real calibration scores available")
    return np.searchsorted(real_sorted, values.astype(np.float64), side="right") / float(len(real_sorted))


def _tail_confidence(values: np.ndarray, real_values: np.ndarray, gamma: float, floor: float) -> np.ndarray:
    pct = _empirical_percentile(values, real_values)
    conf = np.abs(pct - 0.5) * 2.0
    conf = np.power(np.clip(conf, 0.0, 1.0), gamma)
    return np.maximum(conf, floor)


def _logit(x: np.ndarray | float) -> np.ndarray:
    arr = np.asarray(x, dtype=np.float64)
    arr = np.clip(arr, 1e-6, 1.0 - 1e-6)
    return np.log(arr / (1.0 - arr))


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def _alpha_for_method(
    df: pd.DataFrame,
    method: str,
    base_alpha: float,
    min_alpha: float,
    max_alpha: float,
    gamma: float,
    floor: float,
    strength: float,
    disagreement_threshold: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    g = df["global_score"].to_numpy(dtype=np.float64)
    p = df["patch_score"].to_numpy(dtype=np.float64)
    real_mask = df["subset"].str.lower().to_numpy() == "real"
    g_conf = _tail_confidence(g, g[real_mask], gamma=gamma, floor=floor)
    p_conf = _tail_confidence(p, p[real_mask], gamma=gamma, floor=floor)

    if method == "fixed":
        alpha = np.full(len(df), base_alpha, dtype=np.float64)
    elif method == "confidence_ratio":
        alpha = g_conf / np.maximum(g_conf + p_conf, 1e-8)
    elif method == "base_logit":
        alpha = _sigmoid(_logit(base_alpha) + strength * (g_conf - p_conf))
    elif method == "disagreement_gated":
        adaptive = _sigmoid(_logit(base_alpha) + strength * (g_conf - p_conf))
        alpha = np.full(len(df), base_alpha, dtype=np.float64)
        gate = np.abs(g - p) >= disagreement_threshold
        alpha[gate] = adaptive[gate]
    elif method == "patch_anomaly_rescue":
        alpha = np.full(len(df), base_alpha, dtype=np.float64)
        patch_more_anomalous = p + disagreement_threshold < g
        global_more_anomalous = g + disagreement_threshold < p
        alpha[patch_more_anomalous] = _sigmoid(
            _logit(base_alpha) - strength * p_conf[patch_more_anomalous]
        )
        alpha[global_more_anomalous] = _sigmoid(
            _logit(base_alpha) + strength * g_conf[global_more_anomalous]
        )
    else:
        raise ValueError(f"Unknown method: {method}")

    return np.clip(alpha, min_alpha, max_alpha), g_conf, p_conf


def _score_one(
    df: pd.DataFrame,
    method: str,
    base_alpha: float,
    min_alpha: float,
    max_alpha: float,
    gamma: float,
    floor: float,
    strength: float,
    disagreement_threshold: float,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    alpha, g_conf, p_conf = _alpha_for_method(
        df=df,
        method=method,
        base_alpha=base_alpha,
        min_alpha=min_alpha,
        max_alpha=max_alpha,
        gamma=gamma,
        floor=floor,
        strength=strength,
        disagreement_threshold=disagreement_threshold,
    )
    scored = df.copy()
    scored["method"] = method
    scored["base_alpha"] = base_alpha
    scored["alpha"] = alpha
    scored["global_conf"] = g_conf
    scored["patch_conf"] = p_conf
    scored["final_score"] = alpha * scored["global_score"] + (1.0 - alpha) * scored["patch_score"]
    return scored, _metrics(scored)


def run(args: argparse.Namespace) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    global_df = _read_scores(args.global_csv, args.global_score_col, "global_score")
    patch_df = _read_scores(args.patch_csv, args.patch_score_col, "patch_score")
    df = global_df.merge(patch_df, on=KEY_COLUMNS, how="inner", validate="one_to_one")
    if len(df) != len(global_df) or len(df) != len(patch_df):
        print(f"WARNING: merged={len(df)} global={len(global_df)} patch={len(patch_df)}", flush=True)

    methods = [x.strip() for x in args.methods.split(",") if x.strip()]
    summary_rows: list[dict[str, object]] = []
    per_model_frames: list[pd.DataFrame] = []
    best_scored: pd.DataFrame | None = None
    best_key = (-1.0, -1.0)

    for method in methods:
        for base_alpha in _parse_floats(args.base_alphas):
            scored, metrics = _score_one(
                df=df,
                method=method,
                base_alpha=base_alpha,
                min_alpha=args.min_alpha,
                max_alpha=args.max_alpha,
                gamma=args.gamma,
                floor=args.conf_floor,
                strength=args.strength,
                disagreement_threshold=args.disagreement_threshold,
            )
            avg = metrics[metrics["source_model"] == "Average"].iloc[0]
            summary_rows.append(
                {
                    "dataset": args.dataset,
                    "tag": args.tag,
                    "method": method,
                    "base_alpha": base_alpha,
                    "avg_auc": float(avg["auc"]),
                    "avg_ap": float(avg["ap"]),
                    "alpha_mean": float(scored["alpha"].mean()),
                    "alpha_std": float(scored["alpha"].std()),
                    "alpha_min": float(scored["alpha"].min()),
                    "alpha_max": float(scored["alpha"].max()),
                    "global_conf_mean": float(scored["global_conf"].mean()),
                    "patch_conf_mean": float(scored["patch_conf"].mean()),
                    "n_rows": len(scored),
                }
            )
            metrics = metrics.copy()
            metrics.insert(0, "base_alpha", base_alpha)
            metrics.insert(0, "method", method)
            metrics.insert(0, "tag", args.tag)
            metrics.insert(0, "dataset", args.dataset)
            per_model_frames.append(metrics)
            key = (float(avg["auc"]), float(avg["ap"]))
            if key > best_key:
                best_key = key
                best_scored = scored

    summary = pd.DataFrame(summary_rows).sort_values(["avg_auc", "avg_ap"], ascending=False)
    per_model = pd.concat(per_model_frames, ignore_index=True)
    if best_scored is None:
        raise RuntimeError("No fusion rows were scored")
    return summary, per_model, best_scored


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--tag", required=True)
    parser.add_argument("--global-csv", type=Path, required=True)
    parser.add_argument("--patch-csv", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--global-score-col", default="final_score")
    parser.add_argument("--patch-score-col", default="final_score")
    parser.add_argument(
        "--methods",
        default="fixed,confidence_ratio,base_logit,disagreement_gated,patch_anomaly_rescue",
    )
    parser.add_argument("--base-alphas", default="0.50,0.55,0.60,0.65,0.70")
    parser.add_argument("--min-alpha", type=float, default=0.35)
    parser.add_argument("--max-alpha", type=float, default=0.85)
    parser.add_argument("--gamma", type=float, default=1.0)
    parser.add_argument("--conf-floor", type=float, default=0.05)
    parser.add_argument("--strength", type=float, default=2.0)
    parser.add_argument("--disagreement-threshold", type=float, default=0.15)
    parser.add_argument("--save-best-fused-csv", action="store_true")
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary, per_model, best_scored = run(args)
    prefix = f"{args.dataset}_{args.tag}"
    summary_path = args.output_dir / f"{prefix}_reliability_summary.csv"
    per_model_path = args.output_dir / f"{prefix}_reliability_per_model.csv"
    summary.to_csv(summary_path, index=False)
    per_model.to_csv(per_model_path, index=False)
    if args.save_best_fused_csv:
        best_scored.to_csv(args.output_dir / f"{prefix}_reliability_best_fused.csv", index=False)

    best = summary.iloc[0]
    print(
        f"{args.dataset}/{args.tag}: best method={best.method} base_alpha={best.base_alpha:.2f} "
        f"AUC={best.avg_auc:.4f} AP={best.avg_ap:.4f} alpha_mean={best.alpha_mean:.3f}",
        flush=True,
    )
    print(f"  summary: {summary_path}", flush=True)
    print(f"  per_model: {per_model_path}", flush=True)


if __name__ == "__main__":
    main()
