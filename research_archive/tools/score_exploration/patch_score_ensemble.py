#!/usr/bin/env python3
"""Build patch-score ensembles from existing per-video patch CSV files."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score


KEY_COLUMNS = ["subset", "source_model", "filename"]


def _parse_alphas(value: str) -> list[float]:
    if ":" in value:
        start, stop, step = [float(x) for x in value.split(":")]
        count = int(round((stop - start) / step)) + 1
        return [round(start + i * step, 10) for i in range(count)]
    return [float(x) for x in value.split(",") if x.strip()]


def _read_score(path: Path, score_col: str, name: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    missing = [c for c in KEY_COLUMNS + [score_col] if c not in df.columns]
    if missing:
        raise ValueError(f"{path} missing columns: {missing}")
    out = df[KEY_COLUMNS + [score_col]].copy()
    for col in KEY_COLUMNS:
        out[col] = out[col].astype(str)
    return out.rename(columns={score_col: name})


def _rank01(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(len(values), dtype=np.float64)
    if len(values) <= 1:
        return np.zeros_like(values, dtype=np.float64)
    return ranks / float(len(values) - 1)


def _metrics(df: pd.DataFrame, score_col: str) -> pd.DataFrame:
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


def _ensemble_scores(df: pd.DataFrame, score_cols: list[str]) -> dict[str, np.ndarray]:
    mat = df[score_cols].to_numpy(dtype=np.float64)
    rank_mat = np.column_stack([_rank01(mat[:, i]) for i in range(mat.shape[1])])
    scores = {
        "mean": mat.mean(axis=1),
        "median": np.median(mat, axis=1),
        "max": mat.max(axis=1),
        "min": mat.min(axis=1),
        "rank_mean": rank_mat.mean(axis=1),
        "rank_max": rank_mat.max(axis=1),
    }
    if len(score_cols) >= 2:
        sorted_mat = np.sort(mat, axis=1)
        # Higher score means more real. A large low-tail gap indicates a sharp
        # extreme anomaly in at least one scale, so subtract it from mean.
        tail_gap = sorted_mat[:, 1] - sorted_mat[:, 0]
        scores["mean_minus_tail_gap"] = mat.mean(axis=1) - tail_gap
        scores["min_plus_median"] = 0.5 * sorted_mat[:, 0] + 0.5 * np.median(mat, axis=1)
        # Tail-shape proxies derived from score-scale sweeps. These are not
        # true likelihood quantiles, but they approximate whether evidence is
        # concentrated in the extreme tail or shared across broader bottom-k
        # scales. Higher remains more-real.
        scores["tail_gap_neg"] = -tail_gap
        scores["tail_range_neg"] = -(sorted_mat[:, -1] - sorted_mat[:, 0])
        scores["min_score"] = sorted_mat[:, 0]
    return scores


def run(args: argparse.Namespace) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    patch_specs = [x for x in args.patch_specs.split(",") if x.strip()]
    if len(patch_specs) < 1:
        raise ValueError("--patch-specs must contain at least one name=path item")

    merged: pd.DataFrame | None = None
    score_cols: list[str] = []
    for spec in patch_specs:
        if "=" not in spec:
            raise ValueError(f"Invalid patch spec {spec!r}; expected name=path")
        name, raw_path = spec.split("=", 1)
        col = f"patch_{name}"
        part = _read_score(Path(raw_path), args.patch_score_col, col)
        merged = part if merged is None else merged.merge(part, on=KEY_COLUMNS, how="inner", validate="one_to_one")
        score_cols.append(col)

    if merged is None:
        raise RuntimeError("No patch scores loaded")

    ensemble_scores = _ensemble_scores(merged, score_cols)
    patch_rows = []
    patch_metric_frames = []
    fused_rows = []
    fused_metric_frames = []

    out_scores = merged[KEY_COLUMNS].copy()
    for method, values in ensemble_scores.items():
        out_scores[f"patch_ensemble_{method}"] = values.astype(np.float32)
        metrics = _metrics(out_scores.assign(final_score=values), "final_score")
        avg = metrics[metrics["source_model"] == "Average"].iloc[0]
        patch_rows.append(
            {
                "dataset": args.dataset,
                "tag": args.tag,
                "method": method,
                "avg_auc": float(avg["auc"]),
                "avg_ap": float(avg["ap"]),
                "n_rows": len(out_scores),
                "score_cols": ";".join(score_cols),
            }
        )
        metrics = metrics.copy()
        metrics.insert(0, "method", method)
        metrics.insert(0, "tag", args.tag)
        metrics.insert(0, "dataset", args.dataset)
        patch_metric_frames.append(metrics)

    if args.global_csv:
        global_df = _read_score(Path(args.global_csv), args.global_score_col, "global_score")
        fused_base = global_df.merge(out_scores, on=KEY_COLUMNS, how="inner", validate="one_to_one")
        for method in ensemble_scores:
            patch_col = f"patch_ensemble_{method}"
            for alpha in _parse_alphas(args.alphas):
                scored = fused_base.copy()
                scored["final_score"] = alpha * scored["global_score"] + (1.0 - alpha) * scored[patch_col]
                metrics = _metrics(scored, "final_score")
                avg = metrics[metrics["source_model"] == "Average"].iloc[0]
                fused_rows.append(
                    {
                        "dataset": args.dataset,
                        "tag": args.tag,
                        "method": method,
                        "alpha": alpha,
                        "avg_auc": float(avg["auc"]),
                        "avg_ap": float(avg["ap"]),
                        "n_rows": len(scored),
                    }
                )
                metrics = metrics.copy()
                metrics.insert(0, "alpha", alpha)
                metrics.insert(0, "method", method)
                metrics.insert(0, "tag", args.tag)
                metrics.insert(0, "dataset", args.dataset)
                fused_metric_frames.append(metrics)

    patch_summary = pd.DataFrame(patch_rows).sort_values(["avg_auc", "avg_ap"], ascending=False)
    patch_per_model = pd.concat(patch_metric_frames, ignore_index=True)
    fused_summary = (
        pd.DataFrame(fused_rows).sort_values(["avg_auc", "avg_ap"], ascending=False)
        if fused_rows
        else pd.DataFrame()
    )
    if fused_metric_frames:
        fused_per_model = pd.concat(fused_metric_frames, ignore_index=True)
    else:
        fused_per_model = pd.DataFrame()
    return patch_summary, patch_per_model, fused_summary, fused_per_model, out_scores


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--tag", required=True)
    parser.add_argument("--patch-specs", required=True, help="Comma-separated name=path entries")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--patch-score-col", default="final_score")
    parser.add_argument("--global-csv", type=Path, default=None)
    parser.add_argument("--global-score-col", default="final_score")
    parser.add_argument("--alphas", default="0:1:0.025")
    parser.add_argument("--save-scores", action="store_true")
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    patch_summary, patch_per_model, fused_summary, fused_per_model, out_scores = run(args)
    prefix = f"{args.dataset}_{args.tag}"
    patch_summary.to_csv(args.output_dir / f"{prefix}_patch_ensemble_summary.csv", index=False)
    patch_per_model.to_csv(args.output_dir / f"{prefix}_patch_ensemble_per_model.csv", index=False)
    if not fused_summary.empty:
        fused_summary.to_csv(args.output_dir / f"{prefix}_fusion_summary.csv", index=False)
        fused_per_model.to_csv(args.output_dir / f"{prefix}_fusion_per_model.csv", index=False)
    if args.save_scores:
        out_scores.to_csv(args.output_dir / f"{prefix}_patch_ensemble_scores.csv", index=False)

    best_patch = patch_summary.iloc[0]
    msg = (
        f"{args.dataset}/{args.tag}: best patch ensemble={best_patch.method} "
        f"AUC={best_patch.avg_auc:.4f} AP={best_patch.avg_ap:.4f}"
    )
    if not fused_summary.empty:
        best_fused = fused_summary.iloc[0]
        msg += (
            f"; best fused={best_fused.method} alpha={best_fused.alpha:.3f} "
            f"AUC={best_fused.avg_auc:.4f} AP={best_fused.avg_ap:.4f}"
        )
    print(msg, flush=True)


if __name__ == "__main__":
    main()
