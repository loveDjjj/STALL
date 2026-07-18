#!/usr/bin/env python3
"""Apply the fixed STE-STall-Tail universal fusion weights."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score


KEY_COLUMNS = ["subset", "source_model", "filename"]
WEIGHTS = {"global": 0.35, "best_patch": 0.50, "tail": 0.15}


CONFIGS = {
    "comgenvid": {
        "global_csv": "results/comgenvid_results.csv",
        "best_patch_csv": "results/patch_region3_bottomk_sweep/comgenvid_patch_region3_second_order_bottomk0p50_spat10_temp90.csv",
        "tail_csv": "results/patch_score_ensemble/comgenvid_region3_second_order_tail_sweep_patch_ensemble_scores.csv",
        "tail_col": "patch_ensemble_tail_gap_neg",
    },
    "videofeedback": {
        "global_csv": "results/videofeedback_results.csv",
        "best_patch_csv": "results/patch_cross_dataset_ablation/videofeedback_patch_same_grid_second_order_region1_mean_spat10_temp90.csv",
        "tail_csv": "results/patch_score_ensemble/videofeedback_region1_second_order_multiscale_patch_ensemble_scores.csv",
        "tail_col": "patch_ensemble_tail_gap_neg",
    },
    "genvideo": {
        "global_csv": "results/genvideo_results.csv",
        "best_patch_csv": "results/patch_cross_dataset_ablation/genvideo_patch_same_grid_second_order_region2_mean_spat10_temp90.csv",
        "tail_csv": "results/patch_score_ensemble/genvideo_region2_second_order_multiscale_patch_ensemble_scores.csv",
        "tail_col": "patch_ensemble_tail_gap_neg",
    },
}


def _rank01(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(len(values), dtype=np.float64)
    return ranks / max(1.0, float(len(values) - 1))


def _read(path: Path, score_col: str, out_col: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    missing = [c for c in KEY_COLUMNS + [score_col] if c not in df.columns]
    if missing:
        raise ValueError(f"{path} missing columns: {missing}")
    out = df[KEY_COLUMNS + [score_col]].copy()
    for col in KEY_COLUMNS:
        out[col] = out[col].astype(str)
    return out.rename(columns={score_col: out_col})


def _metrics(df: pd.DataFrame, score_col: str) -> pd.DataFrame:
    real = df[df["subset"].str.lower() == "real"]
    fake = df[df["subset"].str.lower() != "real"]
    rows = []
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


def apply_dataset(root: Path, dataset: str, config: dict[str, str], out_dir: Path) -> dict[str, float]:
    global_df = _read(root / config["global_csv"], "final_score", "global_score")
    patch_df = _read(root / config["best_patch_csv"], "final_score", "best_patch_score")
    tail_df = _read(root / config["tail_csv"], config["tail_col"], "tail_score")
    df = global_df.merge(patch_df, on=KEY_COLUMNS, how="inner", validate="one_to_one").merge(
        tail_df, on=KEY_COLUMNS, how="inner", validate="one_to_one"
    )
    for col in ["global_score", "best_patch_score", "tail_score"]:
        df[f"{col}_rank"] = _rank01(df[col].to_numpy(dtype=np.float64))
    df["final_score"] = (
        WEIGHTS["global"] * df["global_score_rank"]
        + WEIGHTS["best_patch"] * df["best_patch_score_rank"]
        + WEIGHTS["tail"] * df["tail_score_rank"]
    )
    df.insert(0, "dataset", dataset)
    df["weight_global"] = WEIGHTS["global"]
    df["weight_best_patch"] = WEIGHTS["best_patch"]
    df["weight_tail"] = WEIGHTS["tail"]

    out_dir.mkdir(parents=True, exist_ok=True)
    fused_path = out_dir / f"{dataset}_ste_stall_tail_universal_fused.csv"
    metrics_path = out_dir / f"{dataset}_ste_stall_tail_universal_metrics.csv"
    df.to_csv(fused_path, index=False)
    metrics = _metrics(df, "final_score")
    metrics.insert(0, "dataset", dataset)
    metrics.to_csv(metrics_path, index=False)

    avg = metrics[metrics["source_model"] == "Average"].iloc[0]
    return {
        "dataset": dataset,
        "auc": float(avg["auc"]),
        "ap": float(avg["ap"]),
        "n_rows": len(df),
        "fused_csv": str(fused_path),
        "metrics_csv": str(metrics_path),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("/data/OneDay/STALL_project/STALL"))
    parser.add_argument("--output-dir", type=Path, default=Path("results/ste_stall_tail_universal"))
    args = parser.parse_args()

    root = args.root.resolve()
    out_dir = args.output_dir
    if not out_dir.is_absolute():
        out_dir = root / out_dir

    rows = [apply_dataset(root, dataset, cfg, out_dir) for dataset, cfg in CONFIGS.items()]
    summary = pd.DataFrame(rows)
    aggregate = {
        "dataset": "Average",
        "auc": summary["auc"].mean(),
        "ap": summary["ap"].mean(),
        "n_rows": int(summary["n_rows"].sum()),
        "fused_csv": "",
        "metrics_csv": "",
    }
    summary = pd.concat([summary, pd.DataFrame([aggregate])], ignore_index=True)
    summary.to_csv(out_dir / "ste_stall_tail_universal_summary.csv", index=False)
    print(summary[["dataset", "auc", "ap", "n_rows"]].to_string(index=False))


if __name__ == "__main__":
    main()
