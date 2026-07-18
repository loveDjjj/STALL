#!/usr/bin/env python3
"""Training-free gated fusion for patch morphology evidence.

The gate is computed without fake labels. It can use:
- morphology confidence from the real-calibrated morphology percentile;
- global/patch disagreement after rank normalization.

The final score starts from global+patch and lets morphology intervene only for
samples where the unsupervised gate is active.
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


def _read(path: Path, score_col: str, out_col: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    missing = [c for c in KEY_COLUMNS + [score_col] if c not in df.columns]
    if missing:
        raise ValueError(f"{path} missing columns: {missing}")
    out = df[KEY_COLUMNS + [score_col]].copy()
    for col in KEY_COLUMNS:
        out[col] = out[col].astype(str)
    return out.rename(columns={score_col: out_col})


def _rank01(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(len(values), dtype=np.float64)
    return ranks / max(1.0, float(len(values) - 1))


def _real_percentile(values: np.ndarray, subset: pd.Series) -> np.ndarray:
    real_values = np.sort(values[subset.str.lower().to_numpy() == "real"].astype(np.float64))
    if len(real_values) == 0:
        raise ValueError("No real rows for calibration")
    return np.searchsorted(real_values, values.astype(np.float64), side="right") / float(len(real_values))


def _metrics(df: pd.DataFrame, score_col: str = "final_score") -> pd.DataFrame:
    real = df[df["subset"].str.lower() == "real"]
    fake = df[df["subset"].str.lower() != "real"]
    rows = []
    for model, fake_group in fake.groupby("source_model", sort=True):
        y = np.concatenate([np.ones(len(real)), np.zeros(len(fake_group))])
        s = np.concatenate([real[score_col].to_numpy(float), fake_group[score_col].to_numpy(float)])
        rows.append(
            {
                "source_model": model,
                "n_real": len(real),
                "n_fake": len(fake_group),
                "auc": roc_auc_score(y, s),
                "ap": average_precision_score(y, s),
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


def _gate_values(df: pd.DataFrame, mode: str, morph_conf: np.ndarray, disagreement: np.ndarray, conf_t: float, dis_t: float) -> np.ndarray:
    if mode == "always":
        return np.ones(len(df), dtype=np.float64)
    if mode == "morph_conf":
        return (morph_conf >= conf_t).astype(np.float64)
    if mode == "disagreement":
        return (disagreement >= dis_t).astype(np.float64)
    if mode == "both":
        return ((morph_conf >= conf_t) & (disagreement >= dis_t)).astype(np.float64)
    if mode == "soft_product":
        return np.clip(morph_conf * disagreement, 0.0, 1.0)
    if mode == "soft_conf":
        return np.clip(morph_conf, 0.0, 1.0)
    raise ValueError(f"Unknown gate mode: {mode}")


def run(args: argparse.Namespace) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    global_df = _read(args.global_csv, args.global_score_col, "global_score")
    patch_df = _read(args.patch_csv, args.patch_score_col, "patch_score")
    morph_df = _read(args.morph_csv, args.morph_score_col, "morph_score")
    df = global_df.merge(patch_df, on=KEY_COLUMNS, how="inner", validate="one_to_one").merge(
        morph_df, on=KEY_COLUMNS, how="inner", validate="one_to_one"
    )

    for col in ["global_score", "patch_score", "morph_score"]:
        df[f"{col}_rank"] = _rank01(df[col].to_numpy(dtype=np.float64))

    if args.morph_score_col.endswith("_pct_high") or args.morph_score_col.endswith("_pct_low"):
        morph_pct = df["morph_score"].to_numpy(dtype=np.float64)
    else:
        morph_pct = _real_percentile(df["morph_score"].to_numpy(dtype=np.float64), df["subset"])
    morph_conf = np.abs(morph_pct - 0.5) * 2.0
    disagreement = np.abs(df["global_score_rank"].to_numpy(float) - df["patch_score_rank"].to_numpy(float))

    summary_rows = []
    per_model_frames = []
    best_scored = None
    best_key = (-1.0, -1.0)
    for base_alpha in _parse_floats(args.base_alphas):
        base = base_alpha * df["global_score_rank"].to_numpy(float) + (1.0 - base_alpha) * df["patch_score_rank"].to_numpy(float)
        for morph_weight in _parse_floats(args.morph_weights):
            for mode in [m.strip() for m in args.gate_modes.split(",") if m.strip()]:
                for conf_t in _parse_floats(args.conf_thresholds):
                    for dis_t in _parse_floats(args.disagreement_thresholds):
                        gate = _gate_values(df, mode, morph_conf, disagreement, conf_t, dis_t)
                        scored = df.copy()
                        scored["dataset"] = args.dataset
                        scored["tag"] = args.tag
                        scored["base_alpha"] = base_alpha
                        scored["morph_weight"] = morph_weight
                        scored["gate_mode"] = mode
                        scored["conf_threshold"] = conf_t
                        scored["disagreement_threshold"] = dis_t
                        scored["morph_conf"] = morph_conf
                        scored["global_patch_disagreement"] = disagreement
                        scored["morph_gate"] = gate
                        # Convex intervention: gate controls whether morphology can replace part of base evidence.
                        scored["final_score"] = (1.0 - morph_weight * gate) * base + (morph_weight * gate) * scored["morph_score_rank"]
                        metrics = _metrics(scored)
                        avg = metrics[metrics["source_model"] == "Average"].iloc[0]
                        summary_rows.append(
                            {
                                "dataset": args.dataset,
                                "tag": args.tag,
                                "morph_score_col": args.morph_score_col,
                                "base_alpha": base_alpha,
                                "morph_weight": morph_weight,
                                "gate_mode": mode,
                                "conf_threshold": conf_t,
                                "disagreement_threshold": dis_t,
                                "avg_auc": float(avg["auc"]),
                                "avg_ap": float(avg["ap"]),
                                "gate_mean": float(gate.mean()),
                                "morph_conf_mean": float(morph_conf.mean()),
                                "disagreement_mean": float(disagreement.mean()),
                                "n_rows": len(scored),
                            }
                        )
                        metrics = metrics.copy()
                        metrics.insert(0, "disagreement_threshold", dis_t)
                        metrics.insert(0, "conf_threshold", conf_t)
                        metrics.insert(0, "gate_mode", mode)
                        metrics.insert(0, "morph_weight", morph_weight)
                        metrics.insert(0, "base_alpha", base_alpha)
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
        raise RuntimeError("No scored rows")
    return summary, per_model, best_scored


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--tag", required=True)
    parser.add_argument("--global-csv", type=Path, required=True)
    parser.add_argument("--patch-csv", type=Path, required=True)
    parser.add_argument("--morph-csv", type=Path, required=True)
    parser.add_argument("--morph-score-col", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--global-score-col", default="final_score")
    parser.add_argument("--patch-score-col", default="final_score")
    parser.add_argument("--base-alphas", default="0.50,0.55,0.60")
    parser.add_argument("--morph-weights", default="0.05,0.10,0.15,0.20")
    parser.add_argument("--gate-modes", default="always,morph_conf,disagreement,both,soft_product,soft_conf")
    parser.add_argument("--conf-thresholds", default="0.30,0.50,0.70")
    parser.add_argument("--disagreement-thresholds", default="0.10,0.20,0.30")
    parser.add_argument("--save-best-fused-csv", action="store_true")
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary, per_model, best_scored = run(args)
    prefix = f"{args.dataset}_{args.tag}"
    summary.to_csv(args.output_dir / f"{prefix}_morph_gate_summary.csv", index=False)
    per_model.to_csv(args.output_dir / f"{prefix}_morph_gate_per_model.csv", index=False)
    if args.save_best_fused_csv:
        best_scored.to_csv(args.output_dir / f"{prefix}_morph_gate_best_fused.csv", index=False)
    best = summary.iloc[0]
    print(
        f"{args.dataset}/{args.tag}: best mode={best.gate_mode} alpha={best.base_alpha:.2f} "
        f"mw={best.morph_weight:.2f} AUC={best.avg_auc:.4f} AP={best.avg_ap:.4f} "
        f"gate_mean={best.gate_mean:.3f}",
        flush=True,
    )


if __name__ == "__main__":
    main()
