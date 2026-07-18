#!/usr/bin/env python3
"""Training-free matching confidence scores from cached patch embeddings."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import average_precision_score, roc_auc_score
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from patch_matching import matching_diagnostics_for_pair  # noqa: E402


def _is_missing(value) -> bool:
    return value is None or (isinstance(value, float) and np.isnan(value)) or str(value) == "nan"


def _cache_path(cache_root: Path, subset: str, source_model: str, video_path: str, duration: int) -> Path:
    return cache_root / subset / source_model / f"{Path(video_path).stem}_{duration}s.pt"


def _stats(prefix: str, arr: np.ndarray) -> dict[str, float]:
    arr = np.asarray(arr, dtype=np.float32).reshape(-1)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return {f"{prefix}_{k}": 0.0 for k in ["mean", "std", "p10", "p50", "p90"]}
    return {
        f"{prefix}_mean": float(arr.mean()),
        f"{prefix}_std": float(arr.std()),
        f"{prefix}_p10": float(np.quantile(arr, 0.10)),
        f"{prefix}_p50": float(np.quantile(arr, 0.50)),
        f"{prefix}_p90": float(np.quantile(arr, 0.90)),
    }


def _video_matching_features(
    patch: np.ndarray,
    grid_size: tuple[int, int],
    radius: int,
    top_m: int,
    temperature: float,
    lambda_dist: float,
    mode: str,
    sample_patches: int | None,
) -> dict[str, float]:
    patch = patch.astype(np.float32)
    if sample_patches is not None and patch.shape[1] > sample_patches:
        idx = np.linspace(0, patch.shape[1] - 1, sample_patches).round().astype(int)
        patch = patch[:, idx]
        # Sampling breaks the original 2D neighborhood, so only use this for all-patch mode.
        # Keep full patches for matching unless sample_patches is None.
        raise ValueError("--sample-patches is not supported for local matching; use full patch grid")

    values: dict[str, list[np.ndarray]] = {
        "max_similarity": [],
        "margin": [],
        "entropy": [],
        "weights_max": [],
        "displacement": [],
        "score": [],
    }
    for t in range(patch.shape[0] - 1):
        diag = matching_diagnostics_for_pair(
            patch[t],
            patch[t + 1],
            grid_size=grid_size,
            radius=radius,
            top_m=top_m,
            temperature=temperature,
            lambda_dist=lambda_dist,
            mode=mode,
        )
        values["max_similarity"].append(diag["top1_cosines"])
        values["margin"].append(diag["top1_scores"] - diag["top2_scores"])
        values["entropy"].append(diag["entropy"])
        values["weights_max"].append(diag["weights_max"])
        values["displacement"].append(diag["displacements"])
        values["score"].append(diag["top1_scores"])

    out = {}
    for name, chunks in values.items():
        out.update(_stats(name, np.concatenate(chunks) if chunks else np.array([])))
    # Fraction of patches that prefer non-zero displacement.
    disp = np.concatenate(values["displacement"]) if values["displacement"] else np.array([])
    out["move_ratio"] = float((disp > 0.0).mean()) if disp.size else 0.0
    out["move_gt1_ratio"] = float((disp > 1.01).mean()) if disp.size else 0.0
    return out


def _score_from_features(df: pd.DataFrame, mode: str) -> np.ndarray:
    if mode == "confidence":
        return (
            df["max_similarity_mean"].to_numpy()
            + 0.5 * df["margin_mean"].to_numpy()
            - 0.2 * df["entropy_mean"].to_numpy()
            - 0.1 * df["displacement_mean"].to_numpy()
        )
    if mode == "uncertainty_real":
        # Some generated videos can be over-smooth and too easy to match; this flips confidence.
        return (
            -df["max_similarity_mean"].to_numpy()
            + 0.5 * df["entropy_mean"].to_numpy()
            + 0.2 * df["move_ratio"].to_numpy()
        )
    if mode == "stable_local":
        return (
            df["max_similarity_p10"].to_numpy()
            + df["margin_p10"].to_numpy()
            - 0.2 * df["move_gt1_ratio"].to_numpy()
        )
    raise ValueError(f"Unknown score mode: {mode}")


def _metrics(df: pd.DataFrame, score_col: str) -> pd.DataFrame:
    real = df[df["subset"].astype(str).str.lower() == "real"]
    fake = df[df["subset"].astype(str).str.lower() != "real"]
    rows = []
    for model, group in fake.groupby("source_model"):
        if len(real) == 0 or len(group) == 0:
            continue
        y = np.concatenate([np.ones(len(real)), np.zeros(len(group))])
        s = np.concatenate([real[score_col].to_numpy(), group[score_col].to_numpy()])
        rows.append(
            {
                "source_model": model,
                "n_real": len(real),
                "n_fake": len(group),
                "auc": roc_auc_score(y, s),
                "ap": average_precision_score(y, s),
                "auc_neg": roc_auc_score(y, -s),
            }
        )
    out = pd.DataFrame(rows)
    if len(out):
        out.loc[len(out)] = {
            "source_model": "Average",
            "n_real": int(round(out["n_real"].mean())),
            "n_fake": int(round(out["n_fake"].mean())),
            "auc": out["auc"].mean(),
            "ap": out["ap"].mean(),
            "auc_neg": out["auc_neg"].mean(),
        }
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", required=True)
    parser.add_argument("--patch-cache", required=True)
    parser.add_argument("--output-csv", required=True)
    parser.add_argument("--duration", type=int, default=2)
    parser.add_argument("--debug", type=int, default=None)
    parser.add_argument("--max-total", type=int, default=None)
    parser.add_argument("--shuffle", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--radius", type=int, default=2)
    parser.add_argument("--top-m", type=int, default=4)
    parser.add_argument("--temperature", type=float, default=0.07)
    parser.add_argument("--lambda-dist", type=float, default=0.01)
    parser.add_argument("--match-mode", choices=["hard", "soft"], default="soft")
    parser.add_argument("--score-mode", default="confidence")
    parser.add_argument("--score-all-modes", action="store_true")
    args = parser.parse_args()

    df = pd.read_csv(args.csv)
    if args.shuffle:
        df = df.sample(frac=1.0, random_state=args.seed).reset_index(drop=True)
    if args.debug:
        df = (
            df.groupby(["subset", "source_model"], group_keys=False)
            .apply(lambda g: g.head(args.debug))
            .reset_index(drop=True)
        )
    if args.max_total:
        df = df.head(args.max_total).reset_index(drop=True)

    cache_root = Path(args.patch_cache)
    rows = []
    for _, row in tqdm(df.iterrows(), total=len(df), desc="Matching confidence"):
        rowd = row.to_dict()
        if _is_missing(rowd.get(f"{args.duration}_sec_idxs")):
            continue
        path = _cache_path(cache_root, str(rowd["subset"]), str(rowd["source_model"]), str(rowd["video_path"]), args.duration)
        if not path.exists():
            continue
        payload = torch.load(path, map_location="cpu", weights_only=True)
        patch = payload["patch"].numpy()
        grid_size = tuple(int(x) for x in payload["grid_size"])
        feats = _video_matching_features(
            patch,
            grid_size,
            radius=args.radius,
            top_m=args.top_m,
            temperature=args.temperature,
            lambda_dist=args.lambda_dist,
            mode=args.match_mode,
            sample_patches=None,
        )
        feats.update(
            {
                "subset": rowd["subset"],
                "source_model": rowd["source_model"],
                "filename": Path(str(rowd["video_path"])).name,
                "match_mode": args.match_mode,
                "radius": args.radius,
                "top_m": args.top_m,
                "temperature": args.temperature,
                "lambda_dist": args.lambda_dist,
            }
        )
        rows.append(feats)

    out = pd.DataFrame(rows)
    modes = ["confidence", "uncertainty_real", "stable_local"] if args.score_all_modes else [args.score_mode]
    for mode in modes:
        out[f"score_{mode}"] = _score_from_features(out, mode)
    out["final_score"] = out[f"score_{args.score_mode}"]
    out["score_mode"] = args.score_mode
    Path(args.output_csv).parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.output_csv, index=False)

    for mode in modes:
        print(f"\n== {mode} ==")
        metrics = _metrics(out, f"score_{mode}")
        print(metrics.to_string(index=False, formatters={"auc": lambda x: f"{x:.4f}", "ap": lambda x: f"{x:.4f}", "auc_neg": lambda x: f"{x:.4f}"}))


if __name__ == "__main__":
    main()
