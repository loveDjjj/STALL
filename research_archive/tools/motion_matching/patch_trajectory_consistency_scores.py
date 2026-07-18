#!/usr/bin/env python3
"""Patch trajectory consistency scores.

This treats local patch matches as short trajectories and measures whether the
motion field has abrupt second-order changes. The intended signal is different
from forward-backward cycle consistency: it asks whether the local patch path
itself has unnatural acceleration/jerk or confidence collapse over time.
"""

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

from patch_matching import matching_diagnostics_for_pair, patch_id_to_rc  # noqa: E402


def _is_missing(value) -> bool:
    return value is None or (isinstance(value, float) and np.isnan(value)) or str(value) == "nan"


def _cache_path(cache_root: Path, subset: str, source_model: str, video_path: str, duration: int) -> Path:
    return cache_root / subset / source_model / f"{Path(video_path).stem}_{duration}s.pt"


def _patch_coords(patch_ids: np.ndarray, grid_size: tuple[int, int]) -> np.ndarray:
    return np.asarray([patch_id_to_rc(int(x), grid_size) for x in patch_ids], dtype=np.float32)


def _stats(prefix: str, values: np.ndarray) -> dict[str, float]:
    arr = np.asarray(values, dtype=np.float32).reshape(-1)
    arr = arr[np.isfinite(arr)]
    keys = ["mean", "std", "p10", "p50", "p90", "p95"]
    if arr.size == 0:
        return {f"{prefix}_{k}": 0.0 for k in keys}
    return {
        f"{prefix}_mean": float(arr.mean()),
        f"{prefix}_std": float(arr.std()),
        f"{prefix}_p10": float(np.quantile(arr, 0.10)),
        f"{prefix}_p50": float(np.quantile(arr, 0.50)),
        f"{prefix}_p90": float(np.quantile(arr, 0.90)),
        f"{prefix}_p95": float(np.quantile(arr, 0.95)),
    }


def _topk_stats(prefix: str, values: np.ndarray, ratios: tuple[float, ...] = (0.05, 0.10, 0.20)) -> dict[str, float]:
    arr = np.asarray(values, dtype=np.float32).reshape(-1)
    arr = arr[np.isfinite(arr)]
    out: dict[str, float] = {}
    if arr.size == 0:
        for ratio in ratios:
            out[f"{prefix}_top{int(round(ratio * 100)):02d}_mean"] = 0.0
        return out
    order = np.sort(arr)[::-1]
    for ratio in ratios:
        k = max(1, int(round(arr.size * ratio)))
        out[f"{prefix}_top{int(round(ratio * 100)):02d}_mean"] = float(order[:k].mean())
    return out


def _trajectory_features(
    patch: np.ndarray,
    grid_size: tuple[int, int],
    radius: int,
    top_m: int,
    temperature: float,
    lambda_dist: float,
    mode: str,
) -> dict[str, float]:
    arr = patch.astype(np.float32, copy=False)
    num_frames, num_patches = arr.shape[:2]
    patch_ids = np.arange(num_patches, dtype=np.int64)
    start_coords = _patch_coords(patch_ids, grid_size)

    matched_ids: list[np.ndarray] = []
    velocity: list[np.ndarray] = []
    margins: list[np.ndarray] = []
    entropy: list[np.ndarray] = []
    cosine: list[np.ndarray] = []

    for t in range(num_frames - 1):
        diag = matching_diagnostics_for_pair(
            arr[t],
            arr[t + 1],
            grid_size=grid_size,
            radius=radius,
            top_m=top_m,
            temperature=temperature,
            lambda_dist=lambda_dist,
            mode=mode,
        )
        dst = diag["matched_ids"]
        dst_coords = _patch_coords(dst, grid_size)
        matched_ids.append(dst)
        velocity.append(dst_coords - start_coords)
        margins.append(diag["top1_scores"] - diag["top2_scores"])
        entropy.append(np.nan_to_num(diag["entropy"], nan=0.0))
        cosine.append(diag["top1_cosines"])

    out: dict[str, float] = {}
    if not velocity:
        return out

    vel = np.stack(velocity, axis=0).astype(np.float32)
    speed = np.sqrt((vel * vel).sum(axis=-1))
    margin_arr = np.stack(margins, axis=0).astype(np.float32)
    entropy_arr = np.stack(entropy, axis=0).astype(np.float32)
    cosine_arr = np.stack(cosine, axis=0).astype(np.float32)

    out.update(_stats("traj_speed", speed))
    out.update(_stats("traj_margin", margin_arr))
    out.update(_stats("traj_entropy", entropy_arr))
    out.update(_stats("traj_cosine", cosine_arr))
    out.update(_topk_stats("traj_speed", speed))

    if vel.shape[0] >= 2:
        accel = vel[1:] - vel[:-1]
        accel_norm = np.sqrt((accel * accel).sum(axis=-1))
        margin_drop = np.maximum(0.0, margin_arr[:-1] - margin_arr[1:])
        entropy_jump = np.maximum(0.0, entropy_arr[1:] - entropy_arr[:-1])
        incoherence = accel_norm + 0.25 * entropy_jump + 0.25 * margin_drop
        out.update(_stats("traj_accel", accel_norm))
        out.update(_topk_stats("traj_accel", accel_norm))
        out.update(_stats("traj_incoherence", incoherence))
        out.update(_topk_stats("traj_incoherence", incoherence))
    else:
        for name in ["traj_accel", "traj_incoherence"]:
            out.update(_stats(name, np.zeros(0, dtype=np.float32)))
            out.update(_topk_stats(name, np.zeros(0, dtype=np.float32)))

    if vel.shape[0] >= 3:
        jerk = vel[2:] - 2.0 * vel[1:-1] + vel[:-2]
        jerk_norm = np.sqrt((jerk * jerk).sum(axis=-1))
        out.update(_stats("traj_jerk", jerk_norm))
        out.update(_topk_stats("traj_jerk", jerk_norm))
    else:
        out.update(_stats("traj_jerk", np.zeros(0, dtype=np.float32)))
        out.update(_topk_stats("traj_jerk", np.zeros(0, dtype=np.float32)))

    # Agreement of consecutive hard destinations. This captures unstable local
    # assignment even when displacement magnitude is small.
    if len(matched_ids) >= 2:
        switches = [(matched_ids[i + 1] != matched_ids[i]).astype(np.float32) for i in range(len(matched_ids) - 1)]
        switch_arr = np.stack(switches, axis=0)
        out.update(_stats("traj_switch", switch_arr))
    else:
        out.update(_stats("traj_switch", np.zeros(0, dtype=np.float32)))

    return out


def _score_from_features(df: pd.DataFrame, mode: str) -> np.ndarray:
    if mode == "traj_jerk_top10":
        return df["traj_jerk_top10_mean"].to_numpy(float)
    if mode == "traj_accel_top10":
        return df["traj_accel_top10_mean"].to_numpy(float)
    if mode == "traj_incoherence_top10":
        return df["traj_incoherence_top10_mean"].to_numpy(float)
    if mode == "traj_unstable_real":
        return (
            df["traj_incoherence_top10_mean"].to_numpy(float)
            + 0.5 * df["traj_switch_mean"].to_numpy(float)
            + 0.25 * df["traj_entropy_mean"].to_numpy(float)
            - 0.25 * df["traj_margin_mean"].to_numpy(float)
        )
    if mode == "anti_traj_incoherence_top10":
        return -df["traj_incoherence_top10_mean"].to_numpy(float)
    raise ValueError(f"Unknown score mode: {mode}")


def _available_score_modes() -> list[str]:
    return [
        "traj_jerk_top10",
        "traj_accel_top10",
        "traj_incoherence_top10",
        "traj_unstable_real",
        "anti_traj_incoherence_top10",
    ]


def _metrics(df: pd.DataFrame, score_col: str) -> pd.DataFrame:
    real = df[df["subset"].astype(str).str.lower() == "real"]
    fake = df[df["subset"].astype(str).str.lower() != "real"]
    rows = []
    for model, group in fake.groupby("source_model", sort=True):
        y = np.concatenate([np.ones(len(real)), np.zeros(len(group))])
        s = np.concatenate([real[score_col].to_numpy(float), group[score_col].to_numpy(float)])
        rows.append(
            {
                "source_model": model,
                "n_real": int(len(real)),
                "n_fake": int(len(group)),
                "auc": float(roc_auc_score(y, s)),
                "ap": float(average_precision_score(y, s)),
                "auc_neg": float(roc_auc_score(y, -s)),
            }
        )
    out = pd.DataFrame(rows)
    if len(out):
        out.loc[len(out)] = {
            "source_model": "Average",
            "n_real": int(round(out["n_real"].mean())),
            "n_fake": int(round(out["n_fake"].mean())),
            "auc": float(out["auc"].mean()),
            "ap": float(out["ap"].mean()),
            "auc_neg": float(out["auc_neg"].mean()),
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
    parser.add_argument("--score-mode", default="traj_incoherence_top10")
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

    rows = []
    cache_root = Path(args.patch_cache)
    for _, row in tqdm(df.iterrows(), total=len(df), desc="Patch trajectory consistency"):
        rowd = row.to_dict()
        if _is_missing(rowd.get(f"{args.duration}_sec_idxs")):
            continue
        path = _cache_path(cache_root, str(rowd["subset"]), str(rowd["source_model"]), str(rowd["video_path"]), args.duration)
        if not path.exists():
            continue
        payload = torch.load(path, map_location="cpu", weights_only=True)
        feats = _trajectory_features(
            payload["patch"].numpy(),
            tuple(int(x) for x in payload["grid_size"]),
            radius=args.radius,
            top_m=args.top_m,
            temperature=args.temperature,
            lambda_dist=args.lambda_dist,
            mode=args.match_mode,
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
    modes = _available_score_modes() if args.score_all_modes else [args.score_mode]
    for mode in modes:
        out[f"score_{mode}"] = _score_from_features(out, mode)
    out["final_score"] = out[f"score_{args.score_mode}"]
    out["score_mode"] = args.score_mode
    output = Path(args.output_csv)
    output.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(output, index=False)

    for mode in modes:
        print(f"\n== {mode} ==")
        metrics = _metrics(out, f"score_{mode}")
        print(
            metrics.to_string(
                index=False,
                formatters={
                    "auc": lambda x: f"{x:.4f}",
                    "ap": lambda x: f"{x:.4f}",
                    "auc_neg": lambda x: f"{x:.4f}",
                },
            )
        )
    print(f"Saved -> {output}")


if __name__ == "__main__":
    main()
