#!/usr/bin/env python3
"""FALCON-style local patch pattern scores for PatchSTALL.

The image-paper inspiration is FALCON-Net: high-pass/noise patterns (INP) plus
3x3 local variation patterns (LVP). This tool adapts that idea to existing video
patch likelihood maps, without training a new network.
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

from dataset_utils import _is_missing_window, load_csv  # noqa: E402
from dataset_utils_patch import _get_patch_cache_path  # noqa: E402
from eval_patch_fast import FastPatchScorer  # noqa: E402


KEY_COLUMNS = ["subset", "source_model", "filename"]


def _stats(prefix: str, values: np.ndarray) -> dict[str, float]:
    arr = np.asarray(values, dtype=np.float32).reshape(-1)
    arr = arr[np.isfinite(arr)]
    keys = ("mean", "std", "p10", "p50", "p90", "p95")
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


def _topk_stats(prefix: str, values: np.ndarray) -> dict[str, float]:
    arr = np.asarray(values, dtype=np.float32).reshape(-1)
    arr = arr[np.isfinite(arr)]
    out = {}
    if arr.size == 0:
        for ratio in (0.05, 0.10, 0.20):
            out[f"{prefix}_top{int(round(ratio * 100)):02d}_mean"] = 0.0
        return out
    order = np.sort(arr)[::-1]
    for ratio in (0.05, 0.10, 0.20):
        k = max(1, int(round(arr.size * ratio)))
        out[f"{prefix}_top{int(round(ratio * 100)):02d}_mean"] = float(order[:k].mean())
    return out


def _gini(values: np.ndarray) -> float:
    arr = np.asarray(values, dtype=np.float64).reshape(-1)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return 0.0
    arr = arr - arr.min()
    if arr.sum() <= 1e-12:
        return 0.0
    arr = np.sort(arr)
    n = arr.size
    idx = np.arange(1, n + 1, dtype=np.float64)
    return float((2.0 * (idx * arr).sum()) / (n * arr.sum()) - (n + 1.0) / n)


def _neighbor_mean(x: np.ndarray) -> np.ndarray:
    # x: [T,H,W]
    padded = np.pad(x, ((0, 0), (1, 1), (1, 1)), mode="edge")
    total = np.zeros_like(x, dtype=np.float32)
    count = 0
    for dy in range(3):
        for dx in range(3):
            if dy == 1 and dx == 1:
                continue
            total += padded[:, dy : dy + x.shape[1], dx : dx + x.shape[2]]
            count += 1
    return total / float(count)


def _directional_diffs(x: np.ndarray) -> np.ndarray:
    padded = np.pad(x, ((0, 0), (1, 1), (1, 1)), mode="edge")
    diffs = []
    center = x
    for dy in range(3):
        for dx in range(3):
            if dy == 1 and dx == 1:
                continue
            neigh = padded[:, dy : dy + x.shape[1], dx : dx + x.shape[2]]
            diffs.append(center - neigh)
    return np.stack(diffs, axis=-1)  # [T,H,W,8]


def _crop_values(x: np.ndarray) -> np.ndarray:
    # x: [T,H,W], lower likelihood means more anomalous. Use high anomaly.
    h, w = x.shape[1:]
    h_mid = h // 2
    w_mid = w // 2
    masks = [
        np.s_[:, :h_mid, :w_mid],
        np.s_[:, :h_mid, w_mid:],
        np.s_[:, h_mid:, :w_mid],
        np.s_[:, h_mid:, w_mid:],
        np.s_[:, h // 4 : max(h // 4 + 1, 3 * h // 4), w // 4 : max(w // 4 + 1, 3 * w // 4)],
        np.s_[:, 1:-1, 1:-1] if h > 2 and w > 2 else np.s_[:, :, :],
    ]
    vals = []
    for mask in masks:
        part = x[mask]
        vals.append(float(part.mean()) if part.size else float(x.mean()))
    return np.asarray(vals, dtype=np.float32)


def _features_for_map(prefix: str, ll: np.ndarray, grid_size: tuple[int, int]) -> dict[str, float]:
    # Higher anomaly means lower likelihood relative to the map mean.
    gh, gw = grid_size
    x = ll.reshape(ll.shape[0], gh, gw).astype(np.float32)
    anomaly = -(x - float(x.mean()))
    neigh = _neighbor_mean(anomaly)
    highpass = np.abs(anomaly - neigh)
    diffs = _directional_diffs(anomaly)
    abs_diffs = np.abs(diffs)
    sign_balance = np.abs((diffs > 0).mean(axis=-1) - 0.5) * 2.0
    local_range = abs_diffs.max(axis=-1) - abs_diffs.min(axis=-1)
    temporal_abs = np.abs(np.diff(anomaly, axis=0)) if anomaly.shape[0] >= 2 else np.zeros_like(anomaly)
    temporal_hp = np.abs(np.diff(highpass, axis=0)) if highpass.shape[0] >= 2 else np.zeros_like(highpass)
    crop = _crop_values(anomaly)

    out: dict[str, float] = {}
    out.update(_stats(f"{prefix}_inp_highpass", highpass))
    out.update(_topk_stats(f"{prefix}_inp_highpass", highpass))
    out.update(_stats(f"{prefix}_lvp_absdiff", abs_diffs))
    out.update(_topk_stats(f"{prefix}_lvp_absdiff", abs_diffs))
    out.update(_stats(f"{prefix}_lvp_sign_balance", sign_balance))
    out.update(_stats(f"{prefix}_lvp_range", local_range))
    out.update(_topk_stats(f"{prefix}_lvp_range", local_range))
    out.update(_stats(f"{prefix}_temporal_abs", temporal_abs))
    out.update(_topk_stats(f"{prefix}_temporal_abs", temporal_abs))
    out.update(_stats(f"{prefix}_temporal_highpass", temporal_hp))
    out.update(_topk_stats(f"{prefix}_temporal_highpass", temporal_hp))
    out[f"{prefix}_inp_gini"] = _gini(highpass)
    out[f"{prefix}_lvp_gini"] = _gini(abs_diffs)
    out[f"{prefix}_crop_std"] = float(crop.std())
    out[f"{prefix}_crop_range"] = float(crop.max() - crop.min())
    out[f"{prefix}_crop_top_gap"] = float(crop.max() - np.median(crop))
    return out


def _iter_jobs(csv_path: str, patch_cache_root: str, duration: int, compact: bool, debug: int | None):
    df = load_csv(csv_path)
    window_col = f"{duration}_sec_idxs"
    if debug is not None:
        df = (
            df.groupby(["subset", "source_model"], group_keys=False)
            .apply(lambda g: g.head(debug))
            .reset_index(drop=True)
        )
    root = Path(patch_cache_root)
    for _, row in df.iterrows():
        if compact and _is_missing_window(row.get(window_col)):
            continue
        stem = Path(row["video_path"]).stem
        cache_path = _get_patch_cache_path(root, row["subset"], row["source_model"], stem, duration, compact)
        if not cache_path.exists():
            continue
        yield {
            "subset": row["subset"],
            "source_model": row["source_model"],
            "filename": Path(row["video_path"]).name,
            "cache_path": cache_path,
        }


@torch.inference_mode()
def _likelihood_maps(
    scorer: FastPatchScorer,
    patch_np: np.ndarray,
    patch_temp_mode: str,
    patch_region_size: int,
) -> tuple[np.ndarray, np.ndarray]:
    device = scorer.device
    patch = torch.from_numpy(patch_np.astype(np.float32))[None].to(device)
    mu_spat, w_spat, mu_temp, w_temp = scorer._params_for_device(device)
    spat_white = torch.matmul(patch - mu_spat, w_spat)
    spat_ll = scorer.log_likelihood_from_white(spat_white)[0].detach().cpu().numpy()
    temp = scorer.temporal_features(patch, patch_temp_mode, patch_region_size)
    temp_white = torch.matmul(temp - mu_temp, w_temp)
    temp_ll = scorer.log_likelihood_from_white(temp_white)[0].detach().cpu().numpy()
    return spat_ll.astype(np.float32), temp_ll.astype(np.float32)


def _score_modes(df: pd.DataFrame) -> dict[str, np.ndarray]:
    scores = {
        "falcon_inp_real": -(
            df["spat_inp_highpass_top10_mean"].to_numpy(float)
            + df["temp_inp_highpass_top10_mean"].to_numpy(float)
        ),
        "falcon_lvp_real": -(
            df["spat_lvp_absdiff_top10_mean"].to_numpy(float)
            + df["temp_lvp_absdiff_top10_mean"].to_numpy(float)
        ),
        "falcon_crop_robust_real": -(
            df["spat_crop_range"].to_numpy(float)
            + df["temp_crop_range"].to_numpy(float)
        ),
        "falcon_temporal_real": -(
            df["spat_temporal_highpass_top10_mean"].to_numpy(float)
            + df["temp_temporal_highpass_top10_mean"].to_numpy(float)
        ),
        "anti_falcon_inp_real": (
            df["spat_inp_highpass_top10_mean"].to_numpy(float)
            + df["temp_inp_highpass_top10_mean"].to_numpy(float)
        ),
        "anti_falcon_lvp_real": (
            df["spat_lvp_absdiff_top10_mean"].to_numpy(float)
            + df["temp_lvp_absdiff_top10_mean"].to_numpy(float)
        ),
    }
    real = df["subset"].astype(str).str.lower() == "real"
    source_features = {
        "falcon_inp": -scores["falcon_inp_real"],
        "falcon_lvp": -scores["falcon_lvp_real"],
        "falcon_crop": -scores["falcon_crop_robust_real"],
        "falcon_temporal": -scores["falcon_temporal_real"],
    }
    for name, values in source_features.items():
        rv = values[real.to_numpy()]
        med = float(np.median(rv))
        mad = float(np.median(np.abs(rv - med)))
        scale = max(mad * 1.4826, 1e-8)
        # Higher-is-real convention: large deviation from the real reference is fake.
        scores[f"{name}_centerdev_real"] = -np.abs(values - med) / scale
    return scores


def _metrics(df: pd.DataFrame, score_col: str) -> pd.DataFrame:
    real = df[df["subset"].astype(str).str.lower() == "real"]
    fake = df[df["subset"].astype(str).str.lower() != "real"]
    rows = []
    for source, group in fake.groupby("source_model", sort=True):
        y = np.concatenate([np.ones(len(real)), np.zeros(len(group))])
        s = np.concatenate([real[score_col].to_numpy(float), group[score_col].to_numpy(float)])
        rows.append(
            {
                "source_model": source,
                "n_real": int(len(real)),
                "n_fake": int(len(group)),
                "auc": float(roc_auc_score(y, s)),
                "ap": float(average_precision_score(y, s)),
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
        }
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", required=True)
    parser.add_argument("--patch-cache", required=True)
    parser.add_argument("--patch-params", required=True)
    parser.add_argument("--output-csv", required=True)
    parser.add_argument("--duration", type=int, default=2)
    parser.add_argument("--compact", action="store_true")
    parser.add_argument("--debug", type=int, default=None)
    parser.add_argument("--score-device", choices=["cuda", "cpu"], default="cuda")
    parser.add_argument("--patch-temp-mode", required=True)
    parser.add_argument("--patch-region-size", type=int, default=None)
    args = parser.parse_args()

    scorer = FastPatchScorer(args.patch_params, device=args.score_device)
    scorer.validate(args.patch_temp_mode)
    patch_region_size = args.patch_region_size if args.patch_region_size is not None else scorer.params_patch_region_size
    if patch_region_size != scorer.params_patch_region_size:
        raise ValueError(
            f"Patch params expect patch_region_size={scorer.params_patch_region_size}, got {patch_region_size}"
        )

    rows = []
    jobs = list(_iter_jobs(args.csv, args.patch_cache, args.duration, args.compact, args.debug))
    for job in tqdm(jobs, desc="FALCON local patterns"):
        payload = torch.load(job["cache_path"], map_location="cpu", weights_only=True)
        patch = payload["patch"].numpy().astype(np.float32)
        grid_size = tuple(int(x) for x in payload["grid_size"])
        if grid_size != scorer.patch_grid_size:
            raise ValueError(f"Params grid={scorer.patch_grid_size}, cache grid={grid_size}")
        spat_ll, temp_ll = _likelihood_maps(scorer, patch, args.patch_temp_mode, patch_region_size)
        feats = {}
        feats.update(_features_for_map("spat", spat_ll, grid_size))
        feats.update(_features_for_map("temp", temp_ll, grid_size))
        feats.update({key: job[key] for key in KEY_COLUMNS})
        rows.append(feats)

    out = pd.DataFrame(rows)
    for mode, scores in _score_modes(out).items():
        out[f"score_{mode}"] = scores
    out["final_score"] = out["score_falcon_inp_real"]
    out["score_mode"] = "falcon_inp_real"
    output = Path(args.output_csv)
    output.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(output, index=False)

    score_cols = [c for c in out.columns if c.startswith("score_") and c != "score_mode"]
    for mode in [c.removeprefix("score_") for c in score_cols]:
        print(f"\n== {mode} ==")
        print(
            _metrics(out, f"score_{mode}").to_string(
                index=False,
                formatters={"auc": lambda x: f"{x:.4f}", "ap": lambda x: f"{x:.4f}"},
            )
        )
    print(f"Saved -> {output}")


if __name__ == "__main__":
    main()
