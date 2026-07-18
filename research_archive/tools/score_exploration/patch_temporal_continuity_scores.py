#!/usr/bin/env python3
"""Evidence-level temporal continuity scores from patch likelihood maps.

This tool does not replace STALL/Patch-STall likelihood aggregation. It derives
auxiliary features describing whether low-likelihood temporal patch anomalies
persist over neighboring transitions.
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


def _jobs(csv_path: str, cache_root: str, duration: int, compact: bool, max_total: int | None, shuffle: bool, seed: int):
    df = load_csv(csv_path)
    if shuffle:
        df = df.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    if max_total is not None:
        df = df.head(max_total).reset_index(drop=True)

    root = Path(cache_root)
    window_col = f"{duration}_sec_idxs"
    out = []
    for _, row in df.iterrows():
        if compact and _is_missing_window(row.get(window_col)):
            continue
        video_path = str(row["video_path"])
        cache_path = _get_patch_cache_path(
            root,
            str(row["subset"]),
            str(row["source_model"]),
            Path(video_path).stem,
            duration,
            compact,
        )
        if not cache_path.exists():
            continue
        out.append(
            {
                "subset": str(row["subset"]),
                "source_model": str(row["source_model"]),
                "filename": Path(video_path).name,
                "cache_path": cache_path,
            }
        )
    return out


@torch.inference_mode()
def _temp_ll(scorer: FastPatchScorer, patch: torch.Tensor, mode: str, region_size: int) -> torch.Tensor:
    device = scorer.device
    patch = patch.unsqueeze(0).float().to(device)
    _, _, mu_temp, W_temp = scorer._params_for_device(device)
    temp = scorer.temporal_features(patch, mode, region_size)
    white = torch.matmul(temp - mu_temp, W_temp)
    return scorer.log_likelihood_from_white(white)[0].detach().cpu()


def _continuity_features(ll: np.ndarray, ratios: list[float]) -> dict[str, float]:
    # ll shape [T, P], lower = more anomalous.
    if ll.ndim != 2:
        raise ValueError(f"Expected [T,P], got {ll.shape}")
    out: dict[str, float] = {}
    t_count, p_count = ll.shape
    flat = ll.reshape(-1)
    frame_min = ll.min(axis=1)
    patch_min = ll.min(axis=0)

    out["frame_min_mean"] = float(frame_min.mean())
    out["frame_min_std"] = float(frame_min.std())
    out["patch_min_mean"] = float(patch_min.mean())
    out["patch_min_std"] = float(patch_min.std())

    for ratio in ratios:
        k = max(1, int(np.ceil(p_count * ratio)))
        masks = []
        for t in range(t_count):
            idx = np.argpartition(ll[t], kth=k - 1)[:k]
            mask = np.zeros(p_count, dtype=bool)
            mask[idx] = True
            masks.append(mask)
        mask_arr = np.stack(masks, axis=0)

        if t_count > 1:
            overlap = (mask_arr[1:] & mask_arr[:-1]).sum(axis=1) / float(k)
            union = (mask_arr[1:] | mask_arr[:-1]).sum(axis=1)
            jaccard = (mask_arr[1:] & mask_arr[:-1]).sum(axis=1) / np.maximum(union, 1)
        else:
            overlap = np.array([0.0], dtype=np.float32)
            jaccard = np.array([0.0], dtype=np.float32)

        # Longest consecutive run length for any patch being in the anomalous set.
        longest = 0
        active = np.zeros(p_count, dtype=np.int32)
        for t in range(t_count):
            active = (active + 1) * mask_arr[t].astype(np.int32)
            longest = max(longest, int(active.max()))

        tag = f"r{str(ratio).replace('.', 'p')}"
        out[f"overlap_mean_{tag}"] = float(overlap.mean())
        out[f"overlap_max_{tag}"] = float(overlap.max())
        out[f"jaccard_mean_{tag}"] = float(jaccard.mean())
        out[f"longest_run_{tag}"] = float(longest)
        out[f"longest_run_norm_{tag}"] = float(longest / max(t_count, 1))

        # Real-score convention: higher = more real. Prior experiments suggest
        # too persistent concentrated anomalies are suspicious, so negate them.
        out[f"score_less_persistent_{tag}"] = float(
            -0.5 * overlap.mean() - 0.3 * jaccard.mean() - 0.2 * (longest / max(t_count, 1))
        )
    return out


def _metrics(df: pd.DataFrame, score_col: str) -> pd.DataFrame:
    real = df[df["subset"].str.lower() == "real"]
    fake = df[df["subset"].str.lower() != "real"]
    rows = []
    for model, group in fake.groupby("source_model", sort=True):
        y = np.concatenate([np.ones(len(real)), np.zeros(len(group))])
        s = np.concatenate([real[score_col].to_numpy(float), group[score_col].to_numpy(float)])
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
    parser.add_argument("--patch-params", required=True)
    parser.add_argument("--output-csv", required=True)
    parser.add_argument("--duration", type=int, default=2)
    parser.add_argument("--compact", action="store_true", default=False)
    parser.add_argument("--patch-temp-mode", default="same_grid_second_order")
    parser.add_argument("--patch-region-size", type=int, default=3)
    parser.add_argument("--score-device", choices=["cuda", "cpu"], default="cuda")
    parser.add_argument("--ratios", default="0.05,0.10,0.20")
    parser.add_argument("--max-total", type=int, default=None)
    parser.add_argument("--shuffle", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    ratios = [float(x) for x in args.ratios.split(",") if x.strip()]
    scorer = FastPatchScorer(args.patch_params, device=args.score_device)
    scorer.validate(args.patch_temp_mode)
    jobs = _jobs(args.csv, args.patch_cache, args.duration, args.compact, args.max_total, args.shuffle, args.seed)

    rows = []
    for job in tqdm(jobs, desc="Temporal continuity"):
        payload = torch.load(job["cache_path"], map_location="cpu", weights_only=True)
        patch = payload["patch"]
        ll = _temp_ll(scorer, patch, args.patch_temp_mode, args.patch_region_size).numpy()
        feats = _continuity_features(ll, ratios)
        feats.update({col: job[col] for col in KEY_COLUMNS})
        rows.append(feats)

    out = pd.DataFrame(rows)
    Path(args.output_csv).parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.output_csv, index=False)
    print(f"Saved temporal continuity scores -> {args.output_csv}")
    for ratio in ratios:
        tag = f"r{str(ratio).replace('.', 'p')}"
        col = f"score_less_persistent_{tag}"
        print(f"\n== {col} ==")
        print(_metrics(out, col).to_string(index=False, formatters={"auc": lambda x: f"{x:.4f}", "ap": lambda x: f"{x:.4f}", "auc_neg": lambda x: f"{x:.4f}"}))


if __name__ == "__main__":
    main()
