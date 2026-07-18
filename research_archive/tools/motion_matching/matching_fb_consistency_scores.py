#!/usr/bin/env python3
"""Forward-backward patch matching consistency scores.

This keeps motion matching as a low-dimensional reliability signal. Instead of
using matched 1024-d deltas, it checks whether a local forward match can be
matched back to the original patch and summarizes confidence/uncertainty.
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


KEY_COLUMNS = ["subset", "source_model", "filename"]


def _is_missing(value) -> bool:
    return value is None or (isinstance(value, float) and np.isnan(value)) or str(value) == "nan"


def _cache_path(cache_root: Path, subset: str, source_model: str, video_path: str, duration: int) -> Path:
    return cache_root / subset / source_model / f"{Path(video_path).stem}_{duration}s.pt"


def _stats(prefix: str, values: np.ndarray) -> dict[str, float]:
    arr = np.asarray(values, dtype=np.float32).reshape(-1)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return {f"{prefix}_{k}": 0.0 for k in ["mean", "std", "p10", "p50", "p90", "p95"]}
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


def _rc_distance(a: np.ndarray, b: np.ndarray, grid_size: tuple[int, int]) -> np.ndarray:
    coords_a = np.asarray([patch_id_to_rc(int(x), grid_size) for x in a], dtype=np.float32)
    coords_b = np.asarray([patch_id_to_rc(int(x), grid_size) for x in b], dtype=np.float32)
    delta = coords_a - coords_b
    return np.sqrt((delta * delta).sum(axis=1)).astype(np.float32)


def _video_fb_features(
    patch: np.ndarray,
    grid_size: tuple[int, int],
    radius: int,
    top_m: int,
    temperature: float,
    lambda_dist: float,
    mode: str,
) -> dict[str, float]:
    arr = patch.astype(np.float32, copy=False)
    values: dict[str, list[np.ndarray]] = {
        "fb_error": [],
        "fb_same": [],
        "fb_uncertain_cycle": [],
        "forward_cosine": [],
        "backward_cosine": [],
        "forward_margin": [],
        "backward_margin": [],
        "forward_entropy": [],
        "backward_entropy": [],
        "forward_disp": [],
        "backward_disp": [],
        "disp_mismatch": [],
    }
    patch_ids = np.arange(arr.shape[1], dtype=np.int64)

    for t in range(arr.shape[0] - 1):
        fwd = matching_diagnostics_for_pair(
            arr[t],
            arr[t + 1],
            grid_size=grid_size,
            radius=radius,
            top_m=top_m,
            temperature=temperature,
            lambda_dist=lambda_dist,
            mode=mode,
        )
        bwd = matching_diagnostics_for_pair(
            arr[t + 1],
            arr[t],
            grid_size=grid_size,
            radius=radius,
            top_m=top_m,
            temperature=temperature,
            lambda_dist=lambda_dist,
            mode=mode,
        )
        back_to = bwd["matched_ids"][fwd["matched_ids"]]
        fb_error = _rc_distance(patch_ids, back_to, grid_size)
        fb_same = (back_to == patch_ids).astype(np.float32)
        fwd_margin = fwd["top1_scores"] - fwd["top2_scores"]
        bwd_margin = bwd["top1_scores"][fwd["matched_ids"]] - bwd["top2_scores"][fwd["matched_ids"]]
        fwd_entropy = np.nan_to_num(fwd["entropy"], nan=0.0)
        bwd_entropy = np.nan_to_num(bwd["entropy"][fwd["matched_ids"]], nan=0.0)
        fwd_disp = fwd["displacements"]
        bwd_disp = bwd["displacements"][fwd["matched_ids"]]
        uncertain_cycle = fb_error + 0.25 * (fwd_entropy + bwd_entropy) - 0.25 * (fwd_margin + bwd_margin)

        values["fb_error"].append(fb_error)
        values["fb_same"].append(fb_same)
        values["fb_uncertain_cycle"].append(uncertain_cycle.astype(np.float32))
        values["forward_cosine"].append(fwd["top1_cosines"])
        values["backward_cosine"].append(bwd["top1_cosines"][fwd["matched_ids"]])
        values["forward_margin"].append(fwd_margin)
        values["backward_margin"].append(bwd_margin)
        values["forward_entropy"].append(fwd_entropy)
        values["backward_entropy"].append(bwd_entropy)
        values["forward_disp"].append(fwd_disp)
        values["backward_disp"].append(bwd_disp)
        values["disp_mismatch"].append(np.abs(fwd_disp - bwd_disp).astype(np.float32))

    out: dict[str, float] = {}
    for name, chunks in values.items():
        merged = np.concatenate(chunks) if chunks else np.zeros(0, dtype=np.float32)
        out.update(_stats(name, merged))
    for name in ["fb_error", "fb_uncertain_cycle", "disp_mismatch"]:
        merged = np.concatenate(values[name]) if values[name] else np.zeros(0, dtype=np.float32)
        out.update(_topk_stats(name, merged))
    out["fb_error_gt0_ratio"] = float((np.concatenate(values["fb_error"]) > 0.0).mean()) if values["fb_error"] else 0.0
    out["fb_error_gt1_ratio"] = float((np.concatenate(values["fb_error"]) > 1.01).mean()) if values["fb_error"] else 0.0
    return out


def _score_from_features(df: pd.DataFrame, mode: str) -> np.ndarray:
    if mode == "fb_consistent_real":
        return (
            -df["fb_error_mean"].to_numpy()
            + df["fb_same_mean"].to_numpy()
            + 0.25 * df["forward_margin_mean"].to_numpy()
            + 0.25 * df["backward_margin_mean"].to_numpy()
        )
    if mode == "fb_uncertainty_real":
        return (
            df["fb_error_mean"].to_numpy()
            + 0.5 * df["forward_entropy_mean"].to_numpy()
            + 0.5 * df["backward_entropy_mean"].to_numpy()
            + 0.25 * df["disp_mismatch_mean"].to_numpy()
            - 0.25 * df["forward_margin_mean"].to_numpy()
            - 0.25 * df["backward_margin_mean"].to_numpy()
        )
    if mode == "fb_cycle_uncertainty_top10":
        return df["fb_uncertain_cycle_top10_mean"].to_numpy()
    if mode == "anti_fb_cycle_uncertainty_top10":
        return -df["fb_uncertain_cycle_top10_mean"].to_numpy()
    if mode == "fb_error_real":
        return df["fb_error_mean"].to_numpy()
    if mode == "anti_fb_error_real":
        return -df["fb_error_mean"].to_numpy()
    raise ValueError(f"Unknown score mode: {mode}")


def _available_score_modes() -> list[str]:
    return [
        "fb_consistent_real",
        "fb_uncertainty_real",
        "fb_cycle_uncertainty_top10",
        "anti_fb_cycle_uncertainty_top10",
        "fb_error_real",
        "anti_fb_error_real",
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
    parser.add_argument(
        "--source-model",
        action="append",
        help="Keep real rows plus the named fake source model. Can be repeated.",
    )
    parser.add_argument(
        "--real-max",
        type=int,
        default=None,
        help="After source filtering, keep at most N real rows with a deterministic sample.",
    )
    parser.add_argument("--real-seed", type=int, default=0)
    parser.add_argument(
        "--output-partial-every",
        type=int,
        default=0,
        help="Write a partial CSV every N processed rows. Disabled by default.",
    )
    parser.add_argument("--radius", type=int, default=2)
    parser.add_argument("--top-m", type=int, default=4)
    parser.add_argument("--temperature", type=float, default=0.07)
    parser.add_argument("--lambda-dist", type=float, default=0.01)
    parser.add_argument("--match-mode", choices=["hard", "soft"], default="soft")
    parser.add_argument("--score-mode", default="fb_uncertainty_real")
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
    if args.source_model:
        keep_sources = set(args.source_model)
        real_mask = df["subset"].astype(str).str.lower() == "real"
        source_mask = df["source_model"].astype(str).isin(keep_sources)
        df = df[real_mask | source_mask].reset_index(drop=True)
    if args.real_max is not None:
        real_mask = df["subset"].astype(str).str.lower() == "real"
        real = df[real_mask]
        nonreal = df[~real_mask]
        if len(real) > args.real_max:
            real = real.sample(n=args.real_max, random_state=args.real_seed)
        df = pd.concat([real, nonreal], ignore_index=True)

    rows = []
    cache_root = Path(args.patch_cache)
    for _, row in tqdm(df.iterrows(), total=len(df), desc="FB matching consistency"):
        rowd = row.to_dict()
        if _is_missing(rowd.get(f"{args.duration}_sec_idxs")):
            continue
        path = _cache_path(cache_root, str(rowd["subset"]), str(rowd["source_model"]), str(rowd["video_path"]), args.duration)
        if not path.exists():
            continue
        payload = torch.load(path, map_location="cpu", weights_only=True)
        feats = _video_fb_features(
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
        if args.output_partial_every and len(rows) % args.output_partial_every == 0:
            partial = pd.DataFrame(rows)
            output = Path(args.output_csv)
            output.parent.mkdir(parents=True, exist_ok=True)
            partial.to_csv(output.with_suffix(output.suffix + ".partial"), index=False)

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
