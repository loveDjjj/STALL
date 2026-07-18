"""Minimal patch-level eval CLI for PatchSTALL.

First version only supports:

- CSV input produced by video_index.py
- patch embedding cache
- patch spatial params
- patch-only scoring
- spatial-only and spatial + same-grid temporal
- global + patch average fusion
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from dataset_utils import _get_cache_path, load_csv
from dataset_utils_patch import (
    count_patch_cache_misses,
    load_csv_with_patch_cache,
    prefill_patch_emb_cache,
)
from metrics import Score, ScoreDirection, get_results_df, print_results
from patch_matching import patch_temporal_delta
from stall import STALL, log_likelihood, whitening_transform as apply_whitening
from stall_patch import PatchSTALL


def bottomk_mean(arr: np.ndarray, ratio: float) -> np.ndarray:
    flat = arr.reshape(arr.shape[0], -1)
    k = max(1, int(np.ceil(flat.shape[1] * ratio)))
    part = np.partition(flat, kth=k - 1, axis=1)[:, :k]
    return part.mean(axis=1)


class PatchSpatialScorer:
    def __init__(self, patch_params_path: str):
        data = np.load(patch_params_path, allow_pickle=True)
        self.mu_patch_spat = data["mu_patch_spat"]
        self.W_patch_spat = data["W_patch_spat"]
        self.calib_patch_spat_scores = np.sort(data["calib_patch_spat_scores"])
        self.mu_patch_temp = data["mu_patch_temp"] if "mu_patch_temp" in data.files else None
        self.W_patch_temp = data["W_patch_temp"] if "W_patch_temp" in data.files else None
        self.calib_patch_temp_scores = (
            np.sort(data["calib_patch_temp_scores"])
            if "calib_patch_temp_scores" in data.files
            else None
        )
        self.patch_grid_size = tuple(int(x) for x in data["patch_grid_size"].tolist())

        agg_raw = data["aggregation_config"]
        if isinstance(agg_raw, np.ndarray):
            agg_raw = agg_raw.item()
        self.aggregation_config = json.loads(str(agg_raw))
        self.bottomk_ratio = float(self.aggregation_config.get("bottomk_ratio", 0.05))
        self.aggregation_mode = self.aggregation_config.get("mode", "bottomk_mean")
        self.expected_patch_temp_mode = self.aggregation_config.get("patch_temp_mode", "same_grid")
        self.temporal_feature_version = self.aggregation_config.get("temporal_feature_version")
        self.match_radius = int(self.aggregation_config.get("match_radius", 2))
        self.top_m = int(self.aggregation_config.get("top_m", 4))
        self.temperature = float(self.aggregation_config.get("temperature", 0.07))
        self.lambda_dist = float(self.aggregation_config.get("lambda_dist", 0.01))
        self.patch_region_size = int(self.aggregation_config.get("patch_region_size", 1))

    def _aggregate_patch_spatial(self, patch_spat_ll: np.ndarray) -> np.ndarray:
        if self.aggregation_mode == "mean":
            return patch_spat_ll.reshape(patch_spat_ll.shape[0], -1).mean(axis=1)
        if self.aggregation_mode == "min":
            return patch_spat_ll.reshape(patch_spat_ll.shape[0], -1).min(axis=1)
        return bottomk_mean(patch_spat_ll, self.bottomk_ratio)

    def _percentile(self, score: np.ndarray) -> np.ndarray:
        positions = np.searchsorted(self.calib_patch_spat_scores, score, side="right")
        return positions / len(self.calib_patch_spat_scores)

    def _temporal_percentile(self, score: np.ndarray) -> np.ndarray:
        if self.calib_patch_temp_scores is None:
            raise ValueError("Patch temporal calibration scores are missing")
        positions = np.searchsorted(self.calib_patch_temp_scores, score, side="right")
        return positions / len(self.calib_patch_temp_scores)

    def validate_temporal_args(self, patch_temp_mode: str, match_radius: int | None = None, top_m: int | None = None, temperature: float | None = None, lambda_dist: float | None = None):
        if patch_temp_mode == "none":
            return
        if patch_temp_mode != self.expected_patch_temp_mode:
            raise ValueError(
                f"Patch params expect patch_temp_mode={self.expected_patch_temp_mode}, "
                f"but eval requested {patch_temp_mode}"
            )
        if self.temporal_feature_version != "l2_normalized_delta_v1":
            raise ValueError(
                "Patch params were created with an old or unknown temporal feature version "
                f"({self.temporal_feature_version!r}). Recreate params with current create_patch_params.py."
            )
        if match_radius is not None and int(match_radius) != self.match_radius:
            raise ValueError(f"Patch params expect match_radius={self.match_radius}, got {match_radius}")
        if patch_temp_mode != "same_grid" and top_m is not None and int(top_m) != self.top_m:
            raise ValueError(f"Patch params expect top_m={self.top_m}, got {top_m}")
        if patch_temp_mode == "motion_soft" and temperature is not None and abs(float(temperature) - self.temperature) > 1e-12:
            raise ValueError(f"Patch params expect temperature={self.temperature}, got {temperature}")
        if patch_temp_mode in {"motion_hard", "motion_soft"} and lambda_dist is not None and abs(float(lambda_dist) - self.lambda_dist) > 1e-12:
            raise ValueError(f"Patch params expect lambda_dist={self.lambda_dist}, got {lambda_dist}")

    def _aggregate_patch_temporal(self, patch_temp_ll: np.ndarray) -> np.ndarray:
        if self.aggregation_mode == "mean":
            return patch_temp_ll.reshape(patch_temp_ll.shape[0], -1).mean(axis=1)
        if self.aggregation_mode == "min":
            return patch_temp_ll.reshape(patch_temp_ll.shape[0], -1).min(axis=1)
        return bottomk_mean(patch_temp_ll, self.bottomk_ratio)

    def score_patch_cache(self, sample: dict, patch_temp_mode: str = "none", patch_spat_weight: float = 0.5, patch_temp_weight: float = 0.5) -> dict:
        patch = sample["patch"][np.newaxis]  # [1, T, P, D]
        patch_spat_ll = log_likelihood(
            apply_whitening(patch, self.mu_patch_spat, self.W_patch_spat)
        )  # [1, T, P]
        patch_spat_agg = self._aggregate_patch_spatial(patch_spat_ll)
        patch_spat_percentile = self._percentile(patch_spat_agg)
        patch_final_score = float(patch_spat_percentile[0])

        result = {
            "patch_spat_ll": patch_spat_ll,
            "patch_spat_agg": float(patch_spat_agg[0]),
            "patch_spat_percentile": patch_final_score,
            "patch_final_score": patch_final_score,
            "final_score": patch_final_score,
        }

        if patch_temp_mode != "none":
            if self.mu_patch_temp is None or self.W_patch_temp is None or self.calib_patch_temp_scores is None:
                raise ValueError("Patch temporal params are missing from patch params file")
            self.validate_temporal_args(patch_temp_mode)
            patch_np = sample["patch"].astype(np.float32)
            grid_size = tuple(int(x) for x in sample["grid_size"])
            patch_temp = patch_temporal_delta(
                patch_np,
                grid_size=grid_size,
                mode=patch_temp_mode,
                radius=self.match_radius,
                top_m=self.top_m,
                temperature=self.temperature,
                lambda_dist=self.lambda_dist,
                region_size=self.patch_region_size,
            )[np.newaxis]
            patch_temp_ll = log_likelihood(
                apply_whitening(patch_temp, self.mu_patch_temp, self.W_patch_temp)
            )  # [1, T-1, P]
            patch_temp_agg = self._aggregate_patch_temporal(patch_temp_ll)
            patch_temp_percentile = self._temporal_percentile(patch_temp_agg)
            denom = max(patch_spat_weight + patch_temp_weight, 1e-8)
            patch_final_score = float(
                (patch_spat_weight * patch_spat_percentile[0] + patch_temp_weight * patch_temp_percentile[0]) / denom
            )
            result.update(
                {
                    "patch_temp_ll": patch_temp_ll,
                    "patch_temp_agg": float(patch_temp_agg[0]),
                    "patch_temp_percentile": float(patch_temp_percentile[0]),
                    "patch_temp_mode": patch_temp_mode,
                    "match_radius": self.match_radius,
                    "top_m": self.top_m,
                    "temperature": self.temperature,
                    "lambda_dist": self.lambda_dist,
                    "patch_region_size": self.patch_region_size,
                    "patch_final_score": patch_final_score,
                    "final_score": patch_final_score,
                }
            )

        return result


class GlobalScorer:
    def __init__(self, global_params_path: str):
        data = np.load(global_params_path, allow_pickle=True)
        self.model = STALL(device="cpu", data_dict=data, load_dino=False)

    def score_global_emb(self, emb: np.ndarray) -> dict:
        result = self.model._scores_from_embs(emb[np.newaxis])
        return {
            "global_spat_percentile": float(result["spat_percentile"][0]),
            "global_temp_percentile": float(result["temp_percentile"][0]),
            "global_final_score": float(result["final_score"][0]),
        }


def make_model(load_dino: bool) -> PatchSTALL:
    import torch

    device = "cuda" if load_dino and torch.cuda.is_available() else "cpu"
    return PatchSTALL(device=device, data_dict=None, load_dino=load_dino)


def load_global_cache_lookup(
    csv_path: str,
    emb_cache_dir: str,
    duration_sec: int,
    compact: bool,
    debug_n: int | None = None,
) -> dict[str, str]:
    df = load_csv(csv_path)
    window_col = f"{duration_sec}_sec_idxs"

    if debug_n is not None:
        df = (
            df.groupby(["subset", "source_model"], group_keys=False)
            .apply(lambda g: g.head(debug_n))
            .reset_index(drop=True)
        )

    cache_root = Path(emb_cache_dir)
    lookup = {}
    for _, row in df.iterrows():
        if compact and row.get(window_col) is None:
            continue
        stem = Path(row["video_path"]).stem
        cache_path = _get_cache_path(
            cache_root, row["subset"], row["source_model"], stem, duration_sec, compact
        )
        if cache_path.exists():
            lookup[row["video_path"]] = str(cache_path)
    return lookup


def load_global_score_lookup(global_score_csv: str) -> dict[tuple[str, str, str], float]:
    df = pd.read_csv(global_score_csv)
    required = {"subset", "source_model", "filename", "final_score"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(
            f"Global score CSV '{global_score_csv}' missing required columns: {sorted(missing)}"
        )
    lookup = {}
    for _, row in df.iterrows():
        key = (row["subset"], row["source_model"], row["filename"])
        lookup[key] = float(row["final_score"])
    return lookup


def run_csv(args) -> pd.DataFrame:
    misses = count_patch_cache_misses(
        args.csv,
        args.patch_emb_cache,
        duration_sec=args.duration,
        debug_n=args.debug_n,
        compact=args.compact,
    )
    if args.no_load_dino and misses > 0:
        raise ValueError(f"--no-load-dino requested but {misses} patch cache files are missing")
    model = make_model(load_dino=not args.no_load_dino)
    scorer = PatchSpatialScorer(args.patch_params)
    scorer.validate_temporal_args(
        args.patch_temp_mode,
        match_radius=args.match_radius,
        top_m=args.top_m,
        temperature=args.temperature,
        lambda_dist=args.lambda_dist,
    )
    global_scorer = None
    global_lookup = {}
    global_score_lookup = {}
    if args.fusion == "avg":
        if args.global_score_csv:
            global_score_lookup = load_global_score_lookup(args.global_score_csv)
        else:
            if not args.emb_cache or not args.global_params:
                raise ValueError(
                    "--fusion avg requires either --global-score-csv or both --emb-cache and --global-params"
                )
            global_scorer = GlobalScorer(args.global_params)
            global_lookup = load_global_cache_lookup(
                args.csv,
                args.emb_cache,
                duration_sec=args.duration,
                compact=args.compact,
                debug_n=args.debug_n,
            )

    total_df = pd.read_csv(args.csv)
    if args.debug_n is not None:
        total_df = (
            total_df.groupby(["subset", "source_model"], group_keys=False)
            .apply(lambda g: g.head(args.debug_n))
            .reset_index(drop=True)
        )
    total = len(total_df)

    print(
        f"Phase 1/2 — Extracting patch embeddings ({misses} cache misses, {total - misses} cached)",
        flush=True,
    )

    from tqdm import tqdm

    for _ in tqdm(
        prefill_patch_emb_cache(
            args.csv,
            args.patch_emb_cache,
            model=model,
            duration_sec=args.duration,
            debug_n=args.debug_n,
            compact=args.compact,
            num_workers=args.workers,
            video_batch=args.video_batch,
        ),
        desc="Extracting",
        unit=" video",
        dynamic_ncols=True,
        total=misses,
    ):
        pass

    print("Phase 2/2 — Scoring", flush=True)
    rows = []
    for sample in tqdm(
        load_csv_with_patch_cache(
            args.csv,
            args.patch_emb_cache,
            model=model,
            duration_sec=args.duration,
            debug_n=args.debug_n,
            compact=args.compact,
        ),
        desc="Scoring",
        unit="video",
        dynamic_ncols=True,
        total=total,
    ):
        result = scorer.score_patch_cache(
            sample,
            patch_temp_mode=args.patch_temp_mode,
            patch_spat_weight=args.patch_spat_weight,
            patch_temp_weight=args.patch_temp_weight,
        )
        global_result = None
        if global_scorer is not None:
            cache_path = global_lookup.get(sample["video_path"])
            if cache_path is not None:
                import torch

                global_emb = torch.load(cache_path, weights_only=True).numpy()
                global_result = global_scorer.score_global_emb(global_emb)
                patch_branch_weight = 1.0 - args.global_weight
                result["final_score"] = (
                    args.global_weight * global_result["global_final_score"] +
                    patch_branch_weight * result["patch_final_score"]
                )
        elif global_score_lookup:
            key = (sample["subset"], sample["source_model"], sample["filename"])
            global_final_score = global_score_lookup.get(key)
            if global_final_score is not None:
                global_result = {
                    "global_spat_percentile": None,
                    "global_temp_percentile": None,
                    "global_final_score": float(global_final_score),
                }
                patch_branch_weight = 1.0 - args.global_weight
                result["final_score"] = (
                    args.global_weight * global_result["global_final_score"] +
                    patch_branch_weight * result["patch_final_score"]
                )

        row = {
            "subset": sample["subset"],
            "source_model": sample["source_model"],
            "filename": sample["filename"],
            "fusion": args.fusion,
            "global_weight": args.global_weight if args.fusion == "avg" else None,
            "patch_spat_weight": args.patch_spat_weight,
            "patch_temp_weight": args.patch_temp_weight if args.patch_temp_mode != "none" else None,
            "patch_temp_mode": args.patch_temp_mode,
            "match_radius": scorer.match_radius if args.patch_temp_mode != "none" else None,
            "top_m": scorer.top_m if args.patch_temp_mode != "none" else None,
            "temperature": scorer.temperature if args.patch_temp_mode != "none" else None,
            "lambda_dist": scorer.lambda_dist if args.patch_temp_mode != "none" else None,
            "patch_spat_percentile": result["patch_spat_percentile"],
            "patch_final_score": result["patch_final_score"],
            "final_score": result["final_score"],
        }
        if "patch_temp_percentile" in result:
            row["patch_temp_percentile"] = result["patch_temp_percentile"]
        if global_result is not None:
            row["global_spat_percentile"] = global_result["global_spat_percentile"]
            row["global_temp_percentile"] = global_result["global_temp_percentile"]
            row["global_final_score"] = global_result["global_final_score"]
        rows.append(row)

    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser(
        description="Minimal patch-level eval for PatchSTALL."
    )
    parser.add_argument("--csv", required=True)
    parser.add_argument("--patch-emb-cache", required=True)
    parser.add_argument("--patch-params", required=True)
    parser.add_argument("--output-csv", required=True)
    parser.add_argument("--emb-cache", default=None, help="Original global embedding cache (required for fusion=avg)")
    parser.add_argument("--global-params", default=None, help="Original global STALL params (required for fusion=avg)")
    parser.add_argument("--global-score-csv", default=None, help="Existing baseline result CSV with final_score to reuse for fusion")
    parser.add_argument("--duration", type=int, default=2, choices=[1, 2, 3, 4])
    parser.add_argument("--compact", action="store_true", default=False)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--video-batch", type=int, default=8)
    parser.add_argument("--debug-n", type=int, default=None)
    parser.add_argument("--no-load-dino", action="store_true", help="Require existing patch cache and skip DINOv3 model loading.")
    parser.add_argument("--fusion", choices=["patch_only", "avg"], default="patch_only")
    parser.add_argument("--global-weight", type=float, default=0.5)
    parser.add_argument("--patch-spat-weight", type=float, default=0.5)
    parser.add_argument("--patch-temp-weight", type=float, default=0.5)
    parser.add_argument("--match-radius", type=int, default=2)
    parser.add_argument("--top-m", type=int, default=4)
    parser.add_argument("--temperature", type=float, default=0.07)
    parser.add_argument("--lambda-dist", type=float, default=0.01)
    parser.add_argument(
        "--patch-temp-mode",
        choices=[
            "none",
            "same_grid",
            "same_grid_lag1",
            "same_grid_multilag",
            "same_grid_second_order",
            "same_grid_multilag_second_order",
            "motion_hard",
            "motion_soft",
        ],
        default="none",
        help="Patch temporal mode. 'none' = spatial only.",
    )
    args = parser.parse_args()

    df = run_csv(args)
    out_path = Path(args.output_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    print(f"Saved per-video scores -> {out_path}")

    results_df = get_results_df(
        df[["subset", "source_model"]],
        {
            "final_score": Score(
                value=df["final_score"].to_numpy(),
                direction=ScoreDirection.HIGHER_IS_REAL,
            )
        },
    )
    print_results(results_df)


if __name__ == "__main__":
    main()
