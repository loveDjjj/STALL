"""Create multiple patch calibration params sharing one whitening fit.

For a fixed dataset / temporal mode / patch region, aggregation choices such
as bottom-k ratios and mean only affect calibration video scores. The sampled
tokens and whitening matrices are identical, so this script fits whitening
once and saves one params file per aggregation.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from create_patch_params import (
    _fit_whitening,
    _get_mu_W,
    aggregate_scores,
    collect_fit_samples,
    iter_real_patch_cache,
)
from patch_matching import patch_temporal_delta
from stall import log_likelihood, whitening_transform as apply_whitening


AGG_PRESETS = {
    "bottomk0p2": ("bottomk_mean", 0.20),
    "bottomk0p5": ("bottomk_mean", 0.50),
    "mean": ("mean", 0.50),
    "strun3x3_bottomk0p2": ("spatiotemporal_run_bottomk_mean", 0.20),
    "strun3x3_bottomk0p5": ("spatiotemporal_run_bottomk_mean", 0.50),
}


def _aggregation_config(
    *,
    aggregation: str,
    bottomk_ratio: float,
    temporal_run_length: int,
    seed: int,
    max_patches_for_fit: int,
    max_real_videos: int | None,
    patch_temp_mode: str,
    match_radius: int,
    top_m: int,
    temperature: float,
    lambda_dist: float,
    patch_region_size: int,
    aggregation_region_size: int,
) -> np.ndarray:
    return np.array(
        json.dumps(
            {
                "mode": aggregation,
                "bottomk_ratio": bottomk_ratio,
                "temporal_run_length": temporal_run_length,
                "seed": seed,
                "max_patches_for_fit": max_patches_for_fit,
                "implementation": "streaming_multiagg_shared_whitening",
                "temporal_feature_version": "l2_normalized_delta_v1",
                "max_real_videos": max_real_videos,
                "patch_temp_mode": patch_temp_mode,
                "match_radius": match_radius,
                "top_m": top_m,
                "temperature": temperature,
                "lambda_dist": lambda_dist,
                "patch_region_size": patch_region_size,
                "aggregation_region_size": aggregation_region_size,
            }
        )
    )


def build_multiagg_params(args: argparse.Namespace) -> dict[str, dict]:
    agg_specs = [(name, *AGG_PRESETS[name]) for name in args.agg]

    patch_fit, temp_fit, grid_size = collect_fit_samples(
        csv_path=args.csv,
        patch_emb_cache_dir=args.patch_emb_cache,
        duration_sec=args.duration,
        compact=args.compact,
        max_patches_for_fit=args.max_patches_for_fit,
        seed=args.seed,
        patch_temp_mode=args.patch_temp_mode,
        match_radius=args.match_radius,
        top_m=args.top_m,
        temperature=args.temperature,
        lambda_dist=args.lambda_dist,
        patch_region_size=args.patch_region_size,
        max_real_videos=args.max_real_videos,
    )

    print(f"Fitting shared patch spatial whitening on {len(patch_fit)} patch tokens", flush=True)
    wt_spat = _fit_whitening(patch_fit)
    mu_patch_spat, W_patch_spat = _get_mu_W(wt_spat)

    print(f"Fitting shared patch temporal whitening on {len(temp_fit)} patch transitions", flush=True)
    wt_temp = _fit_whitening(temp_fit)
    mu_patch_temp, W_patch_temp = _get_mu_W(wt_temp)

    calib_spat = {name: [] for name, _, _ in agg_specs}
    calib_temp = {name: [] for name, _, _ in agg_specs}

    print("Streaming real patch cache for shared calibration scoring...", flush=True)
    for i, (_, payload) in enumerate(
        iter_real_patch_cache(
            args.csv,
            args.patch_emb_cache,
            args.duration,
            args.compact,
            args.max_real_videos,
        ),
        start=1,
    ):
        patch_np = payload["patch"].numpy().astype(np.float32)
        patch = patch_np[np.newaxis]
        grid_size_payload = tuple(int(x) for x in payload["grid_size"])

        patch_spat_ll = log_likelihood(apply_whitening(patch, mu_patch_spat, W_patch_spat))
        patch_temp = patch_temporal_delta(
            patch_np,
            grid_size=grid_size_payload,
            mode=args.patch_temp_mode,
            radius=args.match_radius,
            top_m=args.top_m,
            temperature=args.temperature,
            lambda_dist=args.lambda_dist,
            region_size=args.patch_region_size,
        )[np.newaxis]
        patch_temp_ll = log_likelihood(apply_whitening(patch_temp, mu_patch_temp, W_patch_temp))

        for name, aggregation, bottomk_ratio in agg_specs:
            calib_spat[name].append(
                float(
                    aggregate_scores(
                        patch_spat_ll,
                        aggregation,
                        bottomk_ratio,
                        args.temporal_run_length,
                        grid_size_payload,
                        args.aggregation_region_size,
                    )[0]
                )
            )
            calib_temp[name].append(
                float(
                    aggregate_scores(
                        patch_temp_ll,
                        aggregation,
                        bottomk_ratio,
                        args.temporal_run_length,
                        grid_size_payload,
                        args.aggregation_region_size,
                    )[0]
                )
            )

        if i % 200 == 0:
            print(f"  scored_videos={i}", flush=True)

    params_by_agg = {}
    for name, aggregation, bottomk_ratio in agg_specs:
        params_by_agg[name] = {
            "mu_patch_spat": mu_patch_spat.astype(np.float32),
            "W_patch_spat": W_patch_spat.astype(np.float32),
            "calib_patch_spat_scores": np.array(calib_spat[name], dtype=np.float32),
            "mu_patch_temp": mu_patch_temp.astype(np.float32),
            "W_patch_temp": W_patch_temp.astype(np.float32),
            "calib_patch_temp_scores": np.array(calib_temp[name], dtype=np.float32),
            "patch_grid_size": np.array(grid_size, dtype=np.int32),
            "duration": np.array([args.duration], dtype=np.int32),
            "aggregation_config": _aggregation_config(
                aggregation=aggregation,
                bottomk_ratio=bottomk_ratio,
                temporal_run_length=args.temporal_run_length,
                seed=args.seed,
                max_patches_for_fit=args.max_patches_for_fit,
                max_real_videos=args.max_real_videos,
                patch_temp_mode=args.patch_temp_mode,
                match_radius=args.match_radius,
                top_m=args.top_m,
                temperature=args.temperature,
                lambda_dist=args.lambda_dist,
                patch_region_size=args.patch_region_size,
                aggregation_region_size=args.aggregation_region_size,
            ),
        }
    return params_by_agg


def main() -> None:
    parser = argparse.ArgumentParser(description="Create shared-whitening patch params for multiple aggregations.")
    parser.add_argument("--csv", required=True)
    parser.add_argument("--patch-emb-cache", required=True)
    parser.add_argument("--output-template", required=True, help="Template containing {agg}, e.g. params_{agg}.npz")
    parser.add_argument("--duration", type=int, default=2, choices=[1, 2, 3, 4])
    parser.add_argument("--compact", action="store_true", default=False)
    parser.add_argument("--real-only", action="store_true", default=False)
    parser.add_argument("--max-patches-for-fit", type=int, default=300000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-real-videos", type=int, default=None)
    parser.add_argument("--patch-temp-mode", required=True)
    parser.add_argument("--match-radius", type=int, default=2)
    parser.add_argument("--top-m", type=int, default=4)
    parser.add_argument("--temperature", type=float, default=0.07)
    parser.add_argument("--lambda-dist", type=float, default=0.01)
    parser.add_argument("--patch-region-size", type=int, default=1)
    parser.add_argument("--temporal-run-length", type=int, default=3)
    parser.add_argument("--aggregation-region-size", type=int, default=3)
    parser.add_argument("--agg", nargs="+", choices=sorted(AGG_PRESETS), default=["bottomk0p2", "bottomk0p5", "mean"])
    parser.add_argument("--skip-existing", action="store_true", default=False)
    args = parser.parse_args()

    if not args.real_only:
        raise ValueError("This script only supports --real-only")
    if "{agg}" not in args.output_template:
        raise ValueError("--output-template must contain {agg}")

    outputs = {agg: Path(args.output_template.format(agg=agg)) for agg in args.agg}
    if args.skip_existing and all(path.exists() and path.stat().st_size > 0 for path in outputs.values()):
        print("All requested params already exist; skipping.", flush=True)
        return

    params_by_agg = build_multiagg_params(args)
    for agg, params in params_by_agg.items():
        out = outputs[agg]
        if args.skip_existing and out.exists() and out.stat().st_size > 0:
            print(f"Skip existing params: {out}", flush=True)
            continue
        out.parent.mkdir(parents=True, exist_ok=True)
        np.savez(out, **params)
        print(f"Saved patch params [{agg}]: {out}", flush=True)


if __name__ == "__main__":
    main()
