"""Fast patch-only evaluation for several aggregations in one cache pass."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from tqdm import tqdm

from eval_patch_fast import FastPatchScorer, iter_cache_jobs, load_cache_batch
from metrics import Score, ScoreDirection, get_results_df, print_results


AGG_SPECS = {
    "bottomk0p2": ("bottomk_mean", 0.20),
    "bottomk0p5": ("bottomk_mean", 0.50),
    "mean": ("mean", 0.50),
}


def path_from_pattern(pattern: str, agg_name: str) -> Path:
    if "{agg}" not in pattern:
        raise ValueError("pattern must contain {agg}")
    return Path(pattern.format(agg=agg_name))


def main() -> None:
    parser = argparse.ArgumentParser(description="Fast multi-aggregation patch-only eval.")
    parser.add_argument("--csv", required=True)
    parser.add_argument("--patch-emb-cache", required=True)
    parser.add_argument("--patch-params-pattern", required=True)
    parser.add_argument("--output-csv-pattern", required=True)
    parser.add_argument("--duration", type=int, default=2, choices=[1, 2, 3, 4])
    parser.add_argument("--compact", action="store_true", default=False)
    parser.add_argument("--debug-n", type=int, default=None)
    parser.add_argument("--score-device", choices=["cuda", "cpu"], default="cuda")
    parser.add_argument("--score-batch-size", type=int, default=64)
    parser.add_argument("--patch-temp-mode", required=True)
    parser.add_argument("--patch-region-size", type=int, required=True)
    parser.add_argument("--patch-spat-weight", type=float, default=0.10)
    parser.add_argument("--patch-temp-weight", type=float, default=0.90)
    parser.add_argument(
        "--aggs",
        nargs="+",
        choices=sorted(AGG_SPECS),
        default=["bottomk0p2", "bottomk0p5", "mean"],
    )
    args = parser.parse_args()

    scorers = {}
    rows_by_agg = {agg: [] for agg in args.aggs}
    for agg_name in args.aggs:
        params_path = path_from_pattern(args.patch_params_pattern, agg_name)
        scorer = FastPatchScorer(str(params_path), device=args.score_device)
        scorer.validate(args.patch_temp_mode)
        if scorer.params_patch_region_size != args.patch_region_size:
            raise ValueError(
                f"{params_path} expects patch_region_size={scorer.params_patch_region_size}, "
                f"got {args.patch_region_size}"
            )
        scorers[agg_name] = scorer

    jobs = list(iter_cache_jobs(args.csv, args.patch_emb_cache, args.duration, args.compact, args.debug_n))
    print(
        f"Fast multiagg patch scoring: {len(jobs)} cached videos, "
        f"aggs={','.join(args.aggs)}, batch={args.score_batch_size}, "
        f"mode={args.patch_temp_mode}, region={args.patch_region_size}",
        flush=True,
    )

    for start in tqdm(range(0, len(jobs), args.score_batch_size), desc="Fast multiagg scoring", unit="batch"):
        batch_jobs = jobs[start : start + args.score_batch_size]
        patch_batch, grid_size = load_cache_batch(batch_jobs)

        for agg_name, scorer in scorers.items():
            if grid_size != scorer.patch_grid_size:
                raise ValueError(f"Params grid_size={scorer.patch_grid_size}, cache grid_size={grid_size}")
            aggregation, bottomk_ratio = AGG_SPECS[agg_name]
            scores = scorer.score_batch(
                patch_batch,
                patch_temp_mode=args.patch_temp_mode,
                patch_spat_weight=args.patch_spat_weight,
                patch_temp_weight=args.patch_temp_weight,
                aggregation=aggregation,
                bottomk_ratio=bottomk_ratio,
                temporal_run_length=scorer.params_temporal_run_length,
                patch_region_size=args.patch_region_size,
            )

            for i, job in enumerate(batch_jobs):
                rows_by_agg[agg_name].append(
                    {
                        "subset": job["subset"],
                        "source_model": job["source_model"],
                        "filename": job["filename"],
                        "fusion": "patch_only",
                        "global_weight": None,
                        "patch_spat_weight": args.patch_spat_weight,
                        "patch_temp_weight": args.patch_temp_weight,
                        "patch_temp_mode": args.patch_temp_mode,
                        "match_radius": None,
                        "top_m": None,
                        "temperature": None,
                        "lambda_dist": None,
                        "aggregation": aggregation,
                        "bottomk_ratio": bottomk_ratio,
                        "temporal_run_length": scorer.params_temporal_run_length,
                        "patch_region_size": args.patch_region_size,
                        "patch_spat_percentile": float(scores["patch_spat_percentile"][i]),
                        "patch_temp_percentile": float(scores["patch_temp_percentile"][i]),
                        "patch_final_score": float(scores["patch_final_score"][i]),
                        "final_score": float(scores["final_score"][i]),
                    }
                )

    for agg_name, rows in rows_by_agg.items():
        df = pd.DataFrame(rows)
        out = path_from_pattern(args.output_csv_pattern, agg_name)
        out.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(out, index=False)
        print(f"Saved per-video scores -> {out}")
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
