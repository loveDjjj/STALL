"""Evaluate several patch ablation configs in one cache pass.

This avoids repeatedly loading the same patch-cache files for each aggregation
or region. It reuses FastPatchScorer for score correctness and only changes
the outer scheduling.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from eval_patch_fast import FastPatchScorer, iter_cache_jobs
from metrics import Score, ScoreDirection, get_results_df, print_results


AGG_PRESETS = {
    "bottomk0p2": ("bottomk_mean", 0.20),
    "bottomk0p5": ("bottomk_mean", 0.50),
    "mean": ("mean", 0.50),
    "strun3x3_bottomk0p2": ("spatiotemporal_run_bottomk_mean", 0.20),
    "strun3x3_bottomk0p5": ("spatiotemporal_run_bottomk_mean", 0.50),
}


def _write_result(df: pd.DataFrame, out: Path) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    print(f"Saved per-video scores -> {out}", flush=True)
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


def _job_id(region: int, agg: str) -> str:
    return f"region{region}_{agg}"


def _load_one_patch(job: dict) -> tuple[dict, torch.Tensor, tuple[int, int]]:
    payload = torch.load(job["cache_path"], weights_only=True, map_location="cpu")
    patch = payload["patch"]
    if patch.dtype != torch.float32:
        patch = patch.float()
    return job, patch, tuple(int(x) for x in payload["grid_size"])


def load_cache_batch_parallel(
    jobs: list[dict],
    workers: int,
) -> tuple[list[dict], torch.Tensor, tuple[int, int]]:
    if workers <= 1:
        loaded = [_load_one_patch(job) for job in jobs]
    else:
        with ThreadPoolExecutor(max_workers=workers) as ex:
            loaded = list(ex.map(_load_one_patch, jobs))

    ordered_jobs = []
    patches = []
    grid_size = None
    shape = None
    for job, patch, this_grid in loaded:
        if grid_size is None:
            grid_size = this_grid
            shape = patch.shape
        elif grid_size != this_grid or patch.shape != shape:
            raise ValueError(
                f"Fast batching requires fixed grid/shape. Got {patch.shape}/{this_grid}, "
                f"expected {shape}/{grid_size} for {job['cache_path']}"
            )
        ordered_jobs.append(job)
        patches.append(patch)
    return ordered_jobs, torch.stack(patches, dim=0), grid_size


def _score_region_on_device(
    scorer: FastPatchScorer,
    region_configs: list[dict],
    patch_batch: torch.Tensor,
    device: torch.device,
    patch_temp_mode: str,
    patch_spat_weight: float,
    patch_temp_weight: float,
    region: int,
) -> dict[str, dict[str, np.ndarray]]:
    patch = patch_batch
    if patch.dtype != torch.float32:
        patch = patch.float()
    patch = patch.to(device, non_blocking=True)
    mu_spat, W_spat, mu_temp, W_temp = scorer._params_for_device(device)

    spat_white = torch.matmul(patch - mu_spat, W_spat)
    spat_ll = scorer.log_likelihood_from_white(spat_white)
    temp = scorer.temporal_features(patch, patch_temp_mode, region)
    temp_white = torch.matmul(temp - mu_temp, W_temp)
    temp_ll = scorer.log_likelihood_from_white(temp_white)

    out = {}
    for cfg in region_configs:
        cfg_scorer = cfg["scorer"]
        spat_agg = scorer._aggregate(
            spat_ll,
            cfg["aggregation"],
            cfg["bottomk_ratio"],
            cfg["temporal_run_length"],
        )
        temp_agg = scorer._aggregate(
            temp_ll,
            cfg["aggregation"],
            cfg["bottomk_ratio"],
            cfg["temporal_run_length"],
        )
        spat_agg_np = spat_agg.detach().cpu().numpy()
        temp_agg_np = temp_agg.detach().cpu().numpy()
        spat_pct = cfg_scorer.percentile(spat_agg_np, cfg_scorer.calib_spat).astype(np.float32)
        temp_pct = cfg_scorer.percentile(temp_agg_np, cfg_scorer.calib_temp).astype(np.float32)
        denom = max(patch_spat_weight + patch_temp_weight, 1e-8)
        patch_final = (patch_spat_weight * spat_pct + patch_temp_weight * temp_pct) / denom
        out[cfg["id"]] = {
            "patch_spat_percentile": spat_pct,
            "patch_temp_percentile": temp_pct,
            "patch_final_score": patch_final.astype(np.float32),
            "final_score": patch_final.astype(np.float32),
        }

    del patch, spat_white, spat_ll, temp, temp_white, temp_ll
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return out


def run(args: argparse.Namespace) -> dict[str, pd.DataFrame]:
    jobs = list(iter_cache_jobs(args.csv, args.patch_emb_cache, args.duration, args.compact, args.debug_n))

    configs = []
    for region in args.regions:
        for agg_name in args.agg:
            params_path = Path(args.params_template.format(region=region, agg=agg_name))
            out_path = Path(args.output_template.format(region=region, agg=agg_name))
            if args.skip_existing and out_path.exists() and out_path.stat().st_size > 0:
                print(f"Skip complete result: {out_path}", flush=True)
                continue
            if not params_path.exists():
                raise FileNotFoundError(f"Missing params: {params_path}")

            aggregation, bottomk_ratio = AGG_PRESETS[agg_name]
            scorer = FastPatchScorer(str(params_path), device=args.score_device)
            scorer.validate(args.patch_temp_mode)
            if scorer.params_patch_region_size != region:
                raise ValueError(
                    f"Params {params_path} expect region={scorer.params_patch_region_size}, got {region}"
                )
            configs.append(
                {
                    "id": _job_id(region, agg_name),
                    "region": region,
                    "agg_name": agg_name,
                    "aggregation": aggregation,
                    "bottomk_ratio": bottomk_ratio,
                    "temporal_run_length": scorer.params_temporal_run_length,
                    "scorer": scorer,
                    "out": out_path,
                    "rows": [],
                }
            )

    if not configs:
        print("All requested results already exist; nothing to score.", flush=True)
        return {}

    print(
        f"Fast multi patch scoring: videos={len(jobs)}, configs={len(configs)}, "
        f"batch={args.score_batch_size}, device={args.score_device}, "
        f"cache_workers={args.cache_workers}, mode={args.patch_temp_mode}",
        flush=True,
    )

    configs_by_region: dict[int, list[dict]] = defaultdict(list)
    for cfg in configs:
        configs_by_region[cfg["region"]].append(cfg)

    for start in tqdm(range(0, len(jobs), args.score_batch_size), desc="Fast multi scoring", unit="batch"):
        batch_jobs = jobs[start : start + args.score_batch_size]
        batch_jobs, patch_batch, grid_size = load_cache_batch_parallel(batch_jobs, args.cache_workers)
        if args.pin_memory and torch.cuda.is_available():
            patch_batch = patch_batch.pin_memory()

        # Configs for the same region share the expensive likelihood tensors;
        # only aggregation and calibration percentiles differ.
        for region, region_configs in configs_by_region.items():
            scorer = region_configs[0]["scorer"]
            if grid_size != scorer.patch_grid_size:
                raise ValueError(f"Params grid_size={scorer.patch_grid_size}, cache grid_size={grid_size}")

            if len(scorer.devices) > 1 and patch_batch.shape[0] >= len(scorer.devices):
                chunks = torch.tensor_split(patch_batch, len(scorer.devices), dim=0)
                chunk_sizes = [chunk.shape[0] for chunk in chunks]
                with ThreadPoolExecutor(max_workers=len(scorer.devices)) as ex:
                    parts = list(
                        ex.map(
                            lambda item: _score_region_on_device(
                                scorer=scorer,
                                region_configs=region_configs,
                                patch_batch=item[0],
                                device=item[1],
                                patch_temp_mode=args.patch_temp_mode,
                                patch_spat_weight=args.patch_spat_weight,
                                patch_temp_weight=args.patch_temp_weight,
                                region=region,
                            ),
                            [(chunk, device) for chunk, device in zip(chunks, scorer.devices) if chunk.shape[0] > 0],
                        )
                    )
                scores_by_cfg = {}
                for cfg in region_configs:
                    key = cfg["id"]
                    scores_by_cfg[key] = {
                        score_key: np.concatenate([part[key][score_key] for part in parts], axis=0)
                        for score_key in parts[0][key].keys()
                    }
                if sum(chunk_sizes) != patch_batch.shape[0]:
                    raise RuntimeError("Internal split size mismatch")
            else:
                scores_by_cfg = _score_region_on_device(
                    scorer=scorer,
                    region_configs=region_configs,
                    patch_batch=patch_batch,
                    device=scorer.device,
                    patch_temp_mode=args.patch_temp_mode,
                    patch_spat_weight=args.patch_spat_weight,
                    patch_temp_weight=args.patch_temp_weight,
                    region=region,
                )

            for cfg in region_configs:
                scores = scores_by_cfg[cfg["id"]]

                for i, job in enumerate(batch_jobs):
                    cfg["rows"].append(
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
                            "aggregation": cfg["aggregation"],
                            "bottomk_ratio": cfg["bottomk_ratio"],
                            "temporal_run_length": cfg["temporal_run_length"],
                            "patch_region_size": cfg["region"],
                            "patch_spat_percentile": float(scores["patch_spat_percentile"][i]),
                            "patch_temp_percentile": float(scores["patch_temp_percentile"][i]),
                            "patch_final_score": float(scores["patch_final_score"][i]),
                            "final_score": float(scores["final_score"][i]),
                        }
                    )

    out = {}
    for cfg in configs:
        df = pd.DataFrame(cfg["rows"])
        _write_result(df, cfg["out"])
        out[cfg["id"]] = df
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Fast multi-config GPU patch-only eval from patch cache.")
    parser.add_argument("--csv", required=True)
    parser.add_argument("--patch-emb-cache", required=True)
    parser.add_argument("--params-template", required=True, help="Template containing {region} and {agg}")
    parser.add_argument("--output-template", required=True, help="Template containing {region} and {agg}")
    parser.add_argument("--duration", type=int, default=2, choices=[1, 2, 3, 4])
    parser.add_argument("--compact", action="store_true", default=False)
    parser.add_argument("--debug-n", type=int, default=None)
    parser.add_argument("--score-device", choices=["cuda", "cpu"], default="cuda")
    parser.add_argument("--score-batch-size", type=int, default=64)
    parser.add_argument("--cache-workers", type=int, default=1)
    parser.add_argument("--pin-memory", action="store_true", default=False)
    parser.add_argument("--patch-spat-weight", type=float, default=0.10)
    parser.add_argument("--patch-temp-weight", type=float, default=0.90)
    parser.add_argument("--patch-temp-mode", required=True)
    parser.add_argument("--regions", nargs="+", type=int, default=[1, 2, 3])
    parser.add_argument("--agg", nargs="+", choices=sorted(AGG_PRESETS), default=["bottomk0p2", "bottomk0p5", "mean"])
    parser.add_argument("--skip-existing", action="store_true", default=False)
    args = parser.parse_args()

    if "{region}" not in args.params_template or "{agg}" not in args.params_template:
        raise ValueError("--params-template must contain {region} and {agg}")
    if "{region}" not in args.output_template or "{agg}" not in args.output_template:
        raise ValueError("--output-template must contain {region} and {agg}")
    run(args)


if __name__ == "__main__":
    main()
