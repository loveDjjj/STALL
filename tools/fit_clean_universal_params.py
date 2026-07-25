#!/usr/bin/env python3
"""Build real-only final-layer temporal parameters for U0-U5.

The region-specific whitening transforms were already fitted from the frozen
200-real compact cache. This tool keeps those transforms fixed and recomputes
the temporal calibration distribution for both predeclared aggregations.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from create_patch_params import iter_real_patch_cache
from eval_patch_fast import FastPatchScorer


DATASETS = {
    "comgenvid": {
        "index": REPO_ROOT / "cache/indexes/comgenvid_calib_real200.csv",
        "cache": REPO_ROOT / "cache/patch_embeddings/comgenvid",
    },
    "videofeedback": {
        "index": REPO_ROOT / "cache/indexes/videofeedback_small_calib_real200.csv",
        "cache": REPO_ROOT / "cache/patch_embeddings/videofeedback",
    },
    "genvideo": {
        "index": REPO_ROOT / "cache/indexes/genvideo_calib_real200.csv",
        "cache": REPO_ROOT / "cache/patch_embeddings/genvideo",
    },
}
AGGREGATIONS = {
    "mean": ("mean", 0.2),
    "bottom20": ("bottomk_mean", 0.2),
}


def base_path(dataset: str, region: int) -> Path:
    return (
        REPO_ROOT
        / "results/unified_multiscale_layers/region_params"
        / f"{dataset}_region{region}.npz"
    )


def output_path(output_dir: Path, dataset: str, region: int, name: str) -> Path:
    return output_dir / f"{dataset}_region{region}_{name}.npz"


def score_calibration(dataset: str, device: str) -> dict[tuple[int, str], np.ndarray]:
    spec = DATASETS[dataset]
    scorers = {
        region: FastPatchScorer(str(base_path(dataset, region)), device=device)
        for region in (1, 2, 3)
    }
    scores: dict[tuple[int, str], list[float]] = {
        (region, name): []
        for region in (1, 2, 3)
        for name in AGGREGATIONS
    }
    count = 0
    with torch.inference_mode():
        for _, payload in iter_real_patch_cache(
            str(spec["index"]), str(spec["cache"]), 2, True, None
        ):
            patch = payload["patch"].unsqueeze(0).to(
                scorers[1].device, dtype=torch.float32
            )
            for region, scorer in scorers.items():
                _, _, mu_temp, W_temp = scorer._params_for_device(scorer.device)
                temporal = scorer.temporal_features(
                    patch, "same_grid_second_order", region
                )
                likelihood = scorer.log_likelihood_from_white(
                    torch.matmul(temporal - mu_temp, W_temp)
                )
                for name, (mode, ratio) in AGGREGATIONS.items():
                    raw = scorer._aggregate(
                        likelihood,
                        mode,
                        ratio,
                        scorer.params_temporal_run_length,
                    )
                    scores[(region, name)].append(float(raw.item()))
            count += 1
            if count % 50 == 0:
                print(f"[{dataset}] calibration={count}/200", flush=True)
    if count != 200:
        raise ValueError(f"{dataset}: expected 200 real calibration videos, got {count}")
    for scorer in scorers.values():
        if scorer._executor is not None:
            scorer._executor.shutdown(wait=True)
    return {key: np.asarray(values, dtype=np.float32) for key, values in scores.items()}


def write_params(
    dataset: str,
    output_dir: Path,
    calibration: dict[tuple[int, str], np.ndarray],
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for region in (1, 2, 3):
        source = np.load(base_path(dataset, region), allow_pickle=True)
        source_config = json.loads(str(source["aggregation_config"].item()))
        for name, (mode, ratio) in AGGREGATIONS.items():
            config = dict(source_config)
            config.update(
                {
                    "mode": mode,
                    "bottomk_ratio": ratio,
                    "patch_region_size": region,
                    "calibration_videos": 200,
                    "temporal_only_params": True,
                    "implementation": "clean_universal_real200_v1",
                }
            )
            payload = {key: source[key] for key in source.files}
            same_existing_aggregation = source_config.get("mode") == mode and (
                mode != "bottomk_mean"
                or float(source_config.get("bottomk_ratio", ratio)) == ratio
            )
            temporal_reference = (
                source["calib_patch_temp_scores"]
                if same_existing_aggregation
                else calibration[(region, name)]
            )
            payload.update(
                {
                    "calib_patch_temp_scores": temporal_reference,
                    "aggregation_config": np.array(json.dumps(config, sort_keys=True)),
                }
            )
            destination = output_path(output_dir, dataset, region, name)
            np.savez(destination, **payload)
            print(f"wrote {destination}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=tuple(DATASETS), required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPO_ROOT / "results/clean_universal/params",
    )
    args = parser.parse_args()
    write_params(
        args.dataset,
        args.output_dir,
        score_calibration(args.dataset, args.device),
    )


if __name__ == "__main__":
    main()
