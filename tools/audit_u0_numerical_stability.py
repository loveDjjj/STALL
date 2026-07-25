#!/usr/bin/env python3
"""Audit U0 whitening modes from existing K1 patch-token caches only."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.stats import spearmanr


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
TOOLS_DIR = REPO_ROOT / "tools"
for path in (SRC_DIR, TOOLS_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from build_multi_order_baselines import metric_tables
from stable_whitening import (
    StableGaussianParams,
    configure_strict_fp32,
    l2_normalized_second_order,
    score_mean_gaussian_float64,
    score_mean_gaussian_fp32,
)


KEY_COLUMNS = ["dataset", "protocol_split", "subset", "source_model", "filename"]
BATCH_SIZES = (1, 4, 8, 16)


def stable_order(frame: pd.DataFrame) -> pd.Series:
    def digest(row: pd.Series) -> str:
        text = "\0".join(str(row[column]) for column in KEY_COLUMNS)
        return hashlib.sha256(text.encode("utf-8")).hexdigest()

    return frame.apply(digest, axis=1)


def cache_path(root: Path, row: pd.Series) -> Path:
    stem = Path(str(row["filename"])).stem
    return (
        root
        / str(row["dataset"])
        / str(row["subset"])
        / str(row["source_model"])
        / f"{stem}_2s.pt"
    )


def build_pilot(manifest_path: Path, cache_root: Path, per_group: int) -> pd.DataFrame:
    manifest = pd.read_csv(manifest_path)
    evaluation = manifest[manifest["protocol_split"] == "evaluation"].copy()
    evaluation["filename"] = evaluation["filename"].astype(str)
    evaluation["cache_path"] = evaluation.apply(
        lambda row: str(cache_path(cache_root, row)), axis=1
    )
    evaluation = evaluation[evaluation["cache_path"].map(lambda value: Path(value).exists())]
    evaluation["_hash"] = stable_order(evaluation)
    rows = []
    for dataset, dataset_frame in evaluation.groupby("dataset", sort=True):
        real = dataset_frame[dataset_frame["subset"] == "real"].sort_values("_hash")
        if len(real) < per_group:
            raise ValueError(f"{dataset}: only {len(real)} cached evaluation real videos")
        rows.append(real.head(per_group))
        fake = dataset_frame[dataset_frame["subset"] != "real"]
        for generator, group in fake.groupby("source_model", sort=True):
            ordered = group.sort_values("_hash")
            if len(ordered) < per_group:
                raise ValueError(
                    f"{dataset}/{generator}: only {len(ordered)} cached fake videos"
                )
            rows.append(ordered.head(per_group))
    pilot = pd.concat(rows, ignore_index=True).drop(columns="_hash")
    if pilot.duplicated(KEY_COLUMNS).any():
        raise ValueError("numerical pilot contains duplicate video keys")
    return pilot


def attach_fixed_components(pilot: pd.DataFrame) -> pd.DataFrame:
    frames = []
    for dataset in ("comgenvid", "videofeedback", "genvideo"):
        global_scores = pd.read_csv(
            REPO_ROOT / f"results/paper_scores/{dataset}_global.csv",
            float_precision="round_trip",
        ).rename(columns={"final_score": "global_fixed"})
        patch_scores = pd.read_csv(
            REPO_ROOT / f"results/paper_scores/{dataset}_patch_second_order.csv",
            float_precision="round_trip",
        )
        key = ["subset", "source_model", "filename"]
        source = pilot[pilot["dataset"] == dataset].copy()
        source = source.merge(
            global_scores[[*key, "global_fixed"]],
            on=key,
            how="left",
            validate="one_to_one",
        )
        source = source.merge(
            patch_scores[[*key, "patch_spat_percentile"]].rename(
                columns={"patch_spat_percentile": "patch_spatial_fixed"}
            ),
            on=key,
            how="left",
            validate="one_to_one",
        )
        frames.append(source)
    result = pd.concat(frames, ignore_index=True)
    required = ["global_fixed", "patch_spatial_fixed"]
    if result[required].isna().any().any():
        missing = result[result[required].isna().any(axis=1)][KEY_COLUMNS]
        raise ValueError(f"missing fixed K1 components:\n{missing.to_string(index=False)}")
    return result


def load_patch_batch(paths: list[str]) -> torch.Tensor:
    patches = []
    shape = None
    for value in paths:
        payload = torch.load(value, weights_only=True, map_location="cpu")
        patch = payload["patch"].to(dtype=torch.float32)
        if shape is None:
            shape = tuple(patch.shape)
        elif tuple(patch.shape) != shape:
            raise ValueError(f"cache shape mismatch: {patch.shape} vs {shape}")
        patches.append(patch)
    return torch.stack(patches, dim=0)


def score_mode(
    frame: pd.DataFrame,
    params: StableGaussianParams,
    mode: str,
    batch_size: int,
    device: str,
) -> pd.DataFrame:
    raw_parts = []
    percentile_parts = []
    elapsed = 0.0
    for start in range(0, len(frame), batch_size):
        paths = frame.iloc[start : start + batch_size]["cache_path"].tolist()
        patch = load_patch_batch(paths)
        temporal = l2_normalized_second_order(patch)
        began = time.perf_counter()
        if mode in {"N0", "N1"}:
            raw, percentile = score_mean_gaussian_fp32(
                temporal, params, device=device
            )
        elif mode == "N2":
            raw, percentile = score_mean_gaussian_float64(
                temporal, params, device=device
            )
        else:
            raise ValueError(mode)
        elapsed += time.perf_counter() - began
        raw_parts.append(raw)
        percentile_parts.append(percentile)
    result = frame[KEY_COLUMNS].copy()
    result["mode"] = mode
    result["batch_size"] = batch_size
    result["raw"] = np.concatenate(raw_parts)
    result["percentile"] = np.concatenate(percentile_parts)
    result["local"] = (
        0.1 * frame["patch_spatial_fixed"].to_numpy()
        + 0.9 * result["percentile"].to_numpy()
    )
    result["final"] = (
        0.6 * frame["global_fixed"].to_numpy()
        + 0.4 * result["local"].to_numpy()
    )
    result["score_seconds"] = elapsed
    return result


def covariance_diagnostics(params_dir: Path) -> pd.DataFrame:
    rows = []
    for dataset in ("comgenvid", "videofeedback", "genvideo"):
        path = params_dir / f"{dataset}_region1_mean.npz"
        data = np.load(path, allow_pickle=True)
        whitening = data["W_patch_temp"].astype(np.float64)
        if "whitening_eigenvalues" in data.files:
            eigenvalues = data["whitening_eigenvalues"].astype(np.float64)
            source = "stored"
        else:
            inverse_variance = np.sum(whitening * whitening, axis=0)
            eigenvalues = np.maximum(1.0 / inverse_variance - 1e-5, 0.0)
            source = "recovered_from_W"
        positive = eigenvalues[eigenvalues > 0]
        rows.append(
            {
                "dataset": dataset,
                "dimension": whitening.shape[0],
                "effective_rank": whitening.shape[1],
                "dropped_dimensions": whitening.shape[0] - whitening.shape[1],
                "eigenvalue_min": float(positive.min()),
                "eigenvalue_max": float(positive.max()),
                "condition_number": float(positive.max() / positive.min()),
                "eigenvalue_source": source,
                "zero_eigenvalue_rule": "drop values omitted by fitted whitening rank",
            }
        )
    return pd.DataFrame(rows)


def metrics_for_scores(scores: pd.DataFrame) -> pd.DataFrame:
    frames = []
    for (mode, batch_size), frame in scores.groupby(["mode", "batch_size"]):
        metrics, _ = metric_tables(
            frame,
            42,
            score_columns=("final",),
            config_names={"final": f"{mode}_b{batch_size}"},
        )
        metrics["mode"] = mode
        metrics["batch_size"] = batch_size
        frames.append(metrics)
    return pd.concat(frames, ignore_index=True)


def invariance_summary(scores: pd.DataFrame, metrics: pd.DataFrame) -> pd.DataFrame:
    rows = []
    key = KEY_COLUMNS
    for mode, mode_frame in scores.groupby("mode", sort=True):
        reference = mode_frame[mode_frame["batch_size"] == 1]
        reference_metrics = metrics[
            (metrics["mode"] == mode) & (metrics["batch_size"] == 1)
        ].set_index("dataset")
        for batch_size in BATCH_SIZES:
            target = mode_frame[mode_frame["batch_size"] == batch_size]
            merged = reference.merge(
                target,
                on=key,
                suffixes=("_reference", "_target"),
                validate="one_to_one",
            )
            raw_delta = np.abs(merged["raw_target"] - merged["raw_reference"])
            relative = raw_delta / np.maximum(np.abs(merged["raw_reference"]), 1e-300)
            final_delta = np.abs(
                merged["final_target"] - merged["final_reference"]
            )
            rank_reference = merged["final_reference"].rank(method="average")
            rank_target = merged["final_target"].rank(method="average")
            target_metrics = metrics[
                (metrics["mode"] == mode) & (metrics["batch_size"] == batch_size)
            ].set_index("dataset")
            macro_reference = reference_metrics.loc["Macro-3"]
            macro_target = target_metrics.loc["Macro-3"]
            correlation = spearmanr(
                merged["final_reference"], merged["final_target"]
            ).statistic
            rows.append(
                {
                    "mode": mode,
                    "reference_batch_size": 1,
                    "batch_size": batch_size,
                    "videos": len(merged),
                    "raw_max_abs_error": float(raw_delta.max()),
                    "raw_max_relative_error": float(relative.max()),
                    "percentile_changed_windows": int(
                        (merged["percentile_target"] != merged["percentile_reference"]).sum()
                    ),
                    "final_changed_videos": int((final_delta > 0).sum()),
                    "final_max_abs_error": float(final_delta.max()),
                    "rank_changed_videos": int((rank_reference != rank_target).sum()),
                    "spearman": float(correlation),
                    "macro_auc_delta": float(macro_target.auc - macro_reference.auc),
                    "macro_ap_delta": float(macro_target.ap - macro_reference.ap),
                    "score_seconds": float(
                        target.drop_duplicates("dataset")["score_seconds"].sum()
                    ),
                }
            )
    return pd.DataFrame(rows)


def mode_comparison_summary(
    scores: pd.DataFrame, metrics: pd.DataFrame, reference_mode: str = "N2"
) -> pd.DataFrame:
    reference = scores[
        (scores["mode"] == reference_mode) & (scores["batch_size"] == 1)
    ]
    reference_metrics = metrics[
        (metrics["mode"] == reference_mode) & (metrics["batch_size"] == 1)
    ].set_index("dataset")
    rows = []
    for mode in sorted(scores["mode"].unique()):
        target = scores[(scores["mode"] == mode) & (scores["batch_size"] == 1)]
        merged = reference.merge(
            target,
            on=KEY_COLUMNS,
            suffixes=("_reference", "_target"),
            validate="one_to_one",
        )
        raw_delta = np.abs(merged["raw_target"] - merged["raw_reference"])
        final_delta = np.abs(merged["final_target"] - merged["final_reference"])
        rank_reference = merged["final_reference"].rank(method="average")
        rank_target = merged["final_target"].rank(method="average")
        target_metrics = metrics[
            (metrics["mode"] == mode) & (metrics["batch_size"] == 1)
        ].set_index("dataset")
        macro_reference = reference_metrics.loc["Macro-3"]
        macro_target = target_metrics.loc["Macro-3"]
        rows.append(
            {
                "mode": mode,
                "reference_mode": reference_mode,
                "batch_size": 1,
                "videos": len(merged),
                "raw_max_abs_error": float(raw_delta.max()),
                "raw_median_abs_error": float(np.median(raw_delta)),
                "percentile_changed_windows": int(
                    (merged["percentile_target"] != merged["percentile_reference"]).sum()
                ),
                "final_changed_videos": int((final_delta > 0).sum()),
                "final_max_abs_error": float(final_delta.max()),
                "rank_changed_videos": int((rank_reference != rank_target).sum()),
                "spearman": float(
                    spearmanr(
                        merged["final_reference"], merged["final_target"]
                    ).statistic
                ),
                "macro_auc_delta": float(macro_target.auc - macro_reference.auc),
                "macro_ap_delta": float(macro_target.ap - macro_reference.ap),
            }
        )
    return pd.DataFrame(rows)


def cpu_gpu_float64_check(
    pilot: pd.DataFrame, params_dir: Path, device: str
) -> pd.DataFrame:
    rows = []
    for dataset, frame in pilot.groupby("dataset", sort=True):
        params = StableGaussianParams.from_npz(
            str(params_dir / f"{dataset}_region1_mean.npz")
        )
        patch = load_patch_batch(frame.head(2)["cache_path"].tolist())
        temporal = l2_normalized_second_order(patch)
        raw_cpu, pct_cpu = score_mean_gaussian_float64(
            temporal, params, device="cpu"
        )
        raw_gpu, pct_gpu = score_mean_gaussian_float64(
            temporal, params, device=device
        )
        rows.append(
            {
                "dataset": dataset,
                "windows": len(raw_cpu),
                "raw_max_abs_error": float(np.max(np.abs(raw_cpu - raw_gpu))),
                "percentile_changed_windows": int(np.sum(pct_cpu != pct_gpu)),
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest",
        type=Path,
        default=REPO_ROOT
        / "results/multi_window_joint_typicality/multi_window_manifest.csv",
    )
    parser.add_argument(
        "--cache-root",
        type=Path,
        default=REPO_ROOT / "cache/patch_embeddings",
    )
    parser.add_argument(
        "--params-dir",
        type=Path,
        default=REPO_ROOT / "results/clean_universal/params",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPO_ROOT / "results/u0_numerical_stability",
    )
    parser.add_argument("--per-group", type=int, default=4)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--modes", nargs="+", choices=("N0", "N1", "N2"), default=("N0", "N1", "N2")
    )
    args = parser.parse_args()

    pilot = attach_fixed_components(
        build_pilot(args.manifest, args.cache_root, args.per_group)
    )
    outputs = []
    for dataset, dataset_frame in pilot.groupby("dataset", sort=True):
        params = StableGaussianParams.from_npz(
            str(args.params_dir / f"{dataset}_region1_mean.npz")
        )
        for mode in args.modes:
            if mode == "N1":
                configure_strict_fp32()
            for batch_size in BATCH_SIZES:
                print(
                    f"dataset={dataset} mode={mode} batch={batch_size} "
                    f"videos={len(dataset_frame)}",
                    flush=True,
                )
                outputs.append(
                    score_mode(dataset_frame, params, mode, batch_size, args.device)
                )
    scores = pd.concat(outputs, ignore_index=True)
    metrics = metrics_for_scores(scores)
    summary = invariance_summary(scores, metrics)
    mode_comparison = mode_comparison_summary(scores, metrics)
    covariance = covariance_diagnostics(args.params_dir)
    cpu_gpu = cpu_gpu_float64_check(pilot, args.params_dir, args.device)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    pilot.to_csv(args.output_dir / "pilot_manifest.csv", index=False)
    scores.to_csv(args.output_dir / "per_video_scores.csv", index=False)
    metrics.to_csv(args.output_dir / "mode_metrics.csv", index=False)
    summary.to_csv(args.output_dir / "batch_invariance_summary.csv", index=False)
    mode_comparison.to_csv(
        args.output_dir / "mode_comparison_summary.csv", index=False
    )
    covariance.to_csv(args.output_dir / "covariance_diagnostics.csv", index=False)
    cpu_gpu.to_csv(args.output_dir / "cpu_gpu_float64.csv", index=False)
    metadata = {
        "batch_sizes": list(BATCH_SIZES),
        "per_group": args.per_group,
        "modes": list(args.modes),
        "cache_scope": "existing K1 two-second final-layer patch-token cache",
        "dino_reextraction": False,
        "cdf_tie_policy": "right_inclusive",
        "n1": {
            "cuda_matmul_allow_tf32": False,
            "cudnn_allow_tf32": False,
            "float32_matmul_precision": "highest",
        },
    }
    (args.output_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print("\nBatch invariance")
    print(summary.to_string(index=False))
    print("\nCPU/GPU float64")
    print(cpu_gpu.to_string(index=False))
    print("\nMode comparison against N2")
    print(mode_comparison.to_string(index=False))


if __name__ == "__main__":
    main()
