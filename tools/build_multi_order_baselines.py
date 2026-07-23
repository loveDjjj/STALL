#!/usr/bin/env python3
"""Build protocol-aligned B0-B8 baselines from frozen DINOv3 caches.

The evaluator uses a strict 2 s / 8 FPS window and joins every branch by
``(subset, source_model, filename)`` before computing any metric. Global STALL
components and D3 statistics are recomputed from CLS caches. Frozen local D2
score components are read from the release CSVs because the VideoFeedback
patch-token cache is not present in this workspace.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import average_precision_score, roc_auc_score


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from metrics import _sample_balanced_real


KEY_COLUMNS = ["subset", "source_model", "filename"]
SCORE_COLUMNS = [f"B{i}" for i in range(9)]
CONFIG_NAMES = {
    "B0": "Global spatial",
    "B1": "Global first-order temporal",
    "B2": "Original STALL",
    "B3": "Raw D3 percentile",
    "B4": "STALL + raw D3",
    "B5": "Patch spatial",
    "B6": "Patch same-grid D2",
    "B7": "Current patch",
    "B8": "Current global + patch",
}
GLOBAL_D3_CONFIG_NAMES = {
    "G0": "Original STALL",
    "G1": "STALL + raw D3",
    "G2": "STALL + two-sided D3",
    "G3": "STALL + motion-conditioned D3",
    "G4": "STALL + time-normalized conditional D3",
}
BOOTSTRAP_COMPARISONS = (
    ("B2", "B0", "add_global_t1"),
    ("B2", "B1", "add_global_spatial"),
    ("B4", "B2", "add_raw_d3"),
    ("B7", "B5", "add_local_d2"),
    ("B7", "B6", "add_local_spatial"),
    ("B8", "B2", "add_patch"),
    ("B8", "B7", "add_global"),
)


@dataclass(frozen=True)
class DatasetSpec:
    name: str
    display_name: str
    eval_index: Path
    d3_calib_index: Path
    embedding_cache: Path
    patch_scores: Path


def dataset_specs(root: Path) -> tuple[DatasetSpec, ...]:
    return (
        DatasetSpec(
            "comgenvid",
            "ComGenVid",
            root / "cache/indexes/comgenvid_eval_holdout_real900_all_fake.csv",
            root / "cache/indexes/comgenvid_calib_real200.csv",
            root / "cache/embeddings/comgenvid",
            root / "results/paper_scores/comgenvid_patch_second_order.csv",
        ),
        DatasetSpec(
            "videofeedback",
            "VideoFeedback",
            root / "cache/indexes/videofeedback_small_eval_holdout_real500_fake300permodel.csv",
            root / "cache/indexes/videofeedback_small_calib_real200.csv",
            root / "cache/embeddings/videofeedback",
            root / "results/paper_scores/videofeedback_patch_second_order.csv",
        ),
        DatasetSpec(
            "genvideo",
            "GenVideo",
            root / "cache/indexes/genvideo_eval_holdout_real7984_all_fake.csv",
            root / "cache/indexes/genvideo_calib_real200.csv",
            root / "cache/embeddings/genvideo",
            root / "results/paper_scores/genvideo_patch_second_order.csv",
        ),
    )


def _parse_json_indices(value: object, column: str) -> list[int]:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        raise ValueError(f"missing {column}")
    parsed = json.loads(str(value))
    if not isinstance(parsed, list):
        raise ValueError(f"{column} must contain a JSON list")
    return [int(x) for x in parsed]


def window_positions(downsample_indices: list[int], window_indices: list[int]) -> list[int]:
    """Map native-frame window indices to positions in a full 8 FPS cache."""
    if len(set(downsample_indices)) != len(downsample_indices):
        raise ValueError("downsample indices contain duplicates")
    position = {native: i for i, native in enumerate(downsample_indices)}
    missing = [native for native in window_indices if native not in position]
    if missing:
        raise ValueError(f"window indices absent from downsample indices: {missing[:5]}")
    return [position[native] for native in window_indices]


def _load_index(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    required = {"video_path", "subset", "source_model", "fps", "downsample_idxs", "2_sec_idxs"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"{path} missing columns: {sorted(missing)}")
    df = df.copy()
    df["filename"] = df["video_path"].map(lambda value: Path(str(value)).name)
    for column in KEY_COLUMNS:
        df[column] = df[column].astype(str)
    if df.duplicated(KEY_COLUMNS).any():
        raise ValueError(f"{path} contains duplicate protocol keys")
    return df


def _load_patch_scores(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    required = set(KEY_COLUMNS + ["patch_spat_percentile", "patch_temp_percentile"])
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"{path} missing columns: {sorted(missing)}")
    for column in KEY_COLUMNS:
        df[column] = df[column].astype(str)
    if df.duplicated(KEY_COLUMNS).any():
        raise ValueError(f"{path} contains duplicate protocol keys")
    return df[KEY_COLUMNS + ["patch_spat_percentile", "patch_temp_percentile"]]


def _embedding_path(cache: Path, row: pd.Series, *, compact: bool) -> Path:
    stem = Path(str(row["video_path"])).stem
    suffix = "_2s.pt" if compact else ".pt"
    return cache / str(row["subset"]) / str(row["source_model"]) / f"{stem}{suffix}"


def load_strict_window(cache: Path, row: pd.Series) -> tuple[np.ndarray, np.ndarray]:
    """Load exactly the indexed 2 s window and return embeddings plus timestamps."""
    window = _parse_json_indices(row["2_sec_idxs"], "2_sec_idxs")
    if len(window) != 16:
        raise ValueError(f"expected 16 frames, found {len(window)}")

    compact_path = _embedding_path(cache, row, compact=True)
    full_path = _embedding_path(cache, row, compact=False)
    if compact_path.exists():
        payload = torch.load(compact_path, weights_only=True, map_location="cpu")
        positions = list(range(len(window)))
    elif full_path.exists():
        payload = torch.load(full_path, weights_only=True, map_location="cpu")
        downsample = _parse_json_indices(row["downsample_idxs"], "downsample_idxs")
        positions = window_positions(downsample, window)
    else:
        raise FileNotFoundError(f"missing compact and full cache for {row['video_path']}")

    if isinstance(payload, dict):
        for key in ("embs", "emb", "global"):
            if key in payload:
                payload = payload[key]
                break
        else:
            raise ValueError(f"embedding dict has no recognized global key: {row['video_path']}")
    if isinstance(payload, torch.Tensor):
        payload = payload.numpy()
    emb = np.asarray(payload, dtype=np.float32)
    if emb.ndim == 3 and emb.shape[0] == 1:
        emb = emb[0]
    if emb.ndim != 2:
        raise ValueError(f"embedding must have shape [T,D], found {emb.shape}")
    if max(positions) >= len(emb):
        raise ValueError(f"cache is too short for indexed window: {row['video_path']}")
    sliced = emb[positions]
    timestamps = np.asarray(window, dtype=np.float64) / float(row["fps"])
    return sliced, timestamps


def d3_statistics(emb: np.ndarray, timestamps: np.ndarray) -> dict[str, float]:
    """Compute raw and time-normalized D3 statistics on one strict window."""
    distance = np.linalg.norm(emb[1:] - emb[:-1], axis=1)
    acceleration = np.diff(distance)
    dt = np.diff(timestamps)
    if np.any(dt <= 0):
        raise ValueError("timestamps must be strictly increasing")
    velocity = distance / dt
    acceleration_dt = np.diff(velocity) / ((dt[1:] + dt[:-1]) / 2.0)
    return {
        "d3_raw": float(np.std(acceleration, ddof=0)),
        "motion_raw": float(np.mean(distance)),
        "d3_time_raw": float(np.std(acceleration_dt, ddof=0)),
        "motion_time_raw": float(np.mean(velocity)),
        "actual_dt_mean": float(np.mean(dt)),
        "actual_dt_std": float(np.std(dt, ddof=0)),
    }


class GlobalBatchScorer:
    def __init__(self, params_path: Path, device: str) -> None:
        params = np.load(params_path)
        self.device = torch.device(device)
        self.mu_spat = torch.from_numpy(params["mu_spat"]).to(self.device)
        self.w_spat = torch.from_numpy(params["W_spat"]).to(self.device)
        self.mu_temp = torch.from_numpy(params["mu_temp"]).to(self.device)
        self.w_temp = torch.from_numpy(params["W_temp"]).to(self.device)
        self.calib_spat = np.sort(np.max(params["calib_ll_spat"], axis=1))
        self.calib_temp = np.sort(np.min(params["calib_ll_temp"], axis=1))

    @staticmethod
    def _percentile(values: np.ndarray, calibration: np.ndarray) -> np.ndarray:
        return np.searchsorted(calibration, values, side="right") / float(len(calibration))

    def score(self, windows: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        x = torch.from_numpy(windows).to(self.device)
        with torch.inference_mode():
            spatial = (x - self.mu_spat) @ self.w_spat
            spatial_ll = -0.5 * (
                spatial.shape[-1] * math.log(2.0 * math.pi) + torch.sum(spatial.square(), dim=-1)
            )
            spatial_agg = torch.max(spatial_ll, dim=1).values

            diff = x[:, 1:] - x[:, :-1]
            zero = torch.linalg.vector_norm(diff, dim=-1) == 0
            norm = torch.linalg.vector_norm(diff, dim=-1, keepdim=True).clamp_min(1e-12)
            temporal = (diff / norm - self.mu_temp) @ self.w_temp
            temporal_ll = -0.5 * (
                temporal.shape[-1] * math.log(2.0 * math.pi) + torch.sum(temporal.square(), dim=-1)
            )
            temporal_ll[zero] = torch.inf
            temporal_agg = torch.min(temporal_ll, dim=1).values

        spat = self._percentile(spatial_agg.cpu().numpy(), self.calib_spat)
        temp = self._percentile(temporal_agg.cpu().numpy(), self.calib_temp)
        return spat.astype(np.float64), temp.astype(np.float64)


def _flush_batch(
    scorer: GlobalBatchScorer,
    meta: list[dict],
    windows: list[np.ndarray],
    timestamps: list[np.ndarray],
    rows: list[dict],
) -> None:
    if not windows:
        return
    spat, temp = scorer.score(np.stack(windows))
    for item, emb, time, spat_score, temp_score in zip(meta, windows, timestamps, spat, temp):
        rows.append(
            {
                **item,
                "global_spatial": float(spat_score),
                "global_temporal_t1": float(temp_score),
                "global_stall": float(0.5 * (spat_score + temp_score)),
                **d3_statistics(emb, time),
            }
        )
    meta.clear()
    windows.clear()
    timestamps.clear()


def score_index(
    index: pd.DataFrame,
    cache: Path,
    scorer: GlobalBatchScorer,
    batch_size: int,
) -> tuple[pd.DataFrame, list[str]]:
    rows: list[dict] = []
    skipped: list[str] = []
    batch_meta: list[dict] = []
    batch_windows: list[np.ndarray] = []
    batch_times: list[np.ndarray] = []
    for _, series in index.iterrows():
        try:
            emb, timestamps = load_strict_window(cache, series)
        except (FileNotFoundError, ValueError) as exc:
            skipped.append(f"{series['subset']}/{series['source_model']}/{series['filename']}: {exc}")
            continue
        batch_meta.append({column: str(series[column]) for column in KEY_COLUMNS})
        batch_windows.append(emb)
        batch_times.append(timestamps)
        if len(batch_windows) >= batch_size:
            _flush_batch(scorer, batch_meta, batch_windows, batch_times, rows)
    _flush_batch(scorer, batch_meta, batch_windows, batch_times, rows)
    return pd.DataFrame(rows), skipped


def empirical_cdf(values: np.ndarray, calibration: np.ndarray) -> np.ndarray:
    calibration = np.sort(np.asarray(calibration, dtype=np.float64))
    if len(calibration) == 0 or not np.isfinite(calibration).all():
        raise ValueError("calibration must be non-empty and finite")
    return np.searchsorted(calibration, values, side="right") / float(len(calibration))


def two_sided_realness(values: np.ndarray, calibration: np.ndarray) -> np.ndarray:
    cdf = empirical_cdf(np.asarray(values, dtype=np.float64), calibration)
    return 2.0 * np.minimum(cdf, 1.0 - cdf)


def conditional_two_sided_realness(
    values: np.ndarray,
    conditions: np.ndarray,
    calib_values: np.ndarray,
    calib_conditions: np.ndarray,
    num_bins: int,
) -> tuple[np.ndarray, np.ndarray]:
    if num_bins < 2:
        raise ValueError("num_bins must be at least 2")
    quantiles = np.quantile(calib_conditions, np.linspace(0.0, 1.0, num_bins + 1))
    interior = np.unique(quantiles[1:-1])
    calib_bin = np.searchsorted(interior, calib_conditions, side="right")
    value_bin = np.searchsorted(interior, conditions, side="right")
    scores = np.empty(len(values), dtype=np.float64)
    for bin_id in range(len(interior) + 1):
        target = value_bin == bin_id
        reference = calib_values[calib_bin == bin_id]
        if not np.any(target):
            continue
        if len(reference) < 2:
            reference = calib_values
        scores[target] = two_sided_realness(values[target], reference)
    return scores, value_bin


def add_d3_calibration(eval_df: pd.DataFrame, calib_df: pd.DataFrame, num_bins: int) -> pd.DataFrame:
    out = eval_df.copy()
    raw_calib = calib_df["d3_raw"].to_numpy(dtype=np.float64)
    out["d3_one_sided"] = empirical_cdf(out["d3_raw"].to_numpy(), raw_calib)
    out["d3_two_sided"] = two_sided_realness(out["d3_raw"].to_numpy(), raw_calib)
    out["d3_motion_conditional"], out["motion_bin"] = conditional_two_sided_realness(
        out["d3_raw"].to_numpy(),
        out["motion_raw"].to_numpy(),
        raw_calib,
        calib_df["motion_raw"].to_numpy(),
        num_bins,
    )
    out["d3_time_conditional"], out["motion_time_bin"] = conditional_two_sided_realness(
        out["d3_time_raw"].to_numpy(),
        out["motion_time_raw"].to_numpy(),
        calib_df["d3_time_raw"].to_numpy(),
        calib_df["motion_time_raw"].to_numpy(),
        num_bins,
    )
    return out


def add_baselines(global_d3: pd.DataFrame, patch: pd.DataFrame) -> pd.DataFrame:
    merged = global_d3.merge(patch, on=KEY_COLUMNS, how="inner", validate="one_to_one")
    if merged.empty:
        raise ValueError("global/D3 and patch score tables have no common rows")
    merged["B0"] = merged["global_spatial"]
    merged["B1"] = merged["global_temporal_t1"]
    merged["B2"] = merged["global_stall"]
    merged["B3"] = merged["d3_one_sided"]
    merged["B4"] = 0.5 * merged["B2"] + 0.5 * merged["B3"]
    merged["B5"] = merged["patch_spat_percentile"]
    merged["B6"] = merged["patch_temp_percentile"]
    merged["B7"] = 0.1 * merged["B5"] + 0.9 * merged["B6"]
    merged["B8"] = 0.6 * merged["B2"] + 0.4 * merged["B7"]
    merged["G0"] = merged["B2"]
    merged["G1"] = 0.5 * merged["B0"] + 0.25 * merged["B1"] + 0.25 * merged["B3"]
    merged["G2_two_sided"] = 0.5 * merged["B0"] + 0.25 * merged["B1"] + 0.25 * merged["d3_two_sided"]
    merged["G3_motion_conditional"] = (
        0.5 * merged["B0"] + 0.25 * merged["B1"] + 0.25 * merged["d3_motion_conditional"]
    )
    merged["G4_time_conditional"] = (
        0.5 * merged["B0"] + 0.25 * merged["B1"] + 0.25 * merged["d3_time_conditional"]
    )
    return merged


def pairwise_frames(df: pd.DataFrame, seed: int) -> dict[str, pd.DataFrame]:
    real = df[df["subset"] == "real"]
    frames: dict[str, pd.DataFrame] = {}
    for generator, fake in df[df["subset"] == "annotated"].groupby("source_model", sort=True):
        sampled_real = _sample_balanced_real(real, len(fake), seed)
        fake = fake.head(len(sampled_real))
        frames[str(generator)] = pd.concat([sampled_real, fake], ignore_index=True)
    return frames


def _auc_ap(frame: pd.DataFrame, score: str) -> tuple[float, float]:
    label = frame["subset"].eq("real").astype(np.uint8).to_numpy()
    value = frame[score].to_numpy(dtype=np.float64)
    return float(roc_auc_score(label, value)), float(average_precision_score(label, value))


def metric_tables(
    per_video: pd.DataFrame,
    seed: int,
    score_columns: list[str] | tuple[str, ...] = SCORE_COLUMNS,
    config_names: dict[str, str] = CONFIG_NAMES,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    generator_rows: list[dict] = []
    for dataset, dataset_df in per_video.groupby("dataset", sort=False):
        for generator, pair in pairwise_frames(dataset_df, seed).items():
            counts = pair["subset"].value_counts()
            for config in score_columns:
                auc, ap = _auc_ap(pair, config)
                generator_rows.append(
                    {
                        "dataset": dataset,
                        "generator": generator,
                        "config": config,
                        "config_name": config_names[config],
                        "n_real": int(counts.get("real", 0)),
                        "n_fake": int(counts.get("annotated", 0)),
                        "auc": auc,
                        "ap": ap,
                    }
                )
    generator_metrics = pd.DataFrame(generator_rows)
    dataset_metrics = (
        generator_metrics.groupby(["dataset", "config", "config_name"], as_index=False)
        .agg(n_generators=("generator", "nunique"), auc=("auc", "mean"), ap=("ap", "mean"))
    )
    macro = (
        dataset_metrics.groupby(["config", "config_name"], as_index=False)
        .agg(n_generators=("n_generators", "sum"), auc=("auc", "mean"), ap=("ap", "mean"))
    )
    macro.insert(0, "dataset", "Macro-3")
    dataset_metrics = pd.concat([dataset_metrics, macro], ignore_index=True)
    return dataset_metrics, generator_metrics


def _stable_seed(seed: int, *parts: str) -> int:
    digest = hashlib.sha256("\0".join((str(seed), *parts)).encode("utf-8")).digest()
    return int.from_bytes(digest[:4], "little")


def paired_bootstrap(
    per_video: pd.DataFrame,
    seed: int,
    iterations: int,
    comparisons: tuple[tuple[str, str, str], ...] = BOOTSTRAP_COMPARISONS,
) -> pd.DataFrame:
    rows: list[dict] = []
    for dataset, dataset_df in per_video.groupby("dataset", sort=False):
        pairs = pairwise_frames(dataset_df, seed)
        for new, base, comparison in comparisons:
            point_auc: list[float] = []
            point_ap: list[float] = []
            for pair in pairs.values():
                new_auc, new_ap = _auc_ap(pair, new)
                base_auc, base_ap = _auc_ap(pair, base)
                point_auc.append(new_auc - base_auc)
                point_ap.append(new_ap - base_ap)

            rng = np.random.default_rng(_stable_seed(seed, str(dataset), comparison))
            boot_auc = np.empty(iterations, dtype=np.float64)
            boot_ap = np.empty(iterations, dtype=np.float64)
            split_pairs = []
            for pair in pairs.values():
                real = pair[pair["subset"] == "real"].reset_index(drop=True)
                fake = pair[pair["subset"] == "annotated"].reset_index(drop=True)
                split_pairs.append((real, fake))
            for iteration in range(iterations):
                auc_deltas: list[float] = []
                ap_deltas: list[float] = []
                for real, fake in split_pairs:
                    real_idx = rng.integers(0, len(real), len(real))
                    fake_idx = rng.integers(0, len(fake), len(fake))
                    sample = pd.concat(
                        [real.iloc[real_idx], fake.iloc[fake_idx]], ignore_index=True
                    )
                    new_auc, new_ap = _auc_ap(sample, new)
                    base_auc, base_ap = _auc_ap(sample, base)
                    auc_deltas.append(new_auc - base_auc)
                    ap_deltas.append(new_ap - base_ap)
                boot_auc[iteration] = np.mean(auc_deltas)
                boot_ap[iteration] = np.mean(ap_deltas)
            for metric, point, samples in (
                ("auc", float(np.mean(point_auc)), boot_auc),
                ("ap", float(np.mean(point_ap)), boot_ap),
            ):
                rows.append(
                    {
                        "dataset": dataset,
                        "scope": "generator_macro",
                        "comparison": comparison,
                        "new_config": new,
                        "base_config": base,
                        "metric": metric,
                        "delta": point,
                        "ci95_low": float(np.quantile(samples, 0.025)),
                        "ci95_high": float(np.quantile(samples, 0.975)),
                        "bootstrap_iterations": iterations,
                    }
                )
    return pd.DataFrame(rows)


def _format_metric_table(dataset_metrics: pd.DataFrame) -> list[str]:
    pivot = dataset_metrics.pivot(index="config", columns="dataset", values=["auc", "ap"])
    lines = [
        "| ID | Configuration | ComGenVid AUC/AP | VideoFeedback AUC/AP | GenVideo AUC/AP | Macro-3 AUC/AP |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for config in SCORE_COLUMNS:
        cells = []
        for dataset in ("comgenvid", "videofeedback", "genvideo", "Macro-3"):
            cells.append(f"{pivot.loc[config, ('auc', dataset)]:.4f}/{pivot.loc[config, ('ap', dataset)]:.4f}")
        lines.append(f"| {config} | {CONFIG_NAMES[config]} | " + " | ".join(cells) + " |")
    return lines


def _delta_lines(generator_metrics: pd.DataFrame, dataset: str, new: str, base: str) -> list[str]:
    selected = generator_metrics[
        (generator_metrics["dataset"] == dataset)
        & generator_metrics["config"].isin([new, base])
    ]
    pivot = selected.pivot(index="generator", columns="config", values=["auc", "ap"])
    rows = []
    for generator in pivot.index:
        rows.append(
            (
                str(generator),
                float(pivot.loc[generator, ("auc", new)] - pivot.loc[generator, ("auc", base)]),
                float(pivot.loc[generator, ("ap", new)] - pivot.loc[generator, ("ap", base)]),
            )
        )
    rows.sort(key=lambda item: item[2], reverse=True)
    return [f"- `{name}`: delta AUC {auc:+.4f}, delta AP {ap:+.4f}" for name, auc, ap in rows]


def write_analysis(
    path: Path,
    per_video: pd.DataFrame,
    dataset_metrics: pd.DataFrame,
    generator_metrics: pd.DataFrame,
    bootstrap: pd.DataFrame,
    skips: dict[str, list[str]],
    num_bins: int,
) -> None:
    positive_rows = []
    for new, base, comparison in BOOTSTRAP_COMPARISONS:
        selected = generator_metrics[generator_metrics["config"].isin([new, base])]
        pivot = selected.pivot(index=["dataset", "generator"], columns="config", values="ap")
        positive_rows.append(
            f"- `{comparison}` ({new}-{base}): {(pivot[new] > pivot[base]).sum()}/{len(pivot)} generators improve in AP."
        )

    cg = dataset_metrics[dataset_metrics["dataset"] == "comgenvid"].set_index("config")
    lines = [
        "# Multi-order baseline analysis",
        "",
        "## Protocol",
        "",
        "- Evaluation uses each dataset's frozen holdout index and the strict indexed 2 s / 8 FPS window (16 frames). No 1 s fallback is allowed.",
        "- Every B0-B8 row is the one-to-one key intersection of global CLS-cache scores and the frozen release patch score CSV.",
        "- B0-B2 are recomputed with `precomputed/stall_params_vatex_dino_v3.npz`; B3 is an empirical real-CDF percentile fitted on a disjoint 200-real dataset calibration index.",
        "- B4 is the untuned average `0.5 * B2 + 0.5 * B3`. B7 uses the frozen local weights 0.1 spatial / 0.9 D2; B8 uses 0.6 global / 0.4 patch.",
        f"- Motion-conditioned D3 uses {num_bins} calibration quantile bins. Its Stage-2 candidate columns are included in `per_video_scores.csv` but are not relabeled as B baselines.",
        "- AUC/AP are higher-is-real and macro-averaged over pairwise generator comparisons. Each generator uses the same deterministic real sample for every configuration.",
        "- Paired bootstrap resamples real and fake rows within each generator, preserves score pairing, and reports the generator-macro delta.",
        "",
        "## Important limitation",
        "",
        "The current workspace has no VideoFeedback patch-token cache. B5-B7 therefore reuse the frozen release per-video components, whose patch calibration used all available real videos (4080 for VideoFeedback, 1698 for ComGenVid, 9984 for GenVideo). These branches are retrospectively evaluated on holdout IDs but are not calibration-disjoint. B0-B4 and all D3 variants are reproducible from current caches; a leakage-free B5-B8 rerun requires rebuilding the VideoFeedback patch cache and refitting all three patch models on the same 200-real calibration protocol.",
        "",
        "## Dataset macro metrics",
        "",
        *_format_metric_table(dataset_metrics),
        "",
        "## Positive generator counts",
        "",
        *positive_rows,
        "",
        "## GenVideo: raw D3 contribution (B4-B2)",
        "",
        *_delta_lines(generator_metrics, "genvideo", "B4", "B2"),
        "",
        "## VideoFeedback: raw D3 failures and reversals (B4-B2)",
        "",
        *_delta_lines(generator_metrics, "videofeedback", "B4", "B2"),
        "",
        "## Patch component diagnosis",
        "",
        *_delta_lines(generator_metrics, "comgenvid", "B7", "B5"),
        "",
        "The B7-B5 list isolates the local D2 contribution; `generator_metrics.csv` also supports B7-B6 to isolate local spatial contribution on every dataset.",
        "",
        "## ComGenVid patch-only versus global+patch",
        "",
        f"B7 patch-only reaches AUC/AP {cg.loc['B7', 'auc']:.4f}/{cg.loc['B7', 'ap']:.4f}; B8 reaches {cg.loc['B8', 'auc']:.4f}/{cg.loc['B8', 'ap']:.4f}. The deltas are {cg.loc['B8', 'auc'] - cg.loc['B7', 'auc']:+.4f} AUC and {cg.loc['B8', 'ap'] - cg.loc['B7', 'ap']:+.4f} AP. Per-generator evidence is in the B8-B7 comparison.",
        "",
        "## Bootstrap summary",
        "",
        "| Dataset | Comparison | Metric | Delta | 95% CI |",
        "|---|---|---|---:|---:|",
    ]
    for row in bootstrap.itertuples(index=False):
        lines.append(
            f"| {row.dataset} | {row.new_config}-{row.base_config} | {row.metric.upper()} | "
            f"{row.delta:+.4f} | [{row.ci95_low:+.4f}, {row.ci95_high:+.4f}] |"
        )
    lines.extend(["", "## Missing rows", ""])
    for dataset, messages in skips.items():
        lines.append(f"- `{dataset}`: {len(messages)} strict-window/cache rows skipped.")
        for message in messages[:5]:
            lines.append(f"  - `{message}`")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=REPO_ROOT)
    parser.add_argument("--output-dir", type=Path, default=REPO_ROOT / "results/multi_order_baselines")
    parser.add_argument("--global-params", type=Path, default=REPO_ROOT / "precomputed/stall_params_vatex_dino_v3.npz")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--motion-bins", type=int, default=5)
    parser.add_argument("--bootstrap-iterations", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.bootstrap_iterations < 1:
        raise ValueError("--bootstrap-iterations must be positive")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    scorer = GlobalBatchScorer(args.global_params, args.device)

    all_scores: list[pd.DataFrame] = []
    skips: dict[str, list[str]] = {}
    for spec in dataset_specs(args.root):
        print(f"[{spec.display_name}] loading protocol rows", flush=True)
        index = _load_index(spec.eval_index)
        patch = _load_patch_scores(spec.patch_scores)
        index = index.merge(patch[KEY_COLUMNS], on=KEY_COLUMNS, how="inner", validate="one_to_one")
        calib_index = _load_index(spec.d3_calib_index)

        print(f"[{spec.display_name}] scoring eval={len(index)} calibration={len(calib_index)}", flush=True)
        global_d3, eval_skips = score_index(index, spec.embedding_cache, scorer, args.batch_size)
        calib_d3, calib_skips = score_index(calib_index, spec.embedding_cache, scorer, args.batch_size)
        if calib_d3.empty:
            raise ValueError(f"{spec.name} has no valid D3 calibration rows")
        calibrated = add_d3_calibration(global_d3, calib_d3, args.motion_bins)
        scored = add_baselines(calibrated, patch)
        scored.insert(0, "dataset", spec.name)
        all_scores.append(scored)
        skips[spec.name] = eval_skips + [f"calibration: {item}" for item in calib_skips]
        print(f"[{spec.display_name}] final strict intersection={len(scored)}", flush=True)

    per_video = pd.concat(all_scores, ignore_index=True)
    dataset_metrics, generator_metrics = metric_tables(per_video, args.seed)
    print("running paired bootstrap", flush=True)
    bootstrap = paired_bootstrap(per_video, args.seed, args.bootstrap_iterations)

    per_video.to_csv(args.output_dir / "per_video_scores.csv", index=False)
    dataset_metrics.to_csv(args.output_dir / "dataset_metrics.csv", index=False)
    generator_metrics.to_csv(args.output_dir / "generator_metrics.csv", index=False)
    bootstrap.to_csv(args.output_dir / "bootstrap_deltas.csv", index=False)
    write_analysis(
        args.output_dir / "baseline_analysis.md",
        per_video,
        dataset_metrics,
        generator_metrics,
        bootstrap,
        skips,
        args.motion_bins,
    )
    print(f"wrote {args.output_dir}", flush=True)


if __name__ == "__main__":
    main()
