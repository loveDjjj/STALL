#!/usr/bin/env python3
"""Audit VideoFeedback rank-1023 whitening and batch-shape sensitivity."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from alpha_stalled.calibration import empirical_cdf
from alpha_stalled.historical_window_analysis import (
    KEY_COLUMNS,
    WINDOW_KEYS,
    target_k_reference,
)
from alpha_stalled.metrics import metric_tables
from patch_matching import patch_temporal_delta
from alpha_stalled.legacy_window_scoring import decode_manifest_row_with_retries
from stall_patch import PatchSTALL


OPERATORS = {
    "layer23_region1_mean": {
        "raw": "region1_raw",
        "temporal": "region1_temporal",
        "layer": 23,
        "params": "videofeedback_region1_mean.npz",
    },
    "layer17_region1_mean": {
        "raw": "layer17_raw",
        "temporal": "layer17_temporal",
        "layer": 17,
        "params": "videofeedback_layer17_region1_mean.npz",
    },
}


def read_parts(root: Path) -> pd.DataFrame:
    paths = sorted(root.rglob("part_*.csv"))
    if not paths:
        raise FileNotFoundError(f"no part CSVs under {root}")
    frame = pd.concat(
        [pd.read_csv(path, float_precision="round_trip") for path in paths],
        ignore_index=True,
    )
    if frame.duplicated(WINDOW_KEYS).any():
        raise ValueError(f"duplicate window keys under {root}")
    if len(frame) != 7076:
        raise ValueError(f"expected 7076 VideoFeedback windows, got {len(frame)}")
    return frame


def covariance_diagnostics(params_dir: Path) -> pd.DataFrame:
    rows = []
    for operator, spec in OPERATORS.items():
        data = np.load(params_dir / spec["params"], allow_pickle=True)
        whitening = data["W_patch_temp"].astype(np.float64)
        if "whitening_eigenvalues" in data.files:
            eigenvalues = data["whitening_eigenvalues"].astype(np.float64)
            eigenvalue_source = "stored_fit_eigenvalues"
        else:
            # W = V diag(1/sqrt(lambda + 1e-5)); V has orthonormal columns.
            inverse_variance = np.sum(whitening * whitening, axis=0)
            eigenvalues = np.maximum(1.0 / inverse_variance - 1e-5, 0.0)
            eigenvalue_source = "recovered_from_whitening_column_norms"
        positive = eigenvalues[eigenvalues > 0]
        rows.append(
            {
                "operator": operator,
                "feature_dimension": whitening.shape[0],
                "retained_rank": whitening.shape[1],
                "dropped_dimensions": whitening.shape[0] - whitening.shape[1],
                "full_covariance_condition_number": np.inf,
                "retained_eigenvalue_min": float(positive.min()),
                "retained_eigenvalue_max": float(positive.max()),
                "retained_condition_number": float(positive.max() / positive.min()),
                "whitening_epsilon": 1e-5,
                "eigenvalue_source": eigenvalue_source,
            }
        )
    return pd.DataFrame(rows)


def video_scores(
    baseline: pd.DataFrame,
    raw: pd.DataFrame,
    temporal_column: str,
    score_name: str,
) -> pd.DataFrame:
    source = raw[[*WINDOW_KEYS, temporal_column]].rename(
        columns={temporal_column: "temporal"}
    )
    windows = baseline.merge(source, on=WINDOW_KEYS, how="inner", validate="one_to_one")
    if len(windows) != len(baseline):
        raise ValueError("batch audit changed VideoFeedback window count")
    windows["temporal"] = np.rint(windows["temporal"].to_numpy() * 200.0) / 200.0
    baseline_temporal = np.rint(windows["patch_d2"].to_numpy() * 200.0) / 200.0
    windows["local"] = windows["L_k"] + 0.9 * (
        windows["temporal"] - baseline_temporal
    )
    per_video = (
        windows.groupby(KEY_COLUMNS, sort=False, observed=True)
        .agg(
            effective_k=("effective_k", "first"),
            windows=("window_id", "size"),
            G_raw=("G_k", "mean"),
            L_raw=("local", "mean"),
        )
        .reset_index()
    )
    frames = []
    evaluation = per_video[per_video["protocol_split"] == "evaluation"]
    for effective_k, target in evaluation.groupby("effective_k", sort=True):
        target = target.copy()
        reference = target_k_reference(windows, int(effective_k), "local")
        target["G"] = empirical_cdf(
            target["G_raw"].to_numpy(), reference["G_mean_raw"].to_numpy()
        )
        target["L"] = empirical_cdf(
            target["L_raw"].to_numpy(), reference["L_raw"].to_numpy()
        )
        target[score_name] = 0.6 * target["G"] + 0.4 * target["L"]
        frames.append(target)
    return pd.concat(frames, ignore_index=True)


def compare_batch_outputs(
    batch8: pd.DataFrame,
    batch4: pd.DataFrame,
    baseline: pd.DataFrame,
    frozen_order: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    merged = batch8.merge(
        batch4, on=WINDOW_KEYS, suffixes=("_batch8", "_batch4"), validate="one_to_one"
    )
    score_rows = []
    metric_rows = []
    affected_rows = []
    for operator, spec in OPERATORS.items():
        raw_delta = (
            merged[f"{spec['raw']}_batch8"] - merged[f"{spec['raw']}_batch4"]
        ).abs()
        temporal_delta = (
            merged[f"{spec['temporal']}_batch8"]
            - merged[f"{spec['temporal']}_batch4"]
        ).abs()
        changed = temporal_delta > 0
        score_rows.append(
            {
                "operator": operator,
                "windows": len(merged),
                "raw_score_max_abs_error": float(raw_delta.max()),
                "raw_score_changed_windows": int((raw_delta > 0).sum()),
                "percentile_max_abs_error": float(temporal_delta.max()),
                "percentile_changed_windows": int(changed.sum()),
                "evaluation_percentile_changed_windows": int(
                    (changed & merged["protocol_split"].eq("evaluation")).sum()
                ),
                "evaluation_changed_videos": int(
                    merged.loc[
                        changed & merged["protocol_split"].eq("evaluation"),
                        "filename",
                    ].nunique()
                ),
            }
        )
        affected = merged.loc[
            changed & merged["protocol_split"].eq("evaluation"),
            [*WINDOW_KEYS, f"{spec['raw']}_batch8", f"{spec['raw']}_batch4",
             f"{spec['temporal']}_batch8", f"{spec['temporal']}_batch4"],
        ].copy()
        affected.insert(0, "operator", operator)
        affected_rows.append(affected)

        b8 = video_scores(baseline, batch8, spec["temporal"], "batch8")
        b4 = video_scores(baseline, batch4, spec["temporal"], "batch4")
        order = frozen_order[KEY_COLUMNS].copy()
        order["_order"] = np.arange(len(order), dtype=np.int64)
        b8 = (
            b8.merge(order, on=KEY_COLUMNS, how="inner", validate="one_to_one")
            .sort_values("_order")
            .drop(columns="_order")
            .reset_index(drop=True)
        )
        b4 = (
            b4.merge(order, on=KEY_COLUMNS, how="inner", validate="one_to_one")
            .sort_values("_order")
            .drop(columns="_order")
            .reset_index(drop=True)
        )
        videos = b8.merge(
            b4,
            on=KEY_COLUMNS,
            suffixes=("_batch8", "_batch4"),
            validate="one_to_one",
        )
        videos["rank_batch8"] = videos["batch8"].rank(method="average")
        videos["rank_batch4"] = videos["batch4"].rank(method="average")
        rank_delta = (videos["rank_batch8"] - videos["rank_batch4"]).abs()
        score_delta = (videos["batch8"] - videos["batch4"]).abs()
        metrics8, _ = metric_tables(
            b8, 42, score_columns=("batch8",), config_names={"batch8": "batch8"}
        )
        metrics4, _ = metric_tables(
            b4, 42, score_columns=("batch4",), config_names={"batch4": "batch4"}
        )
        row8 = metrics8[metrics8["dataset"] == "Macro-3"].iloc[0]
        row4 = metrics4[metrics4["dataset"] == "Macro-3"].iloc[0]
        metric_rows.append(
            {
                "operator": operator,
                "evaluation_videos": len(videos),
                "final_score_max_abs_error": float(score_delta.max()),
                "final_score_changed_videos": int((score_delta > 0).sum()),
                "rank_changed_videos": int((rank_delta > 0).sum()),
                "max_absolute_rank_change": float(rank_delta.max()),
                "batch4_auc": float(row4.auc),
                "batch8_auc": float(row8.auc),
                "delta_auc_batch8_minus_batch4": float(row8.auc - row4.auc),
                "batch4_ap": float(row4.ap),
                "batch8_ap": float(row8.ap),
                "delta_ap_batch8_minus_batch4": float(row8.ap - row4.ap),
            }
        )
    return (
        pd.DataFrame(score_rows),
        pd.DataFrame(metric_rows),
        pd.concat(affected_rows, ignore_index=True),
    )


def float64_score(temporal: np.ndarray, params: np.lib.npyio.NpzFile) -> tuple[float, float]:
    centered = temporal.astype(np.float64) - params["mu_patch_temp"].astype(np.float64)
    white = centered @ params["W_patch_temp"].astype(np.float64)
    dim = white.shape[-1]
    likelihood = -0.5 * (
        dim * np.log(2.0 * np.pi) + np.sum(white * white, axis=-1)
    )
    raw = float(likelihood.mean())
    reference = np.sort(params["calib_patch_temp_scores"].astype(np.float64))
    percentile = float(np.searchsorted(reference, raw, side="right") / len(reference))
    return raw, percentile


def float64_recheck(
    affected: pd.DataFrame,
    manifest_path: Path,
    params_dir: Path,
    device: str,
) -> pd.DataFrame:
    keys = affected[KEY_COLUMNS].drop_duplicates()
    manifest = pd.read_csv(manifest_path)
    rows = manifest.merge(keys, on=KEY_COLUMNS, how="inner", validate="one_to_one")
    if len(rows) != len(keys):
        raise ValueError("could not locate every affected evaluation video")
    decoded = [
        decode_manifest_row_with_retries(row, "K3_uniform", 64, 3)
        for _, row in rows.iterrows()
    ]
    extractor = PatchSTALL(device, data_dict=None, load_dino=True)
    outputs = extractor.frames_to_layer_patch_embeddings(
        [item["frames"] for item in decoded], layers=(17, 23), batch_size=32
    )
    lookup = {
        tuple(str(row[column]) for column in KEY_COLUMNS): (item, output)
        for (_, row), item, output in zip(rows.iterrows(), decoded, outputs)
    }
    result = []
    for record in affected.itertuples(index=False):
        spec = OPERATORS[record.operator]
        key = tuple(str(getattr(record, column)) for column in KEY_COLUMNS)
        item, output = lookup[key]
        positions = item["window_positions"][int(record.window_id)]
        patch = output["layers"][spec["layer"]][positions].astype(np.float32)
        temporal = patch_temporal_delta(
            patch,
            grid_size=tuple(output["grid_size"]),
            mode="same_grid_second_order",
            region_size=1,
        )
        params = np.load(params_dir / spec["params"], allow_pickle=True)
        raw64, percentile64 = float64_score(temporal, params)
        raw8 = float(getattr(record, f"{spec['raw']}_batch8"))
        raw4 = float(getattr(record, f"{spec['raw']}_batch4"))
        percentile8 = float(getattr(record, f"{spec['temporal']}_batch8"))
        percentile4 = float(getattr(record, f"{spec['temporal']}_batch4"))
        result.append(
            {
                "operator": record.operator,
                **{column: getattr(record, column) for column in KEY_COLUMNS},
                "window_id": int(record.window_id),
                "batch8_raw": raw8,
                "batch4_raw": raw4,
                "float64_raw": raw64,
                "float64_abs_error_vs_batch8": abs(raw64 - raw8),
                "float64_abs_error_vs_batch4": abs(raw64 - raw4),
                "batch8_percentile": percentile8,
                "batch4_percentile": percentile4,
                "float64_percentile": percentile64,
            }
        )
    return pd.DataFrame(result)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--batch8-root",
        type=Path,
        default=Path("/tmp/alpha_stalled_combined_ms_layers/videofeedback"),
    )
    parser.add_argument(
        "--batch4-root",
        type=Path,
        default=Path("/tmp/alpha_stalled_combined_ms_layers_batch4/videofeedback"),
    )
    parser.add_argument(
        "--baseline-windows",
        type=Path,
        default=REPO_ROOT
        / "results/multi_window_joint_typicality/K3_uniform_window_scores.csv",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=REPO_ROOT
        / "results/multi_window_joint_typicality/multi_window_manifest.csv",
    )
    parser.add_argument(
        "--frozen-evaluation",
        type=Path,
        default=REPO_ROOT
        / "results/multi_window_joint_typicality/K3_uniform_evaluation_scores.csv",
    )
    parser.add_argument(
        "--params-dir",
        type=Path,
        default=REPO_ROOT / "results/clean_universal/params",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPO_ROOT / "results/clean_universal/numerical_audit",
    )
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()

    batch8 = read_parts(args.batch8_root)
    batch4 = read_parts(args.batch4_root)
    baseline = pd.read_csv(args.baseline_windows, float_precision="round_trip")
    baseline = baseline[baseline["dataset"] == "videofeedback"].reset_index(drop=True)
    frozen = pd.read_csv(args.frozen_evaluation, float_precision="round_trip")
    frozen = frozen[frozen["dataset"] == "videofeedback"].reset_index(drop=True)
    condition = covariance_diagnostics(args.params_dir)
    score_diff, metric_diff, affected = compare_batch_outputs(
        batch8, batch4, baseline, frozen
    )
    float64 = float64_recheck(
        affected, args.manifest, args.params_dir, args.device
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    condition.to_csv(args.output_dir / "whitening_condition.csv", index=False)
    score_diff.to_csv(args.output_dir / "batch_shape_score_differences.csv", index=False)
    metric_diff.to_csv(args.output_dir / "batch_shape_metric_differences.csv", index=False)
    affected.to_csv(args.output_dir / "batch_shape_affected_windows.csv", index=False)
    float64.to_csv(args.output_dir / "float64_recheck.csv", index=False)
    print(condition.to_string(index=False))
    print("\nBatch-shape score differences")
    print(score_diff.to_string(index=False))
    print("\nBatch-shape metric differences")
    print(metric_diff.to_string(index=False))
    print("\nFloat64 recheck")
    print(float64.to_string(index=False))


if __name__ == "__main__":
    main()
