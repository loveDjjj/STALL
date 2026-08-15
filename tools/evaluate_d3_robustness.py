#!/usr/bin/env python3
"""Evaluate operator-controlled D3 robustness on frozen strict-window caches."""

from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score


REPO_ROOT = Path(__file__).resolve().parents[1]
TOOLS_DIR = REPO_ROOT / "tools"
for directory in (REPO_ROOT / "src", TOOLS_DIR):
    if str(directory) not in sys.path:
        sys.path.insert(0, str(directory))

from alpha_stalled.metrics import auc_ap as _auc_ap, pairwise_frames
from build_multi_order_baselines import (
    KEY_COLUMNS,
    _load_index,
    dataset_specs,
    d3_statistics,
    empirical_cdf,
    load_strict_window,
    two_sided_realness,
)


CONDITIONS = (
    "reference_8fps_2s",
    "uniform_4fps_2s",
    "central_8fps_1s",
    "frame_drop_one",
    "frame_duplicate_two",
    "remove_exact_duplicates",
)


def _stable_position(key: str, low: int, high: int) -> int:
    digest = hashlib.sha256(key.encode("utf-8")).digest()
    return low + int.from_bytes(digest[:4], "little") % (high - low)


def transformed_windows(
    emb: np.ndarray,
    timestamps: np.ndarray,
    key: str,
) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    drop = _stable_position(key, 3, len(emb) - 3)
    keep = np.ones(len(emb), dtype=bool)
    keep[drop] = False

    duplicate = emb.copy()
    duplicate[5] = duplicate[4]
    duplicate[11] = duplicate[10]

    exact_keep = np.ones(len(emb), dtype=bool)
    exact_keep[1:] = np.any(emb[1:] != emb[:-1], axis=1)
    return {
        "reference_8fps_2s": (emb, timestamps),
        "uniform_4fps_2s": (emb[::2], timestamps[::2]),
        "central_8fps_1s": (emb[4:12], timestamps[4:12]),
        "frame_drop_one": (emb[keep], timestamps[keep]),
        "frame_duplicate_two": (duplicate, timestamps),
        "remove_exact_duplicates": (emb[exact_keep], timestamps[exact_keep]),
    }


def score_rows(index: pd.DataFrame, cache: Path) -> pd.DataFrame:
    rows = []
    for count, (_, row) in enumerate(index.iterrows(), start=1):
        emb, timestamps = load_strict_window(cache, row)
        key = "/".join(str(row[column]) for column in KEY_COLUMNS)
        for condition, (variant_emb, variant_time) in transformed_windows(emb, timestamps, key).items():
            if len(variant_emb) < 3:
                continue
            stats = d3_statistics(variant_emb, variant_time)
            step_distance = np.linalg.norm(variant_emb[1:] - variant_emb[:-1], axis=1)
            rows.append(
                {
                    **{column: str(row[column]) for column in KEY_COLUMNS},
                    "condition": condition,
                    "n_frames": len(variant_emb),
                    "d3_raw": stats["d3_raw"],
                    "motion_raw": stats["motion_raw"],
                    "max_step_distance": float(step_distance.max()),
                }
            )
        if count % 2000 == 0:
            print(f"scored {count}/{len(index)} robustness windows", flush=True)
    return pd.DataFrame(rows)


def calibrate_conditions(eval_scores: pd.DataFrame, calib_scores: pd.DataFrame) -> pd.DataFrame:
    frames = []
    for condition in CONDITIONS:
        current = eval_scores[eval_scores["condition"] == condition].copy()
        calibration = calib_scores[calib_scores["condition"] == condition]
        raw_calib = calibration["d3_raw"].to_numpy(dtype=np.float64)
        current["d3_one_sided"] = empirical_cdf(current["d3_raw"].to_numpy(), raw_calib)
        current["d3_two_sided"] = two_sided_realness(current["d3_raw"].to_numpy(), raw_calib)
        motion_edges = np.unique(
            np.quantile(calibration["motion_raw"], [0.25, 0.5, 0.75])
        )
        current["motion_group"] = np.searchsorted(
            motion_edges,
            current["motion_raw"].to_numpy(),
            side="right",
        )
        scene_cut_threshold = float(np.quantile(calibration["max_step_distance"], 0.95))
        current["scene_cut_proxy"] = current["max_step_distance"] > scene_cut_threshold
        frames.append(current)
    return pd.concat(frames, ignore_index=True)


def build_wide(calibrated: pd.DataFrame, global_scores: pd.DataFrame) -> pd.DataFrame:
    one_sided = calibrated.pivot(
        index=KEY_COLUMNS, columns="condition", values="d3_one_sided"
    ).reset_index()
    motion = calibrated[calibrated["condition"] == "reference_8fps_2s"][
        KEY_COLUMNS + ["motion_group", "scene_cut_proxy"]
    ]
    wide = global_scores[KEY_COLUMNS + ["B0", "B1", "B2"]].merge(
        one_sided,
        on=KEY_COLUMNS,
        how="inner",
        validate="one_to_one",
    ).merge(motion, on=KEY_COLUMNS, how="left", validate="one_to_one")
    for condition in CONDITIONS:
        wide[condition] = 0.5 * wide["B0"] + 0.25 * wide["B1"] + 0.25 * wide[condition]
    return wide


def stability_table(calibrated: pd.DataFrame) -> pd.DataFrame:
    wide = calibrated.pivot(
        index=["dataset", *KEY_COLUMNS],
        columns="condition",
        values=["d3_raw", "d3_one_sided"],
    )
    rows = []
    for condition in CONDITIONS[1:]:
        for score in ("d3_raw", "d3_one_sided"):
            reference = wide[(score, "reference_8fps_2s")]
            variant = wide[(score, condition)]
            rows.append(
                {
                    "condition": condition,
                    "score": score,
                    "pearson_r": float(reference.corr(variant)),
                    "mean_absolute_change": float(np.nanmean(np.abs(reference - variant))),
                    "median_absolute_change": float(np.nanmedian(np.abs(reference - variant))),
                }
            )
    return pd.DataFrame(rows)


def motion_metrics(wide: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (dataset, motion_group), frame in wide.groupby(["dataset", "motion_group"]):
        label = frame["subset"].eq("real").astype(np.uint8).to_numpy()
        if len(np.unique(label)) < 2:
            continue
        score = frame["reference_8fps_2s"].to_numpy(dtype=np.float64)
        rows.append(
            {
                "dataset": dataset,
                "motion_group": int(motion_group),
                "n_real": int(label.sum()),
                "n_fake": int(len(label) - label.sum()),
                "auc": float(roc_auc_score(label, score)),
                "ap": float(average_precision_score(label, score)),
            }
        )
    return pd.DataFrame(rows)


def scene_cut_metrics(wide: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (dataset, scene_cut_proxy), frame in wide.groupby(
        ["dataset", "scene_cut_proxy"], sort=False
    ):
        label = frame["subset"].eq("real").astype(np.uint8).to_numpy()
        if len(np.unique(label)) < 2:
            continue
        score = frame["reference_8fps_2s"].to_numpy(dtype=np.float64)
        rows.append(
            {
                "dataset": dataset,
                "scene_cut_proxy": bool(scene_cut_proxy),
                "n_real": int(label.sum()),
                "n_fake": int(len(label) - label.sum()),
                "auc": float(roc_auc_score(label, score)),
                "ap": float(average_precision_score(label, score)),
            }
        )
    return pd.DataFrame(rows)


def robustness_metric_tables(
    wide: pd.DataFrame,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    generator_rows = []
    for dataset, dataset_frame in wide.groupby("dataset", sort=False):
        for condition in CONDITIONS:
            valid = dataset_frame.replace([np.inf, -np.inf], np.nan).dropna(subset=[condition])
            for generator, pair in pairwise_frames(valid, seed).items():
                auc, ap = _auc_ap(pair, condition)
                counts = pair["subset"].value_counts()
                generator_rows.append(
                    {
                        "dataset": dataset,
                        "generator": generator,
                        "config": condition,
                        "config_name": condition,
                        "n_real": int(counts.get("real", 0)),
                        "n_fake": int(counts.get("annotated", 0)),
                        "auc": auc,
                        "ap": ap,
                    }
                )
    generator = pd.DataFrame(generator_rows)
    dataset = (
        generator.groupby(["dataset", "config", "config_name"], as_index=False)
        .agg(
            n_generators=("generator", "nunique"),
            n_real_min=("n_real", "min"),
            n_fake_min=("n_fake", "min"),
            auc=("auc", "mean"),
            ap=("ap", "mean"),
        )
    )
    macro = (
        dataset.groupby(["config", "config_name"], as_index=False)
        .agg(
            n_generators=("n_generators", "sum"),
            n_real_min=("n_real_min", "min"),
            n_fake_min=("n_fake_min", "min"),
            auc=("auc", "mean"),
            ap=("ap", "mean"),
        )
    )
    macro.insert(0, "dataset", "Macro-3")
    return pd.concat([dataset, macro], ignore_index=True), generator


def write_analysis(
    path: Path,
    dataset_metrics: pd.DataFrame,
    stability: pd.DataFrame,
) -> None:
    metrics = dataset_metrics.pivot(index="config", columns="dataset", values=["auc", "ap"])
    lines = [
        "# D3 robustness analysis",
        "",
        "## Scope",
        "",
        "This is an operator-controlled DINOv3 embedding-cache study. Each perturbation is recalibrated with the matching transform on the disjoint 200-real calibration set. The fused score is the fixed Stage-2 structure `0.5 spatial + 0.25 first-order temporal + 0.25 D3`.",
        "",
        "## Dataset macro metrics",
        "",
        "| Condition | ComGenVid AUC/AP | VideoFeedback AUC/AP | GenVideo AUC/AP | Macro-3 AUC/AP |",
        "|---|---:|---:|---:|---:|",
    ]
    for condition in CONDITIONS:
        cells = []
        for dataset in ("comgenvid", "videofeedback", "genvideo", "Macro-3"):
            cells.append(
                f"{metrics.loc[condition, ('auc', dataset)]:.4f}/{metrics.loc[condition, ('ap', dataset)]:.4f}"
            )
        lines.append(f"| {condition} | " + " | ".join(cells) + " |")

    lines.extend(
        [
            "",
            "## Score stability",
            "",
            "| Condition | Score | Pearson r | Mean absolute change | Median absolute change |",
            "|---|---|---:|---:|---:|",
        ]
    )
    for row in stability.itertuples(index=False):
        lines.append(
            f"| {row.condition} | {row.score} | {row.pearson_r:.4f} | "
            f"{row.mean_absolute_change:.4f} | {row.median_absolute_change:.4f} |"
        )
    lines.extend(
        [
            "",
            "## Interpretation boundary",
            "",
            "- `uniform_4fps_2s`, `central_8fps_1s`, frame drop, duplication, and exact-duplicate removal isolate temporal-operator sensitivity after the encoder. The last condition is the native timestamp sequence after removing adjacent duplicate embeddings.",
            "- JPEG compression and resize cannot be reconstructed from cached embeddings. They require decoding perturbed pixels and rerunning DINOv3, so they are not claimed as completed here.",
            "- Scene-cut annotations are absent. `d3_robustness_scene_cut_proxy.csv` therefore reports a declared proxy: maximum adjacent embedding distance above the 95th percentile of the disjoint real calibration set. It is not treated as ground truth.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--stage1-scores",
        type=Path,
        default=REPO_ROOT / "results/multi_order_baselines/per_video_scores.csv",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPO_ROOT / "results/multi_order_baselines",
    )
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    stage1 = pd.read_csv(args.stage1_scores)
    all_calibrated = []
    all_wide = []
    for spec in dataset_specs(REPO_ROOT):
        index = _load_index(spec.eval_index).merge(
            stage1[stage1["dataset"] == spec.name][KEY_COLUMNS],
            on=KEY_COLUMNS,
            how="inner",
            validate="one_to_one",
        )
        calib_index = _load_index(spec.d3_calib_index)
        print(f"[{spec.name}] robustness eval={len(index)} calibration={len(calib_index)}", flush=True)
        calibrated = calibrate_conditions(
            score_rows(index, spec.embedding_cache),
            score_rows(calib_index, spec.embedding_cache),
        )
        calibrated.insert(0, "dataset", spec.name)
        all_calibrated.append(calibrated)
        global_scores = stage1[stage1["dataset"] == spec.name]
        wide = build_wide(calibrated.drop(columns=["dataset"]), global_scores)
        wide.insert(0, "dataset", spec.name)
        all_wide.append(wide)

    calibrated = pd.concat(all_calibrated, ignore_index=True)
    wide = pd.concat(all_wide, ignore_index=True)
    dataset_metrics, generator_metrics = robustness_metric_tables(wide, args.seed)
    stability = stability_table(calibrated)
    motion = motion_metrics(wide)
    scene_cut = scene_cut_metrics(wide)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    calibrated.to_csv(args.output_dir / "d3_robustness_per_video_scores.csv", index=False)
    dataset_metrics.to_csv(args.output_dir / "d3_robustness_dataset_metrics.csv", index=False)
    generator_metrics.to_csv(args.output_dir / "d3_robustness_generator_metrics.csv", index=False)
    stability.to_csv(args.output_dir / "d3_robustness_stability.csv", index=False)
    motion.to_csv(args.output_dir / "d3_robustness_motion_groups.csv", index=False)
    scene_cut.to_csv(args.output_dir / "d3_robustness_scene_cut_proxy.csv", index=False)
    write_analysis(
        args.output_dir / "d3_robustness_analysis.md",
        dataset_metrics,
        stability,
    )


if __name__ == "__main__":
    main()
