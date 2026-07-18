#!/usr/bin/env python
"""Sanity checks for PatchSTALL patch temporal matching."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from dataset_utils import _is_missing_window, load_csv  # noqa: E402
from dataset_utils_patch import _get_patch_cache_path  # noqa: E402
from eval_patch import PatchSpatialScorer  # noqa: E402
from metrics import predictor_scalar2metrics  # noqa: E402
from patch_matching import (  # noqa: E402
    local_window_indices,
    matching_diagnostics_for_pair,
    patch_id_to_rc,
    patch_temporal_delta,
)


def load_samples_from_cache(csv_path: str, patch_cache: str, duration: int, compact: bool, num_videos: int):
    df = load_csv(csv_path)
    window_col = f"{duration}_sec_idxs"
    samples = []
    cache_root = Path(patch_cache)

    for _, row in df.iterrows():
        if compact and _is_missing_window(row.get(window_col)):
            continue
        stem = Path(row["video_path"]).stem
        cache_path = _get_patch_cache_path(
            cache_root,
            row["subset"],
            row["source_model"],
            stem,
            duration,
            compact,
        )
        if not cache_path.exists():
            continue
        payload = torch.load(cache_path, weights_only=True)
        samples.append(
            {
                "subset": row["subset"],
                "source_model": row["source_model"],
                "filename": Path(row["video_path"]).name,
                "video_path": row["video_path"],
                "patch": payload["patch"].numpy().astype(np.float32),
                "grid_size": tuple(int(x) for x in payload["grid_size"]),
            }
        )
        if len(samples) >= num_videos:
            break

    if not samples:
        raise ValueError(f"No patch cache samples found under {patch_cache}")
    return samples


def load_balanced_samples(csv_path: str, patch_cache: str, duration: int, compact: bool, num_videos: int):
    df = load_csv(csv_path)
    window_col = f"{duration}_sec_idxs"
    cache_root = Path(patch_cache)
    parts = []
    for subset, group in df.groupby("subset"):
        taken = 0
        for _, row in group.iterrows():
            if compact and _is_missing_window(row.get(window_col)):
                continue
            stem = Path(row["video_path"]).stem
            cache_path = _get_patch_cache_path(
                cache_root,
                row["subset"],
                row["source_model"],
                stem,
                duration,
                compact,
            )
            if not cache_path.exists():
                continue
            payload = torch.load(cache_path, weights_only=True)
            parts.append(
                {
                    "subset": row["subset"],
                    "source_model": row["source_model"],
                    "filename": Path(row["video_path"]).name,
                    "video_path": row["video_path"],
                    "patch": payload["patch"].numpy().astype(np.float32),
                    "grid_size": tuple(int(x) for x in payload["grid_size"]),
                }
            )
            taken += 1
            if taken >= num_videos:
                break
    if not parts:
        raise ValueError(f"No patch cache samples found under {patch_cache}")
    return parts


def fmt_stat(arr: np.ndarray) -> str:
    arr = np.asarray(arr, dtype=np.float64)
    return f"min={arr.min():.6g}, mean={arr.mean():.6g}, max={arr.max():.6g}"


def check_radius_zero(samples: list[dict]) -> tuple[list[str], bool]:
    lines = ["## Check 1: radius=0 equivalence", ""]
    ok = True
    for sample in samples:
        patch = sample["patch"]
        grid_size = sample["grid_size"]
        same = patch_temporal_delta(patch, grid_size, mode="same_grid")
        hard0 = patch_temporal_delta(patch, grid_size, mode="motion_hard", radius=0)
        soft0 = patch_temporal_delta(patch, grid_size, mode="motion_soft", radius=0, top_m=1)
        hard_abs = np.abs(same - hard0)
        soft_abs = np.abs(same - soft0)
        hard_max = float(hard_abs.max())
        soft_max = float(soft_abs.max())
        hard_mean = float(hard_abs.mean())
        soft_mean = float(soft_abs.mean())
        ok = ok and hard_max < 1e-5 and soft_max < 1e-5
        lines.append(
            f"- `{sample['filename']}`: hard max={hard_max:.3e}, hard mean={hard_mean:.3e}, "
            f"soft max={soft_max:.3e}, soft mean={soft_mean:.3e}"
        )
    lines.append("")
    lines.append(f"Conclusion: {'PASS' if ok else 'FAIL'}")
    lines.append("")
    return lines, ok


def check_windows(grid_size: tuple[int, int], radius: int) -> list[str]:
    lines = ["## Check 2: patch index and local window", ""]
    probe_ids = [0, 13, 14, grid_size[0] * grid_size[1] - 1]
    for pid in probe_ids:
        row, col = patch_id_to_rc(pid, grid_size)
        lines.append(f"- patch {pid} -> row {row}, col {col}")
    lines.append("")
    for pid in [0, 7, 105, grid_size[0] * grid_size[1] - 1]:
        ids = local_window_indices(pid, grid_size, radius)
        lines.append(f"- patch {pid}, radius={radius}: count={len(ids)}, ids={ids.tolist()}")
    lines.append("")
    return lines


def collect_matching_stats(samples: list[dict], args) -> tuple[list[str], list[dict]]:
    lines = ["## Check 5/6/7: matching score, softmax, displacement", ""]
    mode = "hard" if args.patch_temp_mode == "motion_hard" else "soft"
    all_stats = []
    example_rows = []

    for sample in samples:
        patch = sample["patch"]
        grid_size = sample["grid_size"]
        for frame_id in range(max(0, len(patch) - 1)):
            diag = matching_diagnostics_for_pair(
                patch[frame_id],
                patch[frame_id + 1],
                grid_size,
                radius=args.match_radius,
                top_m=args.top_m,
                temperature=args.temperature,
                lambda_dist=args.lambda_dist,
                mode=mode,
            )
            all_stats.append((sample, frame_id, diag))

    if not all_stats:
        lines.append("No frame pairs available.")
        return lines, example_rows

    def concat(key):
        return np.concatenate([diag[key].reshape(-1) for _, _, diag in all_stats])

    lines.append(f"- cosine: {fmt_stat(concat('cosine_mean'))}")
    lines.append(f"- normalized distance: {fmt_stat(concat('dist_mean'))}")
    lines.append(f"- distance penalty: {fmt_stat(concat('penalty_mean'))}")
    lines.append(f"- final matching score: {fmt_stat(concat('score_mean'))}")
    lines.append(f"- selected displacement: {fmt_stat(concat('displacements'))}")
    lines.append(f"- same-position ratio: {float((concat('displacements') == 0).mean()):.6f}")
    lines.append(f"- move <= 1 patch ratio: {float((concat('displacements') <= 1.0).mean()):.6f}")
    lines.append(f"- move <= 2 patch ratio: {float((concat('displacements') <= 2.0).mean()):.6f}")
    lines.append(f"- top1-top2 margin: {fmt_stat(concat('top1_scores') - concat('top2_scores'))}")
    if args.patch_temp_mode == "motion_soft":
        lines.append(f"- weights_sum: {fmt_stat(concat('weights_sum'))}")
        lines.append(f"- weights_max: {fmt_stat(concat('weights_max'))}")
        lines.append(f"- entropy: {fmt_stat(concat('entropy'))}")
    lines.append("")

    for sample, frame_id, diag in all_stats[: min(3, len(all_stats))]:
        for patch_id in np.argsort(diag["top1_scores"])[:10]:
            matched_id = int(diag["matched_ids"][patch_id])
            row, col = patch_id_to_rc(int(patch_id), sample["grid_size"])
            mrow, mcol = patch_id_to_rc(matched_id, sample["grid_size"])
            example_rows.append(
                {
                    "video_id": sample["filename"],
                    "subset": sample["subset"],
                    "source_model": sample["source_model"],
                    "frame_id": frame_id,
                    "patch_id": int(patch_id),
                    "matched_patch_id": matched_id,
                    "row": row,
                    "col": col,
                    "matched_row": mrow,
                    "matched_col": mcol,
                    "cosine": float(diag["top1_cosines"][patch_id]),
                    "score": float(diag["top1_scores"][patch_id]),
                    "displacement": float(diag["displacements"][patch_id]),
                }
            )
    return lines, example_rows


def check_score_direction(args) -> list[str]:
    lines = ["## Check 3: score direction", ""]
    if not args.patch_params:
        lines.append("Skipped: --patch-params not provided.")
        lines.append("")
        return lines

    scorer = PatchSpatialScorer(args.patch_params)
    try:
        scorer.validate_temporal_args(args.patch_temp_mode)
    except Exception as exc:
        lines.append(f"Skipped: params/eval consistency failed ({exc})")
        lines.append("")
        return lines
    samples = load_balanced_samples(args.csv, args.patch_cache, args.duration, args.compact, args.num_videos)
    rows = []
    for sample in samples:
        result = scorer.score_patch_cache(sample, patch_temp_mode=args.patch_temp_mode)
        row = {
            "subset": sample["subset"],
            "source_model": sample["source_model"],
            "filename": sample["filename"],
            "patch_spat_score": result["patch_spat_percentile"],
            "patch_final_score": result["patch_final_score"],
            "final_score": result["final_score"],
        }
        if "patch_temp_percentile" in result:
            row["patch_temp_score"] = result["patch_temp_percentile"]
        rows.append(row)

    df = pd.DataFrame(rows)
    labels = (df["subset"] == "real").astype(np.uint8).to_numpy()
    for score_col in [c for c in ["patch_spat_score", "patch_temp_score", "patch_final_score", "final_score"] if c in df.columns]:
        scores = df[score_col].to_numpy(dtype=np.float64)
        real_scores = scores[df["subset"].to_numpy() == "real"]
        fake_scores = scores[df["subset"].to_numpy() != "real"]
        if len(np.unique(labels)) == 2:
            auc = predictor_scalar2metrics(scores, labels)["AUC"]
            auc_neg = predictor_scalar2metrics(-scores, labels)["AUC"]
            auc_one_minus = predictor_scalar2metrics(1.0 - scores, labels)["AUC"]
        else:
            auc = auc_neg = auc_one_minus = float("nan")
        lines.append(
            f"- {score_col}: real mean={real_scores.mean():.6f}, fake mean={fake_scores.mean():.6f}, "
            f"real median={np.median(real_scores):.6f}, fake median={np.median(fake_scores):.6f}, "
            f"AUC(score)={auc:.6f}, AUC(-score)={auc_neg:.6f}, AUC(1-score)={auc_one_minus:.6f}"
        )
    lines.append("")
    return lines


def check_param_config(args) -> list[str]:
    lines = ["## Check 4: params temporal mode consistency", ""]
    if not args.patch_params:
        lines.append("Skipped: --patch-params not provided.")
        lines.append("")
        return lines
    data = np.load(args.patch_params, allow_pickle=True)
    config_raw = data["aggregation_config"]
    if isinstance(config_raw, np.ndarray):
        config_raw = config_raw.item()
    config = json.loads(str(config_raw))
    for key in ["patch_temp_mode", "temporal_feature_version", "match_radius", "top_m", "temperature", "lambda_dist", "mode", "bottomk_ratio"]:
        lines.append(f"- params {key}: {config.get(key)}")
    scorer = PatchSpatialScorer(args.patch_params)
    try:
        scorer.validate_temporal_args(args.patch_temp_mode)
        lines.append("- eval args consistency: PASS")
    except Exception as exc:
        lines.append(f"- eval args consistency: FAIL ({exc})")
    lines.append("")
    return lines


def main():
    parser = argparse.ArgumentParser(description="Debug PatchSTALL patch matching sanity checks.")
    parser.add_argument("--csv", default=str(ROOT / "cache/indexes/comgenvid.csv"))
    parser.add_argument("--patch-cache", required=True)
    parser.add_argument("--patch-params", default=None)
    parser.add_argument("--num-videos", type=int, default=5)
    parser.add_argument("--duration", type=int, default=2)
    parser.add_argument("--compact", action="store_true")
    parser.add_argument(
        "--patch-temp-mode",
        choices=[
            "same_grid",
            "same_grid_lag1",
            "same_grid_multilag",
            "same_grid_second_order",
            "same_grid_multilag_second_order",
            "motion_hard",
            "motion_soft",
        ],
        default="motion_soft",
    )
    parser.add_argument("--match-radius", type=int, default=2)
    parser.add_argument("--top-m", type=int, default=4)
    parser.add_argument("--temperature", type=float, default=0.07)
    parser.add_argument("--lambda-dist", type=float, default=0.01)
    parser.add_argument("--output-dir", default=str(ROOT / "debug_outputs"))
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    samples = load_samples_from_cache(args.csv, args.patch_cache, args.duration, args.compact, args.num_videos)
    grid_size = samples[0]["grid_size"]

    report = [
        "# Patch Matching Sanity Report",
        "",
        f"- csv: `{args.csv}`",
        f"- patch_cache: `{args.patch_cache}`",
        f"- patch_params: `{args.patch_params}`",
        f"- num_videos: {len(samples)}",
        f"- patch_temp_mode: `{args.patch_temp_mode}`",
        f"- grid_size: {grid_size}",
        f"- match_radius: {args.match_radius}",
        f"- top_m: {args.top_m}",
        f"- temperature: {args.temperature}",
        f"- lambda_dist: {args.lambda_dist}",
        "",
    ]

    lines, radius_ok = check_radius_zero(samples)
    report.extend(lines)
    report.extend(check_windows(grid_size, radius=1))
    report.extend(check_param_config(args))
    report.extend(check_score_direction(args))
    if args.patch_temp_mode in {"motion_hard", "motion_soft"}:
        lines, examples = collect_matching_stats(samples, args)
        report.extend(lines)
    else:
        examples = []
        report.extend(["## Check 5/6/7: matching score, softmax, displacement", "", "Skipped for same_grid.", ""])

    if not radius_ok:
        report.extend(["## Overall", "", "FAIL: radius=0 is not equivalent to same_grid. Stop motion experiments until fixed.", ""])
    else:
        report.extend(["## Overall", "", "PASS for radius=0 equivalence. Review score direction and matching statistics above before running larger experiments.", ""])

    report_path = output_dir / "patch_matching_sanity_report.md"
    report_path.write_text("\n".join(report), encoding="utf-8")

    examples_path = output_dir / "matching_examples.csv"
    with examples_path.open("w", newline="", encoding="utf-8") as f:
        if examples:
            writer = csv.DictWriter(f, fieldnames=list(examples[0].keys()))
            writer.writeheader()
            writer.writerows(examples)
        else:
            f.write("")

    print(f"Saved sanity report -> {report_path}")
    print(f"Saved matching examples -> {examples_path}")


if __name__ == "__main__":
    main()
