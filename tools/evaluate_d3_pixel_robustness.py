#!/usr/bin/env python3
"""Run bounded JPEG/resize D3 robustness checks by re-encoding source pixels."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
from PIL import Image


REPO_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = REPO_ROOT.parent
SRC_DIR = REPO_ROOT / "src"
TOOLS_DIR = REPO_ROOT / "tools"
for path in (SRC_DIR, TOOLS_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from build_multi_order_baselines import (  # noqa: E402
    KEY_COLUMNS,
    _load_index,
    d3_statistics,
    dataset_specs,
    empirical_cdf,
    two_sided_realness,
)
from alpha_stalled.backbone import load_dinov3_model  # noqa: E402
from alpha_stalled.metrics import auc_ap as _auc_ap, pairwise_frames  # noqa: E402
from alpha_stalled.video_io import load_video_frames  # noqa: E402


CONDITIONS = ("reference_pixels", "jpeg_q30", "resize_half")


def perturb_frames(frames: np.ndarray, condition: str) -> np.ndarray:
    if condition == "reference_pixels":
        return frames
    if condition == "jpeg_q30":
        output = []
        for frame in frames:
            ok, encoded = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 30])
            if not ok:
                raise ValueError("OpenCV JPEG encoding failed")
            decoded = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
            if decoded is None:
                raise ValueError("OpenCV JPEG decoding failed")
            output.append(decoded)
        return np.stack(output)
    if condition == "resize_half":
        output = []
        for frame in frames:
            height, width = frame.shape[:2]
            output.append(
                cv2.resize(
                    frame,
                    (max(2, width // 2), max(2, height // 2)),
                    interpolation=cv2.INTER_AREA,
                )
            )
        return np.stack(output)
    raise ValueError(f"unknown condition: {condition}")


def resolve_video_path(value: object) -> Path:
    path = Path(str(value))
    if path.is_absolute():
        return path
    project_path = PROJECT_ROOT / path
    if project_path.is_file():
        return project_path
    repo_path = REPO_ROOT / path
    if repo_path.is_file():
        return repo_path
    raise FileNotFoundError(str(path))


def select_eval_rows(
    index: pd.DataFrame,
    eligible: pd.DataFrame,
    real_count: int,
    fake_per_generator: int,
    seed: int,
) -> pd.DataFrame:
    frame = index.merge(eligible[KEY_COLUMNS], on=KEY_COLUMNS, how="inner", validate="one_to_one")
    selected = [
        frame[frame["subset"] == "real"].sample(
            n=min(real_count, int(frame["subset"].eq("real").sum())),
            random_state=seed,
        )
    ]
    for offset, (_, group) in enumerate(
        frame[frame["subset"] == "annotated"].groupby("source_model", sort=True), start=1
    ):
        selected.append(
            group.sample(n=min(fake_per_generator, len(group)), random_state=seed + offset)
        )
    return pd.concat(selected, ignore_index=True)


def embed_frames(model, transform, frames: np.ndarray, device: torch.device, batch_size: int) -> np.ndarray:
    output = []
    with torch.inference_mode():
        for start in range(0, len(frames), batch_size):
            tensors = [
                transform(Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)))
                for frame in frames[start : start + batch_size]
            ]
            embeddings = model(torch.stack(tensors).to(device))
            output.append(embeddings.detach().cpu().numpy())
    return np.concatenate(output, axis=0).astype(np.float32, copy=False)


def score_rows(
    index: pd.DataFrame,
    model,
    transform,
    device: torch.device,
    frame_batch_size: int,
) -> pd.DataFrame:
    rows = []
    for count, (_, row) in enumerate(index.iterrows(), start=1):
        base = {column: str(row[column]) for column in KEY_COLUMNS}
        try:
            window = [int(value) for value in json.loads(str(row["2_sec_idxs"]))]
            if len(window) != 16:
                raise ValueError(f"expected 16 frames, found {len(window)}")
            frames = load_video_frames(str(resolve_video_path(row["video_path"])), window)
            if len(frames) != len(window):
                raise ValueError(f"decoded {len(frames)}/{len(window)} frames")
            timestamps = np.asarray(window, dtype=np.float64) / float(row["fps"])
            for condition in CONDITIONS:
                variant = perturb_frames(frames, condition)
                embeddings = embed_frames(model, transform, variant, device, frame_batch_size)
                stats = d3_statistics(embeddings, timestamps)
                rows.append({**base, "condition": condition, "status": "ok", **stats})
        except Exception as exc:
            for condition in CONDITIONS:
                rows.append(
                    {
                        **base,
                        "condition": condition,
                        "status": "failed",
                        "error": f"{type(exc).__name__}: {exc}",
                    }
                )
        if count % 25 == 0 or count == len(index):
            print(f"pixel robustness scored {count}/{len(index)} videos", flush=True)
    return pd.DataFrame(rows)


def calibrate(eval_scores: pd.DataFrame, calibration: pd.DataFrame) -> pd.DataFrame:
    frames = []
    for condition in CONDITIONS:
        current = eval_scores[
            (eval_scores["condition"] == condition) & (eval_scores["status"] == "ok")
        ].copy()
        calib = calibration[
            (calibration["condition"] == condition) & (calibration["status"] == "ok")
        ]["d3_raw"].to_numpy(dtype=np.float64)
        current["d3_one_sided"] = empirical_cdf(current["d3_raw"].to_numpy(), calib)
        current["d3_two_sided"] = two_sided_realness(current["d3_raw"].to_numpy(), calib)
        frames.append(current)
    return pd.concat(frames, ignore_index=True)


def metric_tables(scores: pd.DataFrame, seed: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    for (dataset, condition), frame in scores.groupby(["dataset", "condition"], sort=False):
        for score_name in ("d3_raw", "d3_one_sided", "d3_two_sided"):
            for generator, pair in pairwise_frames(frame, seed).items():
                auc, ap = _auc_ap(pair, score_name)
                rows.append(
                    {
                        "dataset": dataset,
                        "condition": condition,
                        "score": score_name,
                        "generator": generator,
                        "n": len(pair),
                        "auc": auc,
                        "ap": ap,
                    }
                )
    generator = pd.DataFrame(rows)
    dataset = (
        generator.groupby(["dataset", "condition", "score"], as_index=False)
        .agg(n_generators=("generator", "nunique"), auc=("auc", "mean"), ap=("ap", "mean"))
    )
    return dataset, generator


def stability_table(scores: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for dataset, frame in scores.groupby("dataset", sort=False):
        for score in ("d3_raw", "d3_one_sided", "d3_two_sided"):
            wide = frame.pivot(index=KEY_COLUMNS, columns="condition", values=score)
            reference = wide["reference_pixels"]
            for condition in CONDITIONS[1:]:
                variant = wide[condition]
                rows.append(
                    {
                        "dataset": dataset,
                        "condition": condition,
                        "score": score,
                        "n": int((reference.notna() & variant.notna()).sum()),
                        "pearson_r": float(reference.corr(variant)),
                        "mean_absolute_change": float(np.nanmean(np.abs(reference - variant))),
                    }
                )
    return pd.DataFrame(rows)


def write_report(path: Path, metrics: pd.DataFrame, stability: pd.DataFrame, failures: pd.DataFrame) -> None:
    lines = [
        "# Pixel-domain D3 robustness",
        "",
        "This bounded operator-controlled experiment decodes the exact indexed 2 s/16-frame windows and reruns the same DINOv3-L encoder. Evaluation uses 100 real videos and up to 20 videos per generator; calibration uses the disjoint 200-real split. `jpeg_q30` applies an in-memory JPEG encode/decode and `resize_half` downsamples each input dimension by two before the standard 224 x 224 encoder transform.",
        "",
        "## Dataset macro metrics",
        "",
        "| Dataset | Condition | Score | AUC | AP |",
        "|---|---|---|---:|---:|",
    ]
    for row in metrics.itertuples(index=False):
        lines.append(f"| {row.dataset} | {row.condition} | {row.score} | {row.auc:.4f} | {row.ap:.4f} |")
    lines.extend(
        [
            "",
            "## Stability against decoded reference",
            "",
            "| Dataset | Condition | Score | n | Pearson r | Mean absolute change |",
            "|---|---|---|---:|---:|---:|",
        ]
    )
    for row in stability.itertuples(index=False):
        lines.append(
            f"| {row.dataset} | {row.condition} | {row.score} | {row.n} | "
            f"{row.pearson_r:.4f} | {row.mean_absolute_change:.4f} |"
        )
    lines.extend(
        [
            "",
            "## Integrity",
            "",
            f"- Failed condition rows: `{len(failures)}`.",
            "- This is a declared robustness subset, not a replacement for the full strict-protocol result.",
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
        "--output-dir", type=Path, default=REPO_ROOT / "results/multi_order_baselines"
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--frame-batch-size", type=int, default=64)
    parser.add_argument("--real-count", type=int, default=100)
    parser.add_argument("--fake-per-generator", type=int, default=20)
    parser.add_argument("--calibration-real", type=int, default=200)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    stage1 = pd.read_csv(args.stage1_scores)
    device = torch.device(args.device)
    model, transform = load_dinov3_model(args.device)
    all_scores = []
    all_failures = []
    for spec in dataset_specs(REPO_ROOT):
        eligible = stage1[stage1["dataset"] == spec.name]
        eval_index = select_eval_rows(
            _load_index(spec.eval_index),
            eligible,
            args.real_count,
            args.fake_per_generator,
            args.seed,
        )
        calib_index = _load_index(spec.d3_calib_index)
        if len(calib_index) > args.calibration_real:
            calib_index = calib_index.sample(n=args.calibration_real, random_state=args.seed)
        print(
            f"[{spec.name}] pixel eval={len(eval_index)} calibration={len(calib_index)}",
            flush=True,
        )
        eval_scores = score_rows(eval_index, model, transform, device, args.frame_batch_size)
        calibration = score_rows(calib_index, model, transform, device, args.frame_batch_size)
        failures = pd.concat(
            [
                eval_scores[eval_scores["status"] != "ok"].assign(split="eval"),
                calibration[calibration["status"] != "ok"].assign(split="calibration"),
            ],
            ignore_index=True,
        )
        if not failures.empty:
            failures.insert(0, "dataset", spec.name)
            all_failures.append(failures)
        calibrated = calibrate(eval_scores, calibration)
        calibrated.insert(0, "dataset", spec.name)
        all_scores.append(calibrated)

    scores = pd.concat(all_scores, ignore_index=True)
    failures = pd.concat(all_failures, ignore_index=True) if all_failures else pd.DataFrame()
    metrics, generator = metric_tables(scores, args.seed)
    stability = stability_table(scores)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    scores.to_csv(args.output_dir / "d3_pixel_robustness_per_video_scores.csv", index=False)
    metrics.to_csv(args.output_dir / "d3_pixel_robustness_dataset_metrics.csv", index=False)
    generator.to_csv(args.output_dir / "d3_pixel_robustness_generator_metrics.csv", index=False)
    stability.to_csv(args.output_dir / "d3_pixel_robustness_stability.csv", index=False)
    failures.to_csv(args.output_dir / "d3_pixel_robustness_failures.csv", index=False)
    write_report(
        args.output_dir / "d3_pixel_robustness_analysis.md", metrics, stability, failures
    )


if __name__ == "__main__":
    main()
