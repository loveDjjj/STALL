#!/usr/bin/env python3
"""Fit and evaluate leakage-free L0-L3 local D2 variants on three datasets."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
TOOLS_DIR = REPO_ROOT / "tools"
SRC_DIR = REPO_ROOT / "src"
if str(TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(TOOLS_DIR))
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from build_multi_order_baselines import metric_tables, paired_bootstrap
from eval_patch_fast import FastPatchScorer, iter_cache_jobs, load_cache_batch


KEY_COLUMNS = ["subset", "source_model", "filename"]
LOCAL_NAMES = {
    "Ls": "Local spatial",
    "L0": "Same-grid D2",
    "L1": "Global-residual D2",
    "L2": "Spatial-median residual D2",
    "L3": "Persistent same-grid D2",
    "P0": "Leakage-free spatial + D2",
}
LOCAL_COLUMNS = tuple(LOCAL_NAMES)
COMPARISONS = (
    ("L1", "L0", "global_residual"),
    ("L2", "L0", "spatial_median_residual"),
    ("L3", "L0", "persistent_aggregation"),
    ("P0", "L0", "add_local_spatial"),
)


@dataclass(frozen=True)
class LocalDatasetSpec:
    name: str
    calib_index: Path
    eval_index: Path
    patch_cache: Path
    aggregation: str
    bottomk_ratio: float
    patch_region_size: int


@dataclass(frozen=True)
class LocalVariant:
    name: str
    mode: str
    persistent: bool = False


def dataset_specs(root: Path) -> tuple[LocalDatasetSpec, ...]:
    return (
        LocalDatasetSpec(
            "comgenvid",
            root / "cache/indexes/comgenvid_calib_real200.csv",
            root / "cache/indexes/comgenvid_eval_holdout_real900_all_fake.csv",
            root / "cache/patch_embeddings/comgenvid",
            "bottomk_mean",
            0.2,
            3,
        ),
        LocalDatasetSpec(
            "videofeedback",
            root / "cache/indexes/videofeedback_small_calib_real200.csv",
            root / "cache/indexes/videofeedback_small_eval_holdout_real500_fake300permodel.csv",
            root / "cache/patch_embeddings/videofeedback",
            "mean",
            0.5,
            1,
        ),
        LocalDatasetSpec(
            "genvideo",
            root / "cache/indexes/genvideo_calib_real200.csv",
            root / "cache/indexes/genvideo_eval_holdout_real7984_all_fake.csv",
            root / "cache/patch_embeddings/genvideo",
            "mean",
            0.5,
            2,
        ),
    )


VARIANTS = (
    LocalVariant("L0", "same_grid_second_order"),
    LocalVariant("L1", "global_residual_second_order"),
    LocalVariant("L2", "spatial_median_residual_second_order"),
    LocalVariant("L3", "same_grid_second_order", persistent=True),
)


def _with_filename(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    out["filename"] = out["video_path"].map(lambda value: Path(str(value)).name)
    for column in KEY_COLUMNS:
        out[column] = out[column].astype(str)
    return out


def build_strict_eval_index(spec: LocalDatasetSpec, stage1_scores: pd.DataFrame) -> pd.DataFrame:
    index = _with_filename(pd.read_csv(spec.eval_index))
    keys = stage1_scores[stage1_scores["dataset"] == spec.name][KEY_COLUMNS]
    merged = index.merge(keys, on=KEY_COLUMNS, how="inner", validate="one_to_one")
    if len(merged) != len(keys):
        raise ValueError(f"{spec.name}: failed to recover every Stage-1 protocol row")
    return merged.drop(columns=["filename"])


def _run(command: list[str], cwd: Path) -> None:
    print("+ " + " ".join(command), flush=True)
    subprocess.run(command, cwd=cwd, check=True)


def fit_variant(
    spec: LocalDatasetSpec,
    variant: LocalVariant,
    work_dir: Path,
) -> Path:
    params = work_dir / f"{spec.name}_{variant.name}.npz"
    aggregation = "temporal_run_bottomk_mean" if variant.persistent else spec.aggregation
    bottomk = 0.2 if variant.persistent else spec.bottomk_ratio
    run_length = 2 if variant.persistent else 3

    if params.exists():
        payload = np.load(params, allow_pickle=True)
        config = json.loads(str(payload["aggregation_config"].item()))
        if (
            config.get("patch_temp_mode") == variant.mode
            and int(config.get("patch_region_size", 1)) == spec.patch_region_size
            and config.get("mode") == aggregation
        ):
            print(f"[{spec.name}] reuse checkpoint {params}", flush=True)
            return params
        raise ValueError(f"checkpoint config mismatch: {params}")

    _run(
        [
            sys.executable,
            "src/create_patch_params.py",
            "--csv",
            str(spec.calib_index),
            "--patch-emb-cache",
            str(spec.patch_cache),
            "--output",
            str(params),
            "--duration",
            "2",
            "--compact",
            "--real-only",
            "--max-patches-for-fit",
            "300000",
            "--aggregation",
            aggregation,
            "--bottomk-ratio",
            str(bottomk),
            "--temporal-run-length",
            str(run_length),
            "--seed",
            "42",
            "--patch-temp-mode",
            variant.mode,
            "--patch-region-size",
            str(spec.patch_region_size),
        ],
        REPO_ROOT,
    )
    return params


def score_dataset(
    spec: LocalDatasetSpec,
    eval_csv: Path,
    params_by_variant: dict[str, Path],
    score_device: str,
    batch_size: int = 32,
) -> pd.DataFrame:
    scorers = {
        variant.name: FastPatchScorer(str(params_by_variant[variant.name]), device=score_device)
        for variant in VARIANTS
    }
    jobs = list(iter_cache_jobs(str(eval_csv), str(spec.patch_cache), 2, True, None))
    rows = []
    print(
        f"[{spec.name}] one-pass scoring: videos={len(jobs)} variants={len(VARIANTS)}",
        flush=True,
    )
    for start in range(0, len(jobs), batch_size):
        batch_jobs = jobs[start : start + batch_size]
        patch_batch, global_batch, grid_size = load_cache_batch(batch_jobs, include_global=True)
        for variant in VARIANTS:
            scorer = scorers[variant.name]
            if grid_size != scorer.patch_grid_size:
                raise ValueError(
                    f"{spec.name}/{variant.name}: params grid={scorer.patch_grid_size}, cache grid={grid_size}"
                )
            aggregation = scorer.aggregation_config.get("mode", "bottomk_mean")
            scores = scorer.score_batch(
                patch_batch,
                patch_temp_mode=variant.mode,
                patch_spat_weight=0.0,
                patch_temp_weight=1.0,
                aggregation=aggregation,
                bottomk_ratio=scorer.params_bottomk_ratio,
                temporal_run_length=scorer.params_temporal_run_length,
                patch_region_size=scorer.params_patch_region_size,
                global_batch=global_batch,
            )
            for i, job in enumerate(batch_jobs):
                rows.append(
                    {
                        "dataset": spec.name,
                        "subset": job["subset"],
                        "source_model": job["source_model"],
                        "filename": job["filename"],
                        "variant": variant.name,
                        "patch_spat_percentile": float(scores["patch_spat_percentile"][i]),
                        "patch_temp_percentile": float(scores["patch_temp_percentile"][i]),
                    }
                )
        if start == 0 or (start // batch_size + 1) % 50 == 0:
            print(
                f"[{spec.name}] scored {min(start + batch_size, len(jobs))}/{len(jobs)}",
                flush=True,
            )
    for scorer in scorers.values():
        if scorer._executor is not None:
            scorer._executor.shutdown(wait=True)
    return pd.DataFrame(rows)


def to_wide(long_scores: pd.DataFrame) -> pd.DataFrame:
    key = ["dataset", *KEY_COLUMNS]
    temporal = long_scores.pivot(index=key, columns="variant", values="patch_temp_percentile")
    spatial = (
        long_scores[long_scores["variant"] == "L0"]
        .set_index(key)["patch_spat_percentile"]
        .rename("Ls")
    )
    out = temporal.join(spatial).reset_index()
    out["P0"] = 0.1 * out["Ls"] + 0.9 * out["L0"]
    missing = set(LOCAL_COLUMNS).difference(out.columns)
    if missing:
        raise ValueError(f"wide local score table missing columns: {sorted(missing)}")
    return out


def generator_deltas(generator_metrics: pd.DataFrame) -> pd.DataFrame:
    base = generator_metrics[generator_metrics["config"] == "L0"].set_index(
        ["dataset", "generator"]
    )
    rows = []
    for config in ("L1", "L2", "L3", "P0"):
        current = generator_metrics[generator_metrics["config"] == config].set_index(
            ["dataset", "generator"]
        )
        for key in current.index:
            rows.append(
                {
                    "dataset": key[0],
                    "generator": key[1],
                    "config": config,
                    "delta_auc_vs_L0": float(current.loc[key, "auc"] - base.loc[key, "auc"]),
                    "delta_ap_vs_L0": float(current.loc[key, "ap"] - base.loc[key, "ap"]),
                }
            )
    return pd.DataFrame(rows)


def write_analysis(
    path: Path,
    dataset_metrics: pd.DataFrame,
    deltas: pd.DataFrame,
    bootstrap: pd.DataFrame,
) -> None:
    metrics = dataset_metrics.pivot(index="config", columns="dataset", values=["auc", "ap"])
    lines = [
        "# Local D2 residual analysis",
        "",
        "## Protocol",
        "",
        "Every local model is refitted on the same dataset-specific, disjoint 200-real calibration index and evaluated on the exact Stage-1 strict 2 s intersection. L0-L2 preserve each dataset's frozen region pooling and aggregation. L3 changes only aggregation to a length-2 temporal-run bottom-20% mean.",
        "",
        "- L0: normalized same-grid patch D2.",
        "- L1: normalized `(patch D2 - global CLS D2)`.",
        "- L2: normalized `(patch D2 - spatial median patch D2)`.",
        "- L3: L0 with persistent temporal aggregation.",
        "- P0: leakage-free `0.1 local spatial + 0.9 L0`.",
        "",
        "## Dataset macro metrics",
        "",
        "| ID | Configuration | ComGenVid AUC/AP | VideoFeedback AUC/AP | GenVideo AUC/AP | Macro-3 AUC/AP |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for config in LOCAL_COLUMNS:
        cells = []
        for dataset in ("comgenvid", "videofeedback", "genvideo", "Macro-3"):
            cells.append(
                f"{metrics.loc[config, ('auc', dataset)]:.4f}/{metrics.loc[config, ('ap', dataset)]:.4f}"
            )
        lines.append(f"| {config} | {LOCAL_NAMES[config]} | " + " | ".join(cells) + " |")

    lines.extend(["", "## Cross-generator wins versus L0", ""])
    for config in ("L1", "L2", "L3", "P0"):
        selected = deltas[deltas["config"] == config]
        lines.append(
            f"- `{config}`: AP improves on {(selected.delta_ap_vs_L0 > 0).sum()}/{len(selected)} generators; AUC improves on {(selected.delta_auc_vs_L0 > 0).sum()}/{len(selected)}."
        )

    lines.extend(
        [
            "",
            "## Paired bootstrap",
            "",
            "| Dataset | Variant | Metric | Delta | 95% CI |",
            "|---|---|---|---:|---:|",
        ]
    )
    for row in bootstrap.itertuples(index=False):
        lines.append(
            f"| {row.dataset} | {row.new_config}-{row.base_config} | {row.metric.upper()} | "
            f"{row.delta:+.4f} | [{row.ci95_low:+.4f}, {row.ci95_high:+.4f}] |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=REPO_ROOT)
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
    parser.add_argument("--score-device", choices=["cuda", "cpu"], default="cuda")
    parser.add_argument("--bootstrap-iterations", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--work-dir",
        type=Path,
        default=Path("/tmp/alpha_stalled_local_d2_work"),
        help="Durable restart checkpoints; remove after final outputs are verified.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    stage1 = pd.read_csv(args.stage1_scores)
    long_frames = []
    work_root = args.work_dir
    work_root.mkdir(parents=True, exist_ok=True)
    for spec in dataset_specs(args.root):
        eval_csv = work_root / f"{spec.name}_strict_eval.csv"
        strict_index = build_strict_eval_index(spec, stage1)
        strict_index.to_csv(eval_csv, index=False)
        checkpoint = work_root / f"{spec.name}_long_scores.csv"
        if checkpoint.exists():
            frame = pd.read_csv(checkpoint)
            expected = len(strict_index) * len(VARIANTS)
            if len(frame) != expected:
                raise ValueError(
                    f"incomplete score checkpoint {checkpoint}: {len(frame)} != {expected}"
                )
            print(f"[{spec.name}] reuse score checkpoint {checkpoint}", flush=True)
            long_frames.append(frame)
            continue

        params_by_variant = {}
        for variant in VARIANTS:
            print(f"[{spec.name}] {variant.name}: {variant.mode}", flush=True)
            params_by_variant[variant.name] = fit_variant(
                spec,
                variant,
                work_root,
            )
        frame = score_dataset(
            spec,
            eval_csv,
            params_by_variant,
            args.score_device,
        )
        temp_checkpoint = checkpoint.with_suffix(".tmp.csv")
        frame.to_csv(temp_checkpoint, index=False)
        temp_checkpoint.replace(checkpoint)
        long_frames.append(frame)

    per_video = to_wide(pd.concat(long_frames, ignore_index=True))
    dataset_metrics, generator_metrics = metric_tables(
        per_video,
        args.seed,
        score_columns=LOCAL_COLUMNS,
        config_names=LOCAL_NAMES,
    )
    bootstrap = paired_bootstrap(
        per_video,
        args.seed,
        args.bootstrap_iterations,
        comparisons=COMPARISONS,
    )
    deltas = generator_deltas(generator_metrics)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    per_video.to_csv(args.output_dir / "local_d2_per_video_scores.csv", index=False)
    dataset_metrics.to_csv(args.output_dir / "local_d2_dataset_metrics.csv", index=False)
    generator_metrics.to_csv(args.output_dir / "local_d2_generator_metrics.csv", index=False)
    deltas.to_csv(args.output_dir / "local_d2_generator_deltas.csv", index=False)
    bootstrap.to_csv(args.output_dir / "local_d2_bootstrap_deltas.csv", index=False)
    write_analysis(
        args.output_dir / "local_d2_analysis.md",
        dataset_metrics,
        deltas,
        bootstrap,
    )


if __name__ == "__main__":
    main()
