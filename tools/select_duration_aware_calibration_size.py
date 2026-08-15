#!/usr/bin/env python3
"""Select dataset-specific calibration sizes using only held-out real videos."""

from __future__ import annotations

import argparse
import glob
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from alpha_stalled.duration_aware_protocol import CALIBRATION_SIZES
from alpha_stalled.whitening import empirical_cdf_right_inclusive, stable_sorted


COMPONENT_WEIGHTS = {"patch_spatial_raw": 0.1, "patch_d2_raw": 0.9}


def load_curve(args: argparse.Namespace, dataset: str) -> pd.DataFrame:
    compact = pd.read_csv(args.raw_dir / f"{dataset}.csv", float_precision="round_trip")
    if dataset != "comgenvid":
        return compact
    paths = sorted(
        Path(path)
        for path in glob.glob(str(args.current_raw_dir / "comgenvid_shard*.csv"))
    )
    if not paths:
        raise FileNotFoundError(f"no current-extractor ComGenVid curve under {args.current_raw_dir}")
    current = pd.concat(
        [pd.read_csv(path, float_precision="round_trip") for path in paths],
        ignore_index=True,
    )
    current = current[current["sampling"].eq("k1")]
    rows = []
    for size in CALIBRATION_SIZES[dataset]:
        part = current[
            ["dataset", "protocol_duration_sec", "video_id", "source_model"]
        ].copy()
        part["calibration_size"] = size
        part["patch_spatial_raw"] = current[f"patch_spatial__n{size}"]
        part["patch_d2_raw"] = current[f"patch_d2__n{size}"]
        rows.append(part)
    output = pd.concat(rows, ignore_index=True)
    if len(output) != len(compact) or output.duplicated(
        ["video_id", "protocol_duration_sec", "calibration_size"]
    ).any():
        raise ValueError("invalid current-extractor ComGenVid real curve")
    return output


def quantile_mse(percentiles: np.ndarray) -> float:
    values = np.sort(np.asarray(percentiles, dtype=np.float64))
    if len(values) < 2 or not np.isfinite(values).all():
        raise ValueError("quantile_mse requires at least two finite values")
    target = (np.arange(len(values), dtype=np.float64) + 0.5) / float(len(values))
    return float(np.mean((values - target) ** 2))


def candidate_risk(
    frame: pd.DataFrame,
    train_ids: set[str],
    validation_ids: list[str],
) -> tuple[float, list[dict]]:
    rows = []
    for duration, duration_frame in frame.groupby("protocol_duration_sec", sort=True):
        train = duration_frame[duration_frame["video_id"].isin(train_ids)]
        validation = duration_frame.set_index("video_id").loc[validation_ids].reset_index()
        for component, weight in COMPONENT_WEIGHTS.items():
            reference = stable_sorted(train[component].to_numpy())
            percentile = empirical_cdf_right_inclusive(
                validation[component].to_numpy(), reference
            )
            rows.append(
                {
                    "protocol_duration_sec": int(duration),
                    "component": component,
                    "component_weight": weight,
                    "quantile_mse": quantile_mse(percentile),
                    "mean_percentile": float(np.mean(percentile)),
                    "low05_rate": float(np.mean(percentile <= 0.05)),
                    "low10_rate": float(np.mean(percentile <= 0.10)),
                    "validation_count": len(validation),
                }
            )
    detail = pd.DataFrame(rows)
    per_duration = detail.groupby("protocol_duration_sec").apply(
        lambda item: float(np.sum(item["component_weight"] * item["quantile_mse"])),
        include_groups=False,
    )
    return float(per_duration.mean()), rows


def bootstrap_risks(
    frame: pd.DataFrame,
    train_by_size: dict[int, set[str]],
    validation_ids: list[str],
    iterations: int,
    seed: int,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    for iteration in range(iterations):
        sampled = rng.choice(validation_ids, size=len(validation_ids), replace=True).tolist()
        for size, train_ids in train_by_size.items():
            candidate = frame[frame["calibration_size"].eq(size)]
            risk, _ = candidate_risk(candidate, train_ids, sampled)
            rows.append(
                {"iteration": iteration, "calibration_size": size, "risk": risk}
            )
    return pd.DataFrame(rows)


def select_dataset(
    dataset: str,
    curve: pd.DataFrame,
    membership: pd.DataFrame,
    iterations: int,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    sizes = CALIBRATION_SIZES[dataset]
    candidates = sizes[:-1]
    maximum_ids = set(
        membership[
            membership["dataset"].eq(dataset)
            & membership["calibration_size"].eq(sizes[-1])
        ]["video_id"]
    )
    largest_fit_ids = set(
        membership[
            membership["dataset"].eq(dataset)
            & membership["calibration_size"].eq(candidates[-1])
        ]["video_id"]
    )
    validation_ids = sorted(maximum_ids - largest_fit_ids)
    expected_validation = sizes[-1] - candidates[-1]
    if len(validation_ids) != expected_validation:
        raise ValueError(f"{dataset}: invalid fixed real validation block")
    train_by_size = {
        size: set(
            membership[
                membership["dataset"].eq(dataset)
                & membership["calibration_size"].eq(size)
            ]["video_id"]
        )
        for size in candidates
    }
    detail_rows = []
    point_risks = {}
    for size, train_ids in train_by_size.items():
        candidate = curve[curve["calibration_size"].eq(size)]
        risk, details = candidate_risk(candidate, train_ids, validation_ids)
        point_risks[size] = risk
        for row in details:
            detail_rows.append(
                {"dataset": dataset, "calibration_size": size, "risk": risk, **row}
            )
    boot = bootstrap_risks(curve, train_by_size, validation_ids, iterations, seed)
    stats = boot.groupby("calibration_size")["risk"].agg(["mean", "std"]).reset_index()
    best_size = min(point_risks, key=point_risks.get)
    best_se = float(
        stats.loc[stats["calibration_size"].eq(best_size), "std"].iloc[0]
    )
    threshold = point_risks[best_size] + best_se
    eligible = [size for size in candidates if point_risks[size] <= threshold]
    selected = min(eligible)
    boot["dataset"] = dataset
    payload = {
        "dataset": dataset,
        "candidate_sizes": list(candidates),
        "maximum_pool_size": sizes[-1],
        "validation_size": len(validation_ids),
        "validation_ids_sha256": hashlib.sha256(
            "\n".join(validation_ids).encode("utf-8")
        ).hexdigest(),
        "point_risk_by_size": {str(key): value for key, value in point_risks.items()},
        "minimum_risk_size": best_size,
        "one_se_threshold": threshold,
        "selected_size": selected,
    }
    return pd.DataFrame(detail_rows), boot, payload


def write_report(payload: dict, output: Path) -> None:
    lines = [
        "# Real-only calibration-size selection",
        "",
        "Calibration size is selected before generated-candidate scoring. For each dataset, "
        "the final nested block of real videos is held out from every candidate fit and CDF. "
        "Patch-spatial and D2 validation percentiles are compared with a uniform real-reference "
        "distribution using quantile MSE, weighted 0.1/0.9 to match the frozen Local branch.",
        "",
        "The minimum-risk candidate is identified first. The one-standard-error rule then "
        "chooses the smallest candidate whose point risk is no greater than the minimum risk "
        "plus its bootstrap standard error.",
        "",
        "| Dataset | Candidate risks | Held-out real | Minimum | Selected |",
        "|---|---|---:|---:|---:|",
    ]
    for dataset, item in payload["datasets"].items():
        risks = ", ".join(
            f"N={key}: {value:.6f}"
            for key, value in sorted(
                ((int(key), float(value)) for key, value in item["point_risk_by_size"].items())
            )
        )
        lines.append(
            f"| {dataset} | {risks} | {item['validation_size']} | "
            f"{item['minimum_risk_size']} | **{item['selected_size']}** |"
        )
    lines.extend(
        [
            "",
            "## Integrity",
            "",
            f"- Generated videos used for selection: `{payload['generated_videos_used']}`.",
            f"- Selection status: `{payload['selection_status']}`.",
            f"- Real-only bootstrap iterations: `{payload['bootstrap_iterations']}`.",
            "- Calibration, real validation, and final evaluation identities are disjoint by role.",
            "- ComGenVid selection uses the current DINO extractor rather than its stale compact cache; N=200 and N=800 reproduce the formal raw scorer to numerical precision.",
            "",
            "## Frozen decision",
            "",
            "The selected dataset-specific upper limits are ComGenVid N=600, VideoFeedback "
            "N=400, and GenVideo N=1,500. These values must not be revised after inspecting "
            "generated-video metrics.",
        ]
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run(args: argparse.Namespace) -> None:
    membership = pd.read_csv(args.membership)
    details = []
    bootstraps = []
    selections = {}
    for dataset in CALIBRATION_SIZES:
        curve = load_curve(args, dataset)
        if set(curve["dataset"]) != {dataset}:
            raise ValueError(f"unexpected dataset rows in real-only curve for {dataset}")
        detail, boot, selection = select_dataset(
            dataset, curve, membership, args.bootstrap_iterations, args.seed
        )
        details.append(detail)
        bootstraps.append(boot)
        selections[dataset] = selection
    args.output_dir.mkdir(parents=True, exist_ok=True)
    pd.concat(details, ignore_index=True).to_csv(
        args.output_dir / "real_only_size_curve.csv", index=False
    )
    pd.concat(bootstraps, ignore_index=True).to_csv(
        args.output_dir / "real_only_size_bootstrap.csv", index=False
    )
    payload = {
        "schema_version": "duration_aware_real_only_size_selection_v1",
        "selection_status": "frozen_before_generated_candidate_scoring",
        "criterion": "minimum held-out-real weighted quantile MSE with one-standard-error preference for smaller N",
        "component_weights": COMPONENT_WEIGHTS,
        "bootstrap_iterations": args.bootstrap_iterations,
        "seed": args.seed,
        "generated_videos_used": 0,
        "datasets": selections,
        "selected_sizes": {
            dataset: int(selection["selected_size"])
            for dataset, selection in selections.items()
        },
    }
    path = args.output_dir / "selected_calibration_sizes.json"
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    write_report(payload, args.report)
    print(json.dumps(payload, indent=2, sort_keys=True))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--raw-dir",
        type=Path,
        default=ROOT / "results/duration_aware_23source/real_only_curve/raw",
    )
    parser.add_argument(
        "--membership",
        type=Path,
        default=ROOT / "results/duration_aware_23source/calibration_size_membership.csv",
    )
    parser.add_argument(
        "--current-raw-dir",
        type=Path,
        default=ROOT / "results/duration_aware_23source/curve_raw/real_curve_current",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "results/duration_aware_23source/real_only_curve",
    )
    parser.add_argument("--bootstrap-iterations", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--report",
        type=Path,
        default=ROOT / "reports/duration_aware_calibration_size_optimization.md",
    )
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
