#!/usr/bin/env python3
"""Evaluate U0 with 200 calibration real videos and all eligible remaining test real."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from sklearn.metrics import average_precision_score, roc_auc_score


ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from alpha_stalled.metrics import metric_tables
from alpha_stalled.parameters import global_references
from alpha_stalled.u0_calibration_experiments import (
    calibrate_candidate,
    candidate,
    load_parts,
)


DATASETS = ("comgenvid", "videofeedback", "genvideo")
SEEDS = (17, 29, 43)
RAW_REAL_COUNTS = {"comgenvid": 1_700, "videofeedback": 4_080, "genvideo": 9_984}
STRICT_ELIGIBLE_REAL_COUNTS = {
    "comgenvid": 1_698,
    "videofeedback": 4_080,
    "genvideo": 9_984,
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def pooled_metrics(frame: pd.DataFrame, score: str) -> tuple[float, float]:
    labels = frame["subset"].eq("real").astype(np.uint8).to_numpy()
    values = frame[score].to_numpy(dtype=np.float64)
    return float(roc_auc_score(labels, values)), float(average_precision_score(labels, values))


def summarize(metrics: pd.DataFrame) -> pd.DataFrame:
    return (
        metrics.groupby(["metric_protocol", "dataset"], as_index=False)
        .agg(
            auc_mean=("auc", "mean"),
            auc_std=("auc", "std"),
            ap_mean=("ap", "mean"),
            ap_std=("ap", "std"),
            n_real_min=("n_real", "min"),
            n_real_max=("n_real", "max"),
            n_fake_min=("n_fake", "min"),
            n_fake_max=("n_fake", "max"),
        )
        .sort_values(["metric_protocol", "dataset"])
    )


def run(args: argparse.Namespace) -> None:
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    evaluation = load_parts(args.input_dir, "evaluation")
    locked_calibration = load_parts(args.input_dir, "locked_calibration")
    remaining_real = load_parts(args.input_dir, "independent_remaining_real")
    reserve = load_parts(args.input_dir, "reserve")
    locked_calibration_k3 = locked_calibration[
        locked_calibration["sampling"].eq("k3")
    ].copy()
    reserve_k3 = reserve[reserve["sampling"].eq("k3")].copy()
    remaining_real_k3 = remaining_real[remaining_real["sampling"].eq("k3")].copy()
    membership = pd.read_csv(args.membership)
    reserve_manifest = json.loads(args.reserve_manifest.read_text(encoding="utf-8"))
    reserve_by_id = {item["video_id"]: item for item in reserve_manifest["videos"]}
    evaluation_manifest = json.loads(
        args.evaluation_manifest.read_text(encoding="utf-8")
    )
    evaluation_by_id = {
        item["video_id"]: item for item in evaluation_manifest["videos"]
    }
    locked_calibration_manifest = json.loads(
        args.locked_calibration_manifest.read_text(encoding="utf-8")
    )
    locked_calibration_by_id = {
        item["video_id"]: item
        for item in locked_calibration_manifest["videos"]
    }
    remaining_real_manifest = json.loads(
        args.remaining_real_manifest.read_text(encoding="utf-8")
    )
    remaining_real_by_id = {
        item["video_id"]: item for item in remaining_real_manifest["videos"]
    }
    global_spatial_ref, global_t1_ref = global_references(config)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.split_dir.mkdir(parents=True, exist_ok=True)
    per_video_parts: list[pd.DataFrame] = []
    split_rows: list[dict] = []
    pooled_rows: list[dict] = []
    paper_metric_parts: list[pd.DataFrame] = []
    generator_parts: list[pd.DataFrame] = []

    expected_reserve = {dataset: int(count) for dataset, count in reserve_k3.groupby("dataset")["video_id"].nunique().items()}
    expected_evaluation_real = {
        dataset: int(count)
        for dataset, count in evaluation[
            evaluation["subset"].eq("real")
        ].groupby("dataset")["video_id"].nunique().items()
    }
    expected_locked_calibration_real = {
        dataset: int(count)
        for dataset, count in locked_calibration_k3.groupby("dataset")[
            "video_id"
        ].nunique().items()
    }
    expected_remaining_real = {
        dataset: int(count)
        for dataset, count in remaining_real_k3.groupby("dataset")[
            "video_id"
        ].nunique().items()
    }
    for seed in SEEDS:
        seed_scores: list[pd.DataFrame] = []
        for dataset in DATASETS:
            name = candidate(seed, 200)
            selected_ids = set(
                membership[
                    membership["dataset"].eq(dataset)
                    & membership["seed"].eq(seed)
                    & membership["calibration_size"].eq(200)
                ]["video_id"].astype(str)
            )
            all_reserve_ids = set(
                reserve_k3[reserve_k3["dataset"].eq(dataset)]["video_id"].astype(str)
            )
            test_real_ids = all_reserve_ids - selected_ids
            if len(selected_ids) != 200:
                raise ValueError(f"{dataset}/seed={seed}: expected 200 calibration real videos")
            if selected_ids & test_real_ids:
                raise ValueError(f"{dataset}/seed={seed}: calibration/test overlap")
            if len(selected_ids | test_real_ids) != expected_reserve[dataset]:
                raise ValueError(f"{dataset}/seed={seed}: reserve partition is incomplete")

            evaluation_dataset = evaluation[evaluation["dataset"].eq(dataset)]
            locked_calibration_dataset = locked_calibration_k3[
                locked_calibration_k3["dataset"].eq(dataset)
            ]
            remaining_real_dataset = remaining_real_k3[
                remaining_real_k3["dataset"].eq(dataset)
            ]
            target = pd.concat(
                [
                    evaluation_dataset,
                    locked_calibration_dataset,
                    remaining_real_dataset,
                    reserve_k3[
                        reserve_k3["dataset"].eq(dataset)
                        & reserve_k3["video_id"].isin(test_real_ids)
                    ],
                ],
                ignore_index=True,
            )
            scored = calibrate_candidate(
                target,
                reserve[reserve["dataset"].eq(dataset)],
                selected_ids,
                name,
                global_spatial_ref,
                global_t1_ref,
            ).rename(columns={"S": "score"})
            metadata = target[
                [
                    "video_id",
                    "dataset",
                    "protocol_split",
                    "subset",
                    "source_model",
                    "filename",
                ]
            ].drop_duplicates("video_id")
            scored = metadata.merge(scored, on="video_id", validate="one_to_one")
            scored.insert(0, "seed", seed)
            seed_scores.append(scored)

            split_dir = args.split_dir / f"seed_{seed}"
            split_dir.mkdir(parents=True, exist_ok=True)
            split_path = split_dir / f"{dataset}.json"
            evaluation_test_real_ids = set(
                evaluation[
                    evaluation["dataset"].eq(dataset)
                    & evaluation["subset"].eq("real")
                ]["video_id"].astype(str)
            )
            historical_calibration_test_real_ids = set(
                locked_calibration_dataset["video_id"].astype(str)
            )
            supplemental_test_real_ids = set(
                remaining_real_dataset["video_id"].astype(str)
            )
            sources = (
                selected_ids,
                evaluation_test_real_ids,
                historical_calibration_test_real_ids,
                supplemental_test_real_ids,
                test_real_ids,
            )
            for left_index, left in enumerate(sources):
                for right in sources[left_index + 1 :]:
                    if left & right:
                        raise ValueError(
                            f"{dataset}/seed={seed}: real split sources overlap"
                        )
            all_test_real_ids = (
                evaluation_test_real_ids
                | historical_calibration_test_real_ids
                | supplemental_test_real_ids
                | test_real_ids
            )
            payload = {
                "schema_version": "u0_independent_all_eligible_complement_v2",
                "seed": seed,
                "dataset": dataset,
                "calibration_real_count": len(selected_ids),
                "test_real_count": len(all_test_real_ids),
                "raw_index_real_count": RAW_REAL_COUNTS[dataset],
                "strict_2s_eligible_real_count": STRICT_ELIGIBLE_REAL_COUNTS[dataset],
                "strict_2s_excluded_real_count": (
                    RAW_REAL_COUNTS[dataset] - STRICT_ELIGIBLE_REAL_COUNTS[dataset]
                ),
                "fixed_evaluation_test_real_count": len(evaluation_test_real_ids),
                "historical_calibration_test_real_count": len(
                    historical_calibration_test_real_ids
                ),
                "supplemental_test_real_count": len(supplemental_test_real_ids),
                "reserve_complement_test_real_count": len(test_real_ids),
                "generated_test_count": int(
                    evaluation_dataset[evaluation_dataset["subset"].eq("annotated")][
                        "video_id"
                    ].nunique()
                ),
                "generated_calibration_count": 0,
                "overlap_count": 0,
                "calibration_real": [reserve_by_id[value] for value in sorted(selected_ids)],
                "test_real": [
                    evaluation_by_id[value]
                    for value in sorted(evaluation_test_real_ids)
                ]
                + [
                    locked_calibration_by_id[value]
                    for value in sorted(historical_calibration_test_real_ids)
                ]
                + [
                    remaining_real_by_id[value]
                    for value in sorted(supplemental_test_real_ids)
                ]
                + [reserve_by_id[value] for value in sorted(test_real_ids)],
            }
            if payload["test_real_count"] != (
                expected_evaluation_real[dataset]
                + expected_locked_calibration_real[dataset]
                + expected_remaining_real.get(dataset, 0)
                + expected_reserve[dataset]
                - len(selected_ids)
            ):
                raise ValueError(f"{dataset}/seed={seed}: incomplete eligible test-real pool")
            if payload["calibration_real_count"] + payload["test_real_count"] != (
                STRICT_ELIGIBLE_REAL_COUNTS[dataset]
            ):
                raise ValueError(f"{dataset}/seed={seed}: strict real coverage mismatch")
            split_path.write_text(
                json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
            )
            split_rows.append(
                {
                    "seed": seed,
                    "dataset": dataset,
                    "calibration_real": len(selected_ids),
                    "test_real": len(all_test_real_ids),
                    "fixed_evaluation_test_real": len(evaluation_test_real_ids),
                    "historical_calibration_test_real": len(
                        historical_calibration_test_real_ids
                    ),
                    "supplemental_test_real": len(supplemental_test_real_ids),
                    "reserve_complement_test_real": len(test_real_ids),
                    "test_generated": payload["generated_test_count"],
                    "overlap": 0,
                    "split_file": str(split_path.relative_to(ROOT)),
                    "sha256": sha256(split_path),
                }
            )

        seed_frame = pd.concat(seed_scores, ignore_index=True)
        per_video_parts.append(seed_frame)

        seed_dataset_rows = []
        for dataset, frame in seed_frame.groupby("dataset", sort=False):
            auc, ap = pooled_metrics(frame, "score")
            counts = frame["subset"].value_counts()
            seed_dataset_rows.append(
                {
                    "metric_protocol": "pooled_all_generated",
                    "seed": seed,
                    "dataset": dataset,
                    "auc": auc,
                    "ap": ap,
                    "n_real": int(counts.get("real", 0)),
                    "n_fake": int(counts.get("annotated", 0)),
                }
            )
        macro_auc = float(np.mean([row["auc"] for row in seed_dataset_rows]))
        macro_ap = float(np.mean([row["ap"] for row in seed_dataset_rows]))
        pooled_rows.extend(seed_dataset_rows)
        pooled_rows.append(
            {
                "metric_protocol": "pooled_all_generated",
                "seed": seed,
                "dataset": "Macro-3",
                "auc": macro_auc,
                "ap": macro_ap,
                "n_real": sum(row["n_real"] for row in seed_dataset_rows),
                "n_fake": sum(row["n_fake"] for row in seed_dataset_rows),
            }
        )

        paper_metrics, generator_metrics = metric_tables(
            seed_frame,
            seed=int(config["release"]["random_seed"]),
            score_columns=("score",),
            config_names={"score": "LSTL independent all remaining real"},
        )
        paper_metrics.insert(0, "seed", seed)
        paper_metrics.insert(0, "metric_protocol", "paper_pairwise_balanced")
        generator_metrics.insert(0, "seed", seed)
        paper_metric_parts.append(paper_metrics)
        generator_parts.append(generator_metrics)

    pooled = pd.DataFrame(pooled_rows)
    paper = pd.concat(paper_metric_parts, ignore_index=True)
    paper = paper[
        ["seed", "dataset", "auc", "ap", "n_generators"]
    ].copy()
    paper.insert(0, "metric_protocol", "paper_pairwise_balanced")
    paper["n_real"] = np.nan
    paper["n_fake"] = np.nan
    metrics = pd.concat([pooled, paper], ignore_index=True, sort=False)
    summary = summarize(metrics)

    pd.concat(per_video_parts, ignore_index=True).to_csv(
        args.output_dir / "independent_complement_per_video.csv", index=False
    )
    metrics.to_csv(args.output_dir / "independent_complement_seed_metrics.csv", index=False)
    summary.to_csv(args.output_dir / "independent_complement_summary.csv", index=False)
    pd.concat(generator_parts, ignore_index=True).to_csv(
        args.output_dir / "independent_complement_generator_metrics.csv", index=False
    )
    pd.DataFrame(split_rows).to_csv(
        args.output_dir / "independent_complement_split_audit.csv", index=False
    )
    print(metrics.to_string(index=False))
    print("\nSummary\n", summary.to_string(index=False))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", type=Path, default=ROOT / "configs/alpha_stalled_u0_locked.yaml"
    )
    parser.add_argument(
        "--input-dir", type=Path, default=ROOT / "results/u0_calibration_sensitivity"
    )
    parser.add_argument(
        "--membership", type=Path, default=ROOT / "release/u0/calibration_split_membership.csv"
    )
    parser.add_argument(
        "--reserve-manifest",
        type=Path,
        default=ROOT / "release/u0/calibration_reserve_manifest.json",
    )
    parser.add_argument(
        "--evaluation-manifest",
        type=Path,
        default=ROOT / "release/u0/evaluation_manifest.json",
    )
    parser.add_argument(
        "--locked-calibration-manifest",
        type=Path,
        default=ROOT / "release/u0/calibration_manifest.json",
    )
    parser.add_argument(
        "--remaining-real-manifest",
        type=Path,
        default=ROOT / "release/u0/independent_remaining_real_manifest.json",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "results/second_order_independent_calibration",
    )
    parser.add_argument(
        "--split-dir",
        type=Path,
        default=ROOT / "results/second_order_independent_calibration/splits_complement",
    )
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
