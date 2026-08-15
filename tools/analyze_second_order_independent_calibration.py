#!/usr/bin/env python3
"""Analyze locked-U0 Local D1/D2 ablations and independent calibration splits."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
for directory in (ROOT / "src", ROOT / "tools"):
    if str(directory) not in sys.path:
        sys.path.insert(0, str(directory))

from alpha_stalled.calibration import fuse_global_local, fuse_local_components
from alpha_stalled.metrics import metric_tables, pairwise_frames, repeat_by_count
from alpha_stalled.u0_analysis import calibrate_k3_candidate
from alpha_stalled.whitening import empirical_cdf_right_inclusive, stable_sorted


DATASETS = ("comgenvid", "videofeedback", "genvideo")
DISPLAY = {
    "comgenvid": "ComGenVid",
    "videofeedback": "VideoFeedback",
    "genvideo": "GenVideo",
    "Macro-3": "Macro-3",
}
VARIANTS = {
    "local_d1": "Local first-order only",
    "local_d2": "Local second-order only",
    "full_d1": "Full model with first-order",
    "lstl": "LSTL",
}
COMPARISONS = {
    "local_d2_minus_d1": ("local_d2", "local_d1"),
    "lstl_minus_full_d1": ("lstl", "full_d1"),
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def write_provenance(
    args: argparse.Namespace,
    d1: pd.DataFrame,
    d1_calibration: pd.DataFrame,
    regression: dict,
) -> None:
    k3_files = sorted(args.d1_checkpoint_root.glob("*/shard_*/part_*.csv"))
    k1_files = sorted(args.d1_calibration_checkpoint_root.glob("*/shard_*/part_*.csv"))
    params = sorted((args.output_dir / "params").glob("*_d1.npz"))
    independent_splits = sorted(args.split_dir.glob("seed_*/*.json"))
    all_remaining_splits = sorted(
        (args.output_dir / "splits_complement").glob("seed_*/*.json")
    )
    if len(params) != 3 or len(independent_splits) != 15 or len(all_remaining_splits) != 9:
        raise ValueError(
            "incomplete provenance inputs: "
            f"params={len(params)} fixed_splits={len(independent_splits)} "
            f"all_remaining_splits={len(all_remaining_splits)}"
        )
    feature_sources = sorted(set(d1["feature_source"]) | set(d1_calibration["feature_source"]))
    if feature_sources != ["dino_forward"]:
        raise ValueError(f"formal D1 contains unexpected feature sources: {feature_sources}")
    payload = {
        "schema_version": "lstl_local_d1_d2_ablation_provenance_v1",
        "formal_d1_extraction": {
            "frame_batch_size": 32,
            "reuse_k1_cache": False,
            "feature_sources": feature_sources,
            "k3_window_rows": len(d1),
            "k3_video_count": int(d1["video_id"].nunique()),
            "k1_calibration_rows": len(d1_calibration),
            "rejected_nonformal_directory": "results/second_order_independent_calibration/d1_windows",
            "rejection_reason": "backbone frame batch size 8 differs from locked release size 32",
        },
        "regression": regression,
        "input_hashes": {
            "config": sha256(args.config),
            "locked_windows": sha256(args.locked_windows),
            "locked_scores": sha256(args.locked_scores),
            "evaluation_manifest": sha256(args.evaluation_manifest),
            "reserve_manifest": sha256(args.reserve_manifest),
            "remaining_real_manifest": sha256(args.remaining_real_manifest),
            "membership": sha256(args.membership),
            "independent_seed_metrics": sha256(args.seed_metrics),
            "all_remaining_real_seed_metrics": sha256(args.complement_metrics),
            "all_remaining_real_summary": sha256(args.complement_summary),
        },
        "d1_parameter_hashes": {str(path.relative_to(ROOT)): sha256(path) for path in params},
        "d1_k3_scalar_hashes": {str(path.relative_to(ROOT)): sha256(path) for path in k3_files},
        "d1_k1_scalar_hashes": {str(path.relative_to(ROOT)): sha256(path) for path in k1_files},
        "independent_split_hashes": {
            str(path.relative_to(ROOT)): sha256(path)
            for path in independent_splits
        },
        "all_remaining_real_split_hashes": {
            str(path.relative_to(ROOT)): sha256(path)
            for path in all_remaining_splits
        },
    }
    path = args.output_dir / "provenance.json"
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _dataset_shard_directories(root: Path, dataset: str) -> list[Path]:
    directories = sorted(path for path in (root / dataset).glob("shard_*_of_*") if path.is_dir())
    if not directories:
        raise FileNotFoundError(f"no D1 shard directories in {root / dataset}")
    denominators = {int(path.name.rsplit("_", 1)[-1]) for path in directories}
    if len(denominators) != 1:
        raise ValueError(f"mixed shard denominators for {dataset}: {sorted(denominators)}")
    count = denominators.pop()
    indices = {int(path.name.split("_")[1]) for path in directories}
    if indices != set(range(count)) or len(directories) != count:
        raise ValueError(f"incomplete D1 shard set for {dataset}: {sorted(indices)} of {count}")
    return directories


def load_d1_parts(root: Path) -> pd.DataFrame:
    frames = []
    for dataset in DATASETS:
        for directory in _dataset_shard_directories(root, dataset):
            paths = sorted(directory.glob("part_*.csv"))
            if not paths:
                raise FileNotFoundError(f"no D1 score parts in {directory}")
            failures = directory / "failures.csv"
            if failures.is_file() and len(pd.read_csv(failures)):
                raise ValueError(f"D1 failures are present in {failures}")
            frames.extend(pd.read_csv(path, float_precision="round_trip") for path in paths)
    output = pd.concat(frames, ignore_index=True)
    keys = ["video_id", "window_id"]
    if output.duplicated(keys).any():
        raise ValueError("duplicate D1 window keys")
    if len(output) != 58_496 or output["video_id"].nunique() != 22_021:
        raise ValueError(
            f"expected 58,496 D1 windows / 22,021 videos, got {len(output):,} / "
            f"{output['video_id'].nunique():,}"
        )
    if not np.isfinite(output["patch_d1_raw"]).all():
        raise ValueError("D1 raw scores contain non-finite values")
    return output


def load_d1_calibration_parts(root: Path) -> pd.DataFrame:
    frames = []
    for dataset in DATASETS:
        for directory in _dataset_shard_directories(root, dataset):
            paths = sorted(directory.glob("part_*.csv"))
            if not paths:
                raise FileNotFoundError(f"no D1 K1 calibration parts in {directory}")
            failures = directory / "failures.csv"
            if failures.is_file() and len(pd.read_csv(failures)):
                raise ValueError(f"D1 K1 calibration failures are present in {failures}")
            frames.extend(pd.read_csv(path, float_precision="round_trip") for path in paths)
    output = pd.concat(frames, ignore_index=True)
    if len(output) != 600 or output["video_id"].nunique() != 600:
        raise ValueError(
            f"expected 600 D1 K1 calibration rows, got {len(output)} / "
            f"{output['video_id'].nunique()}"
        )
    if not np.isfinite(output["patch_d1_raw"]).all():
        raise ValueError("D1 K1 calibration raw scores contain non-finite values")
    return output


def calibrate_d1_windows(
    d1: pd.DataFrame, d2: pd.DataFrame, calibration: pd.DataFrame
) -> pd.DataFrame:
    keys = ["video_id", "window_id"]
    columns = [
        *keys,
        "dataset",
        "protocol_split",
        "subset",
        "source_model",
        "filename",
        "effective_k",
        "frame_indices",
        "patch_spatial",
        "patch_d2",
        "L_k",
    ]
    merged = d2[columns].merge(
        d1[keys + ["frame_indices", "patch_d1_raw"]],
        on=keys,
        validate="one_to_one",
        suffixes=("_d2", "_d1"),
    )
    if len(merged) != 58_496:
        raise ValueError("D1/D2 windows do not align completely")
    if not merged["frame_indices_d2"].eq(merged["frame_indices_d1"]).all():
        raise ValueError("D1/D2 frame indices differ")
    pieces = []
    for dataset, frame in merged.groupby("dataset", sort=False):
        reference = calibration[calibration["dataset"].eq(dataset)]
        if len(reference) != 200 or reference["video_id"].nunique() != 200:
            raise ValueError(f"{dataset}: expected 200 D1 K1 calibration references")
        frame = frame.copy()
        frame["patch_d1"] = empirical_cdf_right_inclusive(
            frame["patch_d1_raw"].to_numpy(),
            stable_sorted(reference["patch_d1_raw"].to_numpy()),
        )
        frame["L_d1_k"] = fuse_local_components(
            frame["patch_spatial"], frame["patch_d1"]
        )
        pieces.append(frame)
    return pd.concat(pieces, ignore_index=True)


def build_variants(windows: pd.DataFrame, locked_scores: pd.DataFrame) -> pd.DataFrame:
    temporal_d1 = calibrate_k3_candidate(windows, "patch_d1").rename(
        columns={"score": "local_d1"}
    )
    temporal_d2 = calibrate_k3_candidate(windows, "patch_d2").rename(
        columns={"score": "local_d2"}
    )
    branch_d1 = calibrate_k3_candidate(windows, "L_d1_k").rename(
        columns={"score": "local_branch_d1"}
    )
    branch_d2 = calibrate_k3_candidate(windows, "L_k").rename(
        columns={"score": "local_branch_d2"}
    )
    base_columns = [
        "video_id",
        "dataset",
        "protocol_split",
        "subset",
        "source_model",
        "filename",
        "G",
        "L",
        "S",
    ]
    output = locked_scores[base_columns].merge(
        temporal_d1, on="video_id", validate="one_to_one"
    )
    for candidate in (temporal_d2, branch_d1, branch_d2):
        output = output.merge(candidate, on="video_id", validate="one_to_one")
    local_error = float(
        np.max(np.abs(output["local_branch_d2"] - output["L"]))
    )
    if local_error > 1e-15:
        raise ValueError(f"D2 Local regression differs from locked release: {local_error}")
    output["full_d1"] = fuse_global_local(output["G"], output["local_branch_d1"])
    output["lstl"] = fuse_global_local(output["G"], output["local_branch_d2"])
    final_error = float(np.max(np.abs(output["lstl"] - output["S"])))
    if final_error > 1e-15:
        raise ValueError(f"D2 LSTL regression differs from locked release: {final_error}")
    return output, {
        "local_branch_d2_max_abs_error": local_error,
        "lstl_max_abs_error": final_error,
    }


def metric_deltas(metrics: pd.DataFrame) -> pd.DataFrame:
    indexed = metrics.set_index(["dataset", "config"])
    rows = []
    for dataset in (*DATASETS, "Macro-3"):
        for comparison, (new, base) in COMPARISONS.items():
            rows.append(
                {
                    "dataset": dataset,
                    "comparison": comparison,
                    "new_config": new,
                    "base_config": base,
                    "delta_auc": float(indexed.loc[(dataset, new), "auc"] - indexed.loc[(dataset, base), "auc"]),
                    "delta_ap": float(indexed.loc[(dataset, new), "ap"] - indexed.loc[(dataset, base), "ap"]),
                }
            )
    return pd.DataFrame(rows)


def _stable_seed(seed: int, *parts: str) -> int:
    digest = hashlib.sha256("\0".join((str(seed), *parts)).encode()).digest()
    return int.from_bytes(digest[:8], "little")


def _bootstrap_dataset(
    dataset: str, dataset_frame: pd.DataFrame, iterations: int, seed: int
) -> tuple[dict[tuple[str, str], float], dict[tuple[str, str], np.ndarray]]:
    from sklearn.metrics import average_precision_score, roc_auc_score

    pairs = pairwise_frames(dataset_frame, seed)
    real_ids = sorted(
        {
            str(video_id)
            for pair in pairs.values()
            for video_id in pair.loc[pair["subset"].eq("real"), "video_id"]
        }
    )
    split_pairs = [
        (
            str(generator),
            pair[pair["subset"].eq("real")],
            pair[pair["subset"].ne("real")].reset_index(drop=True),
        )
        for generator, pair in pairs.items()
    ]
    points = {
        (comparison, metric): []
        for comparison in COMPARISONS
        for metric in ("auc", "ap")
    }
    functions = {"auc": roc_auc_score, "ap": average_precision_score}
    for pair in pairs.values():
        label = pair["subset"].eq("real").astype(np.uint8).to_numpy()
        for comparison, (new, base) in COMPARISONS.items():
            for metric, function in functions.items():
                points[(comparison, metric)].append(
                    float(function(label, pair[new]) - function(label, pair[base]))
                )
    point_means = {key: float(np.mean(values)) for key, values in points.items()}
    samples = {
        (comparison, metric): np.zeros(iterations, dtype=np.float64)
        for comparison in COMPARISONS
        for metric in ("auc", "ap")
    }
    for iteration in range(iterations):
        real_rng = np.random.default_rng(
            _stable_seed(seed, dataset, str(iteration), "real")
        )
        real_counts = (
            pd.Series(real_rng.choice(real_ids, size=len(real_ids), replace=True))
            .value_counts()
            .astype(int)
            .to_dict()
        )
        generator_deltas = {
            key: [] for key in samples
        }
        for generator, real, fake in split_pairs:
            fake_rng = np.random.default_rng(
                _stable_seed(seed, dataset, str(iteration), generator, "fake")
            )
            sampled = pd.concat(
                [
                    repeat_by_count(real, real_counts),
                    fake.iloc[fake_rng.integers(0, len(fake), size=len(fake))],
                ],
                ignore_index=True,
            )
            sampled_label = sampled["subset"].eq("real").astype(np.uint8)
            values = {
                (config, metric): function(sampled_label, sampled[config])
                for config in VARIANTS
                for metric, function in functions.items()
            }
            for comparison, (new, base) in COMPARISONS.items():
                for metric in functions:
                    generator_deltas[(comparison, metric)].append(
                        values[(new, metric)] - values[(base, metric)]
                    )
        for key, values in generator_deltas.items():
            samples[key][iteration] = float(np.mean(values))
    return point_means, samples


def paired_bootstrap(
    scores: pd.DataFrame, iterations: int, seed: int, workers: int = 3
) -> pd.DataFrame:
    rows = []
    grouped = [
        (str(dataset), frame.copy(), iterations, seed)
        for dataset, frame in scores.groupby("dataset", sort=True)
    ]
    if workers == 1:
        results = [_bootstrap_dataset(*item) for item in grouped]
    else:
        with ProcessPoolExecutor(max_workers=min(workers, len(grouped))) as pool:
            results = list(pool.map(_bootstrap_dataset_star, grouped))
    points: dict[tuple[str, str, str], float] = {}
    samples: dict[tuple[str, str, str], np.ndarray] = {}
    for (dataset, _, _, _), (dataset_points, dataset_samples) in zip(grouped, results):
        for (comparison, metric), value in dataset_points.items():
            points[(dataset, comparison, metric)] = value
        for (comparison, metric), values in dataset_samples.items():
            samples[(dataset, comparison, metric)] = values
    for comparison in COMPARISONS:
        for metric in ("auc", "ap"):
            points[("Macro-3", comparison, metric)] = float(
                np.mean([points[(dataset, comparison, metric)] for dataset in DATASETS])
            )
            samples[("Macro-3", comparison, metric)] = np.mean(
                np.stack([samples[(dataset, comparison, metric)] for dataset in DATASETS]),
                axis=0,
            )
    for key, array in sorted(samples.items()):
        dataset, comparison, metric = key
        new, base = COMPARISONS[comparison]
        rows.append(
            {
                "dataset": dataset,
                "comparison": comparison,
                "new_config": new,
                "base_config": base,
                "metric": metric,
                "delta": points[key],
                "bootstrap_mean": float(array.mean()),
                "ci95_low": float(np.quantile(array, 0.025)),
                "ci95_high": float(np.quantile(array, 0.975)),
                "iterations": iterations,
            }
        )
    return pd.DataFrame(rows)


def _bootstrap_dataset_star(
    arguments: tuple[str, pd.DataFrame, int, int]
) -> tuple[dict[tuple[str, str], float], dict[tuple[str, str], np.ndarray]]:
    return _bootstrap_dataset(*arguments)


def build_split_audits(args: argparse.Namespace) -> pd.DataFrame:
    reserve = json.loads(args.reserve_manifest.read_text(encoding="utf-8"))
    reserve_by_id = {item["video_id"]: item for item in reserve["videos"]}
    membership = pd.read_csv(args.membership)
    evaluation = json.loads(args.evaluation_manifest.read_text(encoding="utf-8"))
    test_real = pd.DataFrame(evaluation["videos"])
    test_real = test_real[test_real["subset"].eq("real")]
    rows = []
    for seed in (17, 29, 43, 71, 101):
        seed_dir = args.split_dir / f"seed_{seed}"
        seed_dir.mkdir(parents=True, exist_ok=True)
        for dataset in DATASETS:
            ids = sorted(
                membership[
                    membership["dataset"].eq(dataset)
                    & membership["seed"].eq(seed)
                    & membership["calibration_size"].eq(200)
                ]["video_id"]
            )
            tests = sorted(test_real[test_real["dataset"].eq(dataset)]["video_id"])
            overlap = set(ids) & set(tests)
            if len(ids) != 200 or overlap:
                raise ValueError(f"invalid independent split {dataset}/seed={seed}")
            payload = {
                "schema_version": "u0_independent_calibration_split_v1",
                "seed": seed,
                "dataset": dataset,
                "calibration_real_count": len(ids),
                "test_real_count": len(tests),
                "generated_calibration_count": 0,
                "overlap_count": 0,
                "test_set_policy": "fixed_locked_evaluation_real",
                "calibration_real": [reserve_by_id[value] for value in ids],
                "test_real_video_ids": tests,
            }
            path = seed_dir / f"{dataset}.json"
            path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
            rows.append(
                {
                    "seed": seed,
                    "dataset": dataset,
                    "calibration_real": len(ids),
                    "test_real": len(tests),
                    "overlap": 0,
                    "split_file": str(path.relative_to(ROOT)),
                    "sha256": sha256(path),
                }
            )
    return pd.DataFrame(rows)


def independent_metrics(path: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    frame = pd.read_csv(path, float_precision="round_trip")
    selected = frame[frame["config"].str.match(r"^S_s(17|29|43|71|101)_n200$")].copy()
    selected["seed"] = selected["config"].str.extract(r"_s(\d+)_")[0].astype(int)
    raw = selected[["dataset", "seed", "auc", "ap"]].sort_values(["seed", "dataset"])
    summaries = []
    for seed_scope, seeds in (("primary_3_seed", (17, 29, 43)), ("extended_5_seed", (17, 29, 43, 71, 101))):
        target = raw[raw["seed"].isin(seeds)]
        summary = target.groupby("dataset", as_index=False).agg(
            auc_mean=("auc", "mean"),
            auc_std=("auc", "std"),
            ap_mean=("ap", "mean"),
            ap_std=("ap", "std"),
        )
        summary.insert(0, "seed_scope", seed_scope)
        summaries.append(summary)
    return raw, pd.concat(summaries, ignore_index=True)


def independent_complement_metrics(
    metrics_path: Path, summary_path: Path
) -> tuple[pd.DataFrame, pd.DataFrame]:
    metrics = pd.read_csv(metrics_path, float_precision="round_trip")
    summary = pd.read_csv(summary_path, float_precision="round_trip")
    expected_protocols = {"paper_pairwise_balanced", "pooled_all_generated"}
    if set(metrics["metric_protocol"]) != expected_protocols:
        raise ValueError("independent-complement metric protocols are incomplete")
    if set(metrics["seed"].astype(int)) != {17, 29, 43}:
        raise ValueError("independent-complement seeds are incomplete")
    return metrics, summary


def write_tex(metrics: pd.DataFrame, deltas: pd.DataFrame, path: Path) -> None:
    indexed = metrics.set_index(["config", "dataset"])
    delta_index = deltas.set_index(["comparison", "dataset"])
    lines = [
        "\\begin{table*}[!t]",
        "  \\centering",
        "  \\caption{Local first- versus second-order temporal ablation under the locked LSTL protocol. All entries are AUC/real-positive AP; only the Local temporal derivative order changes.}",
        "  \\label{tab:local_d1_d2_ablation}",
        "  \\resizebox{\\textwidth}{!}{%",
        "  \\begin{tabular}{lccccc}",
        "    \\toprule",
        "    Variant & ComGenVid & VideoFeedback & GenVideo & Macro-3 & Macro $\\Delta$AUC/$\\Delta$AP \\\\",
        "    \\midrule",
    ]
    for config in VARIANTS:
        cells = [
            f"{indexed.loc[(config, dataset), 'auc']:.4f}/{indexed.loc[(config, dataset), 'ap']:.4f}"
            for dataset in (*DATASETS, "Macro-3")
        ]
        if config == "local_d2":
            delta = delta_index.loc[("local_d2_minus_d1", "Macro-3")]
            delta_cell = f"{delta.delta_auc:+.4f}/{delta.delta_ap:+.4f}"
        elif config == "lstl":
            delta = delta_index.loc[("lstl_minus_full_d1", "Macro-3")]
            delta_cell = f"{delta.delta_auc:+.4f}/{delta.delta_ap:+.4f}"
        else:
            delta_cell = "--"
        lines.append(f"    {VARIANTS[config]} & " + " & ".join(cells) + f" & {delta_cell} \\\\")
    lines.extend(["    \\bottomrule", "  \\end{tabular}%", "  }", "\\end{table*}"])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_independent_tex(
    summary: pd.DataFrame,
    complement_summary: pd.DataFrame,
    metrics: pd.DataFrame,
    path: Path,
) -> None:
    fixed = summary.set_index(["seed_scope", "dataset"])
    complement = complement_summary.set_index(["metric_protocol", "dataset"])
    locked = metrics[metrics["config"].eq("lstl")].set_index("dataset")
    line_break = r" \\"
    lines = [
        "\\begin{table*}[!t]",
        "  \\centering",
        "  \\caption{Independent-real calibration stability over three fixed splits. Values are AUC/real-positive AP; independent rows report mean $\\pm$ sample standard deviation. Fixed evaluation identities isolate calibration-bank variation, while the all-remaining-real diagnostic changes the test-real pool.}",
        "  \\label{tab:independent_real_calibration}",
        "  \\resizebox{\\textwidth}{!}{%",
        "  \\begin{tabular}{lcccc}",
        "    \\toprule",
        "    Protocol & ComGenVid & VideoFeedback & GenVideo & Macro-3" + line_break,
        "    \\midrule",
    ]
    fixed_cells = []
    locked_cells = []
    delta_cells = []
    complement_cells = []
    for dataset in (*DATASETS, "Macro-3"):
        row = fixed.loc[("primary_3_seed", dataset)]
        locked_row = locked.loc[dataset]
        locked_cells.append(f"{locked_row.auc:.4f}/{locked_row.ap:.4f}")
        fixed_cells.append(
            f"{row.auc_mean:.4f}$\\pm${row.auc_std:.4f}/"
            f"{row.ap_mean:.4f}$\\pm${row.ap_std:.4f}"
        )
        delta_cells.append(
            f"{row.auc_mean-locked_row.auc:+.4f}/"
            f"{row.ap_mean-locked_row.ap:+.4f}"
        )
        row = complement.loc[("paper_pairwise_balanced", dataset)]
        complement_cells.append(
            f"{row.auc_mean:.4f}$\\pm${row.auc_std:.4f}/"
            f"{row.ap_mean:.4f}$\\pm${row.ap_std:.4f}"
        )
    lines.append("    Locked original calibration & " + " & ".join(locked_cells) + line_break)
    lines.append("    Independent, fixed evaluation & " + " & ".join(fixed_cells) + line_break)
    lines.append("    $\\Delta$ independent--locked & " + " & ".join(delta_cells) + line_break)
    lines.append(
        "    Independent, all remaining real & " + " & ".join(complement_cells) + line_break)
    lines.extend(["    \\bottomrule", "  \\end{tabular}%", "  }", "\\end{table*}"])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_report(
    metrics: pd.DataFrame,
    generator_metrics: pd.DataFrame,
    deltas: pd.DataFrame,
    bootstrap: pd.DataFrame,
    independent_raw: pd.DataFrame,
    independent_summary: pd.DataFrame,
    complement_raw: pd.DataFrame,
    complement_summary: pd.DataFrame,
    regression: dict,
    split_audit: pd.DataFrame,
    args: argparse.Namespace,
) -> None:
    indexed = metrics.set_index(["config", "dataset"])
    delta_index = deltas.set_index(["comparison", "dataset"])
    locked = {
        dataset: (
            float(indexed.loc[("lstl", dataset), "auc"]),
            float(indexed.loc[("lstl", dataset), "ap"]),
        )
        for dataset in (*DATASETS, "Macro-3")
    }
    lines = [
        "# Second-order temporal ablation and independent real calibration",
        "",
        "## Purpose and locked protocol",
        "",
        "This experiment changes only the Local temporal representation: same-grid patch D1 versus D2. Encoder, final DINOv3 layer 23, 2 s/16-frame windows, nominal 8 FPS, deterministic K=3 sampling, region1, mean aggregation, Local Spatial, Global, alpha=0.6, beta=0.1, real identities, whitening estimator, right-inclusive empirical CDF, and effective-K video calibration remain fixed.",
        "",
        "D1 and D2 use the same fitting algorithm and the same locked 200 real videos per dataset, but each temporal representation fits its own whitening transform and K1 CDF. Reusing the D2 whitening matrix for D1 would not be a valid controlled comparison.",
        "The D1 parameter files copy `mu_patch_spat`, `W_patch_spat`, and the PatchSpatial calibration array exactly from the locked D2 files; only temporal parameters are refit. All three copied Spatial arrays pass elementwise equality checks.",
        "",
        "## Implementation audit",
        "",
        "- Locked final configuration: `configs/alpha_stalled_u0_locked.yaml` (Clean Universal U0/LSTL).",
        "- Patch D1: `src/alpha_stalled/local_branch.py::local_d1_features`.",
        "- Patch D2: `src/alpha_stalled/local_branch.py::local_d2_features`.",
        "- D1 fitting: `tools/fit_u0_local_d1_params.py`.",
        "- Recoverable scalar-only D1 scoring: `tools/score_u0_local_d1_windows.py`.",
        "- The locked 200-real identities per dataset are declared by `release/u0/calibration_manifest.json`; their K1/K3 frame indices are frozen in `release/u0/frame_indices.json`.",
        "- Independent calibration identities are selected by source-stratified seeded SHA-256 rank in `tools/build_u0_calibration_reserve.py` and recorded in `release/u0/calibration_split_membership.csv`.",
        "- Local whitening is fit by `src/create_patch_params.py::build_patch_params`; K1 right-inclusive window CDF and effective-K video CDF are applied by `tools/analyze_u0_calibration_sensitivity.py::calibrate_candidate` and `tools/analyze_u0_core_ablation.py::calibrate_k3_candidate`.",
        "- Formal D1 extraction uses the release frame batch size 32, groups unique frames per video, and disables compact K1-cache reuse for both K1 CDF references and K3 windows. A batch-size-8 pilot was rejected and is not used in any reported result.",
        "- Analysis and split audit: `tools/analyze_second_order_independent_calibration.py`.",
        "- Independent all-remaining-real analysis: `tools/analyze_independent_real_complement.py`.",
        f"- D2 complete-Local regression max error: `{regression['local_branch_d2_max_abs_error']:.3g}`; LSTL final regression max error: `{regression['lstl_max_abs_error']:.3g}`.",
        "- No full K=3 patch-token cache was written; only parameters, per-window scalar scores, and analysis tables are retained.",
        "- Existing patch-token caches contain historical K1 windows only; no cache matches all formal K3 frame indices and release batch-32 grouping, so formal K3 D1 is extracted from source videos.",
        "- Resource policy: K3 patch tokens are discarded after scalar scoring. At launch `/data` had about 248 GB free. GPU 1 was occupied by an external service (about 29.5/32.6 GB), so formal DINO work used GPU 0 only; shard count changes scheduling, not the per-video batch-32 feature definition.",
        "",
        "## D1 versus D2",
        "",
        "A/B isolate the pure calibrated Local Temporal score (PatchD1 versus PatchD2), excluding PatchSpatial and Global. C/D are the complete model: they retain `0.1 PatchSpatial + 0.9 PatchD1/D2`, then add the unchanged calibrated Global branch with the locked 0.6/0.4 fusion.",
        "",
        "| Variant | ComGenVid AUC/AP | VideoFeedback AUC/AP | GenVideo AUC/AP | Macro-3 AUC/AP | Macro delta AUC/AP |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for config in VARIANTS:
        cells = [f"{indexed.loc[(config, d), 'auc']:.4f}/{indexed.loc[(config, d), 'ap']:.4f}" for d in (*DATASETS, "Macro-3")]
        if config == "local_d2":
            delta = delta_index.loc[("local_d2_minus_d1", "Macro-3")]
            delta_cell = f"{delta.delta_auc:+.4f}/{delta.delta_ap:+.4f}"
        elif config == "lstl":
            delta = delta_index.loc[("lstl_minus_full_d1", "Macro-3")]
            delta_cell = f"{delta.delta_auc:+.4f}/{delta.delta_ap:+.4f}"
        else:
            delta_cell = "--"
        lines.append(f"| {VARIANTS[config]} | " + " | ".join(cells) + f" | {delta_cell} |")
    lines.extend(["", "### Per-dataset second-order deltas", "", "| Dataset | Local D2-D1 AUC/AP | LSTL-Full D1 AUC/AP |", "|---|---:|---:|"])
    for dataset in (*DATASETS, "Macro-3"):
        a = delta_index.loc[("local_d2_minus_d1", dataset)]
        b = delta_index.loc[("lstl_minus_full_d1", dataset)]
        lines.append(f"| {DISPLAY[dataset]} | {a.delta_auc:+.4f}/{a.delta_ap:+.4f} | {b.delta_auc:+.4f}/{b.delta_ap:+.4f} |")
    lines.extend(
        [
            "",
            "### Detailed four-variant tables",
            "",
            "The delta columns are populated only for the corresponding second-order row; first-order rows are the reference.",
        ]
    )
    for dataset in (*DATASETS, "Macro-3"):
        lines.extend(
            [
                "",
                f"#### {DISPLAY[dataset]}",
                "",
                "| Variant | AUC | AP | Delta AUC vs corresponding first-order | Delta AP vs corresponding first-order |",
                "|---|---:|---:|---:|---:|",
            ]
        )
        for config in VARIANTS:
            row = indexed.loc[(config, dataset)]
            if config == "local_d2":
                delta = delta_index.loc[("local_d2_minus_d1", dataset)]
                delta_auc, delta_ap = f"{delta.delta_auc:+.4f}", f"{delta.delta_ap:+.4f}"
            elif config == "lstl":
                delta = delta_index.loc[("lstl_minus_full_d1", dataset)]
                delta_auc, delta_ap = f"{delta.delta_auc:+.4f}", f"{delta.delta_ap:+.4f}"
            else:
                delta_auc, delta_ap = "--", "--"
            lines.append(
                f"| {VARIANTS[config]} | {row.auc:.4f} | {row.ap:.4f} | "
                f"{delta_auc} | {delta_ap} |"
            )
    lines.extend(["", "### Paired bootstrap", "", "1,000 paired cluster resamples preserve D1/D2 score pairing and use video IDs, never windows, as sampling units. Following `tools/audit_u0_metric_protocol.py`, real-video cluster counts are drawn once from the dataset-level union and projected into every generator pair, while generated videos are resampled within generator.", "", "| Comparison | Metric | Delta | 95% CI |", "|---|---|---:|---:|"])
    for row in bootstrap[bootstrap["dataset"].eq("Macro-3")].sort_values(["comparison", "metric"]).itertuples(index=False):
        lines.append(f"| {row.comparison} | {row.metric} | {row.delta:+.4f} | [{row.ci95_low:+.4f}, {row.ci95_high:+.4f}] |")
    lines.extend(["", "## Independent real calibration", "", "The locked main result already uses 200 real calibration videos per dataset with zero identity overlap with evaluation. The additional experiment changes only the independent 200-real calibration bank while keeping the 21,421 evaluation identities fixed. This isolates calibration-bank variation; it is a stability test, not a repair of an observed leak.", "", "Each dataset/seed refits Local D2 whitening from its selected 200 reserve real videos, rebuilds the K1 PatchSpatial/PatchD2 CDFs, and rebuilds effective-K video CDFs from the corresponding reserve K3 scores. The predeclared split seed is also the deterministic 300k-token reservoir seed for that fit, so the reported standard deviation measures full calibration-pipeline variation rather than identity variation alone. Global STALL parameters remain the fixed source-domain parameters declared by U0. Generated videos never enter fitting or calibration.", "", "Primary statistics use the requested three fixed seeds 17/29/43. Seeds 71/101 are retained as an extended five-seed audit. Every `std` below is the sample standard deviation across calibration splits (`ddof=1`), not a bootstrap confidence interval; the two uncertainty analyses answer different questions.", "", "| Dataset | 3-seed AUC mean +/- std | 3-seed AP mean +/- std | 5-seed AUC mean +/- std | 5-seed AP mean +/- std |", "|---|---:|---:|---:|---:|"])
    summary = independent_summary.set_index(["seed_scope", "dataset"])
    for dataset in (*DATASETS, "Macro-3"):
        three = summary.loc[("primary_3_seed", dataset)]
        five = summary.loc[("extended_5_seed", dataset)]
        lines.append(f"| {DISPLAY[dataset]} | {three.auc_mean:.4f} +/- {three.auc_std:.4f} | {three.ap_mean:.4f} +/- {three.ap_std:.4f} | {five.auc_mean:.4f} +/- {five.auc_std:.4f} | {five.ap_mean:.4f} +/- {five.ap_std:.4f} |")
    lines.extend(["", "### Per-seed results", "", "| Seed | ComGenVid | VideoFeedback | GenVideo | Macro-3 |", "|---:|---:|---:|---:|---:|"])
    seed_index = independent_raw.set_index(["seed", "dataset"])
    for seed in (17, 29, 43, 71, 101):
        cells = [f"{seed_index.loc[(seed, d), 'auc']:.4f}/{seed_index.loc[(seed, d), 'ap']:.4f}" for d in (*DATASETS, "Macro-3")]
        lines.append(f"| {seed} | " + " | ".join(cells) + " |")
    lines.extend(["", "### Locked versus independent", "", "| Dataset | Locked AUC/AP | Independent 3-seed mean AUC/AP | Delta AUC/AP |", "|---|---:|---:|---:|"])
    for dataset in (*DATASETS, "Macro-3"):
        three = summary.loc[("primary_3_seed", dataset)]
        old_auc, old_ap = locked[dataset]
        lines.append(f"| {DISPLAY[dataset]} | {old_auc:.4f}/{old_ap:.4f} | {three.auc_mean:.4f}/{three.ap_mean:.4f} | {three.auc_mean-old_auc:+.4f}/{three.ap_mean-old_ap:+.4f} |")
    lines.extend(
        [
            "",
            "Every fixed-evaluation split contains 200 calibration real identities, the fixed locked test-real identities, no generated calibration videos, and zero overlap. File hashes are recorded in `split_audit.csv`.",
            "",
            "### Strict all-remaining-real diagnostic",
            "",
            "This second view follows the literal `200 calibration real + all eligible remaining real` protocol. It combines the fixed evaluation-real set, the historical locked-calibration set, the independent reserve complement, and every additional raw-index real video satisfying the same strict 2 s rule. Only VideoFeedback has such an additional pool (3,080 videos). The test-real counts are 1,498/3,880/9,784 for ComGenVid/VideoFeedback/GenVideo, paired with all 3,400/3,000/5,639 locked generated videos. It changes the test-real pool, so its delta from the locked table mixes calibration-bank and test-set sampling effects and is not used as the primary causal comparison.",
            "",
            "| Dataset | Paper protocol AUC mean +/- std | Paper protocol AP mean +/- std | Pooled-all AUC mean +/- std | Pooled-all AP mean +/- std |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    complement = complement_summary.set_index(["metric_protocol", "dataset"])
    for dataset in (*DATASETS, "Macro-3"):
        paper = complement.loc[("paper_pairwise_balanced", dataset)]
        pooled = complement.loc[("pooled_all_generated", dataset)]
        lines.append(
            f"| {DISPLAY[dataset]} | {paper.auc_mean:.4f} +/- {paper.auc_std:.4f} | "
            f"{paper.ap_mean:.4f} +/- {paper.ap_std:.4f} | "
            f"{pooled.auc_mean:.4f} +/- {pooled.auc_std:.4f} | "
            f"{pooled.ap_mean:.4f} +/- {pooled.ap_std:.4f} |"
        )
    lines.extend(
        [
            "",
            "Pooled-all AP is intentionally not compared with the balanced paper AP: real prevalence is 1,498/4,898, 3,880/6,880, and 9,784/15,423, so AP has a different class-prior baseline. AUC is much less sensitive to this prevalence change.",
            "",
            "Strict all-remaining-real split identities and hashes are stored under the backward-compatible path `splits_complement/` and in `independent_complement_split_audit.csv`; every overlap count is zero.",
            "",
            "Per-seed paper-protocol all-remaining-real results:",
            "",
            "| Seed | ComGenVid | VideoFeedback | GenVideo | Macro-3 |",
            "|---:|---:|---:|---:|---:|",
        ]
    )
    complement_seed = complement_raw[
        complement_raw["metric_protocol"].eq("paper_pairwise_balanced")
    ].set_index(["seed", "dataset"])
    for seed in (17, 29, 43):
        cells = [
            f"{complement_seed.loc[(seed, dataset), 'auc']:.4f}/"
            f"{complement_seed.loc[(seed, dataset), 'ap']:.4f}"
            for dataset in (*DATASETS, "Macro-3")
        ]
        lines.append(f"| {seed} | " + " | ".join(cells) + " |")
    lines.extend(
        [
            "",
            "## Modified files",
            "",
            "- `src/alpha_stalled/local_branch.py`: exact same-grid Local D1/D2 primitives.",
            "- `tools/fit_u0_local_d1_params.py`: independent D1 whitening fit on the locked real-only calibration identities.",
            "- `tools/score_u0_local_d1_windows.py`: recoverable K1/K3 scalar scorer.",
            "- `tools/analyze_independent_real_complement.py`: three-seed strict all-remaining-real evaluation.",
            "- `tools/build_independent_remaining_real_manifest.py`: exhaustive strict-2s real-pool supplement audit.",
            "- `tools/score_u0_calibration_candidates.py`: adds recoverable scoring for the supplemental real manifest.",
            "- `tools/analyze_second_order_independent_calibration.py`: unified metrics, bootstrap, split audit, paper table, and report.",
            "- `tests/test_whitening_batch_invariance.py`: explicit D1/D2 formula regression test.",
            "- `tests/test_second_order_independent_calibration.py`: independent-split and metric-protocol integrity tests.",
            "- `scripts/run_second_order_ablation.sh`: reproducible execution wrapper.",
            "",
            "## Interpretation",
            "",
        ]
    )
    full_delta = delta_index.loc[("lstl_minus_full_d1", "Macro-3")]
    local_delta = delta_index.loc[("local_d2_minus_d1", "Macro-3")]
    ci = bootstrap[(bootstrap["dataset"].eq("Macro-3")) & (bootstrap["comparison"].eq("lstl_minus_full_d1")) & (bootstrap["metric"].eq("ap"))].iloc[0]
    generator_ap = generator_metrics.pivot(
        index=["dataset", "generator"], columns="config", values="ap"
    )
    local_wins = int((generator_ap["local_d2"] > generator_ap["local_d1"]).sum())
    full_wins = int((generator_ap["lstl"] > generator_ap["full_d1"]).sum())
    full_losses = (
        generator_ap["lstl"] - generator_ap["full_d1"]
    ).sort_values()
    full_losses = full_losses[full_losses < 0]
    loss_text = ", ".join(
        f"{dataset}/{generator} `{delta:+.4f}`"
        for (dataset, generator), delta in full_losses.items()
    )
    support = full_delta.delta_ap > 0 and ci.ci95_low > 0
    lines.extend([
        f"- Standalone Local D2 changes Macro AUC/AP by `{local_delta.delta_auc:+.4f}/{local_delta.delta_ap:+.4f}` relative to Local D1.",
        f"- In the full model, D2 changes Macro AUC/AP by `{full_delta.delta_auc:+.4f}/{full_delta.delta_ap:+.4f}`; the paired AP interval is `[{ci.ci95_low:+.4f},{ci.ci95_high:+.4f}]`.",
        f"- Local D2 improves AP for `{local_wins}/20` generators; complete LSTL improves `{full_wins}/20`. The only complete-model AP decreases are {loss_text}.",
        f"- The strict experiment {'supports' if support else 'does not establish'} the claim that second-order Local temporal modeling improves the full locked method over an otherwise identical first-order version.",
        "- Current release manifests and all independent splits show zero calibration/test identity overlap. The residual risk is calibration-domain and finite-sample sensitivity, not observed identity leakage.",
        "",
        "## Paper artifacts",
        "",
        "- `paper/ieee_alpha_stalled/tables/local_d1_d2_ablation.tex`: paper-ready D1/D2 table.",
        "- `results/second_order_independent_calibration/dataset_metrics.csv`: per-dataset and Macro metrics.",
        "- `results/second_order_independent_calibration/generator_metrics.csv`: all 20 generators.",
        "- `results/second_order_independent_calibration/bootstrap_deltas.csv`: paired confidence intervals.",
        "- `results/second_order_independent_calibration/independent_seed_metrics.csv`: raw seed results.",
        "- `results/second_order_independent_calibration/splits/`: reproducible calibration/test identity files.",
        "- `results/second_order_independent_calibration/independent_complement_seed_metrics.csv`: strict all-remaining-real raw seed results under both metric protocols.",
        "- `results/second_order_independent_calibration/splits_complement/`: strict all-remaining-real split files (backward-compatible directory name).",
        "",
        "## Commands",
        "",
        "```bash",
        "# Independent-real reserve and fixed split membership (already materialized for this run).",
        "conda run --no-capture-output -n stall python tools/build_u0_calibration_reserve.py",
        "conda run --no-capture-output -n stall python tools/build_independent_remaining_real_manifest.py",
        "# Refit Local whitening for every dataset/seed/size; use SHARD in [0, N).",
        "conda run --no-capture-output -n stall python tools/fit_u0_calibration_sensitivity.py --shard-index SHARD --num-shards N",
        "# Score each split once for all real-only calibration candidates.",
        "conda run --no-capture-output -n stall python tools/score_u0_calibration_candidates.py --split SPLIT --dataset DATASET --shard-index SHARD --num-shards N --extract-device cuda:GPU --score-device cuda:GPU --frame-batch-size 32",
        "# SPLIT must include evaluation, reserve, locked_calibration, and independent_remaining_real for the strict all-remaining-real diagnostic.",
        "# This run reused complete evaluation/reserve scalar caches and newly scored only locked_calibration plus independent_remaining_real.",
        "# This run used N=4 for independent_remaining_real VideoFeedback scoring.",
        "conda run --no-capture-output -n stall python tools/analyze_u0_calibration_sensitivity.py",
        "conda run --no-capture-output -n stall python tools/analyze_independent_real_complement.py",
        "",
        "# Controlled Local D1/D2 ablation.",
        "conda run --no-capture-output -n stall python tools/fit_u0_local_d1_params.py --dataset DATASET --device cuda:0",
        "conda run --no-capture-output -n stall python tools/score_u0_local_d1_windows.py --dataset DATASET --sampling calibration_k1 --shard-index SHARD --num-shards 2 --extract-device cuda:GPU --score-device cuda:GPU --frame-batch-size 32 --no-reuse-k1-cache",
        "# Formal K3 shard counts: ComGenVid=2, VideoFeedback=2, GenVideo=4.",
        "conda run --no-capture-output -n stall python tools/score_u0_local_d1_windows.py --dataset DATASET --sampling k3 --shard-index SHARD --num-shards N --extract-device cuda:GPU --score-device cuda:GPU --frame-batch-size 32 --no-reuse-k1-cache",
        "conda run --no-capture-output -n stall python tools/analyze_second_order_independent_calibration.py --iterations 1000 --workers 3",
        "# Convenience wrapper for the D1 portion: scripts/run_second_order_ablation.sh",
        "```",
    ])
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run(args: argparse.Namespace) -> None:
    d1 = load_d1_parts(args.d1_checkpoint_root)
    d1_calibration = load_d1_calibration_parts(args.d1_calibration_checkpoint_root)
    d2 = pd.read_csv(args.locked_windows, float_precision="round_trip")
    if len(d2) != 58_496:
        raise ValueError(f"expected 58,496 locked windows, got {len(d2)}")
    windows = calibrate_d1_windows(d1, d2, d1_calibration)
    locked = pd.read_csv(args.locked_scores, float_precision="round_trip")
    scores, regression = build_variants(windows, locked)
    metrics, generators = metric_tables(
        scores,
        seed=args.seed,
        score_columns=tuple(VARIANTS),
        config_names=VARIANTS,
    )
    deltas = metric_deltas(metrics)
    bootstrap = paired_bootstrap(scores, args.iterations, args.seed, args.workers)
    split_audit = build_split_audits(args)
    independent_raw, independent_summary = independent_metrics(args.seed_metrics)
    complement_raw, complement_summary = independent_complement_metrics(
        args.complement_metrics, args.complement_summary
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    windows.to_csv(args.output_dir / "per_window_scores.csv", index=False)
    scores.to_csv(args.output_dir / "per_video_scores.csv", index=False)
    metrics.to_csv(args.output_dir / "dataset_metrics.csv", index=False)
    generators.to_csv(args.output_dir / "generator_metrics.csv", index=False)
    deltas.to_csv(args.output_dir / "d1_d2_deltas.csv", index=False)
    bootstrap.to_csv(args.output_dir / "bootstrap_deltas.csv", index=False)
    split_audit.to_csv(args.output_dir / "split_audit.csv", index=False)
    independent_raw.to_csv(args.output_dir / "independent_seed_metrics.csv", index=False)
    independent_summary.to_csv(args.output_dir / "independent_seed_summary.csv", index=False)
    write_provenance(args, d1, d1_calibration, regression)
    write_tex(metrics, deltas, args.tex_table)
    write_independent_tex(
        independent_summary,
        complement_summary,
        metrics,
        args.independent_tex_table,
    )
    write_report(
        metrics,
        generators,
        deltas,
        bootstrap,
        independent_raw,
        independent_summary,
        complement_raw,
        complement_summary,
        regression,
        split_audit,
        args,
    )
    print(metrics.to_string(index=False))
    print("\nDeltas\n", deltas.to_string(index=False))
    print("\nIndependent calibration\n", independent_summary.to_string(index=False))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--iterations", type=int, default=1000)
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--config", type=Path, default=ROOT / "configs/alpha_stalled_u0_locked.yaml"
    )
    parser.add_argument(
        "--d1-checkpoint-root",
        type=Path,
        default=ROOT / "results/second_order_independent_calibration/d1_windows_b32/checkpoints/k3",
    )
    parser.add_argument(
        "--d1-calibration-checkpoint-root",
        type=Path,
        default=ROOT / "results/second_order_independent_calibration/d1_windows_b32/checkpoints/calibration_k1",
    )
    parser.add_argument(
        "--locked-windows",
        type=Path,
        default=ROOT / "results/u0_locked_reproduction/per_window_scores.csv",
    )
    parser.add_argument(
        "--locked-scores", type=Path, default=ROOT / "release/u0/final_video_scores.csv"
    )
    parser.add_argument(
        "--reserve-manifest",
        type=Path,
        default=ROOT / "release/u0/calibration_reserve_manifest.json",
    )
    parser.add_argument(
        "--remaining-real-manifest",
        type=Path,
        default=ROOT / "release/u0/independent_remaining_real_manifest.json",
    )
    parser.add_argument(
        "--membership",
        type=Path,
        default=ROOT / "release/u0/calibration_split_membership.csv",
    )
    parser.add_argument(
        "--evaluation-manifest",
        type=Path,
        default=ROOT / "release/u0/evaluation_manifest.json",
    )
    parser.add_argument(
        "--seed-metrics",
        type=Path,
        default=ROOT / "results/u0_calibration_sensitivity/analysis/dataset_metrics.csv",
    )
    parser.add_argument(
        "--complement-metrics",
        type=Path,
        default=ROOT
        / "results/second_order_independent_calibration/independent_complement_seed_metrics.csv",
    )
    parser.add_argument(
        "--complement-summary",
        type=Path,
        default=ROOT
        / "results/second_order_independent_calibration/independent_complement_summary.csv",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "results/second_order_independent_calibration",
    )
    parser.add_argument(
        "--split-dir",
        type=Path,
        default=ROOT / "results/second_order_independent_calibration/splits",
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=ROOT / "reports/second_order_and_independent_calibration.md",
    )
    parser.add_argument(
        "--tex-table",
        type=Path,
        default=ROOT / "paper/ieee_alpha_stalled/tables/local_d1_d2_ablation.tex",
    )
    parser.add_argument(
        "--independent-tex-table",
        type=Path,
        default=ROOT
        / "paper/ieee_alpha_stalled/tables/independent_real_calibration.tex",
    )
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
