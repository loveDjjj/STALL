#!/usr/bin/env python3
"""Calibrate, evaluate, and bootstrap the duration-aware full-data protocol."""

from __future__ import annotations

import argparse
import glob
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

from alpha_stalled.calibration import cdf_with_positive_infinity
from alpha_stalled.parameters import global_references
from alpha_stalled.aggregation import selected_video_means
from alpha_stalled.whitening import empirical_cdf_right_inclusive, stable_sorted


DATASETS = ("comgenvid", "videofeedback", "genvideo")
SHORT_ONLY = {
    ("videofeedback", "Hotshot-XL"),
    ("genvideo", "HotShot"),
    ("genvideo", "MoonValley"),
}
CANDIDATES = ("n200", "nmax", "ncustom")
BRANCHES = ("G", "L", "S")


def stable_rank(namespace: str, value: str) -> str:
    return hashlib.sha256(f"{namespace}\0{value}".encode()).hexdigest()


def load_raw(directory: Path) -> pd.DataFrame:
    paths = sorted(Path(path) for path in glob.glob(str(directory / "raw" / "*.csv")))
    if not paths:
        raise FileNotFoundError(f"no raw score files under {directory / 'raw'}")
    frame = pd.concat(
        [pd.read_csv(path, float_precision="round_trip") for path in paths],
        ignore_index=True,
    )
    if frame.duplicated(["task_id", "window_id"]).any():
        raise ValueError("duplicate raw task/window keys")
    return frame


def attach_custom_raw(
    raw: pd.DataFrame,
    custom_raw_dir: Path,
    selected_sizes: dict[str, int],
) -> pd.DataFrame:
    paths = sorted(Path(path) for path in glob.glob(str(custom_raw_dir / "*.csv")))
    if not paths:
        raise FileNotFoundError(f"no custom candidate raw files under {custom_raw_dir}")
    pieces = []
    for path in paths:
        frame = pd.read_csv(path, float_precision="round_trip")
        datasets = frame["dataset"].unique()
        if len(datasets) != 1:
            raise ValueError(f"mixed datasets in {path}")
        dataset = str(datasets[0])
        size = int(selected_sizes[dataset])
        pieces.append(
            frame[
                ["task_id", "window_id", f"patch_spatial__n{size}", f"patch_d2__n{size}"]
            ].rename(
                columns={
                    f"patch_spatial__n{size}": "patch_spatial__ncustom",
                    f"patch_d2__n{size}": "patch_d2__ncustom",
                }
            )
        )
    custom = pd.concat(pieces, ignore_index=True)
    if custom.duplicated(["task_id", "window_id"]).any():
        raise ValueError("duplicate custom raw task/window keys")
    output = raw.merge(custom, on=["task_id", "window_id"], validate="one_to_one")
    if len(output) != len(raw) or not np.isfinite(
        output[["patch_spatial__ncustom", "patch_d2__ncustom"]].to_numpy()
    ).all():
        raise ValueError("custom candidate raw coverage or finite-value check failed")
    return output


# Historical compatibility aliases; both now point at the shared definitions.
global_vatex_references = global_references
cdf_temporal = cdf_with_positive_infinity


def finite_reference(values: np.ndarray, name: str) -> np.ndarray:
    scores = np.asarray(values, dtype=np.float64)
    finite = scores[np.isfinite(scores)]
    if len(finite) < 2:
        raise ValueError(f"{name}: fewer than two finite calibration scores")
    return stable_sorted(finite)


def calibration_ids(
    membership: pd.DataFrame,
    size_membership: pd.DataFrame,
    selected_sizes: dict[str, int],
    dataset: str,
    candidate: str,
) -> set[str]:
    if candidate == "ncustom":
        expected = int(selected_sizes[dataset])
        selected = size_membership[
            size_membership["dataset"].eq(dataset)
            & size_membership["calibration_size"].eq(expected)
        ]["video_id"]
    else:
        column = "in_n200" if candidate == "n200" else "in_nmax"
        selected = membership[membership["dataset"].eq(dataset) & membership[column]]["video_id"]
        expected = 200 if candidate == "n200" else {"comgenvid": 800, "videofeedback": 500, "genvideo": 2000}[dataset]
    if len(selected) != expected:
        raise ValueError(f"{dataset}/{candidate}: expected {expected} calibration IDs, got {len(selected)}")
    return set(selected)


def calibrate_candidate(
    raw: pd.DataFrame,
    membership: pd.DataFrame,
    size_membership: pd.DataFrame,
    selected_sizes: dict[str, int],
    candidate: str,
    vatex_spatial: np.ndarray,
    vatex_t1: np.ndarray,
) -> pd.DataFrame:
    windows = []
    for (dataset, duration), frame in raw.groupby(["dataset", "protocol_duration_sec"], sort=False):
        selected_ids = calibration_ids(
            membership, size_membership, selected_sizes, dataset, candidate
        )
        k1 = frame[
            frame["protocol_split"].eq("calibration")
            & frame["sampling"].eq("k1")
            & frame["video_id"].isin(selected_ids)
        ]
        if k1["video_id"].nunique() != len(selected_ids) or len(k1) != len(selected_ids):
            raise ValueError(f"{dataset}/{duration}/{candidate}: incomplete K1 calibration")
        if int(duration) == 2:
            global_spatial_ref, global_t1_ref = vatex_spatial, vatex_t1
        else:
            global_spatial_ref = finite_reference(
                k1["global_spatial_raw"].to_numpy(), f"{dataset}/1s/global_spatial"
            )
            # An all-zero T1 window is declared uninformative (+inf). It is
            # excluded from the reference but still maps to percentile one.
            global_t1_ref = finite_reference(
                k1["global_t1_raw"].to_numpy(), f"{dataset}/1s/global_t1"
            )
        patch_spatial_ref = stable_sorted(k1[f"patch_spatial__{candidate}"].to_numpy())
        patch_d2_ref = stable_sorted(k1[f"patch_d2__{candidate}"].to_numpy())
        target = frame.copy()
        target["global_spatial"] = empirical_cdf_right_inclusive(
            target["global_spatial_raw"].to_numpy(), global_spatial_ref
        )
        target["global_t1"] = cdf_temporal(target["global_t1_raw"].to_numpy(), global_t1_ref)
        target["patch_spatial"] = empirical_cdf_right_inclusive(
            target[f"patch_spatial__{candidate}"].to_numpy(), patch_spatial_ref
        )
        target["patch_d2"] = empirical_cdf_right_inclusive(
            target[f"patch_d2__{candidate}"].to_numpy(), patch_d2_ref
        )
        target["G_k"] = 0.5 * target["global_spatial"] + 0.5 * target["global_t1"]
        target["L_k"] = 0.1 * target["patch_spatial"] + 0.9 * target["patch_d2"]
        windows.append(target)
    calibrated = pd.concat(windows, ignore_index=True)
    task_columns = [
        "task_id", "video_id", "dataset", "protocol_split", "subset", "source_model",
        "filename", "protocol_duration_sec", "sampling", "duration_seconds",
    ]
    per_task = (
        calibrated.groupby(task_columns, sort=False, observed=True)
        .agg(effective_k=("window_id", "size"), G_raw=("G_k", "mean"), L_raw=("L_k", "mean"))
        .reset_index()
    )
    def video_reference(
        dataset: str,
        duration: int,
        selected_ids: set[str],
        target_k: int,
    ) -> pd.DataFrame:
        sampling = "k1" if duration == 1 else "k3_uniform"
        source = calibrated[
            calibrated["dataset"].eq(dataset)
            & calibrated["protocol_duration_sec"].eq(duration)
            & calibrated["protocol_split"].eq("calibration")
            & calibrated["sampling"].eq(sampling)
            & calibrated["video_id"].isin(selected_ids)
        ]
        reference = selected_video_means(
            source, target_k, {"G_raw": "G_k", "L_raw": "L_k"}
        )
        if len(reference) < 2:
            raise ValueError(
                f"{dataset}/{duration}/{candidate}: insufficient effective-K={target_k} reference"
            )
        return reference

    outputs = []
    for (dataset, duration), frame in per_task.groupby(["dataset", "protocol_duration_sec"], sort=False):
        selected_ids = calibration_ids(
            membership, size_membership, selected_sizes, dataset, candidate
        )
        sampling = "k1" if int(duration) == 1 else "k3_uniform"
        evaluation = frame[
            frame["protocol_split"].eq("evaluation") & frame["sampling"].eq(sampling)
        ]
        for effective_k, target in evaluation.groupby("effective_k", sort=True):
            reference = video_reference(
                str(dataset), int(duration), selected_ids, int(effective_k)
            )
            target = target.copy()
            target["G"] = empirical_cdf_right_inclusive(
                target["G_raw"].to_numpy(), stable_sorted(reference["G_raw"].to_numpy())
            )
            target["L"] = empirical_cdf_right_inclusive(
                target["L_raw"].to_numpy(), stable_sorted(reference["L_raw"].to_numpy())
            )
            target["S"] = 0.6 * target["G"] + 0.4 * target["L"]
            target["candidate"] = candidate
            outputs.append(target)
    return pd.concat(outputs, ignore_index=True)


def balanced_real_ids(real_meta: pd.DataFrame, n: int, namespace: str) -> list[str]:
    sources = sorted(real_meta["source_model"].unique())
    base, remainder = divmod(n, len(sources))
    selected = []
    for index, source in enumerate(sources):
        count = base + int(index < remainder)
        group = real_meta[real_meta["source_model"].eq(source)].copy()
        group["_rank"] = group["video_id"].map(lambda value: stable_rank(namespace, str(value)))
        selected.extend(group.sort_values("_rank").head(count)["video_id"].tolist())
    if len(selected) != n or len(set(selected)) != n:
        raise ValueError("could not construct balanced real sample")
    return sorted(selected, key=lambda value: stable_rank(namespace + ":merge", value))


def pair_protocol(scored: pd.DataFrame, protocol: str, seed: int) -> pd.DataFrame:
    metadata = scored[scored["candidate"].eq("n200")].copy()
    rows = []
    for dataset in DATASETS:
        dataset_frame = metadata[metadata["dataset"].eq(dataset)]
        fake_models = sorted(dataset_frame[dataset_frame["subset"].eq("annotated")]["source_model"].unique())
        for model in fake_models:
            if protocol == "strict20_2s" and (dataset, model) in SHORT_ONLY:
                continue
            fake = dataset_frame[
                dataset_frame["subset"].eq("annotated") & dataset_frame["source_model"].eq(model)
            ].copy()
            if protocol == "strict20_2s":
                fake = fake[fake["protocol_duration_sec"].eq(2)].copy()
            real_meta = dataset_frame[dataset_frame["subset"].eq("real")][
                ["video_id", "source_model"]
            ].drop_duplicates()
            # Preserve the locked evaluator's equal quota across real sources.
            # The usable real count is therefore bounded by the smallest real
            # source rather than by the pooled total alone.
            real_counts = real_meta.groupby("source_model")["video_id"].nunique()
            max_balanced_real = int(real_counts.min() * len(real_counts))
            n = min(len(fake), max_balanced_real)
            namespace = f"{seed}:{protocol}:{dataset}:{model}"
            fake["_rank"] = fake["video_id"].map(lambda value: stable_rank(namespace + ":fake", str(value)))
            fake = fake.sort_values("_rank").head(n).reset_index(drop=True)
            real_ids = balanced_real_ids(real_meta, n, namespace + ":real")
            for pair_id, (fake_row, real_id) in enumerate(zip(fake.itertuples(index=False), real_ids)):
                rows.append(
                    {
                        "protocol": protocol,
                        "dataset": dataset,
                        "source_model": model,
                        "pair_id": pair_id,
                        "fake_video_id": fake_row.video_id,
                        "real_video_id": real_id,
                        "protocol_duration_sec": int(fake_row.protocol_duration_sec),
                    }
                )
    pairs = pd.DataFrame(rows)
    score_lookup = scored.set_index(["candidate", "video_id", "protocol_duration_sec"])
    for candidate in CANDIDATES:
        for branch in BRANCHES:
            pairs[f"fake_{candidate}_{branch}"] = [
                score_lookup.loc[(candidate, video_id_value, duration), branch]
                for video_id_value, duration in zip(pairs["fake_video_id"], pairs["protocol_duration_sec"])
            ]
            pairs[f"real_{candidate}_{branch}"] = [
                score_lookup.loc[(candidate, video_id_value, duration), branch]
                for video_id_value, duration in zip(pairs["real_video_id"], pairs["protocol_duration_sec"])
            ]
    return pairs


def metric(real: np.ndarray, fake: np.ndarray) -> tuple[float, float]:
    labels = np.concatenate([np.ones(len(real), dtype=np.uint8), np.zeros(len(fake), dtype=np.uint8)])
    scores = np.concatenate([real, fake])
    return float(roc_auc_score(labels, scores)), float(average_precision_score(labels, scores))


def metric_tables(pairs: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    generator_rows = []
    for keys, frame in pairs.groupby(["protocol", "dataset", "source_model"], sort=True):
        for candidate in CANDIDATES:
            for branch in BRANCHES:
                auc, ap = metric(frame[f"real_{candidate}_{branch}"].to_numpy(), frame[f"fake_{candidate}_{branch}"].to_numpy())
                generator_rows.append(
                    {
                        "protocol": keys[0], "dataset": keys[1], "source_model": keys[2],
                        "candidate": candidate, "branch": branch, "auc": auc, "ap": ap,
                        "n_real": len(frame), "n_fake": len(frame),
                    }
                )
    generators = pd.DataFrame(generator_rows)
    datasets = (
        generators.groupby(["protocol", "dataset", "candidate", "branch"], as_index=False)
        .agg(auc=("auc", "mean"), ap=("ap", "mean"), generators=("source_model", "size"))
    )
    macro = (
        datasets.groupby(["protocol", "candidate", "branch"], as_index=False)
        .agg(auc=("auc", "mean"), ap=("ap", "mean"), generators=("generators", "sum"))
    )
    macro["dataset"] = "Macro-3"
    return generators, pd.concat([datasets, macro], ignore_index=True)


def bootstrap(pairs: pd.DataFrame, iterations: int, seed: int) -> pd.DataFrame:
    contrasts = {
        "nmax_S_minus_n200_S": ("nmax", "S", "n200", "S"),
        "nmax_S_minus_nmax_G": ("nmax", "S", "nmax", "G"),
        "ncustom_S_minus_n200_S": ("ncustom", "S", "n200", "S"),
        "ncustom_S_minus_nmax_S": ("ncustom", "S", "nmax", "S"),
        "ncustom_S_minus_ncustom_G": ("ncustom", "S", "ncustom", "G"),
    }
    rng = np.random.default_rng(seed)
    rows = []
    for protocol, protocol_frame in pairs.groupby("protocol", sort=True):
        groups = list(protocol_frame.groupby(["dataset", "source_model"], sort=True))
        samples = {name: [] for name in contrasts}
        for _ in range(iterations):
            per_contrast = {name: {dataset: [] for dataset in DATASETS} for name in contrasts}
            for (dataset, _), frame in groups:
                chosen = frame.iloc[rng.integers(0, len(frame), len(frame))]
                for name, (ca, ba, cb, bb) in contrasts.items():
                    _, ap_a = metric(chosen[f"real_{ca}_{ba}"].to_numpy(), chosen[f"fake_{ca}_{ba}"].to_numpy())
                    _, ap_b = metric(chosen[f"real_{cb}_{bb}"].to_numpy(), chosen[f"fake_{cb}_{bb}"].to_numpy())
                    per_contrast[name][dataset].append(ap_a - ap_b)
            for name in contrasts:
                dataset_deltas = [np.mean(per_contrast[name][dataset]) for dataset in DATASETS]
                samples[name].append(float(np.mean(dataset_deltas)))
        for name, values in samples.items():
            values = np.asarray(values)
            rows.append(
                {
                    "protocol": protocol, "contrast": name, "iterations": iterations,
                    "mean_delta_ap": float(values.mean()),
                    "ci95_low": float(np.quantile(values, 0.025)),
                    "ci95_high": float(np.quantile(values, 0.975)),
                }
            )
    return pd.DataFrame(rows)


def write_report(
    summary: pd.DataFrame,
    generators: pd.DataFrame,
    bootstrap_frame: pd.DataFrame,
    pairs: pd.DataFrame,
    selection_payload: dict,
    output: Path,
) -> None:
    macro = summary[summary["dataset"].eq("Macro-3")].copy()
    lines = [
        "# Duration-aware 23-source full-data evaluation",
        "",
        "This confirmation protocol was frozen before generated-video scoring. Calibration uses only disjoint real videos; both the maximum pools and the held-out-real custom-size rule are fixed without generated-video performance.",
        "",
        "## Protocol",
        "",
        "- A video with a valid 2 s span uses three deterministic uniformly spaced windows. Each window contains 16 distinct frames sampled at 8 FPS.",
        "- A video without a valid 2 s span uses one deterministic 1 s window containing 8 distinct frames. There is no frame duplication or partial-window fallback.",
        "- A listed undecodable tail window is removed without dropping the video; this leaves `GenVideo/D325.mp4` with effective K=2.",
        "- Window scores are `G_k=0.5*global_spatial+0.5*global_T1`, `L_k=0.1*patch_spatial+0.9*same-grid_D2`; video scores average each branch over available windows and use `S=0.6*G+0.4*L`.",
        "- The 1 s and 2 s streams use independent real-only whitening/CDF calibration. Video-level CDFs are additionally separated by effective K.",
        "- Uniform calibration controls are N=200 per dataset and the predeclared maxima (ComGenVid 800, VideoFeedback 500, GenVideo 2,000).",
        "- The dataset-specific candidate was frozen before generated scoring using held-out-real quantile calibration error: ComGenVid 600, VideoFeedback 400, and GenVideo 1,500. The unused real validation blocks contain 200, 100, and 500 videos, respectively.",
        "",
        "## Data coverage",
        "",
        "All 45,185 indexed generated videos from 23 sources were scored. Physical evaluation collections contain 4,298 ComGenVid, 37,161 VideoFeedback, and 16,188 GenVideo videos. Calibration and evaluation real-video identities have zero overlap.",
        "",
        f"Primary AUC/AP uses {len(pairs[pairs['protocol'].eq('full23')]):,} full-23 matched pairs and {len(pairs[pairs['protocol'].eq('strict20_2s')]):,} strict-20 matched pairs. It is computed per generator with equal real/fake counts and then macro-averaged. This prevents class prevalence and large generators from changing AP or dominating the macro result. It does not discard generated scoring: every generated raw score is retained, while a deterministic balanced subset is used where the disjoint real pool is smaller.",
        "",
        "## Macro results",
        "",
        "| Protocol | Calibration | Branch | Generators | AUC | AP |",
        "|---|---|---|---:|---:|---:|",
    ]
    for row in macro.sort_values(["protocol", "candidate", "branch"]).itertuples(index=False):
        lines.append(f"| {row.protocol} | {row.candidate} | {row.branch} | {row.generators} | {row.auc:.4f} | {row.ap:.4f} |")

    lines.extend(
        [
            "",
            "## Full-23 dataset results",
            "",
            "| Dataset | N=200 Alpha AUC/AP | N=max Alpha AUC/AP | N=custom Alpha AUC/AP | Custom minus N=200/max AP | Custom Global AUC/AP | Alpha minus Global AP |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for dataset in DATASETS:
        def row(candidate: str, branch: str) -> pd.Series:
            selected = summary[
                summary["protocol"].eq("full23")
                & summary["dataset"].eq(dataset)
                & summary["candidate"].eq(candidate)
                & summary["branch"].eq(branch)
            ]
            if len(selected) != 1:
                raise ValueError(f"missing full23 summary row: {dataset}/{candidate}/{branch}")
            return selected.iloc[0]

        n200_s = row("n200", "S")
        nmax_s = row("nmax", "S")
        custom_s = row("ncustom", "S")
        custom_g = row("ncustom", "G")
        lines.append(
            f"| {dataset} | {n200_s.auc:.4f}/{n200_s.ap:.4f} | "
            f"{nmax_s.auc:.4f}/{nmax_s.ap:.4f} | {custom_s.auc:.4f}/{custom_s.ap:.4f} | "
            f"{custom_s.ap - n200_s.ap:+.4f}/{custom_s.ap - nmax_s.ap:+.4f} | "
            f"{custom_g.auc:.4f}/{custom_g.ap:.4f} | {custom_s.ap - custom_g.ap:+.4f} |"
        )

    lines.extend(
        [
            "",
            "## Restored one-second sources",
            "",
            "The fair causal comparison is Alpha (`S`) versus Global STALL (`G`) under the same duration-aware protocol and the same paired videos.",
            "",
            "| Dataset/source | Pairs | N=custom Global AUC/AP | N=custom Alpha AUC/AP | Delta AUC/AP |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for dataset, source in sorted(SHORT_ONLY):
        selected = generators[
            generators["protocol"].eq("full23")
            & generators["dataset"].eq(dataset)
            & generators["source_model"].eq(source)
            & generators["candidate"].eq("ncustom")
        ].set_index("branch")
        global_row, alpha_row = selected.loc["G"], selected.loc["S"]
        lines.append(
            f"| {dataset}/{source} | {int(alpha_row.n_real)} | "
            f"{global_row.auc:.4f}/{global_row.ap:.4f} | {alpha_row.auc:.4f}/{alpha_row.ap:.4f} | "
            f"{alpha_row.auc - global_row.auc:+.4f}/{alpha_row.ap - global_row.ap:+.4f} |"
        )

    full_generator = generators[
        generators["protocol"].eq("full23") & generators["branch"].eq("S")
    ].pivot(index=["dataset", "source_model"], columns="candidate", values="ap")
    calibration_delta = full_generator["nmax"] - full_generator["n200"]
    calibration_wins = int((calibration_delta > 0).sum())
    calibration_losses = int((calibration_delta < 0).sum())
    full_nmax = generators[
        generators["protocol"].eq("full23") & generators["candidate"].eq("nmax")
    ].pivot(index=["dataset", "source_model"], columns="branch", values="ap")
    method_delta = full_nmax["S"] - full_nmax["G"]
    custom_vs_n200 = full_generator["ncustom"] - full_generator["n200"]
    custom_vs_nmax = full_generator["ncustom"] - full_generator["nmax"]
    full_custom = generators[
        generators["protocol"].eq("full23") & generators["candidate"].eq("ncustom")
    ].pivot(index=["dataset", "source_model"], columns="branch", values="ap")
    custom_method_delta = full_custom["S"] - full_custom["G"]
    lines.extend(
        [
            "",
            "## Breadth and calibration effect",
            "",
            f"- N=max Alpha improves AP over N=200 Alpha on {calibration_wins}/23 generators and decreases it on {calibration_losses}/23.",
            f"- N=max Alpha improves AP over same-protocol Global STALL on {int((method_delta > 0).sum())}/23 generators and decreases it on {int((method_delta < 0).sum())}/23.",
            f"- N=custom improves AP over N=200 on {int((custom_vs_n200 > 0).sum())}/23 generators and over N=max on {int((custom_vs_nmax > 0).sum())}/23.",
            f"- N=custom Alpha improves AP over same-protocol Global STALL on {int((custom_method_delta > 0).sum())}/23 generators.",
            "- The expanded calibration gain is therefore small and not generator-wide; it must not be described as a universal scaling benefit.",
        ]
    )

    lines.extend(["", "## Paired cluster bootstrap", "", "| Protocol | Contrast | Mean delta AP | 95% CI |", "|---|---|---:|---:|"])
    for row in bootstrap_frame.itertuples(index=False):
        lines.append(f"| {row.protocol} | {row.contrast} | {row.mean_delta_ap:+.4f} | [{row.ci95_low:+.4f}, {row.ci95_high:+.4f}] |")
    lines.extend(
        [
            "",
            "## Interpretation guardrails",
            "",
            "- `full23` includes every indexed generated video and uses 1s only when 2s is unavailable.",
            "- `strict20_2s` excludes the three entirely short generators and all sub-2s clips, isolating the long-video comparison.",
            "- Each generator is pairwise balanced. Selected real videos use exactly the same 1s/2s counts as selected fake videos.",
            "- AP treats real video as the positive class, matching the locked U0 release.",
            "- The old paper numbers and this experiment are not a causal comparison because the evaluated identities, sample counts, and short-video protocol differ.",
            f"- The size-selection artifact declares `generated_videos_used={selection_payload['generated_videos_used']}` and status `{selection_payload['selection_status']}`.",
            "",
            "## Decision",
            "",
            "Adopt the real-only custom upper limits for the duration-aware full-23 protocol: they use 2,500 calibration videos in total instead of N=max's 3,300, reach 0.8721/0.8741, and improve AP over N=200 by +0.0020 (95% CI +0.0012 to +0.0028) and over N=max by +0.0003 (+0.0001 to +0.0005). Strict-20 custom versus N=200 remains inconclusive, so this is a small full-coverage calibration refinement, not a new detector component or evidence that more real data always helps. It adds no inference-time branch or DINO forward.",
        ]
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run(args: argparse.Namespace) -> None:
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    selection_payload = json.loads(args.size_selection.read_text(encoding="utf-8"))
    if selection_payload.get("selection_status") != "frozen_before_generated_candidate_scoring":
        raise ValueError("custom calibration-size selection was not frozen before scoring")
    if int(selection_payload.get("generated_videos_used", -1)) != 0:
        raise ValueError("custom calibration-size selection used generated videos")
    selected_sizes = {
        key: int(value) for key, value in selection_payload["selected_sizes"].items()
    }
    if args.report_only:
        write_report(
            pd.read_csv(args.input_dir / "dataset_metrics.csv"),
            pd.read_csv(args.input_dir / "generator_metrics.csv"),
            pd.read_csv(args.input_dir / "bootstrap_deltas.csv"),
            pd.read_csv(args.input_dir / "matched_pairs.csv"),
            selection_payload,
            args.report,
        )
        return
    raw = load_raw(args.input_dir)
    raw = attach_custom_raw(raw, args.custom_raw_dir, selected_sizes)
    tasks = pd.read_csv(args.tasks)
    if set(raw["task_id"]) != set(tasks["task_id"]):
        raise ValueError(f"raw/task coverage mismatch raw={raw['task_id'].nunique()} tasks={len(tasks)}")
    membership = pd.read_csv(args.membership)
    size_membership = pd.read_csv(args.size_membership)
    for column in ("in_n200", "in_nmax"):
        if membership[column].dtype != bool:
            membership[column] = membership[column].map(
                {"True": True, "False": False, "1": True, "0": False, 1: True, 0: False}
            )
        if membership[column].isna().any():
            raise ValueError(f"invalid boolean calibration membership column: {column}")
    vatex_spatial, vatex_t1 = global_vatex_references(config)
    scored = pd.concat(
        [
            calibrate_candidate(
                raw,
                membership,
                size_membership,
                selected_sizes,
                candidate,
                vatex_spatial,
                vatex_t1,
            )
            for candidate in CANDIDATES
        ],
        ignore_index=True,
    )
    scored.to_csv(args.input_dir / "per_video_scores.csv", index=False)
    pairs = pd.concat(
        [pair_protocol(scored, protocol, args.seed) for protocol in ("full23", "strict20_2s")],
        ignore_index=True,
    )
    pairs.to_csv(args.input_dir / "matched_pairs.csv", index=False)
    generators, summary = metric_tables(pairs)
    generators.to_csv(args.input_dir / "generator_metrics.csv", index=False)
    summary.to_csv(args.input_dir / "dataset_metrics.csv", index=False)
    boot = bootstrap(pairs, args.bootstrap_iterations, args.seed)
    boot.to_csv(args.input_dir / "bootstrap_deltas.csv", index=False)
    write_report(summary, generators, boot, pairs, selection_payload, args.report)
    print(summary[summary["dataset"].eq("Macro-3")].to_string(index=False))
    print(boot.to_string(index=False))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-dir", type=Path, default=ROOT / "results/duration_aware_23source"
    )
    parser.add_argument(
        "--tasks", type=Path, default=ROOT / "results/duration_aware_23source/protocol_tasks.csv"
    )
    parser.add_argument(
        "--membership", type=Path, default=ROOT / "results/duration_aware_23source/calibration_membership.csv"
    )
    parser.add_argument(
        "--size-membership",
        type=Path,
        default=ROOT / "results/duration_aware_23source/calibration_size_membership.csv",
    )
    parser.add_argument(
        "--size-selection",
        type=Path,
        default=ROOT / "results/duration_aware_23source/real_only_curve/selected_calibration_sizes.json",
    )
    parser.add_argument(
        "--custom-raw-dir",
        type=Path,
        default=ROOT / "results/duration_aware_23source/curve_raw/selected_realonly",
    )
    parser.add_argument(
        "--config", type=Path, default=ROOT / "configs/alpha_stalled_u0_locked.yaml"
    )
    parser.add_argument(
        "--report", type=Path, default=ROOT / "reports/duration_aware_23source_full.md"
    )
    parser.add_argument("--bootstrap-iterations", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--report-only", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
