#!/usr/bin/env python3
"""从Tail likelihood field shards评估三selector、四聚合和standard/crossfit5。"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import torch


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from config import load_config
from evaluation.tables import build_metric_tables
from pipeline import _calibrate_and_aggregate
from tail_evidence import TAIL_RATIOS
from tail_calibration import conformal_tail_authenticity_multi, fit_position_reference
from temporal_selection.evaluation import build_matched_selector_pairwise_table


SELECTORS = ("uniform", "feature_change", "real_anomaly")
CALIBRATION_MODES = ("standard", "crossfit5")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _shard_summaries(index: dict):
    for dataset in index["datasets"].values():
        for shards in dataset.values():
            yield from shards


def _load_shard(field_dir: Path, index: dict, summary: dict) -> dict:
    path = field_dir / summary["path"]
    if _sha256(path) != summary["sha256"]:
        raise ValueError(f"Tail field shard哈希漂移：{path}")
    payload = torch.load(path, weights_only=True)
    if payload.get("run_identity_sha256") != index["identity_sha256"]:
        raise ValueError(f"Tail field shard运行身份漂移：{path}")
    return payload


def _load_records(field_dir: Path) -> tuple[pd.DataFrame, dict, dict]:
    """两遍流式读取：先拟合selector/mode位置CDF，再转换每个field的Tail分数。"""

    index_path = field_dir / "index.json"
    index = json.loads(index_path.read_text(encoding="utf-8"))
    if index.get("status") != "completed":
        raise ValueError("Tail field index尚未完成")
    reference_parts: dict[tuple[str, str, str], list[np.ndarray]] = {}
    for summary in _shard_summaries(index):
        if "/calibration/" not in f"/{summary['path']}/":
            continue
        payload = _load_shard(field_dir, index, summary)
        fields = payload["local_likelihood_fields"].numpy()
        for record in payload["records"]:
            key = (
                str(record["dataset"]),
                str(record["selector"]),
                str(record["calibration_mode"]),
            )
            reference_parts.setdefault(key, []).append(
                fields[int(record["field_index"])].copy()
            )
    position_references = {
        key: fit_position_reference(parts)
        for key, parts in reference_parts.items()
    }
    for dataset in index["datasets"]:
        for selector in SELECTORS:
            if (dataset, selector, "standard") not in position_references:
                raise ValueError(f"缺少位置reference：{dataset}/{selector}/standard")
        if (dataset, "real_anomaly", "crossfit5") not in position_references:
            raise ValueError(f"缺少位置reference：{dataset}/real_anomaly/crossfit5")

    records = []
    tail_ratios = {
        name: ratio for name, ratio in TAIL_RATIOS.items() if name != "mean"
    }
    for summary in _shard_summaries(index):
        payload = _load_shard(field_dir, index, summary)
        fields = payload["local_likelihood_fields"].numpy()
        shard_records = [dict(item) for item in payload["records"]]
        grouped: dict[tuple[str, str], list[int]] = {}
        for record_index, record in enumerate(shard_records):
            grouped.setdefault(
                (str(record["dataset"]), str(record["selector"])), []
            ).append(record_index)
        for (dataset, selector), record_indices in grouped.items():
            field_indices = [
                int(shard_records[index]["field_index"])
                for index in record_indices
            ]
            selected_fields = fields[field_indices]
            modes_by_reference = (
                {"standard": ("standard",), "crossfit5": ("crossfit5",)}
                if selector == "real_anomaly"
                else {"standard": CALIBRATION_MODES}
            )
            for reference_mode, output_modes in modes_by_reference.items():
                reference = position_references[(dataset, selector, reference_mode)]
                values_by_name = conformal_tail_authenticity_multi(
                    selected_fields, reference, tail_ratios
                )
                for mode in output_modes:
                    for name, values in values_by_name.items():
                        for record_index, value in zip(record_indices, values):
                            shard_records[record_index][
                                f"local_raw__{name}__{mode}"
                            ] = float(value)
        records.extend(shard_records)
    frame = pd.DataFrame(records)
    if frame.empty:
        raise ValueError("Tail field shards没有records")
    # Stage FS按selector的原始selection rank分配window_id。effective-K参考会从
    # calibration的完整K3窗口中取子集，因此不能把adaptive窗口改成时间顺序。
    group_columns = [
        "dataset", "split", "video_id", "selector", "calibration_mode"
    ]
    frame = frame.sort_values(
        [*group_columns, "selection_rank"], kind="mergesort"
    ).reset_index(drop=True)
    frame["window_id"] = frame.groupby(
        group_columns, sort=False
    ).cumcount()
    reference_metadata = {
        "/".join(key): {
            "positions": len(value),
            "sha256": hashlib.sha256(value.tobytes()).hexdigest(),
        }
        for key, value in position_references.items()
    }
    return frame, index, reference_metadata


def _variant(selector: str, aggregation: str, calibration_mode: str) -> str:
    return f"{selector}__{aggregation}__{calibration_mode}"


def _attach_frozen_standard_scores(
    records: pd.DataFrame, source_windows: pd.DataFrame
) -> pd.DataFrame:
    """为standard窗口复用Stage FS分量，确保Tail实验只改变Local聚合。"""

    output = records.copy().reset_index(drop=True)
    output["_record_index"] = np.arange(len(output))
    score_columns = [
        "global_spatial_raw", "global_spatial", "global_t1_raw", "global_t1",
        "patch_temporal_raw",
    ]
    for selector in SELECTORS:
        target = output[
            output["selector"].eq(selector)
            & output["calibration_mode"].eq("standard")
        ].copy()
        source = source_windows[source_windows["selector"].eq(selector)].copy()
        keys = ["video_id", "split", "selector"]
        keys.append("window_id" if selector == "uniform" else "candidate_id")
        source = source[keys + score_columns]
        existing_score_columns = [
            column for column in score_columns if column in target.columns
        ]
        merged = target.drop(columns=existing_score_columns).merge(
            source,
            on=keys,
            how="left",
            validate="one_to_one",
        )
        if len(merged) != len(target) or merged[score_columns].isna().any().any():
            raise ValueError(f"{selector}无法与Stage FS窗口分量严格对齐")
        indices = merged["_record_index"].to_numpy(dtype=int)
        for column in score_columns:
            output.loc[indices, column] = merged[column].to_numpy()
        output.loc[indices, "local_raw__mean"] = merged[
            "patch_temporal_raw"
        ].to_numpy()
    return output.drop(columns="_record_index")


def _score_variant(
    records: pd.DataFrame,
    *,
    selector: str,
    aggregation: str,
    calibration_mode: str,
    config: dict,
) -> pd.DataFrame:
    calibration_record_mode = (
        "crossfit5"
        if calibration_mode == "crossfit5" and selector == "real_anomaly"
        else "standard"
    )
    calibration = records[
        records["split"].eq("calibration")
        & records["selector"].eq(selector)
        & records["calibration_mode"].eq(calibration_record_mode)
    ].copy()
    evaluation = records[
        records["split"].eq("evaluation")
        & records["selector"].eq(selector)
        & records["calibration_mode"].eq("standard")
    ].copy()
    raw_column = (
        "local_raw__mean"
        if aggregation == "mean"
        else f"local_raw__{aggregation}__{calibration_mode}"
    )
    if calibration.empty or evaluation.empty or raw_column not in records:
        raise ValueError(
            f"缺少Tail变体输入：{selector}/{aggregation}/{calibration_mode}"
        )
    calibration["patch_temporal_raw"] = calibration[raw_column]
    evaluation["patch_temporal_raw"] = evaluation[raw_column]
    _, videos = _calibrate_and_aggregate(calibration, evaluation, config)
    videos.insert(0, "calibration_mode", calibration_mode)
    videos.insert(0, "aggregation", aggregation)
    videos.insert(0, "selector", selector)
    videos.insert(0, "variant", _variant(selector, aggregation, calibration_mode))
    if not np.isfinite(videos["final_score"].to_numpy(dtype=np.float64)).all():
        raise ValueError("Tail视频分数包含非有限值")
    return videos


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--field-dir", type=Path,
        default=ROOT / "results/caes/tail_fields_v1",
    )
    parser.add_argument(
        "--output-dir", type=Path,
        default=ROOT / "results/runs/caes_tail_matrix",
    )
    parser.add_argument(
        "--source-stage-fs", type=Path,
        default=ROOT / "results/runs/caes_stage_fs",
    )
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if args.output_dir.exists() and not args.overwrite:
        raise FileExistsError(f"Tail评测输出已存在：{args.output_dir}")
    if args.output_dir.exists() and args.overwrite:
        import shutil
        shutil.rmtree(args.output_dir)
    args.output_dir.mkdir(parents=True)
    records, field_index, position_references = _load_records(args.field_dir)
    config = load_config(args.field_dir / "resolved_config.yaml")
    source_windows = pd.read_csv(
        args.source_stage_fs / "window_scores.csv", float_precision="round_trip",
        low_memory=False,
    )
    records = _attach_frozen_standard_scores(records, source_windows)
    videos = []
    for selector in SELECTORS:
        for aggregation in TAIL_RATIOS:
            for calibration_mode in CALIBRATION_MODES:
                videos.append(_score_variant(
                    records,
                    selector=selector,
                    aggregation=aggregation,
                    calibration_mode=calibration_mode,
                    config=config,
                ))
    video_scores = pd.concat(videos, ignore_index=True)
    dataset_tables, generator_tables = [], []
    for variant, frame in video_scores.groupby("variant", sort=False):
        clean = frame.drop(
            columns=["variant", "selector", "aggregation", "calibration_mode"]
        )
        dataset, generator = build_metric_tables(clean, variant)
        metadata = frame[[
            "variant", "selector", "aggregation", "calibration_mode"
        ]].iloc[0].to_dict()
        for table in (dataset, generator):
            for position, (name, value) in enumerate(metadata.items()):
                table.insert(position, name, value)
        dataset_tables.append(dataset)
        generator_tables.append(generator)
    pair_input = video_scores.rename(columns={"variant": "tail_variant"}).copy()
    pair_input["selector"] = pair_input["tail_variant"]
    pairwise = build_matched_selector_pairwise_table(
        pair_input,
        baseline=_variant("uniform", "mean", "standard"),
        seed=int(config["metrics"]["pairwise_seed"]),
    ).rename(columns={"selector": "variant"})
    metadata = video_scores[[
        "variant", "selector", "aggregation", "calibration_mode"
    ]].drop_duplicates()
    pairwise = pairwise.merge(metadata, on="variant", how="left", validate="many_to_one")

    source = pd.read_csv(
        args.source_stage_fs / "video_scores.csv", float_precision="round_trip"
    )
    regression_rows = []
    for selector in SELECTORS:
        current = video_scores[
            video_scores["variant"].eq(_variant(selector, "mean", "standard"))
        ]
        expected = source[
            source["selector"].eq(selector)
            & source["video_id"].isin(set(current["video_id"]))
        ]
        merged = current.merge(
            expected[["video_id", "global_score", "local_score", "final_score"]],
            on="video_id", suffixes=("", "__source"), validate="one_to_one",
        )
        if len(merged) != len(expected):
            raise ValueError(f"{selector} Mean回归视频身份不完整")
        regression_rows.append({
            "selector": selector,
            "videos": len(merged),
            **{
                f"{column}_max_abs": float(np.max(np.abs(
                    merged[column] - merged[f"{column}__source"]
                )))
                for column in ("global_score", "local_score", "final_score")
            },
        })
    controls = video_scores[
        video_scores["selector"].isin(["uniform", "feature_change"])
    ].pivot(
        index=["selector", "aggregation", "video_id"],
        columns="calibration_mode",
        values="final_score",
    )
    control_max = float(np.max(np.abs(controls["standard"] - controls["crossfit5"])))
    if control_max != 0.0:
        raise ValueError(f"Uniform/Feature-change crossfit控制发生漂移：{control_max}")

    video_scores.to_csv(args.output_dir / "video_scores.csv", index=False)
    pd.concat(dataset_tables, ignore_index=True).to_csv(
        args.output_dir / "dataset_metrics.csv", index=False
    )
    pd.concat(generator_tables, ignore_index=True).to_csv(
        args.output_dir / "generator_metrics.csv", index=False
    )
    pairwise.to_csv(args.output_dir / "pairwise_metrics.csv", index=False)
    pd.DataFrame(regression_rows).to_csv(
        args.output_dir / "mean_regression.csv", index=False
    )
    manifest = {
        "schema_version": "caes_tail_evaluation_v1",
        "status": "completed",
        "completed_at": datetime.now(timezone.utc).isoformat(),
        "field_index_sha256": _sha256(args.field_dir / "index.json"),
        "field_run_identity_sha256": field_index["identity_sha256"],
        "selectors": list(SELECTORS),
        "aggregations": TAIL_RATIOS,
        "calibration_modes": list(CALIBRATION_MODES),
        "tail_definition": "negative_mean_of_top_real_nonconformity",
        "position_references": position_references,
        "crossfit_control_max_abs": control_max,
        "artifacts": {
            name: _sha256(args.output_dir / name)
            for name in (
                "video_scores.csv", "dataset_metrics.csv", "generator_metrics.csv",
                "pairwise_metrics.csv", "mean_regression.csv",
            )
        },
    }
    (args.output_dir / "run_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(
        pairwise[pairwise["dataset"].eq("Macro-3")][
            ["selector", "aggregation", "calibration_mode", "auc", "ap_real"]
        ].to_string(index=False)
    )
    print("\nMean回归：")
    print(pd.DataFrame(regression_rows).to_string(index=False))


if __name__ == "__main__":
    main()
