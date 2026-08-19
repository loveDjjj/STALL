"""Alpha STALL 唯一的配置驱动执行路径。"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from artifacts import create_run_directory, write_csv, write_run_manifest
from pipeline import build_bootstrap, run_from_cache
from evaluation.tables import build_metric_tables, normalize_scores


def run(
    repository_root: Path,
    run_name: str,
    config: dict,
    *,
    dry_run: bool,
    overwrite: bool,
    scores_csv: Path | None,
) -> Path:
    output_dir = create_run_directory(repository_root, run_name, overwrite=overwrite)
    if dry_run:
        write_run_manifest(output_dir, repository_root, run_name, config, status="planned")
        return output_dir
    windows = None
    pipeline_metadata = None
    if scores_csv is None:
        windows, scores, pipeline_metadata = run_from_cache(repository_root, config)
    else:
        scores = normalize_scores(pd.read_csv(scores_csv))
        selected = config["data"]["datasets"]
        scores = scores[scores["dataset"].isin(selected)].copy()
        if scores.empty:
            raise ValueError("分数 CSV 不包含当前配置数据集的任何记录")
    dataset_metrics, generator_metrics = build_metric_tables(scores, run_name)
    write_csv(output_dir, "video_scores.csv", scores)
    write_csv(output_dir, "dataset_metrics.csv", dataset_metrics)
    write_csv(output_dir, "generator_metrics.csv", generator_metrics)
    if windows is not None:
        write_csv(output_dir, "window_scores.csv", windows)
        bootstrap = build_bootstrap(scores, config)
        if not bootstrap.empty:
            write_csv(output_dir, "bootstrap_metrics.csv", bootstrap)
    write_run_manifest(
        output_dir,
        repository_root,
        run_name,
        config,
        status="completed",
        input_scores=str(scores_csv) if scores_csv is not None else None,
        pipeline_metadata=pipeline_metadata,
    )
    return output_dir
