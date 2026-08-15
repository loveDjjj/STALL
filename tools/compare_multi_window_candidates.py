#!/usr/bin/env python3
"""Compare the predeclared multi-window candidates on identical video pairs."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
TOOLS_DIR = REPO_ROOT / "tools"
for directory in (REPO_ROOT / "src", TOOLS_DIR):
    if str(directory) not in sys.path:
        sys.path.insert(0, str(directory))

from alpha_stalled.metrics import macro_cluster_bootstrap, metric_tables, paired_bootstrap


KEY_COLUMNS = ["dataset", "subset", "source_model", "filename"]
CONFIG_NAMES = {
    "K3_MW2": "K=3 branch means",
    "K3_MW4": "K=3 local hybrid",
    "K5_MW2": "K=5 branch means",
    "K5_MW4": "K=5 local hybrid",
    "all_MW2": "All non-overlap branch means",
    "all_MW4": "All non-overlap local hybrid",
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--result-dir",
        type=Path,
        default=REPO_ROOT / "results/multi_window_joint_typicality",
    )
    parser.add_argument("--bootstrap-iterations", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    merged: pd.DataFrame | None = None
    for prefix, filename in (
        ("K3", "K3_uniform_evaluation_scores.csv"),
        ("K5", "K5_uniform_evaluation_scores.csv"),
        ("all", "all_nonoverlap_evaluation_scores.csv"),
    ):
        frame = pd.read_csv(
            args.result_dir / filename, float_precision="round_trip"
        )
        frame = frame[KEY_COLUMNS + ["MW2", "MW4"]].rename(
            columns={"MW2": f"{prefix}_MW2", "MW4": f"{prefix}_MW4"}
        )
        merged = frame if merged is None else merged.merge(
            frame, on=KEY_COLUMNS, how="inner", validate="one_to_one"
        )
    assert merged is not None
    if len(merged) != 21421:
        raise ValueError(f"candidate intersection changed: {len(merged)}")

    configs = tuple(CONFIG_NAMES)
    metrics, generator_metrics = metric_tables(
        merged,
        args.seed,
        score_columns=configs,
        config_names=CONFIG_NAMES,
    )
    comparisons = tuple(
        (config, "K3_MW2", f"{config}-K3_MW2")
        for config in configs
        if config != "K3_MW2"
    )
    dataset_bootstrap = paired_bootstrap(
        merged,
        args.seed,
        args.bootstrap_iterations,
        comparisons=comparisons,
    )
    aliased = merged.copy()
    aliased["MW0"] = aliased["K3_MW2"]
    macro_bootstrap = macro_cluster_bootstrap(
        aliased,
        [config for config in configs if config != "K3_MW2"],
        args.seed,
        args.bootstrap_iterations,
    )
    macro_bootstrap["base_config"] = "K3_MW2"
    macro_bootstrap["comparison"] = (
        macro_bootstrap["new_config"] + "-K3_MW2"
    )

    merged.to_csv(args.result_dir / "multi_window_candidate_scores.csv", index=False)
    metrics.to_csv(args.result_dir / "multi_window_candidate_metrics.csv", index=False)
    generator_metrics.to_csv(
        args.result_dir / "multi_window_candidate_generator_metrics.csv", index=False
    )
    pd.concat([dataset_bootstrap, macro_bootstrap], ignore_index=True).to_csv(
        args.result_dir / "multi_window_candidate_bootstrap.csv", index=False
    )
    print(f"videos={len(merged)} saved -> {args.result_dir}")


if __name__ == "__main__":
    main()
