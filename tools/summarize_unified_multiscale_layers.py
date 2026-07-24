#!/usr/bin/env python3
"""Build admission and focus-generator summaries for unified MS/H experiments."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
FOCUS = {
    "ZeroScope-576w",
    "Pika",
    "SoRA-Clip",
    "Crafter",
    "Text2Video-Zero",
}


def summarize_family(root: Path, prefix: str, base: str) -> tuple[list[dict], pd.DataFrame]:
    dataset = pd.read_csv(root / f"{prefix}_dataset_metrics.csv")
    generator = pd.read_csv(root / f"{prefix}_generator_metrics.csv")
    bootstrap = pd.read_csv(root / f"{prefix}_bootstrap_deltas.csv")
    macro = dataset[dataset["dataset"] == "Macro-3"].set_index("config")
    candidates = [config for config in macro.index if config != base]
    rows = []
    for config in candidates:
        dataset_pair = dataset[
            (dataset["dataset"] != "Macro-3")
            & dataset["config"].isin([base, config])
        ].pivot(index="dataset", columns="config", values="ap")
        generator_pair = generator[generator["config"].isin([base, config])].pivot(
            index=["dataset", "generator"], columns="config", values="ap"
        )
        ci = bootstrap[
            (bootstrap["dataset"] == "Macro-3")
            & (bootstrap["new_config"] == config)
            & (bootstrap["metric"] == "ap")
        ].iloc[0]
        delta = float(macro.loc[config, "ap"] - macro.loc[base, "ap"])
        worst = float((dataset_pair[config] - dataset_pair[base]).min())
        nondecline = int((generator_pair[config] >= generator_pair[base]).sum())
        rows.append(
            {
                "family": prefix,
                "config": config,
                "base_config": base,
                "macro_auc": float(macro.loc[config, "auc"]),
                "macro_ap": float(macro.loc[config, "ap"]),
                "delta_macro_ap": delta,
                "worst_dataset_ap_delta": worst,
                "generator_non_decline_count": nondecline,
                "generator_total": len(generator_pair),
                "macro_ap_ci95_low": float(ci.ci95_low),
                "macro_ap_ci95_high": float(ci.ci95_high),
                "passes_macro_ap": delta >= 0.004,
                "passes_worst_dataset": worst >= -0.004,
                "passes_generator_count": nondecline >= 12,
                "passes_bootstrap": float(ci.ci95_low) > 0,
                "admitted": bool(
                    delta >= 0.004
                    and worst >= -0.004
                    and nondecline >= 12
                    and float(ci.ci95_low) > 0
                ),
                "triggers_beta": bool(prefix == "layer" and delta >= 0.004),
            }
        )
    focus = generator[
        generator["generator"].isin(FOCUS)
        & generator["config"].isin([base, *candidates])
    ].copy()
    baseline = focus[focus["config"] == base].set_index(["dataset", "generator"])
    focus["delta_auc_vs_base"] = focus.apply(
        lambda row: row.auc - baseline.loc[(row.dataset, row.generator), "auc"], axis=1
    )
    focus["delta_ap_vs_base"] = focus.apply(
        lambda row: row.ap - baseline.loc[(row.dataset, row.generator), "ap"], axis=1
    )
    focus.insert(0, "family", prefix)
    focus["base_config"] = base
    return rows, focus


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=REPO_ROOT / "results/unified_multiscale_layers",
    )
    args = parser.parse_args()
    admission_rows = []
    focus_frames = []
    for prefix, base in (("multiscale", "MS0"), ("layer", "H0")):
        rows, focus = summarize_family(args.results_dir, prefix, base)
        admission_rows.extend(rows)
        focus_frames.append(focus)
    admission = pd.DataFrame(admission_rows)
    admission.to_csv(args.results_dir / "admission_summary.csv", index=False)
    pd.concat(focus_frames, ignore_index=True).to_csv(
        args.results_dir / "focus_generator_metrics.csv", index=False
    )
    print(admission.to_string(index=False))


if __name__ == "__main__":
    main()
