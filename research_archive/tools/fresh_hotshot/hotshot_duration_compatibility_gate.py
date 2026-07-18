#!/usr/bin/env python3
"""Gate Hotshot-XL duration compatibility against VideoFeedback patch params."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def _params_metadata(path: Path) -> dict[str, object]:
    data = np.load(path, allow_pickle=True)
    config = json.loads(str(data["aggregation_config"].item()))
    duration = int(data["duration"][0]) if "duration" in data.files else -1
    grid = tuple(int(x) for x in data["patch_grid_size"].tolist())
    return {
        "params_duration": duration,
        "params_patch_temp_mode": config.get("patch_temp_mode", ""),
        "params_patch_region_size": int(config.get("patch_region_size", -1)),
        "params_aggregation": config.get("mode", ""),
        "params_grid": f"{grid[0]}x{grid[1]}",
    }


def run(args: argparse.Namespace) -> pd.DataFrame:
    target = pd.read_csv(args.target_csv)
    summary = pd.read_csv(args.runbook_summary_csv)
    meta = _params_metadata(args.patch_params)
    target_duration_min = float(target["duration_seconds"].min())
    target_duration_max = float(target["duration_seconds"].max())
    runbook_duration = int(summary.iloc[0]["duration"])
    params_duration = int(meta["params_duration"])
    compatible = runbook_duration == params_duration
    decision = "PROTOCOL_COMPATIBLE" if compatible else "DURATION_MISMATCH_LOCAL_VARIANT_ONLY"
    return pd.DataFrame(
        [
            {
                "decision": decision,
                "protocol_compatible": compatible,
                "runbook_duration": runbook_duration,
                "params_duration": params_duration,
                "target_rows": int(len(target)),
                "target_duration_min": target_duration_min,
                "target_duration_max": target_duration_max,
                "target_all_one_second": bool((target["duration_seconds"].astype(float) == 1.0).all()),
                "allowed_next": "local_variant_only" if not compatible else "protocol_compatible_variant",
                "required_label": "local variant with duration=1 caveat" if not compatible else "protocol-compatible variant",
                **meta,
            }
        ]
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--target-csv", type=Path, required=True)
    parser.add_argument("--runbook-summary-csv", type=Path, required=True)
    parser.add_argument("--patch-params", type=Path, required=True)
    parser.add_argument("--output-csv", type=Path, required=True)
    args = parser.parse_args()

    decision = run(args)
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    decision.to_csv(args.output_csv, index=False)
    print(decision.to_string(index=False))


if __name__ == "__main__":
    main()
