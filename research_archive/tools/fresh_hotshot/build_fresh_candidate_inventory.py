#!/usr/bin/env python3
"""Build a local inventory of possible fresh-validation candidates."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd


KNOWN_CANDIDATES = [
    {
        "candidate": "comgenvid",
        "global_csv": "results/comgenvid_results.csv",
        "raw_patch_csv": "results/patch_region3_bottomk_sweep/comgenvid_patch_region3_second_order_bottomk0p50_spat10_temp90.csv",
        "persistence_csv": "results/patch_calibrated_persistence/comgenvid_persistence_full.csv",
        "persistence_score_col": "anom_mass_thr0p2_real_pct_real",
        "index_csv": "cache/indexes/comgenvid.csv",
        "note": "internal gate dataset",
    },
    {
        "candidate": "genvideo",
        "global_csv": "results/genvideo_results.csv",
        "raw_patch_csv": "results/patch_cross_dataset_ablation/genvideo_patch_same_grid_second_order_region2_mean_spat10_temp90.csv",
        "persistence_csv": "results/patch_calibrated_persistence/genvideo_persistence_full.csv",
        "persistence_score_col": "temp_pct_mean_real_pct_real",
        "index_csv": "cache/indexes/genvideo.csv",
        "note": "internal gate dataset",
    },
    {
        "candidate": "videofeedback",
        "global_csv": "results/videofeedback_results.csv",
        "raw_patch_csv": "results/patch_cross_dataset_ablation/videofeedback_patch_same_grid_second_order_region1_mean_spat10_temp90.csv",
        "persistence_csv": "results/patch_calibrated_persistence/videofeedback_persistence_full.csv",
        "persistence_score_col": "max_frame_mass_thr0p2_real_pct_real",
        "index_csv": "cache/indexes/videofeedback.csv",
        "note": "internal gate dataset",
    },
    {
        "candidate": "genvideo_1s",
        "global_csv": "results/genvideo_1s_results.csv",
        "raw_patch_csv": "results/patch_cross_dataset_ablation/genvideo_patch_same_grid_second_order_region2_mean_spat10_temp90.csv",
        "persistence_csv": "results/patch_calibrated_persistence/genvideo_persistence_full.csv",
        "persistence_score_col": "temp_pct_mean_real_pct_real",
        "index_csv": "cache/indexes/genvideo.csv",
        "note": "same GenVideo rows with alternate global sampling",
    },
    {
        "candidate": "videofeedback_hotshot",
        "global_csv": "results/videofeedback_hotshot_results.csv",
        "raw_patch_csv": "results/patch_calibrated_persistence/videofeedback_hotshot_patch_same_grid_second_order_region1_mean_spat10_temp90_full.csv",
        "persistence_csv": "results/patch_calibrated_persistence/videofeedback_hotshot_persistence_full.csv",
        "persistence_score_col": "max_frame_mass_thr0p2_real_pct_real",
        "index_csv": "cache/indexes/videofeedback.csv",
        "fresh_status_override": "NOT_FRESH_LOCAL_VARIANT_WITH_DURATION_CAVEAT",
        "note": "completed local variant; duration=1 target with duration=2 real-calibration caveat",
    },
]

KEY_COLUMNS = ["subset", "source_model", "filename"]


def _manifest_used_paths(manifest: dict[str, Any]) -> set[str]:
    used: set[str] = set()
    for protocol in manifest.get("protocols", {}).values():
        for dataset in protocol.get("datasets", []):
            for key in ["global_csv", "raw_patch_csv", "persistence_csv"]:
                value = dataset.get(key)
                if value:
                    used.add(str(value))
    return used


def _csv_stats(path: Path, score_col: str = "final_score") -> dict[str, object]:
    if not path.exists():
        return {"exists": False}
    df = pd.read_csv(path)
    stats: dict[str, object] = {
        "exists": True,
        "rows": int(len(df)),
        "has_key_columns": all(col in df.columns for col in KEY_COLUMNS),
        "has_score_col": score_col in df.columns,
    }
    if all(col in df.columns for col in KEY_COLUMNS):
        keyed = df[KEY_COLUMNS].astype(str)
        stats.update(
            {
                "n_real": int((keyed["subset"].str.lower() == "real").sum()),
                "n_fake": int((keyed["subset"].str.lower() != "real").sum()),
                "n_fake_sources": int(keyed.loc[keyed["subset"].str.lower() != "real", "source_model"].nunique()),
                "duplicate_keys": int(keyed.duplicated(KEY_COLUMNS).sum()),
            }
        )
    return stats


def _candidate_row(root: Path, cand: dict[str, str], used_paths: set[str]) -> dict[str, object]:
    global_path = root / cand["global_csv"]
    raw_patch_path = root / cand["raw_patch_csv"] if cand["raw_patch_csv"] else Path("")
    persistence_path = root / cand["persistence_csv"] if cand["persistence_csv"] else Path("")
    index_path = root / cand["index_csv"]

    global_stats = _csv_stats(global_path)
    raw_patch_stats = _csv_stats(raw_patch_path) if cand["raw_patch_csv"] else {"exists": False}
    persistence_stats = _csv_stats(persistence_path, cand["persistence_score_col"]) if cand["persistence_csv"] else {"exists": False}
    index_stats = _csv_stats(index_path, score_col="")

    has_triplet = bool(global_stats.get("exists") and raw_patch_stats.get("exists") and persistence_stats.get("exists"))
    keys_ready = bool(
        global_stats.get("has_key_columns")
        and raw_patch_stats.get("has_key_columns")
        and persistence_stats.get("has_key_columns")
        and global_stats.get("has_score_col")
        and raw_patch_stats.get("has_score_col")
        and persistence_stats.get("has_score_col")
    )
    no_dupes = bool(
        global_stats.get("duplicate_keys", 1) == 0
        and raw_patch_stats.get("duplicate_keys", 1) == 0
        and persistence_stats.get("duplicate_keys", 1) == 0
    )
    has_labels = bool(global_stats.get("n_real", 0) > 0 and global_stats.get("n_fake", 0) > 0 and global_stats.get("n_fake_sources", 0) > 0)
    scaffold_ready = has_triplet and keys_ready and no_dupes and has_labels
    overlaps_manifest = any(cand.get(key, "") in used_paths for key in ["global_csv", "raw_patch_csv", "persistence_csv"] if cand.get(key))
    if cand.get("fresh_status_override"):
        fresh_status = cand["fresh_status_override"]
    else:
        fresh_status = "NOT_FRESH_INTERNAL_OR_VARIANT" if overlaps_manifest else ("MISSING_REQUIRED_INPUTS" if not scaffold_ready else "LOCAL_READY_NEEDS_PROVENANCE")

    return {
        "candidate": cand["candidate"],
        "fresh_status": fresh_status,
        "scaffold_ready": scaffold_ready,
        "overlaps_manifest_inputs": overlaps_manifest,
        "global_csv": cand["global_csv"],
        "raw_patch_csv": cand["raw_patch_csv"],
        "persistence_csv": cand["persistence_csv"],
        "persistence_score_col": cand["persistence_score_col"],
        "index_csv": cand["index_csv"],
        "global_exists": bool(global_stats.get("exists")),
        "raw_patch_exists": bool(raw_patch_stats.get("exists")),
        "persistence_exists": bool(persistence_stats.get("exists")),
        "index_exists": bool(index_stats.get("exists")),
        "global_rows": int(global_stats.get("rows", 0)),
        "raw_patch_rows": int(raw_patch_stats.get("rows", 0)),
        "persistence_rows": int(persistence_stats.get("rows", 0)),
        "index_rows": int(index_stats.get("rows", 0)),
        "n_real": int(global_stats.get("n_real", 0)),
        "n_fake": int(global_stats.get("n_fake", 0)),
        "n_fake_sources": int(global_stats.get("n_fake_sources", 0)),
        "global_duplicate_keys": int(global_stats.get("duplicate_keys", -1)),
        "raw_patch_duplicate_keys": int(raw_patch_stats.get("duplicate_keys", -1)),
        "persistence_duplicate_keys": int(persistence_stats.get("duplicate_keys", -1)),
        "note": cand["note"],
    }


def run(args: argparse.Namespace) -> pd.DataFrame:
    manifest = json.loads(args.manifest_json.read_text(encoding="utf-8"))
    used_paths = _manifest_used_paths(manifest)
    rows = [_candidate_row(args.root, cand, used_paths) for cand in KNOWN_CANDIDATES]
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--manifest-json", type=Path, required=True)
    parser.add_argument("--output-csv", type=Path, required=True)
    args = parser.parse_args()

    inventory = run(args)
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    inventory.to_csv(args.output_csv, index=False)
    print(inventory.to_string(index=False))
    print(f"Saved inventory -> {args.output_csv}")


if __name__ == "__main__":
    main()
