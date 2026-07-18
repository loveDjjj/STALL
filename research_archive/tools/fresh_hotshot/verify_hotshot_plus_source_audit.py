#!/usr/bin/env python3
"""Verify the expanded source audit that includes the Hotshot local variant."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


REQUIRED_AUDIT_COLUMNS = {
    "dataset",
    "source_model",
    "default_auc",
    "default_ap",
    "n_fake",
    "fallback_auc",
    "fallback_ap",
    "delta_auc",
    "delta_ap",
}

REQUIRED_SUMMARY_COLUMNS = {
    "dataset",
    "n_sources",
    "n_negative_auc",
    "n_negative_ap",
    "min_delta_auc",
    "min_delta_ap",
    "mean_delta_auc",
    "mean_delta_ap",
    "weighted_mean_delta_auc",
    "weighted_mean_delta_ap",
    "worst_auc_source",
    "worst_ap_source",
}

EXPECTED_DATASETS = {
    "comgenvid": 2,
    "genvideo": 8,
    "videofeedback": 10,
    "videofeedback_hotshot_local_variant": 11,
}

EXPECTED_HOTSHOT_SOURCES = {
    "AnimateDiff",
    "Fast-SVD",
    "Hotshot-XL",
    "LVDM",
    "LaVie-base",
    "ModelScope",
    "Pika",
    "SoRA-Clip",
    "Text2Video-Zero",
    "VideoCrafter2",
    "ZeroScope-576w",
}


def _add(rows: list[dict[str, object]], check: str, passed: bool, detail: str) -> None:
    rows.append({"check": check, "passed": bool(passed), "detail": detail})


def _close(a: float, b: float, tol: float = 1e-9) -> bool:
    return abs(float(a) - float(b)) <= tol


def _summary_from_audit(audit: pd.DataFrame, dataset: str) -> dict[str, object]:
    return {
        "dataset": dataset,
        "n_sources": int(len(audit)),
        "n_negative_auc": int((audit["delta_auc"] < 0).sum()),
        "n_negative_ap": int((audit["delta_ap"] < 0).sum()),
        "min_delta_auc": float(audit["delta_auc"].min()),
        "min_delta_ap": float(audit["delta_ap"].min()),
        "mean_delta_auc": float(audit["delta_auc"].mean()),
        "mean_delta_ap": float(audit["delta_ap"].mean()),
        "weighted_mean_delta_auc": float((audit["delta_auc"] * audit["n_fake"]).sum() / audit["n_fake"].sum()),
        "weighted_mean_delta_ap": float((audit["delta_ap"] * audit["n_fake"]).sum() / audit["n_fake"].sum()),
        "worst_auc_source": str(audit.sort_values(["delta_auc", "delta_ap"]).iloc[0]["source_model"]),
        "worst_ap_source": str(audit.sort_values(["delta_ap", "delta_auc"]).iloc[0]["source_model"]),
    }


def _read_required(path: Path, columns: set[str]) -> pd.DataFrame:
    df = pd.read_csv(path)
    missing = sorted(columns - set(df.columns))
    if missing:
        raise ValueError(f"{path} missing columns: {missing}")
    return df


def run(args: argparse.Namespace) -> pd.DataFrame:
    rows: list[dict[str, object]] = []

    paths = [
        args.comgenvid_audit_csv,
        args.genvideo_audit_csv,
        args.videofeedback_audit_csv,
        args.hotshot_audit_csv,
        args.hotshot_summary_csv,
        args.combined_summary_csv,
        args.combined_worst_csv,
    ]
    for path in paths:
        _add(rows, f"exists:{path}", path.exists(), str(path))

    audits = [_read_required(path, REQUIRED_AUDIT_COLUMNS) for path in args.audit_csvs]
    combined_audit = pd.concat(audits, ignore_index=True)
    summary = _read_required(args.combined_summary_csv, REQUIRED_SUMMARY_COLUMNS)
    hotshot_summary = _read_required(args.hotshot_summary_csv, REQUIRED_SUMMARY_COLUMNS)
    worst = _read_required(args.combined_worst_csv, REQUIRED_AUDIT_COLUMNS)

    _add(rows, "combined_summary_row_count", len(summary) == len(EXPECTED_DATASETS) + 1, f"rows={len(summary)}")
    _add(rows, "combined_summary_datasets", set(summary["dataset"].astype(str)) == {*EXPECTED_DATASETS, "ALL"}, str(sorted(summary["dataset"].astype(str).unique())))

    for dataset, expected_sources in EXPECTED_DATASETS.items():
        audit = combined_audit[combined_audit["dataset"].astype(str) == dataset]
        item = summary[summary["dataset"].astype(str) == dataset]
        _add(rows, f"audit_sources:{dataset}", len(audit) == expected_sources, f"actual={len(audit)}, expected={expected_sources}")
        _add(rows, f"summary_row:{dataset}", len(item) == 1, f"rows={len(item)}")
        if len(item) != 1:
            continue
        expected = _summary_from_audit(audit, dataset)
        actual = item.iloc[0]
        for col in ["n_sources", "n_negative_auc", "n_negative_ap"]:
            _add(rows, f"summary_int:{dataset}:{col}", int(actual[col]) == int(expected[col]), f"actual={actual[col]}, expected={expected[col]}")
        for col in ["min_delta_auc", "min_delta_ap", "mean_delta_auc", "mean_delta_ap", "weighted_mean_delta_auc", "weighted_mean_delta_ap"]:
            _add(rows, f"summary_float:{dataset}:{col}", _close(float(actual[col]), float(expected[col])), f"actual={actual[col]}, expected={expected[col]}")
        for col in ["worst_auc_source", "worst_ap_source"]:
            _add(rows, f"summary_source:{dataset}:{col}", str(actual[col]) == str(expected[col]), f"actual={actual[col]}, expected={expected[col]}")

    hotshot_audit = combined_audit[combined_audit["dataset"].astype(str) == "videofeedback_hotshot_local_variant"]
    hotshot_row = hotshot_summary.iloc[0] if len(hotshot_summary) else None
    _add(rows, "hotshot_summary_single_row", len(hotshot_summary) == 1, f"rows={len(hotshot_summary)}")
    if hotshot_row is not None:
        _add(rows, "hotshot_summary_dataset", str(hotshot_row["dataset"]) == "videofeedback_hotshot_local_variant", str(hotshot_row["dataset"]))
        _add(rows, "hotshot_summary_sources_11", int(hotshot_row["n_sources"]) == 11, str(hotshot_row["n_sources"]))
        _add(rows, "hotshot_summary_no_negative_auc", int(hotshot_row["n_negative_auc"]) == 0, str(hotshot_row["n_negative_auc"]))
        _add(rows, "hotshot_summary_no_negative_ap", int(hotshot_row["n_negative_ap"]) == 0, str(hotshot_row["n_negative_ap"]))
    _add(rows, "hotshot_sources_expected", set(hotshot_audit["source_model"].astype(str)) == EXPECTED_HOTSHOT_SOURCES, str(sorted(hotshot_audit["source_model"].astype(str).unique())))
    _add(rows, "hotshot_delta_auc_all_positive", bool((hotshot_audit["delta_auc"] > 0).all()), f"min={hotshot_audit['delta_auc'].min()}")
    _add(rows, "hotshot_delta_ap_all_positive", bool((hotshot_audit["delta_ap"] > 0).all()), f"min={hotshot_audit['delta_ap'].min()}")

    all_expected = _summary_from_audit(combined_audit, "ALL")
    all_row = summary[summary["dataset"].astype(str) == "ALL"]
    _add(rows, "all_summary_single_row", len(all_row) == 1, f"rows={len(all_row)}")
    if len(all_row) == 1:
        actual = all_row.iloc[0]
        for col in ["n_sources", "n_negative_auc", "n_negative_ap"]:
            _add(rows, f"all_summary_int:{col}", int(actual[col]) == int(all_expected[col]), f"actual={actual[col]}, expected={all_expected[col]}")
        for col in ["min_delta_auc", "min_delta_ap", "mean_delta_auc", "mean_delta_ap", "weighted_mean_delta_auc", "weighted_mean_delta_ap"]:
            _add(rows, f"all_summary_float:{col}", _close(float(actual[col]), float(all_expected[col])), f"actual={actual[col]}, expected={all_expected[col]}")
        for col in ["worst_auc_source", "worst_ap_source"]:
            _add(rows, f"all_summary_source:{col}", str(actual[col]) == str(all_expected[col]), f"actual={actual[col]}, expected={all_expected[col]}")

    sorted_worst = combined_audit.sort_values(["delta_auc", "delta_ap", "dataset", "source_model"]).reset_index(drop=True)
    actual_worst = worst.reset_index(drop=True)
    _add(rows, "worst_sources_nonempty", len(actual_worst) > 0, f"rows={len(actual_worst)}")
    _add(rows, "worst_sources_sorted_prefix", actual_worst.equals(sorted_worst.head(len(actual_worst))), f"rows={len(actual_worst)}")
    if len(actual_worst) > 0:
        first = actual_worst.iloc[0]
        _add(rows, "worst_auc_source_gen2", str(first["source_model"]) == "Gen2", str(first.to_dict()))
        _add(rows, "worst_auc_delta_matches_all_min", _close(float(first["delta_auc"]), float(all_expected["min_delta_auc"])), f"actual={first['delta_auc']}, expected={all_expected['min_delta_auc']}")

    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--comgenvid-audit-csv", type=Path, default=Path("results/patch_calibrated_persistence/comgenvid_alpha060_rawbase_splitminus_vs_default_by_source.csv"))
    parser.add_argument("--genvideo-audit-csv", type=Path, default=Path("results/patch_calibrated_persistence/genvideo_alpha060_rawbase_splitminus_vs_default_by_source.csv"))
    parser.add_argument("--videofeedback-audit-csv", type=Path, default=Path("results/patch_calibrated_persistence/videofeedback_alpha060_rawbase_splitminus_vs_default_by_source.csv"))
    parser.add_argument("--hotshot-audit-csv", type=Path, default=Path("results/patch_calibrated_persistence/videofeedback_hotshot_rawbase_splitminus_vs_default_source_audit.csv"))
    parser.add_argument("--hotshot-summary-csv", type=Path, default=Path("results/patch_calibrated_persistence/videofeedback_hotshot_rawbase_splitminus_vs_default_source_audit_summary.csv"))
    parser.add_argument("--combined-summary-csv", type=Path, default=Path("results/patch_calibrated_persistence/rawbase_splitminus_vs_default_plus_hotshot_local_source_audit_summary.csv"))
    parser.add_argument("--combined-worst-csv", type=Path, default=Path("results/patch_calibrated_persistence/rawbase_splitminus_vs_default_plus_hotshot_local_worst_sources.csv"))
    parser.add_argument("--output-csv", type=Path, required=True)
    args = parser.parse_args()
    args.audit_csvs = [
        args.comgenvid_audit_csv,
        args.genvideo_audit_csv,
        args.videofeedback_audit_csv,
        args.hotshot_audit_csv,
    ]

    checks = run(args)
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    checks.to_csv(args.output_csv, index=False)
    n_passed = int(checks["passed"].sum())
    print(f"PASS {n_passed}/{len(checks)}" if n_passed == len(checks) else f"FAIL {n_passed}/{len(checks)}")
    print(checks.to_string(index=False))
    if n_passed != len(checks):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
