#!/usr/bin/env python3
"""检查 Alpha-STALLED release 资产是否存在且内部一致。

这是轻量级提交前/release 前 sanity check。不需要 DINOv3、torch、原始视频或
patch cache；只检查由已有 score CSV 生成的自包含 release 资产。
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


DATASETS = ("comgenvid", "videofeedback", "genvideo")
EXPECTED_ALPHA = 0.60
REQUIRED_CALIBRATION_FILES = [
    "precomputed/stall_params_vatex_dino_v3.npz",
    "precomputed/patch_params_comgenvid_real_second_order_region3_bottomk0p20_v2.npz",
    "precomputed/patch_params_videofeedback_real_same_grid_second_order_region1_mean_v2.npz",
    "precomputed/patch_params_genvideo_real_same_grid_second_order_region2_mean_v2.npz",
]
REQUIRED_DOCS = [
    "FILE_STRUCTURE.md",
    "configs/alpha_stalled.yaml",
    "results/README.md",
    "results/alpha_stalled_project_manuscript_zh.md",
    "results/paper_tables/alpha_stalled_main_summary.md",
    "results/paper_tables/ablation_summary.md",
    "results/paper_tables/patch_coverage_gaps.csv",
    "results/paper_tables/patch_coverage_gaps.md",
    "results/paper_sweeps/alpha_sweep_summary.md",
    "docs/restructure/asset_manifest.md",
    "docs/restructure/reproducibility_audit.md",
    "docs/restructure/release_staging_manifest.md",
    "scripts/reproduce/README.md",
    "scripts/reproduce/rebuild_paper_assets.sh",
    "scripts/ablations/README.md",
]
REQUIRED_COMGENVID_ABLATION_SCORES = [
    "results/paper_scores/comgenvid_patch_spatial.csv",
    "results/paper_scores/comgenvid_patch_lag1.csv",
    "results/paper_scores/comgenvid_patch_multilag.csv",
    "results/paper_scores/comgenvid_patch_motionhard.csv",
    "results/paper_scores/comgenvid_patch_motionsoft.csv",
    "results/paper_scores/comgenvid_patch_second_order_ablation.csv",
]


def _require_path(root: Path, rel: str, errors: list[str]) -> Path:
    path = root / rel
    if not path.exists():
        errors.append(f"缺失: {rel}")
    elif path.is_file() and path.stat().st_size == 0:
        errors.append(f"空文件: {rel}")
    return path


def _require_columns(path: Path, columns: list[str], errors: list[str]) -> pd.DataFrame:
    try:
        df = pd.read_csv(path)
    except Exception as exc:  # pragma: no cover - 防御性 release 诊断
        errors.append(f"无法读取 CSV {path}: {exc}")
        return pd.DataFrame()
    missing = [c for c in columns if c not in df.columns]
    if missing:
        errors.append(f"{path} 缺少列: {missing}")
    return df


def _average_row(metrics: pd.DataFrame, path: Path, errors: list[str]) -> pd.Series | None:
    if "Generative Model" not in metrics.columns:
        errors.append(f"{path} 缺少 'Generative Model' 列")
        return None
    avg = metrics[metrics["Generative Model"] == "Average"]
    if avg.empty:
        errors.append(f"{path} 缺少 Average 行")
        return None
    return avg.iloc[0]


def verify(root: Path) -> list[str]:
    errors: list[str] = []

    for rel in REQUIRED_CALIBRATION_FILES + REQUIRED_DOCS:
        _require_path(root, rel, errors)
    for rel in REQUIRED_COMGENVID_ABLATION_SCORES:
        path = _require_path(root, rel, errors)
        if path.exists():
            _require_columns(path, ["subset", "source_model", "filename", "patch_final_score"], errors)

    coverage_csv = root / "results/paper_tables/patch_coverage_gaps.csv"
    if coverage_csv.exists():
        coverage = _require_columns(
            coverage_csv,
            ["dataset", "source_model", "status", "release_decision", "reason"],
            errors,
        )
        expected = {
            ("VideoFeedback", "Hotshot-XL"),
            ("GenVideo", "HotShot-XL"),
            ("GenVideo", "MoonValley"),
        }
        seen = set(zip(coverage.get("dataset", []), coverage.get("source_model", [])))
        missing = sorted(expected - seen)
        if missing:
            errors.append(f"patch_coverage_gaps.csv 缺少行: {missing}")

    for dataset in DATASETS:
        global_csv = _require_path(root, f"results/paper_scores/{dataset}_global.csv", errors)
        patch_csv = _require_path(root, f"results/paper_scores/{dataset}_patch_second_order.csv", errors)
        fused_csv = _require_path(root, f"results/paper_scores/{dataset}_alpha_stalled.csv", errors)
        metrics_csv = _require_path(root, f"results/paper_tables/{dataset}_alpha_stalled_metrics.csv", errors)
        global_metrics_csv = _require_path(root, f"results/paper_tables/{dataset}_global_only_metrics.csv", errors)
        patch_metrics_csv = _require_path(root, f"results/paper_tables/{dataset}_patch_only_metrics.csv", errors)
        sweep_csv = _require_path(
            root,
            f"results/paper_sweeps/{dataset}_alpha/{dataset}_same_grid_second_order_summary.csv",
            errors,
        )
        _require_path(
            root,
            f"results/paper_sweeps/{dataset}_alpha/{dataset}_same_grid_second_order_per_model.csv",
            errors,
        )
        _require_path(
            root,
            f"results/paper_sweeps/{dataset}_alpha/{dataset}_same_grid_second_order_summary.md",
            errors,
        )

        if global_csv.exists():
            _require_columns(global_csv, ["subset", "source_model", "filename", "final_score"], errors)
        if patch_csv.exists():
            _require_columns(
                patch_csv,
                ["subset", "source_model", "filename", "patch_final_score"],
                errors,
            )
        if fused_csv.exists():
            fused = _require_columns(
                fused_csv,
                ["subset", "source_model", "filename", "global_score", "patch_score", "alpha", "final_score"],
                errors,
            )
            if not fused.empty and abs(float(fused["alpha"].iloc[0]) - EXPECTED_ALPHA) > 1e-12:
                errors.append(f"{fused_csv} alpha 不是 {EXPECTED_ALPHA}")

        if metrics_csv.exists() and sweep_csv.exists():
            metrics = _require_columns(metrics_csv, ["Generative Model", "final_score AUC", "final_score AP"], errors)
            sweep = _require_columns(sweep_csv, ["alpha", "avg_auc", "avg_ap"], errors)
            avg = _average_row(metrics, metrics_csv, errors)
            alpha_row = sweep[sweep["alpha"].round(10) == EXPECTED_ALPHA] if not sweep.empty else pd.DataFrame()
            if alpha_row.empty:
                errors.append(f"{sweep_csv} 缺少 alpha={EXPECTED_ALPHA:.2f} 行")
            elif avg is not None:
                row = alpha_row.iloc[0]
                if abs(float(avg["final_score AUC"]) - float(row["avg_auc"])) > 5e-12:
                    errors.append(f"{dataset}: alpha=0.60 sweep AUC 与主指标不一致")
                if abs(float(avg["final_score AP"]) - float(row["avg_ap"])) > 5e-12:
                    errors.append(f"{dataset}: alpha=0.60 sweep AP 与主指标不一致")

        if global_metrics_csv.exists():
            _average_row(
                _require_columns(global_metrics_csv, ["Generative Model", "final_score AUC", "final_score AP"], errors),
                global_metrics_csv,
                errors,
            )
        if patch_metrics_csv.exists():
            _average_row(
                _require_columns(
                    patch_metrics_csv,
                    ["Generative Model", "patch_final_score AUC", "patch_final_score AP"],
                    errors,
                ),
                patch_metrics_csv,
                errors,
            )

    return errors


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    args = parser.parse_args()

    root = args.root.resolve()
    errors = verify(root)
    if errors:
        print("Alpha-STALLED release 验证失败:")
        for err in errors:
            print(f"  - {err}")
        raise SystemExit(1)
    print(f"Alpha-STALLED release 资产已通过验证: {root}")


if __name__ == "__main__":
    main()
