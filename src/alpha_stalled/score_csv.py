"""Shared score-CSV loading, fusion, and pairwise metric operations."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from .metrics import ScoreDirection, build_results_table


KEY_COLUMNS = ("subset", "source_model", "filename")


def evaluate_score_csv(
    csv_path: Path,
    score_col: str,
    seed: int = 42,
    higher_is: str = "real",
    skip_global_compare: bool = True,
) -> pd.DataFrame:
    """Evaluate one score column with the historical pairwise-balanced protocol."""

    df = pd.read_csv(csv_path)
    required = ["subset", "source_model", score_col]
    missing = [column for column in required if column not in df.columns]
    if missing:
        raise ValueError(f"{csv_path} 缺少列: {missing}")
    try:
        direction = {
            "real": ScoreDirection.HIGHER_IS_REAL,
            "fake": ScoreDirection.HIGHER_IS_FAKE,
        }[higher_is]
    except KeyError as exc:
        raise ValueError(f"higher_is must be 'real' or 'fake', got {higher_is!r}") from exc
    return build_results_table(
        df[["subset", "source_model", score_col]].copy(),
        {score_col: direction},
        seed=seed,
        skip_global_compare=skip_global_compare,
        verbose=False,
    )


def read_keyed_scores(path: Path, score_col: str, prefix: str) -> pd.DataFrame:
    """Load one score with the canonical three-column historical video identity."""

    df = pd.read_csv(path)
    required = [*KEY_COLUMNS, score_col]
    missing = [column for column in required if column not in df.columns]
    if missing:
        raise ValueError(f"{path} 缺少列: {missing}")
    out = df[required].copy()
    for column in KEY_COLUMNS:
        out[column] = out[column].astype(str)
    return out.rename(columns={score_col: f"{prefix}_score"})


def fuse_score_csvs(
    global_csv: Path,
    patch_csv: Path,
    alpha: float,
    global_score_col: str,
    patch_score_col: str,
) -> pd.DataFrame:
    """Strictly join and linearly fuse matching global and patch score CSVs."""

    if not 0.0 <= alpha <= 1.0:
        raise ValueError(f"alpha 必须位于 [0,1]，当前为 {alpha}")
    global_df = read_keyed_scores(global_csv, global_score_col, "global")
    patch_df = read_keyed_scores(patch_csv, patch_score_col, "patch")
    merged = global_df.merge(
        patch_df,
        on=list(KEY_COLUMNS),
        how="inner",
        validate="one_to_one",
    )
    if len(merged) != len(global_df) or len(merged) != len(patch_df):
        raise ValueError(
            f"Global/patch CSV 交集不一致: merged={len(merged)}, "
            f"global={len(global_df)}, patch={len(patch_df)}"
        )
    merged["alpha"] = float(alpha)
    merged["final_score"] = (
        alpha * merged["global_score"] + (1.0 - alpha) * merged["patch_score"]
    )
    return merged


def compute_fused_metrics(df: pd.DataFrame, seed: int) -> pd.DataFrame:
    """Compute the paper's historical pairwise metrics for ``final_score``."""

    return build_results_table(
        df[["subset", "source_model", "final_score"]],
        {"final_score": ScoreDirection.HIGHER_IS_REAL},
        seed=seed,
        skip_global_compare=True,
        verbose=False,
    )


def metric_rows(df: pd.DataFrame, score_col: str, seed: int) -> pd.DataFrame:
    """Return normalized per-generator metric rows for a score sweep."""

    metrics = build_results_table(
        df[["subset", "source_model", score_col]],
        {score_col: ScoreDirection.HIGHER_IS_REAL},
        seed=seed,
        skip_global_compare=True,
        verbose=False,
    )
    out = metrics.rename(
        columns={
            "Generative Model": "source_model",
            f"{score_col} AUC": "auc",
            f"{score_col} AP": "ap",
            "n_annotated": "n_fake",
        }
    )
    keep = [
        column
        for column in ("source_model", "n_real", "n_fake", "n_total", "auc", "ap")
        if column in out.columns
    ]
    return out[keep]


__all__ = [
    "KEY_COLUMNS",
    "compute_fused_metrics",
    "evaluate_score_csv",
    "fuse_score_csvs",
    "metric_rows",
    "read_keyed_scores",
]
