"""Compatibility imports for the shared Alpha-STALLED metric implementation.

New code should import :mod:`alpha_stalled.metrics`. This module remains so the
original STALL entry points and archived experiment scripts keep working.
"""

from alpha_stalled.metrics import (
    Score,
    ScoreDirection,
    _auc_flipped_row,
    _balance_datasets,
    _calculate_averages,
    _compute_metrics,
    _run_pairwise,
    _sample_balanced_real,
    _sample_proportional,
    _to_numpy,
    build_results_table,
    get_results_df,
    predictor_scalar2metrics,
    print_results,
    sample_balanced_real,
)


__all__ = [
    "Score",
    "ScoreDirection",
    "build_results_table",
    "get_results_df",
    "predictor_scalar2metrics",
    "print_results",
    "sample_balanced_real",
]
