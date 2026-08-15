"""Compatibility imports for Alpha-STALLED's stable numerical primitives.

New code should import :mod:`alpha_stalled.whitening`. This module remains for
historical experiment scripts and external callers using the original path.
"""

from alpha_stalled.whitening import (
    CDF_TIE_POLICY,
    GaussianMeanCandidateScorerFloat64,
    StableGaussianParams,
    configure_strict_fp32,
    empirical_cdf_right_inclusive,
    l2_normalized_first_order,
    l2_normalized_patch_first_order,
    l2_normalized_second_order,
    score_gaussian_aggregate_float64,
    score_gaussian_mean_candidates_float64,
    score_mean_gaussian_float64,
    score_mean_gaussian_fp32,
    stable_sorted,
)


__all__ = [
    "CDF_TIE_POLICY",
    "GaussianMeanCandidateScorerFloat64",
    "StableGaussianParams",
    "configure_strict_fp32",
    "empirical_cdf_right_inclusive",
    "l2_normalized_first_order",
    "l2_normalized_patch_first_order",
    "l2_normalized_second_order",
    "score_gaussian_aggregate_float64",
    "score_gaussian_mean_candidates_float64",
    "score_mean_gaussian_float64",
    "score_mean_gaussian_fp32",
    "stable_sorted",
]
