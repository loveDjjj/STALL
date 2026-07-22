# Patch likelihood statistical assumption audit

Dataset: ComGenVid real videos; sampled real videos: 256.

This diagnostic checks whether the Gaussian likelihood machinery inherited from STALL is at least approximately reasonable for patch-level representations.
It is not a proof of exact multivariate normality. With thousands of samples and high-dimensional neural features, strict normality tests are expected to be sensitive to small deviations.

## Summary

| representation | whitening | rows | dim | diag mean | offdiag RMS | AD pass@5% | DP p>0.01 | cosine std / expected | position norm² std |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| patch_token | release_muW_patch_spatial | 49920 | 1023 | 0.992 | 0.017 | 0.180 | 0.156 | 0.0346/0.0313 | 0.088 |
| region3_pooled_token | diagnostic_half_split_muW | 24960 | 1024 | 1.047 | 0.010 | 0.039 | 0.023 | 0.0335/0.0312 | 0.071 |
| region3_second_order_diff | release_muW_patch_temporal | 49920 | 1024 | 0.975 | 0.009 | 0.883 | 0.906 | 0.0319/0.0312 | 0.009 |

## Interpretation for the manuscript

- The covariance and direction-cosine diagnostics are the most relevant checks for using a whitened Gaussian score. They test whether whitening approximately sphericalizes the selected representation.
- AD and D'Agostino-Pearson tests are reported as stress diagnostics rather than pass/fail proof. Low pass rates should be written as a modeling boundary, not as a failure of the detector.
- Position-sharing is assessed by the spread of per-position whitened norm² and log-likelihood means. A small spread supports using one shared `(mu, W)`; a large spread would motivate per-position calibration.

## Files

- `patch_likelihood_assumption_summary.csv`
- `patch_likelihood_position_summary.csv`
- `patch_likelihood_position_detail.csv`
- `patch_likelihood_assumption_audit.pdf/.png/.svg`
