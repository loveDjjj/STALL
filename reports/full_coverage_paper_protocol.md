# Full-coverage and paper-count protocol audit

This analysis reuses frozen duration-aware per-video scores. No DINO inference, whitening fit, CDF fit, or generated-video parameter selection is performed.

## Full-coverage results

All 45,185 generated videos are included. AP_std50 uses sample weights so real and fake each contribute total weight 0.5; raw AP is retained only as a prevalence diagnostic. AP_fake matches the positive-class convention stated in the original STALL paper, while AP_real preserves this repository's historical convention.

| Scope | Global AUC/AP_fake/AP_real | Alpha AUC/AP_fake/AP_real | Delta AUC/AP_fake/AP_real |
|---|---:|---:|---:|
| comgenvid | 0.8628/0.8449/0.8709 | 0.9045/0.8973/0.9117 | +0.0417/+0.0525/+0.0408 |
| videofeedback | 0.8478/0.8248/0.8608 | 0.8605/0.8418/0.8710 | +0.0127/+0.0170/+0.0102 |
| genvideo | 0.8208/0.8085/0.8111 | 0.8550/0.8495/0.8417 | +0.0342/+0.0411/+0.0306 |
| Macro-3 | 0.8438/0.8260/0.8476 | 0.8733/0.8629/0.8748 | +0.0295/+0.0368/+0.0272 |
| All-23 | 0.8373/0.8194/0.8401 | 0.8619/0.8500/0.8618 | +0.0246/+0.0305/+0.0217 |

## Paper-count-matched fake cohort

The deterministic cohort contains 36,641 generated videos, matching the per-generator counts in the STALL supplementary tables. Video identities cannot be claimed identical because the original complete manifests are unavailable. Target-domain calibration remains disjoint, so the original paper's full real counts cannot simultaneously be reused for Alpha-STALLED without leakage.

| Scope | Global AUC/AP_fake/AP_real | Alpha AUC/AP_fake/AP_real | Delta AUC/AP_fake/AP_real |
|---|---:|---:|---:|
| comgenvid | 0.8628/0.8449/0.8709 | 0.9045/0.8973/0.9117 | +0.0417/+0.0525/+0.0408 |
| videofeedback | 0.8478/0.8247/0.8612 | 0.8604/0.8414/0.8709 | +0.0126/+0.0167/+0.0098 |
| genvideo | 0.8214/0.8084/0.8128 | 0.8552/0.8488/0.8436 | +0.0338/+0.0404/+0.0308 |
| Macro-3 | 0.8440/0.8260/0.8483 | 0.8734/0.8625/0.8754 | +0.0294/+0.0365/+0.0271 |
| All-23 | 0.8376/0.8193/0.8410 | 0.8620/0.8495/0.8626 | +0.0243/+0.0301/+0.0216 |

## Balanced primary and seed sensitivity

The balanced row uses the current disjoint-real constraints. One hundred deterministic selection seeds change only which eligible real/fake identities enter each generator comparison; scores and parameters stay frozen.

Seed 42 Macro-3 AUC/AP_fake/AP_real is Global 0.8404/0.8200/0.8454 and Alpha 0.8721/0.8608/0.8741.

Across 100 seeds, Alpha-minus-Global AP_real has mean +0.0278, standard deviation 0.0007, and range [+0.0262, +0.0294].
Alpha-minus-Global AP_fake has mean +0.0378, standard deviation 0.0014, and range [+0.0350, +0.0425].

Bootstrap and selection-seed sensitivity answer different questions. Bootstrap estimates finite-sample uncertainty within a fixed cohort; seed sensitivity measures dependence on which eligible identities were selected. Neither creates coverage of unseen videos.

## Artifacts

- `full_coverage_generator_metrics.csv` and `full_coverage_summary.csv`
- `paper_count_fake_manifest.csv` and `paper_count_summary.csv`
- `balanced_seed_metrics.csv` and `balanced_seed_deltas.csv`
- `score_distribution_summary.csv`, `all_fake_scores.csv`, and `failure_cases.csv`
