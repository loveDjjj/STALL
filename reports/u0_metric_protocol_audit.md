# U0 metric protocol audit

Stored `S` is higher-is-real. Historical AP is real-positive AP. Fake-positive AP uses `1-S` with fake label 1; the two AP values are not interchangeable.

| Scope | Dataset | AUC | Fake-positive AP | Real-positive AP | BAcc@0.5 | TPR@1% | TPR@5% | EER |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| generator_pairwise_dataset_macro | comgenvid | 0.8968 | 0.8837 | 0.9064 | 0.6971 | 0.2233 | 0.4421 | 0.1868 |
| unique_video_pooled | comgenvid | 0.8970 | 0.9647 | 0.7706 | 0.6960 | 0.2197 | 0.4509 | 0.1839 |
| generator_pairwise_dataset_macro | genvideo | 0.8657 | 0.8698 | 0.8416 | 0.7183 | 0.2379 | 0.5152 | 0.2080 |
| unique_video_pooled | genvideo | 0.8702 | 0.8327 | 0.8851 | 0.7148 | 0.1917 | 0.4965 | 0.2069 |
| generator_pairwise_dataset_macro | videofeedback | 0.8597 | 0.8390 | 0.8689 | 0.6578 | 0.1343 | 0.3970 | 0.2068 |
| unique_video_pooled | videofeedback | 0.8579 | 0.9672 | 0.5855 | 0.6605 | 0.1343 | 0.4210 | 0.2163 |
| generator_pairwise_macro3 | Macro-3 | 0.8741 | 0.8642 | 0.8723 | 0.6911 | 0.1985 | 0.4514 | 0.2006 |
| unique_video_pooled_macro3 | Macro-3 | 0.8750 | 0.9215 | 0.7471 | 0.6904 | 0.1819 | 0.4561 | 0.2024 |
| generator_macro20 | All-20 | 0.8658 | 0.8558 | 0.8617 | 0.6859 | 0.1846 | 0.4488 | 0.2053 |

The original STALL paper states that generated video is the positive class for AP. This repository's frozen Alpha-STALLED main table instead uses real-positive AP for continuity with its historical evaluation scripts. Fake-positive AP is therefore the orientation aligned with STALL Table 1; real-positive AP is an internal endpoint, and the two must be named explicitly. AUC is unchanged when both score and label orientation are reversed. Pooled AP is prevalence-sensitive and must not be compared numerically with balanced pairwise AP.

| Bootstrap scope | Metric | Mean | Std | 95% CI |
|---|---|---:|---:|---:|
| generator_macro:All-20 | auc | 0.8656 | 0.0067 | [0.8537, 0.8788] |
| generator_macro:All-20 | fake_positive_ap | 0.8561 | 0.0091 | [0.8386, 0.8738] |
| generator_macro:All-20 | real_positive_ap | 0.8623 | 0.0064 | [0.8504, 0.8743] |
| generator_pairwise:Macro-3 | auc | 0.8740 | 0.0051 | [0.8642, 0.8835] |
| generator_pairwise:Macro-3 | fake_positive_ap | 0.8645 | 0.0069 | [0.8509, 0.8781] |
| generator_pairwise:Macro-3 | real_positive_ap | 0.8728 | 0.0050 | [0.8630, 0.8824] |
| unique_video_pooled:Macro-3 | auc | 0.8751 | 0.0038 | [0.8677, 0.8824] |
| unique_video_pooled:Macro-3 | fake_positive_ap | 0.9216 | 0.0019 | [0.9179, 0.9254] |
| unique_video_pooled:Macro-3 | real_positive_ap | 0.7473 | 0.0083 | [0.7309, 0.7623] |

Bootstrap resamples video IDs as clusters. Repeated real videos in generator-pairwise evaluation share one resampling multiplicity; windows are never bootstrap units.

Machine-readable bootstrap rows: 63.
