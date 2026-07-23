# Global-local fusion analysis

## Protocol

Global D3 G1-G4 failed the Stage-2 admission criteria, so the global branch is frozen as original STALL (B2). Fixed fusion scans only the specified alpha grid `{0.2, 0.4, 0.5, 0.6, 0.8}`. Local models are the leakage-free Stage-3 scores.

The fixed-alpha main rule selects `P0` by three-dataset mean AP at alpha 0.6, then uses `0.6 * B2 + 0.4 * P0` without target-specific tuning.

## Selection protocols

| Protocol | Target | Local | Alpha | Selection AP | Target AUC/AP |
|---|---|---|---:|---:|---:|
| worst_case | comgenvid | P0 | 0.5 | 0.8285 | 0.8728/0.8857 |
| worst_case | genvideo | P0 | 0.5 | 0.8285 | 0.8400/0.8285 |
| worst_case | videofeedback | P0 | 0.5 | 0.8285 | 0.8569/0.8641 |
| fixed_alpha_0.6 | comgenvid | P0 | 0.6 | 0.8600 | 0.8699/0.8819 |
| fixed_alpha_0.6 | genvideo | P0 | 0.6 | 0.8600 | 0.8382/0.8274 |
| fixed_alpha_0.6 | videofeedback | P0 | 0.6 | 0.8600 | 0.8631/0.8708 |
| leave_one_dataset_out | comgenvid | P0 | 0.6 | 0.8491 | 0.8699/0.8819 |
| target_oracle_diagnostic | comgenvid | P0 | 0.2 | 0.8945 | 0.8824/0.8945 |
| leave_one_dataset_out | genvideo | P0 | 0.6 | 0.8764 | 0.8382/0.8274 |
| target_oracle_diagnostic | genvideo | P0 | 0.5 | 0.8285 | 0.8400/0.8285 |
| leave_one_dataset_out | videofeedback | P0 | 0.2 | 0.8590 | 0.8255/0.8313 |
| target_oracle_diagnostic | videofeedback | P0 | 0.8 | 0.8757 | 0.8682/0.8757 |

## Fixed alpha 0.6

| Dataset | Local | AUC | AP |
|---|---|---:|---:|
| comgenvid | P0 | 0.8699 | 0.8819 |
| genvideo | P0 | 0.8382 | 0.8274 |
| videofeedback | P0 | 0.8631 | 0.8708 |

## Paired bootstrap for selected fixed fusion

| Dataset | Comparison | Metric | Delta | 95% CI |
|---|---|---|---:|---:|
| comgenvid | final_selected-B2 | AUC | +0.0148 | [+0.0126, +0.0173] |
| comgenvid | final_selected-B2 | AP | +0.0213 | [+0.0178, +0.0249] |
| comgenvid | final_selected-B8 | AUC | -0.0500 | [-0.0562, -0.0439] |
| comgenvid | final_selected-B8 | AP | -0.0411 | [-0.0461, -0.0361] |
| videofeedback | final_selected-B2 | AUC | +0.0103 | [+0.0045, +0.0161] |
| videofeedback | final_selected-B2 | AP | +0.0068 | [+0.0018, +0.0122] |
| videofeedback | final_selected-B8 | AUC | +0.0070 | [+0.0042, +0.0098] |
| videofeedback | final_selected-B8 | AP | +0.0021 | [-0.0003, +0.0046] |
| genvideo | final_selected-B2 | AUC | +0.0296 | [+0.0242, +0.0356] |
| genvideo | final_selected-B2 | AP | +0.0237 | [+0.0183, +0.0295] |
| genvideo | final_selected-B8 | AUC | -0.0071 | [-0.0115, -0.0029] |
| genvideo | final_selected-B8 | AP | -0.0059 | [-0.0095, -0.0025] |

## Reliability fusion decision

A per-video inverse-variance reliability weight is not reported because the frozen protocol currently contains one compact 2 s patch window per video. Estimating variance from a single scalar would be invalid, and `|score - 0.5|` is explicitly disallowed. Reliability fusion therefore does not enter the method until multiple independently cached windows are available.
