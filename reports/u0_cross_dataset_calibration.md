# U0 cross-dataset real calibration

The detector configuration is fixed. Only the real calibration bank changes; no generated video enters whitening, CDFs, source selection, or parameter fitting.

## Calibration banks

| Bank | Videos | Dataset composition | Duration mean/median (s) | Resolution median (px) |
|---|---:|---|---:|---:|
| ComGenVid | 200 | ComGenVid:200 | 8.73/8.01 | 172,800 |
| VideoFeedback | 200 | VideoFeedback:200 | 2.70/3.00 | 230,400 |
| GenVideo | 200 | GenVideo:200 | 14.73/13.00 | 76,800 |
| Pooled-200 | 200 | ComGenVid:67, GenVideo:66, VideoFeedback:67 | 8.55/8.01 | 172,800 |
| Pooled-600 | 600 | ComGenVid:200, GenVideo:200, VideoFeedback:200 | 8.72/8.01 | 172,800 |

## Global AUC/AP matrix

| Calibration source | ComGenVid | VideoFeedback | GenVideo | Macro-3 |
|---|---:|---:|---:|---:|
| ComGenVid | 0.8627/0.8694 | 0.8347/0.8504 | 0.8400/0.8184 | 0.8458/0.8461 |
| VideoFeedback | 0.8628/0.8694 | 0.8442/0.8582 | 0.8537/0.8275 | 0.8535/0.8517 |
| GenVideo | 0.8631/0.8706 | 0.8289/0.8489 | 0.8386/0.8181 | 0.8435/0.8459 |
| Pooled-200 | 0.8624/0.8699 | 0.8334/0.8529 | 0.8451/0.8239 | 0.8470/0.8489 |
| Pooled-600 | 0.8630/0.8717 | 0.8366/0.8537 | 0.8433/0.8212 | 0.8476/0.8489 |

## Local AUC/AP matrix

| Calibration source | ComGenVid | VideoFeedback | GenVideo | Macro-3 |
|---|---:|---:|---:|---:|
| ComGenVid | 0.8914/0.8912 | 0.7104/0.7038 | 0.7928/0.7790 | 0.7982/0.7913 |
| VideoFeedback | 0.7628/0.7564 | 0.7828/0.7849 | 0.7881/0.7598 | 0.7779/0.7670 |
| GenVideo | 0.7835/0.7692 | 0.6725/0.6867 | 0.8328/0.8114 | 0.7629/0.7558 |
| Pooled-200 | 0.8742/0.8722 | 0.7431/0.7495 | 0.8372/0.8199 | 0.8182/0.8139 |
| Pooled-600 | 0.8759/0.8789 | 0.7600/0.7677 | 0.8387/0.8157 | 0.8249/0.8208 |

## Final AUC/AP matrix

| Calibration source | ComGenVid | VideoFeedback | GenVideo | Macro-3 |
|---|---:|---:|---:|---:|
| ComGenVid | 0.8968/0.9064 | 0.8367/0.8510 | 0.8537/0.8365 | 0.8624/0.8646 |
| VideoFeedback | 0.8683/0.8726 | 0.8597/0.8689 | 0.8648/0.8377 | 0.8643/0.8597 |
| GenVideo | 0.8572/0.8679 | 0.8098/0.8277 | 0.8657/0.8416 | 0.8442/0.8457 |
| Pooled-200 | 0.8855/0.8966 | 0.8324/0.8477 | 0.8662/0.8440 | 0.8614/0.8628 |
| Pooled-600 | 0.8887/0.8998 | 0.8365/0.8497 | 0.8640/0.8413 | 0.8631/0.8636 |

## Distribution shift

Machine-readable real/fake score means, standard deviations, and quantiles for every source-target-branch combination are stored in `score_distribution_shift.csv`.

## Answers

1. Target-domain real calibration gives diagonal mean final AP `0.8723`. The mean off-domain final AP is `0.8489`.
2. Global target/off-domain AP is `0.8486/0.8475`; Local is `0.8292/0.7425`. The larger gap identifies the more domain-dependent branch.
3. Pooled-200/Pooled-600 Macro AP is `0.8628/0.8636` versus target-bank diagonal `0.8723`.
4. Positioning must follow the matrix: use target-domain real calibration unless the pooled bank is empirically close enough across all three targets; do not call the detector target-data-free.

## Integrity

- Diagonal reconstructed locked-score max error: `0`.
- Distribution rows: 90; calibration videos probed: 600.
