# U0 calibration size and seed stability

All splits are sampled from a real-only reserve that is disjoint from both the locked 600-video calibration set and the 21,421-video evaluation set. Splits from different seeds may overlap each other; they are independent evaluation-disjoint calibration draws, not five mutually disjoint banks.

## Reserve audit

| Dataset | Eligible reserve | Sources | Duration mean/median (s) |
|---|---:|---|---:|
| ComGenVid | 600 | MSVD:600 | 9.36/8.01 |
| VideoFeedback | 300 | DiDeMo:145, Panda70M:155 | 2.69/3.00 |
| GenVideo | 1800 | MSR-VTT:1800 | 14.57/13.00 |

Membership rows: 5,625; locked overlap: 0. Sizes 25/50/100 are nested in size 200 within each dataset/seed and source-stratified before stable SHA-256 ranking.

## Final-score stability

Values are mean +/- sample standard deviation across seeds 17, 29, 43, 71, 101; bracketed values are seed min/max. AP remains real-positive.

| Size | ComGenVid AUC/AP | VideoFeedback AUC/AP | GenVideo AUC/AP | Macro-3 AUC/AP |
|---:|---:|---:|---:|---:|
| 25 | 0.8540+/-0.0140 / 0.8529+/-0.0126 [0.8311,0.8609] | 0.8324+/-0.0132 / 0.8395+/-0.0111 [0.8281,0.8545] | 0.7976+/-0.0165 / 0.7785+/-0.0127 [0.7639,0.7952] | 0.8280+/-0.0060 / 0.8237+/-0.0050 [0.8194,0.8312] |
| 50 | 0.8714+/-0.0054 / 0.8801+/-0.0051 [0.8738,0.8857] | 0.8358+/-0.0092 / 0.8472+/-0.0087 [0.8361,0.8561] | 0.8411+/-0.0130 / 0.8188+/-0.0078 [0.8084,0.8301] | 0.8495+/-0.0027 / 0.8487+/-0.0026 [0.8448,0.8514] |
| 100 | 0.8871+/-0.0063 / 0.8983+/-0.0047 [0.8911,0.9040] | 0.8411+/-0.0075 / 0.8529+/-0.0067 [0.8447,0.8618] | 0.8611+/-0.0073 / 0.8371+/-0.0071 [0.8294,0.8459] | 0.8631+/-0.0054 / 0.8628+/-0.0052 [0.8566,0.8668] |
| 200 | 0.8964+/-0.0038 / 0.9065+/-0.0027 [0.9021,0.9093] | 0.8489+/-0.0028 / 0.8578+/-0.0021 [0.8556,0.8608] | 0.8683+/-0.0038 / 0.8395+/-0.0039 [0.8352,0.8440] | 0.8712+/-0.0027 / 0.8680+/-0.0022 [0.8652,0.8705] |

## Branch calibration dependence

| Size | Global Macro AP | Local Macro AP | Final Macro AP |
|---:|---:|---:|---:|
| 25 | 0.8301+/-0.0052 | 0.5193+/-0.0075 | 0.8237+/-0.0050 |
| 50 | 0.8406+/-0.0033 | 0.6421+/-0.0211 | 0.8487+/-0.0026 |
| 100 | 0.8440+/-0.0018 | 0.7699+/-0.0221 | 0.8628+/-0.0052 |
| 200 | 0.8464+/-0.0011 | 0.8175+/-0.0023 | 0.8680+/-0.0022 |

## Gain relative to Original STALL

| Size | Macro AUC delta | Macro AP delta |
|---:|---:|---:|
| 25 | -0.0108 | -0.0191 |
| 50 | +0.0107 | +0.0059 |
| 100 | +0.0244 | +0.0200 |
| 200 | +0.0324 | +0.0251 |

## Decision

The independent-reserve N=200 Macro AP is `0.867953 +/- 0.002210`; the locked release split is `0.872299`. Relative to Original STALL AP `0.842810`, retaining 95% of the N=200 mean improvement requires Macro AP >= `0.866696`; the smallest tested size meeting it is `200`.

The seed experiment refits Local whitening, K1 window CDFs, and effective-K video CDFs for every split. It is separate from the paired video-cluster bootstrap.

Mean per-video score standard deviation across seeds at N=200:
- G: dataset mean `0.016534`.
- L: dataset mean `0.022005`.
- S: dataset mean `0.015021`.

## Numerical and integrity checks

- Evaluation windows: 56,812; reserve K1/K3 rows: 10,611; score/decode failures: 0.
- Accelerated locked raw max errors: GlobalSpatial `0`, GlobalT1 `0`, PatchSpatial `1.5e-11`, PatchD2 `2.27e-12`.
- Generated videos are used only for final evaluation metrics; none enter whitening, CDF construction, split selection, or parameter fitting.
