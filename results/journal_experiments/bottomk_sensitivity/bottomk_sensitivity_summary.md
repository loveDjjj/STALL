# Bottom-k 敏感性实跑进展

本表只汇总已经完成的全量 patch eval。未完成的 bottom-k 点不填补、不插值。

| dataset | region | aggregation | bottom-k | 平均 AUC | 平均 AP | 状态 |
|---|---:|---|---:|---:|---:|---|
| comgenvid | 3 | bottomk_mean | 0.10 | 0.9245 | 0.9285 | journal_full_eval |
| comgenvid | 3 | bottomk_mean | 0.15 | 0.9262 | 0.9300 | journal_full_eval |
| comgenvid | 3 | bottomk_mean | 0.20 | 0.9273 | 0.9309 | journal_full_eval |
| comgenvid | 3 | bottomk_mean | 0.30 | 0.9292 | 0.9323 | journal_full_eval |
| comgenvid | 3 | bottomk_mean | 0.50 | 0.9312 | 0.9334 | journal_full_eval |

初步结论：ComGenVid region=3 下，当前已完成 bottom-k 点中，AUC 最优为 bottomk=0.50 (0.9312)，AP 最优为 bottomk=0.50 (0.9334)。该结果用于敏感性分析，不改变 release 默认配置。
