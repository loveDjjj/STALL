# Aggregation / bottom-k 敏感性实跑进展

mean aggregation 作为对应数据集主线 region 的基线；bottom-k 点只汇总已经完成的全量 patch eval。

| dataset | region | aggregation | bottom-k | 平均 AUC | 平均 AP | 状态 |
|---|---:|---|---:|---:|---:|---|
| genvideo | 2 | bottomk_mean | 0.20 | 0.7652 | 0.7571 | journal_full_eval |
| genvideo | 2 | bottomk_mean | 0.50 | 0.7795 | 0.7746 | journal_full_eval |
| genvideo | 2 | mean | 1.00 | 0.8072 | 0.8092 | region_mean_baseline |
| videofeedback | 1 | mean | 1.00 | 0.8228 | 0.8302 | region_mean_baseline |

genvideo: 当前已完成点中，AUC 最优为 mean (bottomk=1.00, AUC=0.8072)；AP 最优为 mean (bottomk=1.00, AP=0.8092)。
videofeedback: 当前已完成点中，AUC 最优为 mean (bottomk=1.00, AUC=0.8228)；AP 最优为 mean (bottomk=1.00, AP=0.8302)。
