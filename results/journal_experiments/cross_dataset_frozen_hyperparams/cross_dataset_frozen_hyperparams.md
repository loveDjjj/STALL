# Cross-dataset frozen hyperparameter 分析

本分析只复用已有敏感性结果，不重新提取特征、不重新评测视频。
目标是区分目标数据集 oracle sweep 与跨数据集冻结配置的泛化性。

选择准则统一为平均 AUC：transfer matrix 使用 source dataset 上的最优参数；
leave-one-dataset-out 使用另外两个数据集平均 AUC 最优的参数，再报告目标数据集性能。

## alpha

| held-out target | train datasets | frozen param | frozen AUC/AP | target oracle param | oracle AUC/AP | ΔAUC | ΔAP |
|---|---|---:|---:|---:|---:|---:|---:|
| ComGenVid | videofeedback+genvideo | 0.75 | 0.9071 / 0.9061 | 0.20 | 0.9296 / 0.9327 | -0.0224 | -0.0266 |
| VideoFeedback | comgenvid+genvideo | 0.50 | 0.8575 / 0.8703 | 0.80 | 0.8687 / 0.8784 | -0.0112 | -0.0081 |
| GenVideo | comgenvid+videofeedback | 0.55 | 0.8360 / 0.8273 | 0.65 | 0.8380 / 0.8281 | -0.0020 | -0.0009 |

- 平均 |ΔAUC| = 0.0119；最差 ΔAUC = -0.0224。

## beta

| held-out target | train datasets | frozen param | frozen AUC/AP | target oracle param | oracle AUC/AP | ΔAUC | ΔAP |
|---|---|---:|---:|---:|---:|---:|---:|
| ComGenVid | videofeedback+genvideo | 0.20 | 0.9178 / 0.9194 | 0 | 0.9205 / 0.9220 | -0.0027 | -0.0026 |
| VideoFeedback | comgenvid+genvideo | 0.10 | 0.8628 / 0.8750 | 0.10 | 0.8628 / 0.8750 | +0.0000 | +0.0000 |
| GenVideo | comgenvid+videofeedback | 0.05 | 0.8362 / 0.8271 | 0.30 | 0.8396 / 0.8311 | -0.0034 | -0.0040 |

- 平均 |ΔAUC| = 0.0020；最差 ΔAUC = -0.0034。

## region

| held-out target | train datasets | frozen param | frozen AUC/AP | target oracle param | oracle AUC/AP | ΔAUC | ΔAP |
|---|---|---:|---:|---:|---:|---:|---:|
| ComGenVid | videofeedback+genvideo | 1 | 0.9088 / 0.9075 | 3 | 0.9299 / 0.9288 | -0.0211 | -0.0212 |
| VideoFeedback | comgenvid+genvideo | 2 | 0.7941 / 0.7946 | 1 | 0.8228 / 0.8302 | -0.0288 | -0.0356 |
| GenVideo | comgenvid+videofeedback | 1 | 0.7976 / 0.7985 | 2 | 0.8072 / 0.8092 | -0.0097 | -0.0108 |

- 平均 |ΔAUC| = 0.0198；最差 ΔAUC = -0.0288。

## aggregation

| held-out target | train datasets | frozen param | frozen AUC/AP | target oracle param | oracle AUC/AP | ΔAUC | ΔAP |
|---|---|---:|---:|---:|---:|---:|---:|
| ComGenVid | videofeedback+genvideo | mean | 0.9299 / 0.9288 | bottomk_0.50 | 0.9312 / 0.9334 | -0.0013 | -0.0046 |
| VideoFeedback | comgenvid+genvideo | mean | 0.8228 / 0.8302 | mean | 0.8228 / 0.8302 | +0.0000 | +0.0000 |
| GenVideo | comgenvid+videofeedback | mean | 0.8072 / 0.8092 | mean | 0.8072 / 0.8092 | +0.0000 | +0.0000 |

- 平均 |ΔAUC| = 0.0004；最差 ΔAUC = -0.0013。

## 论文写作建议

- alpha 的 frozen gap 直接反映全局 STALL 与 patch 二阶时序证据的融合比例是否可跨数据集迁移。
- beta 的 frozen gap 用于说明 patch 内部空间证据只提供辅助作用，局部二阶时序仍是主贡献。
- region 的 frozen gap 通常会更大，因为它对应局部时序证据的空间支持域，受视频分辨率、运动尺度和生成器类型影响。
- aggregation 的 frozen gap 用于说明 mean 与 bottom-k 不是普适优劣关系；应将其写为数据集局部异常分布的边界分析，而不是事后重选默认配置。

结论上，手稿应报告 oracle sweep 作为敏感性上界，同时给出 leave-one-dataset-out frozen 结果作为泛化性证据；
默认配置不应声称是所有数据集的全局最优，而应描述为在不使用测试批次 rank、标签或生成器来源的前提下，将全局校准与局部二阶时序证据稳定结合的固定推理规则。
