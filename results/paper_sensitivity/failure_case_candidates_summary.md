# 失败样本候选摘要

本文件由 `results/paper_sensitivity/failure_case_candidates.csv` 汇总而来，用于选择后续 patch anomaly map 或人工审计案例。候选只用于事后分析，不参与推理或调参。

## 优先审计对象

### Paired bootstrap 判定的生成器级负迁移

| 数据集 | 生成器 | ΔAUC mean | 95% CI |
|---|---|---:|---:|
| videofeedback | Text2Video-Zero | -0.0709 | [-0.0767, -0.0649] |
| videofeedback | VideoCrafter2 | -0.0304 | [-0.0354, -0.0253] |

这些生成器应优先做失败案例图，因为其负迁移在 paired bootstrap 下比较稳定。

### CI 跨 0 的边界生成器

| 数据集 | 生成器 | ΔAUC mean | 95% CI |
|---|---|---:|---:|
| videofeedback | LaVie-base | -0.0013 | [-0.0076, +0.0054] |
| videofeedback | SoRA-Clip | +0.0051 | [-0.0056, +0.0173] |
| genvideo | Lavie | +0.0091 | [-0.0014, +0.0201] |
| genvideo | MorphStudio | +0.0094 | [-0.0041, +0.0239] |
| genvideo | Crafter | +0.0162 | [-0.0092, +0.0453] |

这些生成器不是稳定负迁移，但适合检查方法边界。

## 生成视频被 Alpha-STALLED 打得更像真实

| 数据集 | subset/source | 文件 | global | patch | alpha | alpha-global |
|---|---|---|---:|---:|---:|---:|
| videofeedback | annotated/LaVie-base | `4001104.mp4` | 0.0234 | 0.9839 | 0.4076 | +0.3842 |
| videofeedback | annotated/VideoCrafter2 | `2001966.mp4` | 0.0013 | 0.9104 | 0.3649 | +0.3636 |
| videofeedback | annotated/LaVie-base | `4002630.mp4` | 0.0861 | 0.9948 | 0.4496 | +0.3635 |
| videofeedback | annotated/VideoCrafter2 | `2000555.mp4` | 0.0589 | 0.9663 | 0.4219 | +0.3630 |
| videofeedback | annotated/VideoCrafter2 | `2002317.mp4` | 0.0038 | 0.9099 | 0.3662 | +0.3624 |
| videofeedback | annotated/LaVie-base | `4004413.mp4` | 0.0672 | 0.9677 | 0.4274 | +0.3602 |
| videofeedback | annotated/VideoCrafter2 | `2002745.mp4` | 0.0412 | 0.9400 | 0.4008 | +0.3595 |
| videofeedback | annotated/AnimateDiff | `5000419.mp4` | 0.0166 | 0.9082 | 0.3733 | +0.3566 |
| videofeedback | annotated/AnimateDiff | `5000254.mp4` | 0.0380 | 0.9218 | 0.3915 | +0.3535 |
| videofeedback | annotated/SoRA-Clip | `s000095.mp4` | 0.0622 | 0.9454 | 0.4155 | +0.3533 |
| videofeedback | annotated/LaVie-base | `4004408.mp4` | 0.1039 | 0.9866 | 0.4570 | +0.3531 |
| videofeedback | annotated/VideoCrafter2 | `2006829.mp4` | 0.0062 | 0.8883 | 0.3591 | +0.3528 |

## patch 分支显著高于 global

| 数据集 | subset/source | 文件 | global | patch | alpha | alpha-global |
|---|---|---|---:|---:|---:|---:|
| videofeedback | annotated/LaVie-base | `4001104.mp4` | 0.0234 | 0.9839 | 0.4076 | +0.3842 |
| videofeedback | annotated/VideoCrafter2 | `2001966.mp4` | 0.0013 | 0.9104 | 0.3649 | +0.3636 |
| videofeedback | annotated/LaVie-base | `4002630.mp4` | 0.0861 | 0.9948 | 0.4496 | +0.3635 |
| videofeedback | annotated/VideoCrafter2 | `2000555.mp4` | 0.0589 | 0.9663 | 0.4219 | +0.3630 |
| videofeedback | annotated/VideoCrafter2 | `2002317.mp4` | 0.0038 | 0.9099 | 0.3662 | +0.3624 |
| videofeedback | annotated/LaVie-base | `4004413.mp4` | 0.0672 | 0.9677 | 0.4274 | +0.3602 |
| videofeedback | annotated/VideoCrafter2 | `2002745.mp4` | 0.0412 | 0.9400 | 0.4008 | +0.3595 |
| videofeedback | annotated/AnimateDiff | `5000419.mp4` | 0.0166 | 0.9082 | 0.3733 | +0.3566 |
| videofeedback | annotated/AnimateDiff | `5000254.mp4` | 0.0380 | 0.9218 | 0.3915 | +0.3535 |
| videofeedback | annotated/SoRA-Clip | `s000095.mp4` | 0.0622 | 0.9454 | 0.4155 | +0.3533 |
| videofeedback | annotated/LaVie-base | `4004408.mp4` | 0.1039 | 0.9866 | 0.4570 | +0.3531 |
| videofeedback | annotated/VideoCrafter2 | `2006829.mp4` | 0.0062 | 0.8883 | 0.3591 | +0.3528 |

## Alpha-STALLED 仍最难识别的生成视频

| 数据集 | subset/source | 文件 | global | patch | alpha | alpha-global |
|---|---|---|---:|---:|---:|---:|
| videofeedback | annotated/Text2Video-Zero | `1000945.mp4` | 0.9293 | 0.9042 | 0.9193 | -0.0100 |
| videofeedback | annotated/Text2Video-Zero | `1002458.mp4` | 0.7584 | 0.9152 | 0.8211 | +0.0627 |
| videofeedback | annotated/LaVie-base | `4000576.mp4` | 0.8148 | 0.7852 | 0.8030 | -0.0118 |
| videofeedback | annotated/ModelScope | `3000848.mp4` | 0.8951 | 0.6400 | 0.7931 | -0.1020 |
| videofeedback | annotated/ModelScope | `3004838.mp4` | 0.7920 | 0.7895 | 0.7910 | -0.0010 |
| videofeedback | annotated/Text2Video-Zero | `1000968.mp4` | 0.7043 | 0.8567 | 0.7653 | +0.0609 |
| videofeedback | annotated/Text2Video-Zero | `1002250.mp4` | 0.6764 | 0.8939 | 0.7634 | +0.0870 |
| videofeedback | annotated/Text2Video-Zero | `1001779.mp4` | 0.7628 | 0.7502 | 0.7577 | -0.0050 |
| videofeedback | annotated/Text2Video-Zero | `1000937.mp4` | 0.7251 | 0.7877 | 0.7501 | +0.0250 |
| videofeedback | annotated/Text2Video-Zero | `1001056.mp4` | 0.6318 | 0.9157 | 0.7453 | +0.1136 |
| videofeedback | annotated/Text2Video-Zero | `1000341.mp4` | 0.5977 | 0.9617 | 0.7433 | +0.1456 |
| videofeedback | annotated/Text2Video-Zero | `1002423.mp4` | 0.6115 | 0.9382 | 0.7422 | +0.1307 |

## 真实视频被 Alpha-STALLED 打得更像生成

| 数据集 | subset/source | 文件 | global | patch | alpha | alpha-global |
|---|---|---|---:|---:|---:|---:|
| genvideo | real/MSR-VTT | `video8981.mp4` | 0.9321 | 0.1964 | 0.6378 | -0.2943 |
| genvideo | real/MSR-VTT | `video7428.mp4` | 0.8543 | 0.1475 | 0.5716 | -0.2827 |
| genvideo | real/MSR-VTT | `video9338.mp4` | 0.7357 | 0.0849 | 0.4754 | -0.2603 |
| genvideo | real/MSR-VTT | `video9148.mp4` | 0.7785 | 0.1726 | 0.5361 | -0.2424 |
| genvideo | real/MSR-VTT | `video1039.mp4` | 0.8014 | 0.1972 | 0.5597 | -0.2417 |
| comgenvid | real/MSVD | `jjl2ZMdFCsw_130_142.mp4` | 0.9317 | 0.3456 | 0.6973 | -0.2344 |
| genvideo | real/MSR-VTT | `video1994.mp4` | 0.8534 | 0.2694 | 0.6198 | -0.2336 |
| genvideo | real/MSR-VTT | `video2622.mp4` | 0.7878 | 0.2234 | 0.5620 | -0.2257 |
| genvideo | real/MSR-VTT | `video5681.mp4` | 0.9181 | 0.3621 | 0.6957 | -0.2224 |
| genvideo | real/MSR-VTT | `video7312.mp4` | 0.7567 | 0.2053 | 0.5361 | -0.2206 |
| comgenvid | real/MSVD | `_1vy2HIN60A_32_40.mp4` | 0.7331 | 0.1859 | 0.5142 | -0.2189 |
| genvideo | real/MSR-VTT | `video5930.mp4` | 0.6950 | 0.1484 | 0.4763 | -0.2186 |

## Alpha-STALLED 最容易误伤的真实视频

| 数据集 | subset/source | 文件 | global | patch | alpha | alpha-global |
|---|---|---|---:|---:|---:|---:|
| genvideo | real/MSR-VTT | `video9844.mp4` | 0.0030 | 0.0036 | 0.0032 | +0.0002 |
| videofeedback | real/Panda70M | `p106966.mp4` | 0.0048 | 0.0014 | 0.0035 | -0.0014 |
| genvideo | real/MSR-VTT | `video3085.mp4` | 0.0051 | 0.0015 | 0.0037 | -0.0015 |
| genvideo | real/MSR-VTT | `video2150.mp4` | 0.0034 | 0.0044 | 0.0038 | +0.0004 |
| genvideo | real/MSR-VTT | `video1032.mp4` | 0.0040 | 0.0041 | 0.0040 | +0.0001 |
| genvideo | real/MSR-VTT | `video5272.mp4` | 0.0054 | 0.0020 | 0.0040 | -0.0013 |
| videofeedback | real/Panda70M | `p106439.mp4` | 0.0067 | 0.0019 | 0.0048 | -0.0019 |
| genvideo | real/MSR-VTT | `video113.mp4` | 0.0033 | 0.0072 | 0.0049 | +0.0016 |
| comgenvid | real/MSVD | `Z19zFlPah-o_6_11.mp4` | 0.0024 | 0.0091 | 0.0051 | +0.0027 |
| genvideo | real/MSR-VTT | `video5946.mp4` | 0.0017 | 0.0102 | 0.0051 | +0.0034 |
| comgenvid | real/MSVD | `dP15zlyra3c_0_10.mp4` | 0.0042 | 0.0067 | 0.0052 | +0.0010 |
| genvideo | real/MSR-VTT | `video4032.mp4` | 0.0058 | 0.0045 | 0.0053 | -0.0005 |

## 建议的案例图选择规则

1. 优先选 VideoFeedback / Text2Video-Zero 和 VideoFeedback / VideoCrafter2，因为 paired bootstrap 已显示稳定负迁移。
2. 每个负迁移生成器至少选一个 `generated_score_increased_by_alpha` 样本和一个 `generated_patch_global_conflict` 样本。
3. 同时选一个 `real_score_decreased_by_alpha` 真实样本，说明 patch 分支也可能影响真实视频召回。
4. 案例图不要只画分数条形图，应回到 patch cache 或原视频，展示 patch-level anomaly map 随时间变化。
