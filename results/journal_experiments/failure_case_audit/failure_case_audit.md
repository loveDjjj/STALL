# Failure / boundary case 审计

本审计合并 failure candidate、逐生成器 paired bootstrap、生成器宏平均 bootstrap 和 index 元数据。
它不观看视频、不重新提取特征、不参与调参，只用于选择论文 failure-mode 案例和限制性分析。

## 审计口径

- `P0`：优先进入正文或主补充图的案例；包括稳定负迁移生成器的生成样本，以及极低分真实样本。
- `P1`：适合进入补充材料的边界案例；包括 CI 跨 0 的生成器或强 patch/global 冲突。
- `P2`：保留为扩展审计，不建议占用正文版面。

## 数量汇总

| dataset | priority | risk type | generator CI class | cases |
|---|---|---|---|---:|
| ComGenVid | P0 | real_false_positive_risk | no_generator_ci | 15 |
| ComGenVid | P1 | generated_false_real_risk | stable_positive_transfer | 20 |
| ComGenVid | P1 | patch_global_conflict | stable_positive_transfer | 20 |
| ComGenVid | P1 | real_recall_degradation_risk | no_generator_ci | 20 |
| ComGenVid | P1 | residual_generated_hard_case | stable_positive_transfer | 7 |
| ComGenVid | P2 | real_false_positive_risk | no_generator_ci | 5 |
| ComGenVid | P2 | residual_generated_hard_case | stable_positive_transfer | 13 |
| GenVideo | P0 | real_false_positive_risk | no_generator_ci | 20 |
| GenVideo | P1 | generated_false_real_risk | boundary_ci_crosses_zero | 17 |
| GenVideo | P1 | generated_false_real_risk | stable_positive_transfer | 3 |
| GenVideo | P1 | patch_global_conflict | boundary_ci_crosses_zero | 17 |
| GenVideo | P1 | patch_global_conflict | stable_positive_transfer | 3 |
| GenVideo | P1 | real_recall_degradation_risk | no_generator_ci | 20 |
| GenVideo | P1 | residual_generated_hard_case | boundary_ci_crosses_zero | 1 |
| GenVideo | P1 | residual_generated_hard_case | stable_positive_transfer | 2 |
| GenVideo | P2 | residual_generated_hard_case | stable_positive_transfer | 17 |
| VideoFeedback | P0 | generated_false_real_risk | stable_negative_transfer | 8 |
| VideoFeedback | P0 | patch_global_conflict | stable_negative_transfer | 8 |
| VideoFeedback | P0 | real_false_positive_risk | no_generator_ci | 20 |
| VideoFeedback | P0 | residual_generated_hard_case | stable_negative_transfer | 15 |
| VideoFeedback | P1 | generated_false_real_risk | boundary_ci_crosses_zero | 9 |
| VideoFeedback | P1 | generated_false_real_risk | stable_positive_transfer | 3 |
| VideoFeedback | P1 | patch_global_conflict | boundary_ci_crosses_zero | 9 |
| VideoFeedback | P1 | patch_global_conflict | stable_positive_transfer | 3 |
| VideoFeedback | P1 | real_recall_degradation_risk | no_generator_ci | 20 |
| VideoFeedback | P1 | residual_generated_hard_case | boundary_ci_crosses_zero | 1 |
| VideoFeedback | P1 | residual_generated_hard_case | stable_positive_transfer | 1 |
| VideoFeedback | P2 | residual_generated_hard_case | stable_positive_transfer | 3 |

## P0 推荐案例

| dataset | source | file | risk | global | patch | alpha | gen ΔAUC CI | recommended use |
|---|---|---|---|---:|---:|---:|---:|---|
| ComGenVid | real/MSVD | `1QVn0FffExM_14_24.mp4` | real_false_positive_risk | 0.0093 | 0.0062 | 0.0081 | n/a | failure figure: real-video false-positive / recall boundary |
| ComGenVid | real/MSVD | `Z19zFlPah-o_6_11.mp4` | real_false_positive_risk | 0.0024 | 0.0091 | 0.0051 | n/a | failure figure: real-video false-positive / recall boundary |
| ComGenVid | real/MSVD | `ZbzDGXEwtGc_6_15.mp4` | real_false_positive_risk | 0.0087 | 0.0011 | 0.0056 | n/a | failure figure: real-video false-positive / recall boundary |
| ComGenVid | real/MSVD | `dP15zlyra3c_0_10.mp4` | real_false_positive_risk | 0.0042 | 0.0067 | 0.0052 | n/a | failure figure: real-video false-positive / recall boundary |
| GenVideo | real/MSR-VTT | `video1032.mp4` | real_false_positive_risk | 0.0040 | 0.0041 | 0.0040 | n/a | failure figure: real-video false-positive / recall boundary |
| GenVideo | real/MSR-VTT | `video2150.mp4` | real_false_positive_risk | 0.0034 | 0.0044 | 0.0038 | n/a | failure figure: real-video false-positive / recall boundary |
| GenVideo | real/MSR-VTT | `video3085.mp4` | real_false_positive_risk | 0.0051 | 0.0015 | 0.0037 | n/a | failure figure: real-video false-positive / recall boundary |
| GenVideo | real/MSR-VTT | `video9844.mp4` | real_false_positive_risk | 0.0030 | 0.0036 | 0.0032 | n/a | failure figure: real-video false-positive / recall boundary |
| VideoFeedback | annotated/VideoCrafter2 | `2000555.mp4` | generated_false_real_risk | 0.0589 | 0.9663 | 0.4219 | [-0.0354, -0.0253] | failure figure: stable generator-level negative transfer or hard generated case |
| VideoFeedback | annotated/VideoCrafter2 | `2001966.mp4` | generated_false_real_risk | 0.0013 | 0.9104 | 0.3649 | [-0.0354, -0.0253] | failure figure: stable generator-level negative transfer or hard generated case |
| VideoFeedback | annotated/VideoCrafter2 | `2002317.mp4` | generated_false_real_risk | 0.0038 | 0.9099 | 0.3662 | [-0.0354, -0.0253] | failure figure: stable generator-level negative transfer or hard generated case |
| VideoFeedback | annotated/VideoCrafter2 | `2002745.mp4` | generated_false_real_risk | 0.0412 | 0.9400 | 0.4008 | [-0.0354, -0.0253] | failure figure: stable generator-level negative transfer or hard generated case |
| VideoFeedback | real/Panda70M | `p100187.mp4` | real_false_positive_risk | 0.0052 | 0.0091 | 0.0067 | n/a | failure figure: real-video false-positive / recall boundary |
| VideoFeedback | real/Panda70M | `p100866.mp4` | real_false_positive_risk | 0.0084 | 0.0049 | 0.0070 | n/a | failure figure: real-video false-positive / recall boundary |
| VideoFeedback | real/Panda70M | `p106439.mp4` | real_false_positive_risk | 0.0067 | 0.0019 | 0.0048 | n/a | failure figure: real-video false-positive / recall boundary |
| VideoFeedback | real/Panda70M | `p106966.mp4` | real_false_positive_risk | 0.0048 | 0.0014 | 0.0035 | n/a | failure figure: real-video false-positive / recall boundary |
| VideoFeedback | annotated/Text2Video-Zero | `1000945.mp4` | residual_generated_hard_case | 0.9293 | 0.9042 | 0.9193 | [-0.0767, -0.0649] | failure figure: stable generator-level negative transfer or hard generated case |
| VideoFeedback | annotated/Text2Video-Zero | `1000968.mp4` | residual_generated_hard_case | 0.7043 | 0.8567 | 0.7653 | [-0.0767, -0.0649] | failure figure: stable generator-level negative transfer or hard generated case |
| VideoFeedback | annotated/Text2Video-Zero | `1002250.mp4` | residual_generated_hard_case | 0.6764 | 0.8939 | 0.7634 | [-0.0767, -0.0649] | failure figure: stable generator-level negative transfer or hard generated case |
| VideoFeedback | annotated/Text2Video-Zero | `1002458.mp4` | residual_generated_hard_case | 0.7584 | 0.9152 | 0.8211 | [-0.0767, -0.0649] | failure figure: stable generator-level negative transfer or hard generated case |

## 写作建议

- 正文可以用宏平均 paired bootstrap 说明三数据集总体提升均稳定为正。
- failure-mode 小节不要否认 VideoFeedback 内部负迁移；应明确 Text2Video-Zero / VideoCrafter2 是稳定负迁移来源。
- patch-global conflict 案例适合展示局部二阶时序证据可能过强，从而把生成视频推向真实侧。
- 真实视频低分案例适合写成 real-domain / low-motion / compression boundary，下一步若要更强证据需人工观看关键帧。
