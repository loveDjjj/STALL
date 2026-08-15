# Alpha-STALLED 当前锁定协议

本文档是 `u0_locked_v1` 的唯一可读协议说明。发生冲突时，机器可读的
`configs/alpha_stalled_u0_locked.yaml`、`release/u0/` 清单和哈希优先于本文档；
本文档优先于历史 README、旧报告和 `paper_scores`。

## 1. 协议身份

| 字段 | 值 |
|---|---|
| 方法名 | Alpha-STALLED |
| protocol ID | `u0_locked_v1` |
| release 名称 | `Alpha-STALLED-U0-locked` |
| 锁定日期 | 2026-07-24 |
| score direction | higher is real |
| 主指标 | Macro-3 real-positive AP |
| 主评测范围 | strict 2-second, 20 generators |

`U0` 是协议/发布版本，不是另一个检测器。实验报告中的 `LSTL` 指使用 Local
D2 的完整锁定 Alpha-STALLED，不应再作为第三个方法名扩散。

## 2. 数据身份

| 数据集 | 校准真实 | 评测真实 | 评测生成 | 评测总数 | 生成器 |
|---|---:|---:|---:|---:|---:|
| ComGenVid | 200 | 898 | 3,400 | 4,298 | 2 |
| VideoFeedback | 200 | 500 | 3,000 | 3,500 | 10 |
| GenVideo | 200 | 7,984 | 5,639 | 13,623 | 8 |
| 合计 | 600 | 9,382 | 12,039 | 21,421 | 20 |

校准清单与评测清单的 `video_id` 交集为 0。校准集只包含真实视频；测试真实和
生成视频不参与 whitening、Gaussian 参数、CDF 或阈值拟合，locked run 也不再
根据它们调整配置。必须同时保留开发边界：`alpha/beta/K` 和总体结构曾在这三个
开发基准上查看生成视频结果后冻结，因此这三个数据集不是 untouched confirmation
sets。锁定后 GenVidBench 才承担外部确认角色。

## 3. 窗口采样

1. 将视频按名义 8 FPS 建立可用帧索引。
2. 每个窗口取连续 16 个索引，对应约 2 秒。
3. 在闭区间 `[0, number_of_8fps_frames - 16]` 上均匀选择三个整数起点。
4. 映射后完全相同的 16 帧窗口去重。
5. 不复制帧、不使用 1 秒 fallback、不保留不足 16 帧的尾部。

评测 effective-K 分布是 K=1/2/3：3,689/73/17,659。`effective_K` 是唯一有效
窗口数，不是请求窗口数。8 FPS 是抽样密度，不改变源视频时长或播放速度。

## 4. 表征与分支

所有分支共享冻结 DINOv3 ViT-L/16 final layer 23：

- 输入 BGR 经 OpenCV 解码后转 RGB；
- `ToTensor -> Resize(224,224) -> ImageNet normalization`；
- final CLS/global token 和 14x14=196 个 final patch token；
- 特征维度 1024；
- Local 不包含 CLS 或 register token。

窗口级定义：

```text
GlobalSpatial: final CLS Gaussian likelihood, max over frames
GlobalT1: L2-normalized lag-1 CLS difference likelihood, min over time
G_k = 0.5 * GlobalSpatial + 0.5 * GlobalT1

PatchSpatial: final patch-token Gaussian likelihood, mean over patch/time
PatchD2[t,i] = P[t+2,i] - 2*P[t+1,i] + P[t,i]
PatchD2: feature-axis L2 normalize, Gaussian likelihood, mean over patch/time
L_k = 0.1 * PatchSpatial + 0.9 * PatchD2
```

Local 固定 region 1、mean aggregation。region 1 表示保留原始 14x14 token，
不是选择一个图像区域。旧 region 2/3 是 D2 前的 raw-token 空间平均池化，不属于
正式配置。

## 5. 真实分布校准

Global 的基础 whitening 和窗口分量参考来自锁定 VATEX 参数。Local 的
PatchSpatial/PatchD2 whitening 由每个目标数据集各 200 条独立真实视频拟合。

Local 窗口 ECDF 使用每个校准视频一个锁定 K1-current 窗口。随后对 K=3
轨迹分别计算：

```text
G_raw(V) = mean_k G_k
L_raw(V) = mean_k L_k
```

根据目标评测视频的 effective-K，从真实校准轨迹选择同样数量的窗口并建立
`G_raw`、`L_raw` 视频级 ECDF。每个真实校准视频每个分支只贡献一个值：

```text
G(V) = ECDF_real,effective-K(G_raw(V))
L(V) = ECDF_real,effective-K(L_raw(V))
S(V) = 0.6 * G(V) + 0.4 * L(V)
```

## 6. 数值规则

- DINO backbone、差分和 L2 normalization：float32。
- whitening mean/matrix、Gaussian likelihood、raw score、CDF：float64。
- CDF：stable mergesort，右闭合 `P(reference <= score)`。
- DINO 每视频独立组织 unique frames，release frame batch size 为 32。
- 外层 video batch 不得改变单视频 DINO frame grouping 或 score GEMM shape。

这些规则属于协议，不是可自由替换的性能优化。

## 7. 指标

主表按生成器分别构建确定性的真假平衡比较，计算 AUC/AP；随后：

1. 数据集内对生成器等权平均；
2. 三个数据集等权平均得到 Macro-3。

锁定主表使用 `real=1` 和 higher-is-real score，因此 AP 必须写为 `AP_real`。
原 STALL Table 1 使用 generated-positive AP，必须写为 `AP_fake`。Macro-3 是三个
数据集等权；All-23 是 23 个生成器等权，二者不可互换。

Bootstrap 在视频 ID 层进行配对 cluster resampling，窗口从不作为独立样本。
Bootstrap 估计固定 cohort 的有限样本不确定性，不增加数据覆盖率。

## 8. 权威资产

```text
configs/alpha_stalled_u0_locked.yaml
release/u0/calibration_manifest.json
release/u0/evaluation_manifest.json
release/u0/frame_indices.json
release/u0/params/*.npz
release/u0/final_video_scores.csv
release/u0/reproduction_metadata.json
release/u0/config_and_checkpoint_hashes.json
release/u0/validation.json
```

锁定结果为 Macro-3 AUC/AP_real `0.8740724396350226/0.8722992273320395`。
K3 traversal 覆盖 22,021 个 calibration/evaluation 视频和 58,496 个窗口；
`release/u0/validation.json` 记录 61 项检查全部通过，且没有缺失视频、重复键、
解码/评分失败或非有限必需分数。

## 9. 23-source 扩展

`duration_aware_23source` 是独立扩展协议，不替换 strict-20 主协议。有效 2 秒的
视频仍使用 2 秒/16 帧/K<=3；三个短视频来源使用原 STALL 的 1 秒/8 帧/K=1。
该扩展覆盖全部 45,185 条生成视频分数，并使用独立 1 秒/2 秒真实校准流。

## 10. 变更规则

以下任一变化都必须产生新 protocol ID，不能继续写作 `u0_locked_v1`：

- 数据身份或 split；
- 帧索引、FPS、窗口长度、K 或短视频策略；
- backbone/checkpoint/layer/preprocessing/batch grouping；
- region、aggregation、D1/D2、alpha、beta；
- whitening、CDF、tie policy、effective-K 处理；
- label/AP 方向或 Macro 定义。

新实验不得覆盖 `release/u0/`，应写入独立 `results/<experiment_id>/` 后再登记。
