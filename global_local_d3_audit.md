# Global--Local--D3 实现与协议审计

审计日期：2026-07-23

本审计对应 Alpha-STALLED 新主线的阶段 0。结论来自当前源码、冻结配置、校准
参数和本地 index/cache 的只读检查；没有启动新的全量特征提取或大规模评测。

## 结论摘要

1. 原版 global STALL 使用 DINOv3 ViT-L/16 的归一化 CLS token，不使用
   register token，也不做 patch-token mean。
2. 当前 Local D2 是选项 A：同网格向量有限差分
   `z[t+2,p] - 2*z[t+1,p] + z[t,p]`，随后沿特征维 L2 normalize。
3. 所有 STALL/patch percentile 与最终分数均为越高越真实。`beta=0.10` 表示
   patch spatial 占 10%，patch temporal 占 90%；最终 `alpha=0.60` 表示 global
   占 60%，完整 patch 分支占 40%。
4. 两条 D3 路径的标准差定义不一致：operator-controlled 使用 `ddof=0`，
   D3-exact 使用 `ddof=1`。阶段 1 前必须冻结为同一口径并写入输出元数据。
5. 当前 raw D3 只做单侧经验 CDF，隐含“volatility 越大越真实”；附件要求的
   two-sided、motion-conditioned 和 time-normalized calibration 尚未实现。
6. index 中没有重复帧号，但 cache 抽样发现 GenVideo real 存在明显的相邻
   embedding 完全重复。公平协议必须增加内容级重复帧检测。
7. 当前 patch D2 是 1024 维向量加速度，而 global D3 是标量距离加速度；二者
   不能直接相减形成 local residual。阶段 3 必须先统一为向量定义或标量定义。

## 1. 现有 Global STALL

### 1.1 表征与张量形状

`src/stall.py:244-276` 对每帧调用 `self.model(x)`，输出 `[B,1024]`，再还原为
视频张量 `[N,T,1024]`。本地 DINOv3 的 `forward()` 在
`dinov3/dinov3/models/vision_transformer.py:324-329` 明确返回
`head(x_norm_clstoken)`；`forward_features()` 同时暴露 CLS、storage/register 和
patch token，但 global 路径只取 CLS。因此：

| 表征 | 是否用于 global STALL | 证据 |
|---|---|---|
| normalized CLS token | 是 | `vision_transformer.py:254,324-329` |
| register/storage token | 否 | 只在 feature dict 的 `x_storage_tokens` 中出现 |
| mean patch token | 否 | global 路径没有读取 `x_norm_patchtokens` |

冻结配置为 224x224 输入、ViT-L/16、14x14 patch grid、特征维 1024，见
`configs/alpha_stalled.yaml:23-30`。

### 1.2 空间似然

对 frame embedding `g[t]`：

```text
y_s[t] = (g[t] - mu_s) W_s
LL_s[t] = -0.5 * (D_s log(2*pi) + ||y_s[t]||_2^2)
S_s_raw = max_t LL_s[t]
G_s = ECDF_real_s(S_s_raw)
```

实现位置为 `src/stall.py:99-113,280-282,304-311`。当前参数形状为
`mu_s=[1024]`、`W_s=[1024,1024]`；校准矩阵为 `[33976,16]`。空间聚合是
`max`，即选择一个视频中 likelihood 最高、最像真实分布的帧，不是 mean 或
bottom-k。

### 1.3 一阶时序似然

```text
delta[t] = g[t+1] - g[t]
u[t] = delta[t] / max(||delta[t]||_2, eps)
y_t[t] = (u[t] - mu_t) W_t
LL_t[t] = -0.5 * (D_t log(2*pi) + ||y_t[t]||_2^2)
S_t_raw = min_t LL_t[t]
G_t1 = ECDF_real_t(S_t_raw)
```

`src/stall.py:116-136,284-295` 证明差分在白化前做 L2 normalize；聚合为
`min`，即最异常 frame pair。完全相同的连续 embedding 被置为 `+inf`，从 min
聚合中排除。当前 `W_t=[1024,1023]`，因为白化拟合发生 rank truncation；校准
矩阵为 `[33976,15]`。

Global 最终分数为：

```text
G = 0.5 * (G_s + G_t1)
```

见 `src/stall.py:299-322`。经验 CDF 使用 `searchsorted(..., side="right")`，所以
likelihood 越高、percentile 越高、越接近真实视频。

## 2. 现有 Local Patch 分支

### 2.1 Patch token 与区域池化

`src/stall_patch.py:63-122` 从 `forward_features()` 读取
`x_norm_patchtokens`，每帧为 `[196,1024]`；视频为 `[T,196,1024]`。global token
仍由 `self.model(x)` 单独取得，因此没有把 CLS/register 混入 patch grid。

`patch_region_size=r` 只在 temporal feature 前做非重叠区域 mean pooling：

```text
[B,T,14*14,1024] -> [B,T,floor(14/r)*floor(14/r),1024]
```

实现见 `src/eval_patch_fast.py:102-150`。边界不能整除时会裁掉底部/右侧；空间
patch likelihood 仍在原始 14x14 token 上计算。

### 2.2 D2 的准确公式

当前 `same_grid_second_order` 是：

```text
a[t,p] = z[t+2,p] - 2*z[t+1,p] + z[t,p]
u2[t,p] = a[t,p] / max(||a[t,p]||_2, eps)
```

对应附件选项 A，不是距离序列差分，也不是 lag=2 embedding difference。
NumPy 实现在 `src/patch_matching.py:113-136,323-324`，GPU fast scorer 在
`src/eval_patch_fast.py:133-150`。输入 `[B,T,P,1024]`，输出
`[B,T-2,P',1024]`；`P'` 由 temporal region pooling 决定。

### 2.3 Patch likelihood、聚合和 beta

空间和时序分别白化、计算标准高斯 log likelihood、用相同的 dataset-specific
aggregation 压成每视频标量，再分别映射为真实校准 ECDF：

```text
L_s = ECDF_real(aggregate(LL(raw_patch)))
L_t2 = ECDF_real(aggregate(LL(normalized_D2)))
L = beta * L_s + (1 - beta) * L_t2
```

`src/create_patch_params.py:267-344` 给出校准路径，
`src/eval_patch_fast.py:285-306` 给出推理路径。冻结 `beta=0.10`，所以空间占
10%，D2 temporal 占 90%，不是显著性阈值或 bottom-k 比例。

| 数据集 | temporal region | 聚合 | 形状/含义 |
|---|---:|---|---|
| ComGenVid | 3 | bottom 20% mean | D2 为 `[B,T-2,16,1024]`；空间仍为 196 patch |
| VideoFeedback | 1 | mean | D2 为 `[B,T-2,196,1024]` |
| GenVideo | 2 | mean | D2 为 `[B,T-2,49,1024]` |

冻结配置见 `configs/alpha_stalled.yaml:38-70`。Bottom-k 取最低 likelihood 项，
实现为 `ceil(total_items * ratio)` 后求 mean；mean 配置则对全部时间和 patch 求均值。

Global--Local 最终融合为：

```text
S = alpha * G + (1 - alpha) * L, alpha = 0.60
```

`tools/eval_alpha_stalled.py:40-66` 用 one-to-one inner merge 强制两分支视频 ID
完全一致，并声明 `HIGHER_IS_REAL`。因此 global、patch、fusion 和 percentile 的
方向完全一致：越高越真实。

## 3. D3-style 实现审计

### 3.1 Operator-controlled DINOv3 路径

`tools/eval_global_second_order_volatility.py:110-135` 当前计算：

```text
d[t] = ||g[t+1] - g[t]||_2
a[t] = d[t+1] - d[t]
V = std(a, ddof=0)
```

默认是 raw DINOv3 CLS embedding + L2。可选项包括先对 embedding 行归一化、
cosine distance 和 cosine similarity。对 std 而言 similarity/distance 只差符号，
数值相同；raw L2 与 normalized L2 不同。

当前 real calibration 是单侧：

```text
D3_score = ECDF_real(V)
```

见 `tools/eval_global_second_order_volatility.py:186-200`。这把高 volatility 固定
解释为更真实，尚未实现附件要求的 `2*min(F,1-F)`、motion-conditioned CDF 或
时间尺度归一化。

短视频不足 3 帧时返回 NaN 并跳过。CLI 可用 `--fallback-durations 2,1`，逐视频
优先选 2s、缺失时选 1s，并在输出记录实际 duration；因此 fallback 结果会混合
1s/2s，但不会在同一个视频内混合两个窗口。`_slice_window()` 在 index 超出旧 cache
长度时退化为 cache 前 N 帧，这个兼容策略可能静默改变窗口，阶段 1 应改为显式
cache layout 元数据或直接报错。

### 3.2 D3-exact / official-baseline 路径

`tools/prepare_genvideo_d3_exact_protocol.py:75-138` 固定 3s@8fps 抽帧计划；超过
3s 的视频使用固定 seed 的随机整数起点。`extract_d3_frames_from_runlist.py:35-50`
调用 ffmpeg `fps=8` 写 JPEG。评测器执行长边两侧各裁 10%、resize 224，并读取：

- 少于 8 帧：失败；
- 8-15 帧：读取前 8 帧；
- 至少 16 帧：读取前 16 帧。

见 `tools/eval_d3_exact_from_frames.py:77-111`。默认 encoder 为
`microsoft/xclip-base-patch16`，也支持 CLIP、DINOv2 和本地 DINOv3；embedding
优先取 `pooler_output`，否则取 CLS。D3-exact 的二阶公式与上面一致，但使用
`np.std(..., ddof=1)`，见 `tools/eval_d3_exact_from_frames.py:175-189`。

当前风险：

- official XCLIP 与 operator DINOv3 必须作为不同结果类别，不能混表解释为同一算子；
- `ddof=0` 与 `ddof=1` 不一致；
- ffmpeg `fps=8` 对低 FPS 输入可能复制帧；当前工具没有检测或删除重复 JPEG；
- exact 路径读取前 8/16 帧，而 operator 路径使用 index 中的随机 1s/2s 窗口；
- exact 输出的 raw std 直接当 realness，未经过 STALL real calibration。

## 4. 抽帧、时间间隔和重复帧实证

对三个完整 index 的全部 `downsample_idxs` 检查结果：重复 index 视频为 0，
非递增 index 视频为 0。由于 `src/video_index.py:32-54` 只允许源 FPS >= 8，并按
`round(source_fps/8 * j)` 取样，索引层不会通过重复 native frame 上采样。

但是实际时间间隔不恒等于 0.125 s：

| 数据集 | mean dt | std dt | min--max dt | 偏离 125ms 超过 1ms |
|---|---:|---:|---:|---:|
| ComGenVid | 0.124950 | 0.012123 | 0.066298--0.200000 | 64.00% |
| VideoFeedback | 0.125000 | 0.000000 | 0.125000--0.125000 | 0.00% |
| GenVideo | 0.125011 | 0.013451 | 0.066667--0.240096 | 75.37% |

因此普通 D3 不能假设每个相邻 native index 的实际 `delta_t` 完全相同；方案 D
应使用 index 与源 FPS 计算每一步时间间隔，而不是只除以名义 0.125 s。

进一步对每个数据集 real/fake 各确定性抽样最多 500 个 cache 视频，用相邻
DINOv3 embedding 是否完全相等作为内容重复代理：

| 数据集/subset | 视频 | 含重复对的视频 | 重复相邻对 / 总对 | 比例 |
|---|---:|---:|---:|---:|
| ComGenVid annotated | 500 | 0 | 0/7500 | 0.0000% |
| ComGenVid real | 499 | 1 | 1/7485 | 0.0134% |
| VideoFeedback annotated | 500 | 7 | 14/7100 | 0.1972% |
| VideoFeedback real | 500 | 0 | 0/7500 | 0.0000% |
| GenVideo annotated | 500 | 0 | 0/6380 | 0.0000% |
| GenVideo real | 500 | 19 | 100/7500 | 1.3333% |

这不是像素级重复帧证明，但足以否定“没有重复内容”的假设。阶段 1 前应对冻结
manifest 的实际解码帧做精确像素 hash 和感知 hash，并同时报告删除前/后的结果。

## 5. 阶段 1 公平协议

先生成唯一 manifest，至少包含：

```text
dataset, subset, source_model, video_id, video_path,
source_fps, target_fps, frame_indices, frame_timestamps,
window_duration, calibration_split, evaluation_split,
global_cache_path, patch_cache_path, duplicate_frame_count, missing_reason
```

所有 B0-B8 必须从同一 manifest 左连接，不允许每个方法自行 drop：

1. 同一视频 ID 和每生成器相同 real/fake 数量；
2. 同一个 DINOv3 embedding cache 与相同 CLS/patch token 版本；
3. 同一窗口、目标 FPS、frame indices 和实际 timestamps；
4. 同一 calibration split，且 calibration real 不进入 evaluation real；
5. 同一缺失规则：短视频、cache miss、重复帧处理都写入 `missing_reason`；
6. 每个配置保存逐视频分数与组件分数，不只保存宏平均；
7. AUC/AP 使用 `src/metrics.py` 的 pairwise-balanced、fake-positive 口径；
8. paired bootstrap 在同一视频重采样上计算 delta，并报告逐生成器 win rate。

结果必须分成两类：

- `operator-controlled`：统一 DINOv3、统一 manifest，只比较 STALL/D3 算子；
- `official-baseline`：XCLIP-B/16、官方 crop/resize/read-8-or-16 协议，单独报告，
  不与 DINOv3 内部消融混为同一配置。

## 6. 阶段 1 前的阻塞项和决策

| 优先级 | 问题 | 必须动作 |
|---|---|---|
| P0 | operator D3 `ddof=0`，exact D3 `ddof=1` | 冻结一个定义；建议同时输出 population/sample，但主表固定一种 |
| P0 | raw D3 单侧 percentile | 实现 raw、two-sided、motion-conditioned、time-normalized 四个独立列 |
| P0 | global scalar acceleration 与 local vector D2 不兼容 | residual 前明确统一为 vector residual 或 scalar magnitude residual |
| P0 | 现有结果协议混杂 | 建立唯一 manifest 后重跑 B0-B8，不能拼接旧结果 |
| P1 | GenVideo real 有重复 embedding | 增加像素/感知 hash，报告过滤前后 D3 |
| P1 | 实际 `delta_t` 非均匀 | 从 timestamps 实现逐步时间归一化，不用名义 FPS 常数替代 |
| P1 | cache prefix fallback 会静默改窗口 | 改为严格 cache schema 验证，错误时记录 missing |
| P1 | `compute_windows()` 每个视频重置相同 seed | 改为由稳定 video ID 派生 seed，manifest 固化窗口 |
| P1 | global 用 VATEX 校准，patch 使用 dataset real 校准 | 每个表明确校准源；新增统一 real-only calibration 对照 |

阶段 0 判定：代码中没有发现需要恢复 motion-hard/soft、高阶 D3、多 lag 或简单
gate/cap 的实现错误。下一步应先修复上述协议项并建立 B0-B8 manifest/runner，再做
可信基线；不应直接启动大规模实验。
