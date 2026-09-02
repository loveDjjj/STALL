# Alpha-STALLED 当前代码执行流程审计

> 审计日期：2026-09-02  
> 审计分支：`explore/local-dynamics-likelihood`  
> 基线提交：`718f780`  
> 审计原则：以下流程来自实际调用关系和当前结果产物，不根据 README 或论文图反推代码。

## 1. 审计结论

当前唯一主执行链为：

```text
scripts/run_alpha_stall.sh
  -> scripts/run_experiment.py
  -> src.runner.run
  -> src.pipeline.run_from_cache
       -> manifest 读取与互斥检查
       -> packed patch cache 读取与窗口重建
       -> 官方 VATEX Global 参数加载
       -> 目标域真实视频 Local 参数拟合
       -> calibration/evaluation 窗口评分
       -> 窗口 CDF、视频均值、effective-K 视频 CDF
       -> 0.60 Global + 0.40 Local D2
  -> src.evaluation.tables / metrics
  -> results/runs/<run_name>/
```

当前正式方法不是旧说明中的四分支方法，而是：

```text
Global = 0.5 * Global Spatial + 0.5 * Global T1
Local  = Local D2-only
Final  = 0.6 * Global + 0.4 * Local
```

Local Spatial 已由受控实验否定并关闭。当前 Local D2 使用固定网格位置，不包含显式 patch 对应、光流或区域池化。

## 2. 数据与资源边界

| 资源 | 当前事实 | 审计判断 |
|---|---|---|
| 开发数据集 | ComGenVid、VideoFeedback、GenVideo | 三者同时承担方法开发和报告，不应称为完全 untouched test |
| 校准集 | 每个数据集 200 个真实视频，seed=17 | 只使用真实视频，且与 evaluation 路径互斥 |
| 评测集 | 4,298 + 3,500 + 13,623 = 21,421 条可用视频 | GenVideo 另排除 2,565 条不足 16 个降采样帧的视频 |
| Global 统计 | 官方 `precomputed/stall_params_vatex_dino_v3.npz` | 来自 VATEX，不使用目标数据集重新拟合 Global 窗口统计 |
| Local 统计 | 每次运行从目标域 calibration real 重拟合 | 不使用 fake；当前拟合参数未独立落盘 |
| 特征缓存 | `cache/patch_embeddings_k3_2s_8fps`，约 630 GB | 已含 Global 与 `14x14` Patch token，可直接支持 C0-C3 |
| GPU | GPU 0 当前被其他任务占用；GPU 1 空闲约 32 GB | 后续低成本实验应固定使用 `cuda:1` |
| 剩余磁盘 | `/data` 约 69 GB | 禁止重建同规模特征缓存；新增实验只写小型配置、日志和分数 |

## 3. 真实执行流程

### 3.1 启动、配置与结果目录

| 步骤 | 文件与符号 | 输入 | 输出 | 默认/来源 | fake/eval 使用 | 泄漏风险 |
|---|---|---|---|---|---|---|
| Shell 启动 | `scripts/run_alpha_stall.sh` | 命令行参数 | `run_experiment.py` 参数 | 基础配置 `configs/benchmark.yaml`，实验差异用 `--set` | 否 | 低；前提是 run 名唯一 |
| 配置读取 | `src/config.py::load_config/apply_overrides/validate_config` | YAML + 点分覆盖 | resolved config | `method.name=alpha_stall` | 否 | 中；当前未校验 2 秒/8 FPS/16 帧三者一致，也未校验权重和为 1 |
| 运行编排 | `src/runner.py::run` | config、run_name | 结果目录与日志 | `results/runs/<run_name>` | 只在最终评价读取标签 | 低；已有 config hash 和 Git commit |
| 主流水线 | `src/pipeline.py::run_from_cache` | repository_root、config | window/video DataFrame、pipeline metadata | 唯一主路径 | 是，评价阶段 | 见各步骤 |

`runner.run` 在评分前写入 `command.txt`、`resolved_config.yaml`、`run_manifest.json` 和 `progress.json`，并将 stdout/stderr 同时写入 `logs/run.log`。异常会生成 `failure.json`；外部信号还会生成 `termination.json`。

### 3.2 Manifest、FPS 与视频划分

| 步骤 | 文件与符号 | 输入 shape/类型 | 输出 shape/类型 | 默认参数 | fake/eval 使用 | 泄漏风险 |
|---|---|---|---|---|---|---|
| 视频元数据 | `scripts/build_manifest.py::get_video_metadata` | 单视频路径 | fps、duration、估算帧数 | `ffprobe` | 构建 manifest 时读取全部视频元信息 | 中；估算 `num_frames=round(fps*duration)` 不是逐帧真值 |
| 8 FPS 索引 | `scripts/build_manifest.py::downsample_frames` | `num_frames,current_fps,target_fps` | 原视频帧索引列表 `[L]` | `target_fps=8` | calibration/eval 同规则 | 中；低于 8 FPS 的视频被排除，不上采样 |
| split 读取 | `src/data/manifest.py::load_manifest` | CSV | DataFrame | 开发/外部 manifest 根目录来自 config | 同时读取 calibration/evaluation 清单 | 低 |
| calibration 抽样 | `src/pipeline.py::_choose_calibration` | calibration real DataFrame | 200 条 real | seed 经 dataset 名哈希派生 | 不读取 fake | 低 |
| 互斥检查 | `src/pipeline.py::_ensure_disjoint` | 两个 DataFrame | 无 | 无条件执行 | 读取路径字段 | 低；即使 YAML 写 false，代码仍强制互斥 |
| 短视频策略 | `src/pipeline.py::_apply_short_video_policy` | DataFrame | 可用行 + 排除数 | 至少 16 个降采样帧 | calibration/eval 同规则 | 中；不同类别/生成器的排除率可能形成选择偏差 |

`calibration.disjoint_from_evaluation` 当前只是声明字段；真实代码始终调用 `_ensure_disjoint`。这是保守行为，但配置与行为之间并非一一控制关系。

### 3.3 多窗口采样

`src/data/sampling.py::uniform_windows` 对长度为 `L` 的 8 FPS 索引序列构造窗口：

```text
window_frames = 16
max_start = L - 16
starts = round(linspace(0, max_start, K))
```

随后去除完全重复的窗口。因此请求 `K=3` 不保证实际得到 3 个不同窗口，短视频会得到 effective-K=1 或 2。

| 属性 | 当前实现 |
|---|---|
| 每窗时长 | 代码固定 16 帧；在 manifest 为 8 FPS 时等价于 2 秒 |
| K=3 位置 | 首、约中、尾三个确定性起点，使用 NumPy `rint` |
| 重叠 | 允许；只删除完全相同的窗口 |
| 短视频 | `<16` 帧排除；`>=16` 帧但起点重复时 effective-K 下降 |
| 配置控制 | 实际只直接读取 `sampling.num_windows`；`window_seconds/fps/frames_per_window/strategy` 未贯穿到窗口函数 |

这一点是当前最重要的配置真实性问题之一：修改 YAML 中的 FPS、窗口秒数或每窗帧数，不会自动改变主评分代码。

### 3.4 DINOv3 与特征缓存

缓存由 `src/features.py::AlphaStallFeatureExtractor` 生成。单帧输入经过：

```text
BGR/RGB frame
 -> resize 224x224
 -> ImageNet normalization
 -> frozen DINOv3 ViT-L/16 forward_features
 -> x_norm_clstoken      [B, 1024]
 -> x_norm_patchtokens   [B, 196, 1024]
```

| 特征 | 单窗口输入 | 单窗口输出 | 说明 |
|---|---:|---:|---|
| Global token | 16 帧 | `[16,1024]` | 最终归一化 CLS token，DINO head 为 Identity |
| Patch token | 16 帧 | `[16,196,1024]` | 最终归一化 patch token，网格 `14x14` |
| CLS/register 是否混入 Patch | - | 否 | Patch 直接来自 `x_norm_patchtokens` |
| layer | final block output | 1024 维 | cache contract 记录 output layer 23 |

缓存根 contract 绑定 encoder repo commit、checkpoint SHA256、预处理、层、dtype、特征描述和帧选择协议。缓存内容与方法证据构造解耦，因此 hard/soft correspondence 可以直接复用 Patch token。

但是，缓存审计仍有三个缺口：

1. cache key 为 `subset/source_model/stem`，不包含 dataset 名；依赖打包前碰撞审计。
2. unpacked strict cache 会校验源视频 SHA；packed cache 读取仅核对 frame indices 与 payload descriptor，不重新核对源 SHA。
3. run manifest 记录根 contract hash，但不记录 manifest 内容 hash、packed index hash和逐条实际缓存键集合。

### 3.5 Global 分支

参数由 `src/branches/global_branch.py::load_official_stall_parameters` 从官方 VATEX NPZ 读取：

| 组件 | 输入 | evidence | 窗口聚合 | CDF |
|---|---:|---|---|---|
| Global Spatial | `[B,16,1024]` | 原始 Global token 的高斯 log-likelihood | 时间维 `max` | 官方 VATEX spatial reference |
| Global T1 | `[B,15,1024]` | `normalize(g[t+1]-g[t])` 的高斯 log-likelihood | 时间维 `min` | 官方 VATEX temporal reference |

零差分在 Global T1 中被标记为无效并赋 `+inf`，使其不会成为 `min`；若整窗全为零差分，允许该窗口 percentile 映射为 1。Global 窗口分数为：

```text
G_k = 0.5 * CDF_global_spatial + 0.5 * CDF_global_t1
```

### 3.6 Local D2 分支

当前实现位于 `src/branches/local_branch.py` 和 `src/math_utils.py`。对输入：

```text
P in R^[B,16,196,1024]
```

计算：

```text
A[t,i] = P[t+2,i] - 2*P[t+1,i] + P[t,i]
A_hat[t,i] = A[t,i] / max(||A[t,i]||_2, eps)
```

输出 shape 为 `[B,14,196,1024]`。这不是在归一化 D1 上再次差分，也不是对标量帧间距离做二阶差分。它是 raw patch feature vector 的直接二阶有限差分，随后沿 1024 维特征轴 L2 归一化。

Local 的默认拟合与评分为：

| 步骤 | 文件与符号 | 输入 | 输出 | 数据来源 |
|---|---|---|---|---|
| 样本收集 | `pipeline::_fit_local_parameters` | calibration real 窗口 D2 `[14,196,1024]` | 最多 300,000 行 `[M,1024]` | 仅目标域 real |
| 抽样 | `pipeline::_reservoir_add` | 全部时空 patch 行 | 固定上限 reservoir | seed=17 |
| 白化 | `math_utils::WhiteningTransform` | `[M,1024]` | `mu [1024]`, `W [1024,r]` | 经验 covariance；`r<=min(M-1,1024)` |
| 似然 | `math_utils::score_gaussian_aggregate_float64` | D2 + `mu/W` | 每窗 raw scalar | float64 score |
| 局部聚合 | 同上 | `[14,196]` likelihood field | mean scalar | 固定 mean |
| 窗口校准 | `pipeline::_calibrate_component` | calibration/eval raw | percentile | calibration real windows |

Local D2 的零向量经 `torch.nn.functional.normalize(..., eps=1e-12)` 后保持零向量，当前没有像 Global T1 一样的 invalid mask。因此重复 patch 可能在白化后被当作一个合法观测，而不是“无时序信息”。

### 3.7 白化、协方差与似然

`src/math_utils.py::WhiteningTransform` 当前默认经验协方差：

```text
mu = mean(X)
Sigma = torch.cov((X-mu)^T)
Sigma = V diag(lambda) V^T
r_max = min(M-1, D)
保留 lambda > 0
W = V diag(1/sqrt(lambda + 1e-5))
y = (x-mu) W
log p(y) = -0.5 * (r*log(2*pi) + ||y||^2)
```

代码也提供 OAS，但默认主实验为 empirical。Local 拟合参数只在内存中存在，最终 run 只记录参数名称，不保存 `mu/W`、reservoir 样本身份或参数 hash。这会妨碍逐位复现和后续复用，应在 Stage 1 最小重构中修复。

### 3.8 窗口校准、视频聚合与融合

`src/pipeline.py::_calibrate_and_aggregate` 的实际顺序为：

```text
1. 对 Local raw window likelihood 建立 calibration-real 窗口 CDF
2. Global 使用官方 VATEX component CDF
3. 窗口内形成 G_k 与 L_k
4. 每个视频对各分支窗口分数取 mean
5. 按 effective-K=1/2/3，从 calibration real 构造对应的视频均值参考分布
6. 对 Global/Local 视频均值分别做目标域 real CDF
7. Final = 0.60 * Global_video_percentile + 0.40 * Local_video_percentile
```

因此 Global 有两层参考：窗口组件使用官方 VATEX CDF，视频层又使用目标域真实视频按 effective-K 重校准。论文必须明确这一点，否则“Global 完全固定为官方 STALL”会让读者误以为目标域 real 不影响 Global 最终分数。

### 3.9 指标与配对协议

| 输出 | 文件与函数 | 统计口径 |
|---|---|---|
| pooled dataset | `evaluation.tables::build_metric_tables` | 每数据集全部 real 与 fake |
| per-generator | 同上 | 每个生成器 fake 与该数据集全部 real |
| 论文 pairwise | `build_pairwise_metric_table` | 每生成器与确定性选择的等量 real 配对，再做数据集内宏平均 |
| Macro-3 | 同上 | 三个数据集的 pairwise dataset macro 等权平均 |
| AUC/AP | `evaluation.metrics::binary_metrics` | 分数越高越 real；AP 正类为 real |
| 低 FPR | 同上 | Fake TPR @ Real FPR 1% |
| bootstrap | `paired_bootstrap` | 当前只对 AUC 差异做视频行重采样 |

当前尚未在核心 runner 中输出 Fake Recall@0.1% FPR、FPR@95TPR、AP 的置信区间和 calibration split 方差。论文 pairwise 真实样本选择也未保存成显式 pair manifest，而是依赖 DataFrame 顺序、`fake.head()` 和 seed。

## 4. 训练自由与数据泄漏审计

| 问题 | 结论 | 证据边界 |
|---|---|---|
| 是否训练 DINO | 否 | 冻结 DINOv3，只读缓存 |
| 是否训练分类器 | 否 | 高斯白化与经验 CDF，不含真假监督优化 |
| 是否用 fake 拟合参数/CDF | 否 | `_fit_local_parameters` 和校准参考只接收 calibration real |
| 是否用 evaluation real 拟合 | 否 | manifest 路径互斥检查后只从 calibration 选择 |
| 是否用 generator identity 评分 | 否 | `source_model` 只用于分组评价和缓存路径 |
| 是否严格 zero-shot | 需限定 | 对 fake 参数拟合是 zero-shot，但 Local 和视频 CDF 使用目标域 real，应称 target-real-calibrated |
| 是否完全未看开发 fake 调方法 | 否 | D1/D2、K、融合权重和 Spatial 删除均参考开发 benchmark 结果 |
| 是否存在测试集选参风险 | 有 | 若把三开发集同时称最终测试集，会形成 post-selection 偏差；需要新 benchmark 或预注册 confirmation cell |

## 5. 历史/旁路代码

以下模块不在当前 `runner -> pipeline` 主路径中：

- `src/aggregation.py`
- `src/calibration.py`
- `src/scoring.py`
- `src/calibration_fit.py`

其中 `src/calibration_fit.py::patch_temporal_delta` 暴露过 `match_radius/top_m/temperature/lambda_dist` 参数，但当前实现未使用这些参数，并未实现 hard/soft matching。继续保留会造成“已有 correspondence 实现”的错误印象。Stage 1 应先将其标记为历史接口或删除无效参数，而不是再复制一套平行流水线。

## 6. Stage 1 前必须修复的最小问题

1. 将 evidence 构造从 `_temporal_patch` 抽象为配置驱动的单一入口，支持 `same_grid/hard_local/soft_local`。
2. 保存每次 Local 拟合的 `mu/W`、证据配置、样本计数和内容 hash。
3. 将 16 帧、uniform 策略与 8 FPS 协议纳入强校验，拒绝“配置改了但代码没改”的运行。
4. 为 packed cache 补充 index hash 和源 manifest hash 到 run manifest；不重建特征。
5. 输出 0.1%/1% FPR 指标、峰值显存、纯 evidence 计算耗时。
6. 保存 calibration video IDs 和 pairwise real IDs，消除隐式顺序依赖。

## 7. 可复用性结论

当前主干可以作为 Stage 1 的稳定基线：数据拆分、严格缓存、官方 Global、real-only Local 拟合、effective-K 校准和结果落盘均已贯通。最合理的改造不是重写目录树，而是在现有 `pipeline.py` 周围新增小型 `correspondence/` 与 `dynamics/` 模块，并保持所有 C0-C3 共享同一评分、校准和评价路径。
