# Alpha-STALLED 创新验证与最小重构计划

> 版本：2026-09-02  
> 分支：`explore/local-dynamics-likelihood`  
> 核心约束：冻结 DINO、不训练分类器、不用 fake 拟合、不按生成器调参、calibration real 与 evaluation real 互斥。

## 0. 一句话研究命题

在生成视频检测中，我们将检验：**真实视频的局部表示动态是否具有依赖运动状态与对应置信度的条件结构，而生成视频是否稳定偏离该结构；这一偏离能否只用真实视频校准，在严格 shortcut control 和跨域条件下检测。**

当前证据只支持 same-grid Local D2 是 Global STALL 的有效补充；它尚不支持 correspondence、trajectory geometry 或 conditional likelihood 已经成立。

## 1. Current Repository

### 1.1 真实代码框架

```text
scripts/run_alpha_stall.sh
  -> scripts/run_experiment.py
  -> src/runner.py
  -> src/pipeline.py
       |- data/manifest.py + data/sampling.py
       |- data/packed_cache.py + data/cache_contract.py
       |- branches/global_branch.py
       |- branches/local_branch.py
       |- math_utils.py
       `- evaluation/{metrics,tables,bootstrap}.py
  -> results/runs/<run_name>/
```

当前缓存已经提供 `[T,1024]` Global token 和 `[T,196,1024]` Patch token；后续局部 correspondence/dynamics 不需再跑 DINO，也不需扩大 630 GB 缓存。

### 1.2 当前正式方法

```text
Global window:
  G_k = 0.5 * VATEX_CDF(max_t LL(g_t))
      + 0.5 * VATEX_CDF(min_t LL(normalize(g_{t+1}-g_t)))

Local window:
  a_{t,i} = normalize(P_{t+2,i} - 2P_{t+1,i} + P_{t,i})
  L_k = target-real-CDF(mean_{t,i} LL(a_{t,i}))

Video:
  分支窗口均值 -> effective-K target-real CDF
  S = 0.60 * G + 0.40 * L
```

正式 run `alpha_stall_full_d2_k3_no_spatial_refit` 在 21,421 条可用视频上得到 Macro-3 AUC/AP `0.8741/0.8729`。Local Spatial 已被否定，不再进入候选主线。

### 1.3 当前资源约束

| 项目 | 状态 | 决策 |
|---|---|---|
| GPU 0 | 被其他任务占用 | 不使用 |
| GPU 1 | RTX 5090，约 32 GB 空闲 | Stage 1 固定 `cuda:1` |
| `/data` 剩余 | 约 69 GB | 禁止新建大规模 feature cache |
| 现有 cache | 630 GB，22,820 条 packed entries | 只读复用 |
| 结果空间 | 当前 232 MB | 新实验只保存参数、CSV、日志和诊断摘要 |

## 2. Paper vs Code

核心一致，但有六项必须修复的可复现性缺口：

1. `sampling.window_seconds/fps/frames_per_window/strategy` 没有完整驱动评分，代码实际硬编码 16 帧与 uniform。
2. Local real-only 拟合的 `mu/W` 未保存，历史 run 无法逐位重放。
3. packed cache 读取不重新验证源视频 SHA，cache key 不含 dataset。
4. run manifest 缺 manifest hash、packed index hash、local parameter hash 和 calibration IDs。
5. pairwise table 依赖隐式 DataFrame 顺序，未保存显式 real/fake 配对清单。
6. 核心 runner 缺 0.1% FPR、FPR@95TPR、AP CI、seed/split variance 和效率指标。

科学表述还需限定：Global 窗口组件来自官方 VATEX，但 Global 视频分数仍经过目标域 real effective-K CDF；当前方法应称 `target-real-calibrated training-free`，而不是 calibration-free zero-shot。

## 3. Latest Related Work

### 3.1 必须承认的先验

| Work | 已占据的核心概念 | 对当前项目的约束 |
|---|---|---|
| D3, ICCV 2025 | 全局 embedding 标量距离序列的二阶变化 | 不可声称首次 second-order detection |
| ReStraV, NeurIPS 2025 | 表示轨迹 step distance、turning angle、21-D descriptor | curvature/straightening 不是新概念；其方法使用 fake-supervised MLP |
| Over-Coherence, WACV 2026 | cosine transition sequence 的过度平滑/变化 criterion | 不能只以 roughness/smoothness 作为新故事 |
| STALL, CVPR 2026 | real-only whitening、Global Spatial/T1 likelihood、percentile fusion | Global 是继承，不是贡献 |
| SPLIT, ECCV 2026 | same-grid patch TTR、局部 motion gradient LSMI、部分 fake、0.1% FPR、cross-real threshold | 对 Patch roughness/Local motion/低 FPR 构成最直接碰撞 |
| MotionPhys, arXiv 2026 | sparse optical-flow trajectory 的多尺度物理几何 | “trajectory physics/geometry”叙事已拥挤 |
| VidAudit 与 motion-bias audit, 2026 | FPS、时长、编码、重复帧、motion magnitude shortcut | 新方法必须通过控制实验，而不能只报 AUC/AP |

### 3.2 研究机会

现有工作分别覆盖了手工二阶统计、监督轨迹 descriptor、patch roughness、光流物理轨迹和无条件 Gaussian likelihood。仍有待验证的交叉区域是：

```text
frozen patch correspondence
  + real-only conditional dynamics density
  + correspondence uncertainty
  + sample-efficient/cross-real calibration
```

这不是已证明的文献空白，只是截至当前检索未发现同构公开方案。最终 novelty 必须由代码级复现与实验结果共同确认。

## 4. Novelty Collision

### 4.1 标红删除

- Local Spatial likelihood：内部负结果，外部又与 SPLIT LSMI 叙事拥挤。
- same-grid Patch D2 本身：可保留为 baseline，不再作为论文标题级创新。
- curvature、speed、path/chord 单项：分别与 ReStraV、Over-Coherence、SPLIT、MotionPhys 重叠。
- multi-scale patch、partial fake、0.1% FPR：可作为实验，不可作为首次贡献。
- 简单 bottom-k：既有内部失败，又与局部部分伪造检测目标接近。

### 4.2 可以防守但需要强证据

- correspondence-aware D2 likelihood；
- confidence-weighted real likelihood field；
- motion-state-conditioned real dynamics likelihood；
- 在 25/50 real 条件下仍稳定的 robust density；
- 不依赖 fake 调权的 calibrated evidence fusion。

## 5. Candidate Innovations

### C. Correspondence-Aware Local Dynamics

**Hypothesis**：same-grid D2 混入物体/相机运动造成的对应误差；局部匹配后，真实动态分布更集中，fake 偏离更稳定。

对 patch `i=(x,y)` 与下一帧邻域 `N_r(i)`：

```text
C_ij = cosine(P[t,i], P[t+1,j]) - lambda_spatial * d(i,j)
A_ij = softmax(C_ij/tau)
P_hat[t+1,i] = sum_j A_ij P[t+1,j]
v[t,i] = P_hat[t+1,i] - P[t,i]
a[t,i] = normalize(v[t+1,i] - v[t,i])
```

confidence：

```text
rho[t,i] = 1 - H(A[t,i]) / log(|N_r(i)|)
q = sum rho[t,i] * LL(a[t,i]) / sum rho[t,i]
```

| 项目 | 计划 |
|---|---|
| Code integration | 新增 `src/correspondence/`，由 `pipeline` 的唯一 evidence factory 调用 |
| 固定参数 | 第一轮 `r=1`、`tau=0.07`、简单归一化空间惩罚；来源明确记录 |
| Expected benefit | 消除网格漂移噪声，使 Local likelihood 对真实快速运动更稳健 |
| Risk | DINO patch 语义过粗；soft match 可能只做平滑；重复纹理导致错误对应 |
| Overlap | MotionPhys 有光流对应；SPLIT 是 same-grid。差异在 frozen-token match + real likelihood |
| Compute | 不跑 backbone；3x3 局部匹配约为 same-grid 的常数倍，32 GB GPU 可承受 |
| Minimum experiment | C0-C3，固定三个数据集与所有非 correspondence 参数 |
| Continue gate | C3-C0 Macro AUC 或 AP >=0.005，且至少 2/3 数据集提升 |

### T. Trajectory Geometry Likelihood

**Hypothesis**：D2 只描述加速度方向，低维 speed/turning/speed-ratio 可能更稳健且更省 real calibration。

```text
speed      = ||v_t||
curvature  = 1 - cos(v_{t-1},v_t)
speedRatio = log((||v_t||+eps)/(||v_{t-1}||+eps))
pathChord  = (||x1-x0||+||x2-x1||)/(||x2-x0||+eps)-1
```

| 项目 | 计划 |
|---|---|
| Code integration | `src/dynamics/trajectory_geometry.py`，输出低维 descriptor |
| Expected benefit | 减少 1024-D covariance 小样本问题，增强可解释性 |
| Risk | 与 ReStraV/SPLIT/MotionPhys novelty 高碰撞；可能丢失语义方向信息 |
| Overlap | 高；不能单独作为主创新 |
| Compute | 低 |
| Minimum experiment | T0-T5；必须以 Stage 1 最佳 correspondence 为基础 |
| Continue gate | 低维模型在 25/50 real 或 cross-real 明显优于 D2，而不只在 200-real AUC 持平 |

### D. Conditional Dynamics Likelihood

**Hypothesis**：相同加速度在慢运动与高速运动中含义不同；无条件 `p(a)` 将多个真实 motion regimes 混在一起。

最低成本模型：只用 real calibration 的 speed quantile 划分三个固定 bin：

```text
b = bin_real_quantile(speed; 0, 1/3, 2/3, 1)
score(a,s) = log p_real(a | b(s))
```

后续可扩展为 `p(curvature|speed)` 或低维 `p(phi|motion regime)`，但第一阶段不使用 neural density。

| 项目 | 计划 |
|---|---|
| Code integration | `src/likelihood/conditional.py` 或现有 `math_utils` 的小模块；bin 和参数全落盘 |
| Expected benefit | 对 fast motion/camera shake 降低 false positive；提高跨域和少样本稳定性 |
| Risk | 每 bin 样本减少；speed 本身受 FPS/内容偏差影响；可能过拟合 target real |
| Overlap | ReStraV 联合 descriptor、MotionPhys motion geometry；未发现相同 real-only conditional density |
| Compute | 中低；主要是多组小型统计 |
| Minimum experiment | D0-D3 + motion-stratified metrics |
| Continue gate | conditional-unconditional >=0.005，至少 2/3 数据集同向，并改善 fast-motion real FPR |

### S. Robust Real Likelihood

**Hypothesis**：1024-D empirical covariance 对 200 real 和 domain shift 不稳定；shrinkage 或低维 robust density 可降低真实样本需求。

| Candidate | 规则 | 参数选择 |
|---|---|---|
| Empirical | 当前 PCA whitening | baseline |
| OAS | 无监督 shrinkage | closed-form real-only |
| Ledoit-Wolf | 无监督 shrinkage | closed-form real-only |
| Student-t/robust Mahalanobis | 低维 descriptor 重尾模型 | 只在 real calibration 拟合 |
| kNN | 到 real descriptor 的 kNN distance | 预设 k=5/10，不以 fake 选择 |

其贡献定位应是“conditional dynamics 的可靠统计实现”，不是新 covariance estimator。

### A/F/W/U. 次级方向

| 方向 | 最小合法实验 | 当前优先级 |
|---|---|---:|
| Tail/CVaR | mean vs 预注册 top anomaly 10%，并在 calibration real 同规则 | 低 |
| Conformal fusion | weighted vs equal/Cauchy，依赖性不声称严格独立 | 中 |
| Adaptive windows | calibration 与 test 同一 Top-K selection | 低 |
| Universal real bank | 完整 calibration-domain x eval-domain transfer matrix | 高，但在方法候选收敛后执行 |

## 6. Recommended Mainline

### Top 1：Conditional Correspondence-Aware Local Dynamics Likelihood

| 维度 | 评分（5分） | 理由 |
|---|---:|---|
| Scientific novelty | 4 | correspondence + conditional real density 的组合比单 D2 更清晰 |
| Feasibility | 4 | 直接复用现有 patch cache，无需 backbone |
| Expected performance | 3 | 理论合理，但 matching 也可能破坏 DINO 轨迹 |
| Reviewer defensibility | 4 | 可用 C0-C3 与 D0-D3 严格拆分因果来源 |
| Implementation cost | 3 | 局部匹配与 batched likelihood 需谨慎实现 |

推荐主标题方向仅在两个 gate 都通过后启用：

```text
Training-Free Generated Video Detection via
Correspondence-Aware Conditional Local Dynamics Likelihoods
```

### Top 2：Sample-Efficient Robust Real Dynamics Modeling

| 维度 | 评分（5分） | 理由 |
|---|---:|---|
| Scientific novelty | 3 | 统计工具不新，但针对真实动力学样本效率的问题明确 |
| Feasibility | 5 | 低成本复用已提取 evidence |
| Expected performance | 3 | 可能主要改善 25/50 real 与跨域，而非 200-real AUC |
| Reviewer defensibility | 5 | 直接回答 target-real dependence 和 deployment 成本 |
| Implementation cost | 4 | sklearn/低维统计即可 |

### Top 3：Shortcut-Resistant Real-Only Calibration and Evaluation

| 维度 | 评分（5分） | 理由 |
|---|---:|---|
| Scientific novelty | 3 | 更偏协议贡献，但 2026 文献显示这是领域核心缺口 |
| Feasibility | 3 | 需统一转码/motion matching/新 confirmation 数据 |
| Expected performance | 2 | 可能降低表面 AUC，但提高可信度 |
| Reviewer defensibility | 5 | 可正面回应 VidAudit 与 motion-bias 批评 |
| Implementation cost | 3 | 数据处理成本高于算法成本 |

不推荐把 Trajectory Geometry 单独列入 Top 3，因为与 ReStraV、SPLIT、MotionPhys 的碰撞过强。它只作为 Top 1/2 内部可替换 descriptor。

## 7. Experiment Matrix

### Stage 1：Correspondence

| ID | Correspondence | Confidence | 其他设置 |
|---|---|---|---|
| C0 | same-grid | 无 | 当前 D2 baseline |
| C1 | hard local 3x3 | 无 | 只改 correspondence |
| C2 | soft local 3x3 | 无 | `tau=0.07`，固定 spatial penalty |
| C3 | soft local 3x3 | aggregation weighting | entropy confidence |

先跑 calibration + 小型 development subset smoke；数值稳定后跑三开发集。保存 AUC/AP、per-generator、0.1%/1% FPR、runtime、peak VRAM。

### Stage 2：Trajectory Geometry

`T0 D2 / T1 curvature / T2 speed ratio / T3 path-chord / T4 D2+curvature / T5 low-D descriptor`。Stage 1 若通过则使用最佳 correspondence；若否定则回到 C0 same-grid，避免把无效 matching 带入 H2。

实际 Stage 1 否定 correspondence 后，Stage 2 固定回到 C0 same-grid，仍执行 T1-T5
以独立检验 H2，但采用以下预注册决策：

- 主线继续门槛：候选相对 T0 的 Macro AUC 或 AP 至少 `+0.005`，且同一指标
  至少 2/3 数据集提升；
- T5 四维 descriptor 若相对 T0 的 Macro AUC/AP 均不低于 `-0.002`，只保留到
  few-real-shot/cross-real 稳定性验证，不能仅凭持平声称创新；
- T3 path/chord 与 SPLIT TTR 高度重叠，无论性能如何都只作诊断，不单独进入主线；
- 若全部几何候选低于 T0，则 H2 被否定，Stage 3 直接以 T0 D2 检验 conditional
  likelihood，不继续增加几何组合。

### Stage 3：Conditional Dynamics

`D0 unconditional D2 / D1 speed-conditioned D2 / D2 unconditional geometry / D3 speed-conditioned geometry`。bin 只由 calibration real quantile 决定。

### Stage 4：Statistical Model

`S0 empirical / S1 Ledoit-Wolf / S2 OAS / S3 low-D Student-t or robust Mahalanobis / S4 kNN`。重点报告 n_real=25/50/100/200 曲线。

### Stage 5：Aggregation/Fusion

`A0 mean / A1 fixed tail-CVaR`；`F0 current weights / F1 equal calibrated / F2 Cauchy-conformal`。不根据每数据集 fake 选择。

### Stage 6：Cross-Domain Calibration

构造 calibration domain x evaluation domain 矩阵，并报告目标 FPR 的实际漂移。Universal bank 在此阶段冻结。

### Stage 7：New Dataset

只对最终 candidate 跑 untouched confirmation。优先级：CoCoVideo-26K 或可获取的 ViF-Bench small > AIGVDBench 预定义小规模 cell > FakeParts（仅 localization/tail 成立时）。

## 8. Code Refactor Plan

### 8.1 保留

- `src/runner.py`：唯一运行与产物入口。
- `src/pipeline.py`：保留数据级编排、校准和评价，不再直接实现所有 evidence 数学。
- `src/data/`：manifest、packed cache、strict contract。
- `src/features.py`：冻结 DINO 提取与 cache contract 来源。
- `src/branches/global_branch.py`：官方 STALL Global 兼容实现。
- `src/evaluation/`：扩展而非复制。

### 8.2 最小新增

```text
src/
  correspondence/
    __init__.py
    local.py          # same-grid / hard / soft / confidence
  dynamics/
    __init__.py
    local.py          # D1 / D2，后续可加 geometry
```

第一阶段不创建 likelihood/aggregation/fusion 多层目录；只有出现第二个真实实现并显著降低复杂度时再拆分。避免为满足理想目录树而重写稳定主干。

### 8.3 抽象点

新增一个配置驱动 evidence factory：

```text
build_local_dynamics(patch, correspondence_config, dynamics_config)
  -> features [B,T',P,D]
  -> optional weights [B,T',P]
  -> diagnostics
```

`pipeline::_fit_local_parameters` 与 `_score_windows` 必须调用同一个 factory，确保 calibration/test 完全同路径。任何 adaptive selection/confidence 也必须同步作用于 calibration real。

### 8.4 配置

继续使用唯一 `configs/benchmark.yaml`，新增少量预留字段：

```yaml
method:
  local:
    correspondence:
      type: same_grid
      radius: 1
      temperature: 0.07
      spatial_penalty: 0.0
      confidence: none
```

C0-C3 由一个 `scripts/run_correspondence.sh` 的实验矩阵/参数控制，不创建四份 YAML，也不创建四套 Python 入口。

### 8.5 归档/删除候选

- `src/calibration_fit.py` 的无效 matching 参数必须删除或显式标记 historical；当前它们被接收但未使用。
- `src/aggregation.py`、`src/calibration.py`、`src/scoring.py` 不在主路径；在 Stage 1 测试通过后统一迁入 `historical/` 或删除。
- 历史 `run_*matrix.sh` 暂不在同一提交中删除，先保证结果可追溯；新增实验只保留一个 Stage 1 launcher。

### 8.6 产物与 cache

每个新 run 额外保存：

- `fitted_local_params.npz` 与 SHA256；
- `calibration_ids.csv`；
- `pairwise_pairs.csv`；
- manifest、packed index、cache contract hash；
- correspondence 参数与来源；
- runtime/peak VRAM；
- non-finite 和 zero-norm 计数。

不修改现有 packed feature cache，不创建 per-video 新 `.pt`。中间 correspondence/evidence 默认流式计算；只有复用次数证明值得时，才保存远小于 patch token 的低维 evidence cache。

## 9. 决策规则

1. C3 不过 correspondence gate：停止 hard/soft/OT 方向，C0 保留。
2. Geometry 仅提高 200-real AUC、但不改善 few-real/cross-real：不作为主线。
3. Conditional gain `<0.002` 或只在一个数据集出现：停止 conditional 复杂化。
4. 任何方法在 motion-matched/统一重编码后崩溃：不得作为生成痕迹 claim。
5. 最终候选必须在 untouched confirmation benchmark 上一次性评测；失败则按失败结果报告，不回到该 benchmark 调参。

这套规则的目标不是保证新方法成立，而是尽早、低成本地淘汰不能经受文献和实验双重检验的故事。

## 10. Stage 1 已执行决策（2026-09-02）

C1 hard、C2 soft 和 C3 soft+confidence 相对 C0 的 Macro AUC/AP 均下降；C3 为
`-0.0029/-0.0046`，且没有在至少两个数据集提升同一指标。1,000 次论文配对口径
bootstrap 的三个 Macro AUC/AP 区间也均完全低于零。

因此 H1 被否定：当前 same-grid correspondence 不是可由简单局部 hard/soft matching
修复的主要瓶颈。停止 radius=2、OT 和 learned matcher 扩展，Stage 2 固定使用 C0
same-grid。完整结果见 `reports/stage1_correspondence_results.md`。
