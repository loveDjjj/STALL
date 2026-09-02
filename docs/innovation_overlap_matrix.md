# Alpha-STALLED 创新重叠矩阵

> 判定日期：2026-09-02。  
> 颜色含义：<span style="color:#c62828"><b>高碰撞</b></span> 表示不能单独作为主创新；**中碰撞** 表示必须限定数学对象与训练协议；**低碰撞** 表示仍需实验支持，但现有文献中未发现同构方案。

## 1. 候选与已有工作的碰撞

| 候选想法 | D3 | ReStraV | Over-Coherence | STALL | SPLIT | MotionPhys | 判定 | 可保留的差异 |
|---|---|---|---|---|---|---|---|---|
| Global Spatial + Global T1 | 低 | 低 | 中 | 完全相同来源 | 低 | 低 | <span style="color:#c62828"><b>高碰撞</b></span> | 只能作为继承 baseline |
| Local same-grid D2 | 二阶 cue 高重叠 | 轨迹加速度中重叠 | 平滑变化中重叠 | 新于 STALL | patch temporal roughness 高重叠 | 轨迹动力学高重叠 | <span style="color:#c62828"><b>高碰撞</b></span> | 可作为 correspondence framework 的 C0 特例 |
| D2 向量 Gaussian likelihood | D3 无 likelihood | ReStraV 用监督 MLP | 无 density | 概率框架相近 | 无 Gaussian density | 细节待核实 | **中碰撞** | real-only local vector density 是组合差异，不足以单独支撑强创新 |
| Local Spatial likelihood | 低 | 低 | 低 | Spatial likelihood 概念相近 | LSMI 强局部空间 cue | 低 | <span style="color:#c62828"><b>高碰撞且内部否定</b></span> | 不保留；内部 factorial 已显示负贡献 |
| Region pooling / multi-scale patch | 低 | 全局展平 | 低 | 低 | 多 backbone/patch cue；局部尺度主题接近 | 多尺度轨迹 | <span style="color:#c62828"><b>高碰撞</b></span> | 仅作诊断，不作主线 |
| Bottom-k / tail aggregation | 低 | 统计 descriptor | 极值 criterion | min/max | partial fake 与局部聚合目标相同 | 可能局部异常 | **中高碰撞** | 若用 likelihood field + 预注册 CVaR 并做 localization，才可能成立 |
| Path/chord ratio | D2 物理叙事 | trajectory straightening | 平滑度 | 低 | TTR 数学上直接相关 | trajectory geometry | <span style="color:#c62828"><b>高碰撞</b></span> | 不能单独 claim novelty |
| Curvature/turning angle | 间接 | 直接核心 | 间接 | 低 | roughness 相关 | 直接轨迹几何 | <span style="color:#c62828"><b>高碰撞</b></span> | 只能作为受控 descriptor component |
| Speed/speed ratio | 全局距离变化 | step distance | cosine sequence | T1 | TTR 含路径长度 | motion trajectory | <span style="color:#c62828"><b>高碰撞</b></span> | 可作为 conditional state，不可作为最终 novelty |
| Hard local patch correspondence | 无 | 无显式 local match | 无 | 无 | same-grid，无匹配 | 光流轨迹可能对应 | **中碰撞** | frozen-token local search + real-only likelihood 的具体组合可能不同 |
| Soft local correspondence | 无 | 无 | 无 | 无 | 无 | 光流不同 | **中低碰撞** | 需证明不是简单平滑器，且在 motion-matched audit 下有效 |
| Correspondence entropy/confidence 聚合 | 无 | 无 | 无 | 无 | 无显式 match confidence | 待核实 | **低碰撞** | `rho` 同时用于不确定性诊断和 likelihood 聚合，需 calibration-matched |
| `p(D2 | speed_bin)` | 无 | descriptor 含 speed/angle，但监督联合学习 | 无 | 无条件 LL | 无 conditional density | 可能按运动状态分析，待核实 | **低至中碰撞** | real-only conditional dynamics likelihood 是当前最可防守候选 |
| 低维 geometry real-only likelihood | 无 | 21-D + fake-supervised MLP | 无 | Gaussian LL 框架 | 手工 score | compact trajectory representation | **中碰撞** | 重点应是 sample efficiency/robust calibration，而非 descriptor 本身 |
| Shrinkage/OAS covariance | 无 | 无 | 无 | empirical PCA whitening | 无 | 待核实 | **低创新、强工程价值** | 作为可靠统计组件与少样本证据，不宜单列核心贡献 |
| Student-t/robust Mahalanobis | 无 | 无 | 无 | Gaussian | 无 | 待核实 | **低至中碰撞** | 只有在 fast motion/camera shake 和 few-real-shot 上稳定改善才值得保留 |
| kNN real density | 无 | 无 | 无 | Gaussian | 无 | 待核实 | **低创新** | 作为低维 density baseline |
| Conformal evidence fusion | 无 | 无 | 无 | percentile average | real threshold | 无 | **低至中碰撞** | 能去除 alpha/beta fake-guided choice时有方法论价值；依赖性必须诚实处理 |
| Fisher/Cauchy p-value fusion | 无 | 无 | 无 | 无 | 无 | 无 | **低至中碰撞** | 统计方法本身成熟，贡献只能是无 fake 调权的稳定协议 |
| Adaptive top-K windows | 无 | 固定中心 | max temporal | 单窗 | 固定片段 | 多尺度 | **中碰撞** | calibration real 必须执行完全相同 selection；否则非法 |
| Universal real bank | 无 | 不适用 | blind/real threshold | VATEX 已是 universal Global bank | cross-real threshold | 待核实 | **中碰撞** | Local conditional model 的跨域 sample efficiency 才可能形成差异 |
| Partial fake localization | 无 | 无 dense field | 无 | 无 | 已重点完成 | trajectory 可定位 | <span style="color:#c62828"><b>高碰撞</b></span> | 只能作为应用验证，必须与 SPLIT 对比 |
| Fake Recall @ 0.1% FPR | 无 | 非核心 | 部分阈值 | 非核心 | 已重点完成 | 待核实 | <span style="color:#c62828"><b>已是协议基线</b></span> | 必须补，但不能作为创新 |
| Motion-matched/shortcut audit | 被审计对象 | 被审计对象 | 被审计对象 | 需审计 | 需审计 | 需审计 | **低算法创新、极高可信度价值** | 可成为论文实验设计的显著强项 |

## 2. 已不能作为独立创新的想法

以下表述应从标题、摘要和贡献列表中删除或降级：

1. “首次使用二阶时序特征检测 AI 生成视频”：D3 已明确提出。
2. “首次建模表示轨迹曲率/直线性”：ReStraV 已明确提出，MotionPhys 又扩展到物理轨迹。
3. “首次在 patch token 上分析局部时序粗糙度”：SPLIT 已明确提出 TTR。
4. “首次分析局部空间运动不一致”：SPLIT 已明确提出 LSMI。
5. “首次 training-free 检测部分编辑视频”：SPLIT 已在 FakeParts 上完成。
6. “首次 real-only 低 FPR 校准”：SPLIT 已报告 0.1%/1%/5% 与 cross-real threshold transfer。
7. “多尺度 patch/路径-弦比本身是新方法”：与 SPLIT TTR、ReStraV/MotionPhys 轨迹几何高度接近。

## 3. 仍值得验证的三条差异

### 3.1 Correspondence-aware local dynamics likelihood

不是把 same-grid 换成 matching 后就自动成为贡献。必须同时满足：

- local search 规则完全预设，不使用 fake 选择；
- hard/soft/confidence 只改变 correspondence，其余协议严格固定；
- 相对 C0 至少提高 Macro AUC 或 AP `0.005`，且至少 2/3 数据集同向；
- 在 motion-matched 或统一重编码控制下仍保留收益；
- 额外耗时和显存可接受。

若不满足，matching 只说明 same-grid 并非主要瓶颈，应停止 OT/learned matcher 扩展。

### 3.2 Conditional real dynamics likelihood

候选核心不是“又一个 curvature descriptor”，而是：

```text
p(local dynamics | real motion state, correspondence confidence)
```

最低成本版本以 real calibration 的 speed quantile 构造 slow/medium/fast 三个 bin，在每个 bin 内拟合 D2 或低维 geometry 的真实分布。其可防守点是条件统计和 fake-free fitting，而非 speed/D2 本身。

### 3.3 Sample-efficient robust real calibration

如果低维条件动力学配合 shrinkage/robust density 能在 25/50 real 下接近当前 200-real 表现，并改善 cross-real transfer，这一结果比单纯 `+0.002` AUC 更有论文价值。它直接回答 deployment 中真实参考库大小和域偏移问题。

## 4. 内部已否定方向

| 方向 | 当前证据 | 决策 |
|---|---|---|
| Local Spatial | 加入 Local 或完整模型均降低 Macro | 删除，不重复 |
| region pooling | 不存在跨数据集统一最优，且 Spatial 已删除 | 归档 |
| K=5/all windows | 未超过 K=3，成本更高 | 不作主线 |
| bottom-k/hybrid 旧实验 | 未形成稳定主结果，且与 partial/tail 文献拥挤 | 仅在新 likelihood field 有明确假设时重开 |
| OAS 单独替代 | 现有结果基本持平 | 不作为创新；保留统计 baseline |
| D4/D5 高阶差分 | 缺乏独立科学假设 | 不执行 |

## 5. Reviewer defensibility 检查

任何新候选进入论文前，必须能回答：

1. 数学上与 D3 的 scalar D2、SPLIT TTR、ReStraV curvature 有何不同？
2. 性能是否来自 real/fake FPS、时长、编码、重复帧或 motion magnitude 偏差？
3. 所有超参数是否来自理论、先验文献、real-only rule 或预注册 development setting？
4. 是否在选择候选后保留一个未用于选参的 confirmation benchmark？
5. 0.1% FPR、cross-real threshold 和 calibration sample efficiency 是否可接受？
6. 新模块在已有 frozen cache 上的额外时间/显存是否相对 backbone 足够小？

只有 correspondence 或 conditional likelihood 在这些问题下仍成立，才应升级论文主线。
