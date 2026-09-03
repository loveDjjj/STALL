# Stage 2：轨迹几何实验结果

> 完成日期：2026-09-02  
> 实现提交：`59746c3`  
> 基础 correspondence：Stage 1 胜出的 C0 same-grid  
> 数据范围：三个开发数据集固定 21,421 条可用视频

## 1. 受控设置

| ID | Local dynamics | 维度 | 说明 |
|---|---|---:|---|
| T0 | normalized vector D2 | 1024 | 当前正式基线 |
| T1 | curvature `1-cos(v_prev,v_cur)` | 1 | 零速度转向定义为 0 |
| T2 | log speed ratio | 1 | 固定 `eps=1e-6` |
| T3 | path/chord ratio | 1 | 与 SPLIT TTR 高重叠，仅作诊断 |
| T4 | normalized D2 + curvature | 1025 | 检验曲率是否补充 D2 |
| T5 | log-speed、curvature、speed-ratio、path/chord | 4 | 低维 geometry descriptor |

所有候选只使用 real calibration 拟合 Gaussian likelihood，共享 C0 的 Global 窗口分数、K=3、CDF 和 `0.6/0.4` 融合。五个候选通过一次 cache 扫描共同评分，避免重复读取五次 630 GB Patch cache。

## 2. 最终融合结果

| Variant | ComGenVid AUC/AP | VideoFeedback AUC/AP | GenVideo AUC/AP | Macro-3 AUC/AP | Delta vs T0 |
|---|---:|---:|---:|---:|---:|
| T0 D2 | **0.9015/0.9110** | 0.8621/0.8687 | 0.8586/0.8391 | 0.8741/0.8729 | - |
| T1 curvature | 0.8854/0.8897 | 0.8108/0.8296 | 0.8144/0.8157 | 0.8368/0.8450 | -0.0372/-0.0279 |
| T2 speed ratio | 0.7302/0.7688 | 0.7607/0.8037 | 0.5866/0.6323 | 0.6925/0.7349 | -0.1816/-0.1380 |
| T3 path/chord | 0.7894/0.8103 | 0.7777/0.8035 | 0.6745/0.6984 | 0.7472/0.7708 | -0.1268/-0.1022 |
| T4 D2+curvature | 0.9014/0.9110 | **0.8621/0.8686** | **0.8586/0.8392** | **0.8741/0.8729** | +0.000002/+0.000009 |
| T5 geometry | 0.8313/0.8383 | 0.8074/0.8304 | 0.5866/0.6323 | 0.7418/0.7670 | -0.1323/-0.1060 |

T4 与 T0 的差异只有百万分之一量级，不具有实际意义；其余几何候选在三个数据集上均低于 T0。

## 3. Local-only 证据

| Variant | Local Macro AUC | Local Macro AP |
|---|---:|---:|
| T0 D2 | **0.8270** | **0.8229** |
| T1 curvature | 0.6919 | 0.6764 |
| T2 speed ratio | 0.3307 | 0.4502 |
| T3 path/chord | 0.4401 | 0.4797 |
| T4 D2+curvature | 0.8269 | 0.8230 |
| T5 geometry | 0.4411 | 0.5026 |

低维几何量并未被 Global 融合掩盖，而是自身判别能力明显不足。T2/T3 甚至在部分数据上方向接近反转，说明简单单变量 Gaussian typicality 对 motion magnitude、重复帧和数据域差异非常敏感。

## 4. 成对 bootstrap

按论文生成器平衡配对规则进行 1,000 次视频级成对重采样。T4 相对 T0：

| Metric | Delta mean | 95% CI |
|---|---:|---:|
| Macro AUC | +0.0000008 | [-0.0000900, +0.0000969] |
| Macro AP | +0.0000060 | [-0.0000905, +0.0000967] |

区间高度对称跨零且幅度小于 `1e-4`，说明给 1024-D D2 再拼接一个 curvature 标量没有可检测增益。其他候选的大幅负差无需靠边界显著性解释。

## 5. 部署指标

几何候选没有统一改善超低 FPR。部分单项例外，例如 T5 在 ComGenVid 的 0.1% FPR recall 达到 `0.0729`，但其 pooled AUC 与 pairwise Macro 大幅下降；T2 在 VideoFeedback 1% FPR recall 达到 `0.154`，同时整体 AUC/AP 显著更差。不能据单一 operating point 选择候选。

## 6. 效率与存储

- 五候选共同评分总计 `4,194.5 s`，约 69.9 分钟。
- 峰值显存 `6.05 GiB`（PyTorch allocated 口径）。
- 完整矩阵结果约 109 MB；未增加 feature cache。
- 相比五个独立 run，单次扫描将预计约 3.15 TB 的重复 cache 读取降为约 630 GB。
- 每个数据集、每个候选均保存独立 Local 参数、200 个 calibration IDs 与 SHA256。

## 7. 科学解释与决策

结果支持以下判断：

1. D2 的有效信息不仅是转向角或速度标量，而包含 1024 维特征方向上的异常结构。
2. curvature 与 normalized D2 高度冗余；加入 D2 后几乎完全没有边际信息。
3. speed ratio/path-chord 更易受到视频运动幅度、静止段和预处理差异影响，与 2026 motion-shortcut 审计的风险一致。
4. T5 的低维统计虽然样本效率理论上更好，但当前 200-real 性能已远低于 T0，未通过 `-0.002` 保留门槛，不值得进入 robust-density 主线。
5. T3 与 SPLIT TTR、T1 与 ReStraV/MotionPhys 的创新重叠本已很高，负结果进一步排除其作为论文主贡献。

**H2 被否定。** Stage 3 固定使用 T0 same-grid normalized vector D2 检验 `p(D2|speed state)`；conditioned geometry 只作为任务要求的必要对照。如果条件化不能达到 `+0.005` 且至少 2/3 数据集提升，则停止 conditional dynamics，不为故事增加更多 bins 或 neural density。
