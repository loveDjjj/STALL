# Alpha-STALLED 论文与代码一致性审计

> 审计基准：当前 D2-only 论文源文件、`configs/benchmark.yaml`、提交 `718f780`。  
> 旧版“Local Spatial + Local D2，0.1/0.9”说明不再视为当前论文定义。

## 1. 总结

当前代码与最新论文的核心公式基本一致：DINOv3 ViT-L/16、`14x14x1024` patch token、8 FPS、16 帧、至多 K=3、same-grid raw-vector D2 后归一化、Local mean、Global max/min、D2-only Local 和 `0.6/0.4` 最终融合均可在主运行产物中追溯。

但“一致”不等于“可充分复现”。当前仍有四类高风险问题：

1. 配置中的 FPS/秒数/帧数/strategy 没有真正驱动评分代码。
2. Local 拟合参数未保存，无法对历史运行逐位重放。
3. packed cache 和 run manifest 的内容身份记录不完整。
4. 三个开发 benchmark 已参与方法选择，不能再无条件称为 untouched zero-shot test。

## 2. 逐项核对

| Item | 当前论文 | 当前代码 | Match? | Risk |
|---|---|---|---|---|
| 方法名 | Alpha-STALLED | `method.name=alpha_stall` | 是 | 低；展示名与配置标识不同但含义清楚 |
| Encoder | DINOv3 ViT-L/16 | `dinov3_vitl16`，官方权重 SHA 绑定 | 是 | 低 |
| Feature dim | 1024 | CLS/Patch 均为 1024 | 是 | 低 |
| Patch grid | `14x14` | 224 输入、16 patch size，196 tokens | 是 | 低 |
| Patch layer | 最终归一化 patch token | `x_norm_patchtokens`，contract 记 layer 23 | 是 | 低 |
| Patch 是否含 CLS/register | 不含 | 直接读取 `x_norm_patchtokens` | 是 | 低 |
| Feature normalize | DINO 最终 norm token | 使用 `x_norm_*` | 是 | 中；论文应区分 backbone LayerNorm 与后续 D2 L2 normalize |
| FPS | 8 FPS | manifest 按 8 FPS 生成 | 条件一致 | 高；评分只信任 manifest，修改 YAML `sampling.fps` 不生效 |
| Window duration | 2 秒 | 固定 16 个降采样帧 | 条件一致 | 高；修改 `window_seconds` 不生效 |
| Frames/window | 16 | `_window_features(... window_frames=16)` | 是 | 中；值硬编码而非 config 驱动 |
| Window strategy | uniform | `uniform_windows` | 是 | 中；修改 YAML `strategy` 不生效 |
| K | `K<=3`，正式 K=3 | 读取 `sampling.num_windows=3` | 是 | 低 |
| K=3 位置 | 首/中/尾均匀窗口 | `rint(linspace(0,L-16,3))` | 是 | 低；论文建议给出确定性定义 |
| 重复窗口 | effective-K 去重 | `deduplicate_windows` | 是 | 低 |
| 短视频 | `<16` 帧排除 | `_apply_short_video_policy` | 是 | 中；排除比例需按类别报告 |
| effective-K CDF | K=1/2/3 分开 | 按实际窗口数构造目标域真实视频参考 | 是 | 低 |
| Global 参数 | 官方 VATEX | NPZ 路径与 SHA256 强校验 | 是 | 低 |
| Global Spatial | per-frame LL，时间 max | `GLOBAL_SPATIAL_AGGREGATION=max` | 是 | 低 |
| Global T1 | 归一化一阶向量差分，时间 min | raw T1 后 L2 normalize，min LL | 是 | 低 |
| Global zero norm | 无时序信息应排除 | zero mask 设 `+inf` 后 min | 是 | 低；整窗全零会映射为 real percentile 1，应在边界中说明 |
| Global fusion | 0.5 Spatial + 0.5 T1 | config 0.5/0.5，活跃分量归一化 | 是 | 低 |
| Local Spatial | 正式方法删除 | `spatial_enabled=false`, weight 0 | 是 | 低 |
| Region pooling | 正式方法不使用 | 主路径没有 region pooling | 是 | 低；旧表/历史模块仍含 region 术语 |
| Local correspondence | same-grid，无显式匹配 | 相同 patch index 直接差分 | 是 | 高；resize 后位置不等于物体对应，属于方法限制 |
| D2 定义 | raw feature vector 二阶有限差分 | `P[t+2]-2P[t+1]+P[t]` | 是 | 低 |
| D2 是否在 normalized D1 上做 | 否 | 否 | 是 | 低 |
| D2 是否 scalar distance difference | 否 | 否 | 是 | 低；必须与 D3 明确区分 |
| D2 后 normalize | 沿 1024 维 L2 normalize | `F.normalize(..., eps=1e-12)` | 是 | 低 |
| Local zero norm | 论文未明确 | 零向量保留为零并参与高斯评分 | 不充分 | 高；应定义或做敏感性检查 |
| Local aggregation | time x patch mean | Gaussian LL 对所有 14x196 位置 mean | 是 | 低 |
| Local covariance | empirical | 默认 empirical；也支持 OAS | 是 | 中；rank/正则需在方法中完整写出 |
| Local fit sample cap | 论文未明确 | reservoir 上限 300,000 行 | 不充分 | 高；这是影响白化结果的实质参数 |
| Local fit seed | 17 | reservoir RNG seed=17 | 是 | 中；需与校准抽样 seed 的双重作用一起说明 |
| Local window CDF | target-real empirical CDF | calibration real raw windows | 是 | 低 |
| 视频内聚合 | 分支内窗口均值 | Global/Local 分别 mean | 是 | 低 |
| 视频级 CDF | target-real、effective-K matched | 两个分支都重新做目标域 CDF | 是 | 中；Global 最终并非只由 VATEX 决定 |
| Final fusion | 0.60 Global + 0.40 Local | config 0.6/0.4 | 是 | 高；权重曾参考开发 benchmark，应避免“无调参”表述 |
| Calibration count | 每数据集 200 real | 200 | 是 | 低 |
| Calibration seed | 17 | dataset-hashed seed 基于 17 | 是 | 中；并非三个数据集都直接给 pandas `17` |
| Calibration/eval overlap | 严格互斥 | `_ensure_disjoint` 无条件执行 | 是 | 低 |
| Fake 是否参与拟合 | 否 | 否 | 是 | 低 |
| Eval label 是否参与单视频评分 | 否 | 评分后才用于 metrics | 是 | 低 |
| 主表样本 | 21,421 条、20 generators | final run 相符 | 是 | 低 |
| GenVideo generator list | 8 个可用 generator | 2 个来源因全部不足 16 帧等原因不进入最终 run | 是但需解释 | 高；不能把 manifest 10 个与最终 8 个混写 |
| ComGenVid exclusions | 排除 2 个短 real | run manifest 记录 2 | 是 | 低 |
| GenVideo exclusions | 排除 2,565 条 | run manifest 记录 2,565 | 是 | 高；选择偏差必须进入正文或补充材料 |
| Score direction | 越高越 real | `higher_is_real` | 是 | 低 |
| AP positive class | real | `positive_class=real` | 是 | 中；与 SPLIT/D3 等 fake-positive AP 不可直接混比 |
| 论文主指标 | generator-pairwise macro AUC/AP | `pairwise_metrics.csv` | 是 | 中；另有 pooled 表，必须标清口径 |
| Pairwise real selection | 固定 seed、来源平衡 | seed=42 + DataFrame order + `fake.head()` | 部分 | 高；未保存显式 pair manifest |
| 低 FPR | 当前正文主要 AUC/AP，1% 为附加 | 只实现 1% | 部分 | 高；SPLIT/VidAudit 后 0.1% 已是关键部署指标 |
| Bootstrap | 受控比较 CI | 核心函数只输出 AUC delta CI | 部分 | 中；AP CI 依赖旁路脚本而非统一 runner |
| Local 参数可复现 | 应可冻结/发布 | 运行时拟合但不保存 `mu/W` | 否 | 高 |
| Cache config-aware | 应绑定关键提取协议 | 根 contract 较完整 | 部分 | 高；packed read 不重新校验源 SHA，key 不含 dataset |
| Run provenance | config、code、data、params 可追溯 | config hash + Git commit + root contract hash | 部分 | 高；缺 manifest/index/local-param hash |

## 3. D2 与相关方法的准确边界

### 3.1 当前 Local D2

```text
a[t,i] = P[t+2,i] - 2P[t+1,i] + P[t,i]
a_hat[t,i] = normalize(a[t,i])
```

它是固定网格 patch feature vector 的二阶差分，再进入 real-only Gaussian likelihood。

### 3.2 D3 的“二阶”

D3 官方代码先计算相邻全局帧 embedding 的标量 cosine similarity 或 L2 distance：

```text
d1[t] = distance(z[t], z[t+1])
d2[t] = d1[t+1] - d1[t]
score = std(d2)  # 官方评测主要使用该统计
```

因此二者都利用二阶时间变化，但数学对象不同：D3 对标量距离序列差分；当前方法对局部特征向量轨迹差分并建模真实似然。论文可以陈述这一差异，但不能声称“首次使用二阶动态”。

### 3.3 SPLIT 的重叠

SPLIT 的 TTR 也从 same-grid patch token 轨迹计算一阶/两步路径长度，并以 log ratio 聚合；LSMI 对同网格 patch motion field 做空间梯度。即使公式不等价，以下宽泛 claim 已不可防守：

- 首次在 patch token 上使用时序粗糙度检测生成视频；
- 首次用局部运动不一致检测部分生成视频；
- 首次 training-free patch-level temporal detection；
- 首次用 real-only threshold 做低 FPR 视频检测。

## 4. 论文措辞修订建议

| 当前/潜在说法 | 问题 | 建议 |
|---|---|---|
| “完全 zero-shot” | Local 使用目标域 real，结构与权重参考开发 fake | “不使用生成视频拟合的 target-real-calibrated training-free detector” |
| “Global 严格等同原版 STALL” | 视频层又做目标域 effective-K CDF，且 K=3 | “窗口组件继承官方 STALL；再按统一多窗口协议做目标域视频级校准” |
| “局部二阶是主要创新” | D3、ReStraV、SPLIT、MotionPhys 已覆盖二阶/轨迹/局部粗糙度 | 将 same-grid D2 降为基线或框架特例 |
| “固定权重不调参” | 0.6/0.4 来自开发过程 | “预先固定并跨数据集统一使用”；同时补 real-only/equal fusion 对照 |
| “跨生成器泛化” | 当前三个数据集已参与开发，外部 GenVidBench 含重叠 ModelScope | 限定为现有 benchmark 泛化；新增 untouched confirmation benchmark |
| “所有视频” | GenVideo 有 2,565 条被排除 | 明确 eligible intersection 和排除原因 |

## 5. 投稿前一致性门槛

必须完成：

1. 将运行时 Local 参数、manifest hash、packed index hash 和 calibration IDs 落盘。
2. 让 `sampling.window_seconds/fps/frames_per_window/strategy` 要么真正驱动代码，要么被严格锁定并拒绝覆盖。
3. 在主 runner 中统一加入 Fake Recall@0.1%/1% Real FPR，并报告真实域阈值转移。
4. 对 GenVideo 大规模短视频排除做来源/标签分层统计。
5. 将现有三数据集定位为 development suite；新方向只在其上做阶段筛选，最后预注册一个未用于选参的 confirmation benchmark。

在这些问题修复前，核心数值可以复现到 run 级别，但方法定义和数据身份还未达到可审计发布包的标准。
