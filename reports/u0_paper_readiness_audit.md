# Alpha-STALLED U0 学术论文就绪性审计

日期：2026-07-25

Submission readiness：**`ready_with_author_checks`**（限定为相对 STALL 的方法论文）；
若标题或摘要要使用 SOTA 论点，则为 **`blocked`**，直到同协议竞争 baseline 完成。

## 审计结论

当前指标提升、核心消融、统计检验和锁定外部验证，已经满足以下论文论点：

> Alpha-STALLED U0 是 Original STALL 的训练自由 Global--Local 与
> multi-window 扩展；在固定三数据集协议下，它相对 STALL 和同核心 K1 都有
> 视频级配对置信区间为正的 Macro AP 增益，并在一个锁定新域上得到有限确认。

当前证据**不满足**以下论点：

> U0 优于所有现有生成视频检测方法、达到 SOTA、无需目标域真实视频、或能通用
> 定位语义伪影。

原因是尚缺同协议竞争方法重跑，外部确认只有两个生成器且需要新域 real 校准，
合成注入也没有支持单调异常响应。

## 一句话论点与术语锁定

一句话论点：在训练自由、目标域 real-only calibration 条件下，Alpha-STALLED U0
通过 same-grid PatchD2 和三个确定性视频窗口扩展 Original STALL，并在固定协议上
获得可复现的 Macro AP 增益；该结论止于相对 STALL 的受控比较。

| Canonical term | 本文含义 | 不再混用的写法 |
|---|---|---|
| Alpha-STALLED U0 | 完全统一、锁定的 K3 正式方法 | final K3、historical tuned K3 |
| Original STALL | 原方法在固定交集上的正式对照 | Global-only（后者是 U0 消融） |
| Unified K1 | 与 U0 同核心、仅 K=1 的直接对照 | historical clean K1 |
| Global / Local | 全局分支 / patch 局部分支 | 第三 volatility 分支 |
| same-grid PatchD2 | 同空间网格 patch token 二阶差分 | hard/soft matching、D3 |
| Macro-3 AUC/AP | 三数据集 generator-pairwise 宏平均 | unique-video pooled AP |
| real-positive AP | real 为正类、真实性分数为方向 | fake-positive AP |
| target-domain real-only calibration | 只用目标域独立真实视频校准 | target-data-free |

## 投稿材料矩阵

| Item | Status | 当前来源或缺失输入 | 输出/行动 |
|---|---|---|---|
| Main manuscript | required | `paper/ieee_alpha_stalled/main.tex` | 科学内容就绪，待版面编译 |
| Anonymous manuscript | venue-dependent | 未指定双盲规则 | AUTHOR_INPUT_NEEDED |
| Title page | required | 作者、单位、通讯作者均缺失 | AUTHOR_INPUT_NEEDED |
| Cover letter | venue-dependent | 目标 venue/article type 未指定 | 选择 venue 后起草 |
| Highlights/summary | venue-dependent | 未知字数和格式 | 选择 venue 后起草 |
| Figures/tables | required | 主文输入齐全，静态检查通过 | 待最终 PDF 逐页检查 |
| Supplementary information | likely required | 负结果和协议审计已有素材 | 需按 venue 页数拆分 |
| Data/code availability | required | 声明草稿已有，URL/license 缺失 | AUTHOR_INPUT_NEEDED |
| Contributions/funding/conflicts | required | 均为占位符 | AUTHOR_INPUT_NEEDED |
| Ethics/misuse statement | recommended | 已有范围声明 | 作者确认 |
| Reviewer suggestions | venue-dependent | 无候选及冲突核查 | AUTHOR_INPUT_NEEDED |

## 论文条件检查

| 条件 | 证据 | 状态 |
|---|---|---|
| 明确冻结的方法 | U0 统一 region1/mean、layer23、alpha=0.6、beta=0.1、K=3 | 满足 |
| 独立校准/评测 | 每数据集 200 calibration real；与 21,421 evaluation 零重叠 | 满足 |
| 不使用 fake 拟合 | fake 不进入 whitening、CDF、阈值、权重或参数 | 满足 |
| 公平主基线 | Original STALL + 同核心 Unified K1 | 满足 |
| 主要效应量 | U0-STALL AP +0.0295；U0-K1 AP +0.0087 | 满足 |
| 统计不确定性 | 两个主要 AP CI 分别为 [+0.0241,+0.0358]、[+0.0057,+0.0120] | 满足 |
| 组件必要性 | 加 Local +0.0237 AP；加 Global +0.0431 AP，CI 均为正 | 满足 |
| 多数据集/生成器 | 3 个数据集、20 个生成器；相对 STALL 16/20 不下降 | 满足 |
| 校准稳定性 | N=25/50/100/200，5 seed；N=200 AP std=0.0022 | 满足 |
| 锁定外部确认 | GenVidBench 900 视频、2 个 fake generators | 部分满足 |
| 扰动不确定性 | 1,600 视频，Scenario A 1,000 次 paired cluster bootstrap | 满足 |
| 同协议 SOTA 横向表 | D3/AEROBLADE/RIGID/ZED/T2VE/AIGVDet 尚未公平重跑 | 缺失 |
| 完整投稿元数据/PDF | 作者、单位、venue、release URL 待定；本机缺 TeX 字体 | 待作者完成 |

## 指标是否足够

### 相对 Original STALL

- Macro AUC/AP：`0.8388/0.8428 -> 0.8741/0.8723`。
- Delta：`+0.0353/+0.0295`。
- AP 95% paired cluster-bootstrap CI：`[+0.0241,+0.0358]`。
- 16/20 个生成器 AP 不下降。

这不是只有零点几百分点的波动，效应量和区间均足以支撑相对 STALL 的主论点。

### 相对公平 Unified K1

- Macro AUC/AP：`0.8632/0.8636 -> 0.8741/0.8723`。
- Delta：`+0.0109/+0.0087`。
- AP 95% CI：`[+0.0057,+0.0120]`。
- 15/20 个生成器提升，但 VideoFeedback AP 下降 `0.0045`。

该结果足以支撑 K=3 的平均覆盖增益，但不能写成逐数据集一致提升。K3 使用约
`2.30x` K1 帧数，因此论文必须同时报告计算代价。

## 消融是否足够

核心消融已经形成闭环：

1. GlobalSpatial、GlobalT1、完整 Global 均有独立 K1 行。
2. PatchSpatial、PatchD2、完整 Local 均有独立 K1 行。
3. K3 Global-only、Local-only、Global+Local 隔离分支贡献。
4. Unified K1 与 locked K3 隔离窗口覆盖贡献。
5. K5/all、bottom-2/hybrid、residual、multiscale、intermediate layer、Joint、OAS
   均有明确拒绝证据。
6. 失败组件没有在观察结果后堆叠调权重恢复。

需要诚实披露的组件问题是 PatchSpatial：K1 中加入冻结的 0.1 PatchSpatial 使
Macro AP 下降 `0.0070`，CI `[-0.0088,-0.0054]`。因此创新贡献应放在 PatchD2，
不能把 PatchSpatial 写成有效组件。U0 保留 beta=0.1 是 release 锁定选择，而不是
后验最优证明。

## 本次优化

- 主表新增 Unified K1，避免用历史 clean K1 归因 K3 增益。
- 新增协议角色表，将 historical audit、正式对照和 leakage-excluded 结果分开。
- 明确主终点、两个主比较、次要终点和探索性分析；不作多重比较下的全族显著性声明。
- 删除“预注册式消融”措辞，改为可验证的“运行前准入门槛”。
- 收紧 D2 对 lag/high-order 的结论，只称代表性历史机制对照。
- 新增鲁棒性 1,000 次视频级 paired cluster bootstrap；机器可读结果位于
  `results/research_summary/u0_robustness_bootstrap_deltas.csv`。
- 将 CRF23/drop10 的结论改为“小点损失且区间跨 0”，明确这不是等效性证明。
- 正文局限性和结论明确禁止 SOTA、target-data-free 和通用语义定位外推。

## 仍需补充的实验

如果目标期刊/会议只要求相对原方法的严谨扩展，当前实验主体可以进入投稿整理。

如果目标是竞争性较强的生成内容检测 venue，优先级最高的新增实验只有一类：

1. 冻结统一样本交集、帧预算、正类方向和校准资源。
2. 重跑至少一组有代表性的公开方法，例如 D3、AEROBLADE、RIGID、ZED、T2VE、
   AIGVDet 中可复现者。
3. 同时报告失败、缺失覆盖和推理资源，不为不同 baseline 单独选有利子集。
4. 在运行前确定主竞争方法和指标，避免看过 U0 结果后选择对手。

不同论文中的公开数字不能直接拼入主表替代这一步。

## 验证记录

- `python -m unittest discover -s tests -v`：117/117 通过。
- `tools/verify_u0_locked_release.py`：61/61 通过，Macro 精确复现
  `0.874072/0.872299`。
- 鲁棒性 bootstrap：80 个 dataset/metric/condition 结果，全部 1,000 次；点估计
  与锁定 robustness metric table 误差小于 `1e-12`。
- LaTeX 静态检查：35 个 tex 文件，48 个 labels，输入、引用、花括号和 citation
  key 零错误。
- 本机 PDF 编译未完成：`pdflatex/lualatex` 在正文前因缺少 `ptmr7t.tfm` 和
  `luaotfload-main` 失败；需要在完整 XeLaTeX/TeX Live 环境中完成版面验证。

## 最终判断

**满足相对 STALL 的严谨方法论文条件；不满足 SOTA 竞争性声明条件。**

当前最合理的投稿策略不是继续堆叠已经失败的 Local 结构，而是保持 U0 锁定，补齐
同协议竞争 baseline，并把目标域校准、VideoFeedback 负迁移、严重扰动退化和注入
机制边界完整写入论文。
