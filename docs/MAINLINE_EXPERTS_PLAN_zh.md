# 当前论文主线上的协方差专家：受控试验方案

状态：开发全量四格及独立验收已完成，未通过继续门槛，保留原主线。三域专家拟合、源级真实留出、6个原图拟合样本位置核验、15569视频完全回退公式回归及8组配对区间完成。原生单窗Global专家的停止结论保留；这里检验的是目标适配Global＋Local D2＋FC K3下的增量，不复用不同协议的检测分数。最终结果见第10节。

## 1. 问题与固定基线

唯一基线R0为最新23单元：目标Global＋目标Local归一化D2，FC K3，等权融合，Average AUC/AP-real **0.874472/0.877075**。0.881462/0.882444是旧20单元回归锚点，不用于本轮主要对比；native Global的0.835/0.846也不是本轮基线。

保留每域200拟合片段，ComGenVid实际132独立源；不得改为近期来源诊断的每源一片段132/200/200，否则R0数据条件改变。拟合Uniform K3、每视频最多256 Local D2、224、batch8固定尾批、fp32差分/归一化、fp64评分、当前ridge、位置均值与0.5G＋0.5L全部固定。

核心新增假设：**在原主线中，用内容相关协方差替换总体协方差，能否让同一份Local D2证据更有判别力？**

Local协方差专家是主要候选。历史M25只改变内容均值、共享白化，没有验证这个候选。GT专家是辅助控制，用来检验此前Global时序的小收益是否还能叠加。GS保持当前目标域空间模型，不改成官方VATEX、不增加空间专家。

## 2. 最小四格矩阵

| 配置 | GS空间 | GT一阶时序 | Local D2 | 目的 |
| --- | --- | --- | --- | --- |
| R0 | 原目标模型 | 原目标模型 | 原目标模型 | 精确基线 |
| R1 | 不变 | 目标协方差专家 | 原目标模型 | GT专家独立作用 |
| R2 | 不变 | 原目标模型 | 目标协方差专家 | Local专家独立作用，主要新假设 |
| R3 | 不变 | 目标协方差专家 | 目标协方差专家 | 两者组合 |

这不是增加检测分支。顶层依然只有Global和Local，两者权重不变。GT/Local的评分矩阵根据当前窗口内容选择。

$$S_0=\tfrac14\overline{S_{GS}}+\tfrac14\overline{S_{GT}}+\tfrac12L,$$
$$S_3=\tfrac14\overline{S_{GS}}+\tfrac14\overline{S_{GT,E}}+\tfrac12L_E.$$

所有原始分数/校准固定后，每视频应满足S3−S0=(S1−S0)+(S2−S0)，在约定浮点容差内验证。AUC/AP不是线性函数，不能相加各自指标增益预测R3。

## 3. 第一版专家定义

本轮先固定**两个目标域内容专家**，不扫描2/4/8。旧研究2200条VATEX的4专家直接用于每域200目标片段并不具有同样支持，因此专家数与参考协议明确改变。

对每个真实拟合窗口，用其Global帧特征形成内容描述：

$$c(W)=\operatorname{normalize}\left(\operatorname{mean}_{t\in W}\operatorname{normalize}(g_t)\right).$$

只用目标fit real学习两中心球面聚类，seed17，聚类中按片段等权（每片段的窗口权重总和为1）。测试窗口沿同一余弦距离硬路由。GT与Local共用路由；不读取fake/CDF/evaluation更新中心，不增加软路由、在线近邻、预测或额外内容网络。

各分支共享原总体均值μ_d，只更换方向协方差：

$$C_{d,m}^{E}=0.5\widehat C_{d,m}+0.5\widehat C_d+10^{-5}I.$$

第一版固定0.5总体收缩，不根据fake调整；它使小簇统计向已验证目标总体模型靠近。它保证数值可解的条件，不保证统计估计可靠或检测必然提高。原总体协方差与簇内协方差均按主线分母/权重定义、在未加ridge前组合，不能重复加入正则。

专家均值不单独拟合进评分器；GT仍取窗口min，Local仍按所有有效位置的原均值规则。GT零T1与Local零D2的原分支规则分别保留。

Gaussian分组沿用每观察原有权重，按簇条件归一化。不能把只贡献少数样本到某簇的视频重新赋予一整份视频权重。报告每簇片段数、窗口数、独立源数、向量数与源权重有效量；不能把几万个相关Patch当成几万个独立视频。

## 4. 已完成的只读可行性检查

本次仅用已有fit Global在内存中做了两簇支持诊断，无新模型检测。窗口加权球面聚类得到下列独立源支持（同一源可出现在两簇，不能将两列相加当新增来源）：

| 域 | 两簇独立源数 | 风险 |
| --- | --- | --- |
| ComGenVid | 25 / 109 | 一簇77片段仅25源，重复源影响明显 |
| VideoFeedback | 38 / 162 | 支持不均衡 |
| GenVideo | 90 / 133 | 一些视频跨窗口属于不同簇 |
| GenVidBench | 80 / 45 | 外部拟合预算更小 |
| ViF | 46 / 35 | 必须保留小样本边界 |

这些计数不代表专家已经有效。不能沿用native实验128源的硬门槛后让全部回退，再称已检验主线专家。本轮用上述显式总体收缩，并先检查实际独立源支持与fit内部源级留出；若统计严重不稳定，整体判定该构造不适合当前预算，不根据fake偷偷降低门槛或换专家数。

真实留出时，总体Gaussian、聚类和专家都只能用该折训练源重拟合；不能把全200片段拟合的总体先验用于留出验证。留出NLL只作诊断，不代替真假检测。

## 5. 拟合缓存怎样复用

优先读取与主线回归绑定的paper_fit_<domain>/fit_features及prepared_fit.csv；当前600份小资产有Global窗口、帧索引、raw_d2[256,1024]和sampling_identity。无需重新提完整拟合Patch。

抽样D2所属窗口可精确重建。原代码按window/time/patch/channel展平，设K为真实Uniform窗口数：

$$N=K\times14\times196.$$

种子为SHA256('17:'+sampling_identity)前8字节小端整数；用同一np.random.default_rng(seed).choice(N,256,replace=False)重建索引j：

$$window=\lfloor j/(14\cdot196)\rfloor,\quad time=\lfloor j/196\rfloor\bmod14,\quad patch=j\bmod196.$$

每个样本绑定原所属窗口的内容描述，不能用新的物理路径重新随机抽样。必须核验旧采样算法、K、展平顺序、hash及抽样身份；必要时少量原视频验证，不靠形状相同默认兼容。

8帧短视频继承主线：总体Gaussian仍来自16帧拟合观察，8帧评价/参考使用对应短窗流程。专家路由函数对可用帧求同一内容描述，但不声称16帧拟合与8帧查询的内容估计方差相同；短窗结果单列，不为了它另外改变原Gaussian预算。M=1必须恢复该现有短窗协议。

## 6. CDF保持现有层级，分别重算

- GS：原模型与原CDF逐视频不变。
- GT专家：2000条独立VATEX的主线Uniform首窗执行冻结路由/专家评分，形成GT整体算法CDF；评价仍在FC窗上评分。保留主线已有的Uniform参考/FC评价差异，不同时修正选择器。
- Local专家：同2000条VATEX执行FC K3、按窗口路由/专家D2评分，先按主线均值/effective-K规则构造视频参考。保持ranked-linspace子窗规则，不擅自变成Top-k。
- 8/16帧各用现有匹配长度的参考方式；不加Local窗口CDF、Global视频CDF或同簇C/soft CDF。
- R1与R3共享同一个GT专家CDF，R2与R3共享同一个Local专家CDF；原分支分别复用原CDF。不能把新Gaussian分数查询旧总体数组。

CDF、阈值和评价角色继续分开。首次只报告排序与评价ROC操作点；若使用独立VATEX阈值评估固定FPR，冻结所有评分规则后另算，不用于选模型。

## 7. 评价成本与代码接入

现有coarse_1fps和native单窗Global不能替代主线FC K3密集特征；paper_evidence_fc3和cache/scores主要是特定Gaussian的scalar。因此GT-only也不是CPU零提取实验。旧约630GiB Uniform Patch只在帧索引、数值合同完全匹配时复用，不能假设全命中。

最省成本做法：**一次原主线batch8/union顺序的DINO前向，同时评分R0/R1/R2/R3**，包括独立CDF。GT/Local专家先各自得到所有专家窗口分数和路由，再形成四格，不为GT跑完整轮后又为Local重提一轮。

保留小型FC Global、窗口索引/路由、各专家窗口raw与配置hash；不再保存全量巨大Patch或1024×1024每视频方向矩。当前原视频、旧缓存保持不变。

建议接入主线reference_fit与SharedEvidenceScorer的统计接口，增加独立专家参考模块和研究配置，不调用native Engine冒充主线实现。沿用现有StableGaussianParams及主线归一化/白化规则；新run独立输出。configs/paper.yaml与正式参数不覆盖。

## 8. 执行顺序与停止条件

1. **仅fit real的准备与回归。** 冻结两簇/总体收缩、恢复D2样本索引；M=1、专家完全退回总体、R0逐视频及各分支回归。检查簇支持及源级真实留出。此阶段不把小样本AUC拿来与0.874比较。
2. **成本探针。** 用预定少量参考/评价身份测完整四格的端到端速度与缓存命中，只用于估计时间和核验R0，不选择专家参数。FC Local新评分可能是小时级，先实测，不能沿用native Global的分钟级预算。
3. **开发全量四格。** 最新固定23单元、15569评价身份，报告每个生成器、三域和Average；包括GT-only、Local raw/calibrated-only、Final、ROC操作点与源级配对区间。
4. **有独立增益后再补内容必要性控制。** 打乱fit窗口/样本的内容簇归属、保留同数量与相同路由器，再拟合对应专家并重算CDF；避免只在测试时打乱专家。明确这是拟合内容关联打乱，不是新随机选择器。其作用是区分内容匹配与随机统计分割。
5. **冻结外部验证。** 对保留候选使用同一专家定义及实际外部fit预算，不按域改设置。不将开发增益直接当外部收益。

主比较是R2−R0、R1−R0、R3−R0及组合对单独升级。至少一项主要指标有可靠正增量、另一项不明显损害，且不能只靠某个域或预算不一致。外部不支持时不升级论文主线。不因R3某个域最高而按域拼模型。

如果四格没有清楚收益，就保留原模型，结束“主线加专家”这次假设；不继续扫簇数、收缩、CDF、融合权重。如果仅GT或仅Local有效，只保留那一个专家化位置，不为形式完整强行保留两者。

本方案区分了此前native Global专家失败与当前主线协方差专家假设，但这种区别不是成功证据。当前没有新检测结果或涨分承诺。

## 9. 当前执行入口

```bash
python -m src.mainline_experts.run fit --dataset comgenvid
python -m src.mainline_experts.run fit --dataset videofeedback
python -m src.mainline_experts.run fit --dataset genvideo
python -m src.mainline_experts.run prepare
python -m src.mainline_experts.run verify --pilot
python -m src.mainline_experts.run dense --pilot --rank 0 --world-size 1
python -m src.mainline_experts.run dense --rank 0 --world-size 2
python -m src.mainline_experts.run dense --rank 1 --world-size 2
python -m src.mainline_experts.run evaluate
python -m src.mainline_experts.run analyze
python -m src.mainline_experts.run verify
python -m src.mainline_experts.run report
```

两个dense rank并行，完成检查点按物理窗口key独立，pilot通过的数据可复用。独立VATEX相同窗口跨目标模型去重，共19569物理任务服务15569评价身份和每域/长度2000CDF。拟合CPU任务与GPU探针已完成，不应因中断或权限切换重提。

真实留出Local在ComGenVid/GenVideo改善、VideoFeedback近似持平；GT在VideoFeedback变差，全部保留为诊断。全量R0原始与最终分数误差均为0。降低FFmpeg解码线程的6视频像素hash对照一致但更慢，因此没有采用该提速尝试。

## 10. 已完成结果与预定停止决定

| 配置 | Average AUC | AP-real |
| --- | ---: | ---: |
| R0 当前主线 | 0.874472 | 0.877075 |
| R1 仅GT专家 | 0.870729 | 0.875437 |
| R2 仅Local专家 | 0.874814 | 0.878930 |
| R3 双专家 | 0.866651 | 0.874063 |

R2相对R0的ΔAUC为+0.000343，95%区间[-0.002128,+0.002603]；ΔAP-real为+0.001855，区间[-0.000110,+0.004007]。两个区间均跨零。VideoFeedback的R2 Final AUC/AP差值为-0.009647/-0.005454，区间均在零以下；该域Local raw AUC也下降0.013918，不是只在CDF之后出现损失。ComGenVid的正结果和GenVideo的部分正结果保留，但不据此按域启用专家。

R1两个Average指标区间为负，R3的AUC区间为负。三个候选都未达到第8节的可靠新增收益门槛，所以不触发条件性的拟合内容打乱与外部确认；这些后续实验本轮未做，不声称外部失败。结束本次主线协方差专家假设，不扫簇数、收缩、融合或CDF，正式主线不变。

76项tests通过；独立核验65684个路由、31138个专家百分位、36个原图专家评分探针。矩阵恒等式最大误差8.66e-14，直接逐位置Local评分最大误差6.83e-13。双卡密集阶段约48.53/46.72分钟，共19569个物理任务，服务15569评价身份及各域/长度匹配的CDF；不是四次独立特征提取。

[完整报告](../results/runs/mainline_experts/RESULTS_zh.md)、[23生成器结果](../results/runs/mainline_experts/generator_metrics.csv)、[配对区间](../results/runs/mainline_experts/confidence_intervals.csv)、[验收](../results/runs/mainline_experts/verification.json)。
