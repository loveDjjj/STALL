# Alpha-STALLED论文主线与实验协议

2026-09-12状态：Looped保持停止，退役研究源码已保存在Git恢复点efede85。高维方向、位置数匹配、新真实拟合池和重编码验证均已完成并写入[IEEE正文](../paper/ieee_alpha_stalled/main.tex)。本文件保留方法与历史回归合同；下文三个组成部分不作为三个预设成立的独立创新，目标适配与观察策略作为条件和归因控制。不要根据旧计划再次启动已结束实验。

最新完整评价：主实验与全部消融均已扩展至23单元，且按原文对齐VideoFeedback动态等级筛选。Macro-3 AUC/AP-real为0.874472/0.877075，详见[完整结果](../results/paper_complete/RESULTS_zh.md)、[论文正文](MANUSCRIPT_REVISION_PLAN_zh.md)与[协议核验](MANUSCRIPT_PROTOCOL_NOTES_zh.md)。下文0.881462/0.882444及20单元数量保留为既有模型回归合同，不再代表当前完整主表覆盖。

状态：用户确认的新主线。本文定义科学协议，不把资料清理或工程回归当作新的检测实验。
前五组论文实验现已用当前主线重跑并完成验收，正式表格与说明见[本轮实验报告](../results/paper/RESULTS_zh.md)。主线21421条视频的Final、Global和Local分数均与冻结锚点逐条一致；后续不要把下面的历史来源说明当成“仍需重跑”。
当前使用见[运行手册](RUNBOOK_zh.md)，数据与缓存见[DATA_zh.md](DATA_zh.md)，修改规则见[仓库规范](REPOSITORY_RULES_zh.md)。

## 1. 唯一主方法与术语

论文方法名保留 **Alpha-STALLED**。新主线采用目标域适配Global + Local D2，固定Feature-change K3、0.5/0.5融合。
配置身份建议记为`target_global_local_d2_fc3_v1`（实验协议ID，不是新算法名称）。
现有结果行是`adapted_Global_plus_Local`，完整开发配对Macro AUC/AP-real为 **0.8814615038969483 / 0.8824439675246946**。
正文显示0.881462/0.882444或0.8815/0.8824，不用四舍五入值做回归测试。

| 名称 | 含义 | 不应混称 |
| --- | --- | --- |
| 当前主线 | 目标Global + 目标Local D2 | 旧B3 |
| 旧B3 | 官方VATEX Global + 目标Local D2，0.878177/0.877191 | 新论文最终默认 |
| 原版STALL | 官方Global检测器及其原生观察与参考协议 | 将观察改为FC K3后的Global控制 |
| GS / GT / LT | 全局空间、全局归一化T1、局部归一化D2 | Local Spatial（不在当前主线） |
| 目标域适配 | 用独立目标real拟合Gaussian度量 | 目标域CDF、测试标签调参、每视频自动选域 |
| 真实拟合集 / CDF参考集 / 阈值集 | 参数拟合 / 百分位映射 / 操作阈值 | 三种角色的一般“校准集” |
| 片段 / 源视频组 | 文件单位 / 同源片段的分组单位 | 256个Patch向量就是256个独立样本 |

采用新主线是已观察开发与外部结果后的研究决定，不能把旧实验追认成“此前预注册的新主线确认”。
旧结果中的B3身份不重新解释。当前唯一入口为scripts/run.py，当前五域参考包位于precomputed/target_reference；旧B3入口已清理。

## 2. 论文要回答的问题

核心问题：不微调视觉编码器、不训练真假分类器时，局部二阶动态方向能否补充全局证据？这种局部信息应由怎样的真实统计度量解释，在有限观察预算下能保留多少检测能力？

三个组成部分按以下顺序说明；研究主张集中于局部高维方向的独立价值，不能为了凑数量而给相同强度的原创性声明：

| 方面 | 解决什么具体问题 | 方法内容 | 必须有的证据 |
| --- | --- | --- | --- |
| 局部二阶动态方向 | 全局特征与标量变化未显式保留局部方向 | 同网格归一化向量D2、位置均值 | 新Global固定时D1/D2、幅度、Global D2、Local移除 |
| 目标真实统计度量适配 | 外部参考未必匹配目标真实方向分布 | 同一目标real池拟合GS/GT/LT，CDF角色分离 | 源/目标度量、均值/白化、公平目标real预算 |
| 固定预算动态片段观察 | 有限计算下避免只观察单个固定片段 | 1FPS粗扫、FC最多3个连续窗口 | 新主线下K1/Uniform K3/FC K3和实际成本 |

“对基线有利”指检验能够支持其独立价值的合理假设，不是筛掉不利域、翻转分数、只选最佳seed或用fake挑超参数。
每项实验预先固定改变什么、配对身份、指标和停止条件；失败结果保留并收窄表述。

### 2.1 原版STALL参考与创新边界

核对[STALL原文](https://arxiv.org/html/2603.15026v2)第4节、附录A.3、D.1、D.6.6及[官方代码](https://github.com/OmerBenHayun/STALL)：
原文已有真实白化/百分位融合、高阶整体差分与参考来源实验；采用8FPS、16帧并做逐生成器平衡比较。不能声称首次使用Gaussian、二阶差分或目标真实参考适配。
我们具体研究局部向量方向及双分支真实统计的受控贡献，不将代数性质称为分类性能定理。
下一阶段需核对D3/SPLIT既有逐窗分数并补齐当前23单元口径，优先构建同编码器、同Global、同窗口的强标量控制；具体范围按最新研究方案。原生方法与替换编码器后的控制分别命名，不将旧20单元表直接作为新23单元结果。

## 3. 精确数据流

```text
目标域独立real拟合集（开发每域200片段；源数另计）
  Uniform K3 / 8FPS / 16帧 / 冻结DINOv3
    ├─ Global token → GS Gaussian
    ├─ 归一化Global T1 → GT Gaussian
    └─ 同网格归一化Patch D2 → LT Gaussian（每视频最多256位置）

独立VATEX CDF 2000视频
    ├─ Uniform K1首窗 → 目标GS/GT评分 → 两个Global窗口CDF
    └─ Feature-change K3 → 目标LT评分 → effective-K Local视频CDF

待测视频 → 原始帧索引 → 1FPS Global粗扫 → FC至多3窗
    ├─ 每窗GS(max) / GT(min) → 各自Global窗口CDF → 窗内0.5/0.5 → 窗均值G
    └─ 每窗LT位置均值 → 窗均值qL → 匹配effective-K的外部视频CDF → L
  S = 0.5G + 0.5L

再独立VATEX threshold 2000视频 → 完整冻结评分器 → 可选阈值
evaluation标签只进入指标计算；不参与Gaussian、CDF、窗口选择与融合。
```

重要：新Global已不使用官方VATEX Gaussian或官方CDF数组作为自己的主线统计。它沿用STALL评分结构，但使用目标参数以及由目标参数重评分的独立VATEX窗口CDF。
官方NPZ仍为STALL比较必需资产，不能在清理时删除。
Global CDF的Uniform首窗与测试FC窗并非完全selector-matched；为精确复现0.881462，本轮保留这一事实，不能悄悄改成FC CDF。
Local所有域复用同一批VATEX视频身份，但每个域必须用自身Gaussian重评分，不能共享另一Gaussian的raw分布。

## 4. 方法与公式

### 4.1 输入与观察

冻结DINOv3 ViT-L/16。224×224输入得到Global token $g_t\in\mathbb R^{1024}$，Patch网格 $p_{t,i}\in\mathbb R^{1024}$，$i=1,\ldots,196$。
14×14是Patch网格数量；每个Patch为输入上的16×16像素。不是“14×14像素Patch”。
ratio下采样索引为 $i_j=\operatorname{round}(f_{orig}j/8)$，只保留合法互异帧；不将低FPS视频复制帧上采样。
严格复现实验优先使用冻结manifest索引，不能重新探测元数据后假定一致。时长不足、缺帧及VFR需独立记录。

粗扫为dense索引每隔8个位置取得Global特征，名义1FPS；相邻变化 $r_t=\|g^{coarse}_{t+1}-g^{coarse}_t\|_2$。
候选窗口长度16个dense帧，步长4（名义0.5秒），尾对齐去重。
候选分数为落入其范围的粗转移中点对应$r_t$均值；没有中点时取离窗口中心最近的转移。
按变化分数降序、起点升序确定性排序，取前3个候选。重叠允许，不是三个互不相交窗口。
保留**选择排名顺序**及时间起点两个字段，不能只因按时间排序更直观而改变effective-K参考取法。

### 4.2 真实统计拟合

对任一分支输入向量$x_{v,i}$，每个拟合片段内部给权重$a_{v,i}=1/m_v$。令$a_1=\sum a_{v,i}$、$a_2=\sum a_{v,i}^2$：

$$\mu=\frac{\sum a_{v,i}x_{v,i}}{a_1},\qquad
C=\frac{\sum a_{v,i}(x_{v,i}-\mu)(x_{v,i}-\mu)^T}{a_1-a_2/a_1}.$$

对应当前`np.cov(aweights=..., ddof=1)`。Local每片段固定256位置时退化为普通无偏协方差。
令$C=Q\operatorname{diag}(\lambda_j)Q^T$，白化矩阵

$$W=Q\operatorname{diag}((\max(\lambda_j,0)+10^{-5})^{-1/2}),\quad A=WW^T.$$

保留1024维，微小负特征值截到0，实质非PSD时报错。不得与官方降秩或其他正则规则混用。
Local按固定视频路径/seed抽256位置；路径重命名会影响抽样身份，因此归档不能顺手更名视频。
Global使用相同拟合片段的全部可用Global观察，各片段总权重相同；观察向量数不等于Local的51200。

| 开发域 | 拟合片段/源组 | GS观察数 | GT有效观察数 | Local D2观察数 |
| --- | --- | ---: | ---: | ---: |
| ComGenVid | 200/132 | 9600 | 8991 | 51200 |
| VideoFeedback | 200/200 | 7744 | 7242 | 51200 |
| GenVideo | 200/200 | 9600 | 8937 | 51200 |

源组隔离和片段等预算是两件事；当前不是对源视频等权拟合。均值/协方差拟合是real-only统计估计，不声称完全无参数学习。

### 4.3 Global空间和一阶时序

$$d_t=g_{t+1}-g_t,\qquad v_t=d_t/\max(\|d_t\|_2,10^{-12}).$$

Gaussian型分数统一写成

$$\ell_b(x)=-\tfrac12\{\|(x-\mu_b)^TW_b\|_2^2+D_b\log(2\pi)\}.$$

这是当前实现的白化空间评分，不包含原特征密度的Jacobian/logdet常数；同模型匹配CDF时常数不改变排名。
若比较不同协方差的真实留出NLL，必须另加logdet，不能复用此raw均值挑模型。

每窗$W_j$：$q_{GS,j}=\max_t\ell_{GS}(g_t)$，$q_{GT,j}=\min_{t:\|d_t\|>0}\ell_{GT}(v_t)$。
Global严格零转移不参与Gaussian拟合及时序min；若全窗零转移，GT raw为$+\infty$，CDF保留相应质量。NaN和负无穷拒绝。
令$F_{GS,d},F_{GT,d}$由独立VATEX Uniform K1首窗的同类max/min raw构造：

$$G_j=\tfrac12F_{GS,d}(q_{GS,j})+\tfrac12F_{GT,d}(q_{GT,j}),\qquad G(V)=K_{eff}^{-1}\sum_jG_j.$$

右包含经验CDF为$F(q)=N^{-1}\sum_n\mathbf1[q_n\le q]$，不插值，不对$G(V)$再做视频CDF。

### 4.4 Local二阶动态方向

$$a_{t,i}=p_{t+2,i}-2p_{t+1,i}+p_{t,i},\qquad u_{t,i}=a_{t,i}/\max(\|a_{t,i}\|_2,10^{-12}).$$

差分及归一化在当前特征float32中执行，评分使用float64；$a=0$得到零向量并按现有Local路径参与均值，不能套用Global的$+\infty$策略。
窗口$T=16,P=196$，$q_{L,j}=\frac1{14P}\sum_{t=1}^{14}\sum_i\ell_{LT,d}(u_{t,i})$，视频$q_L=K_{eff}^{-1}\sum_jq_{L,j}$。
不加Local Spatial，不加区域池化/匹配/位置Tail，也不使用Local窗口CDF。

对查询有效窗数$k\in\{1,2,3\}$，每条VATEX参考有$m\ge k$个FC窗时，按**保存的选择排名**取
$I_k(m)=\operatorname{unique}(\operatorname{round}(\operatorname{linspace}(0,m-1,k)))$，构造其子窗raw均值。
这是既有effective-K规则，不等同于重新运行每条参考视频的Top-$k$。
将这些参考均值组成$F_{L,d,k}$，最后$L(V)=F_{L,d,K_{eff}}(q_L(V))$。

### 4.5 方向矩解释与边界

令每窗口位置均值再窗口均值诱导的权重为$\omega_i$，$\sum\omega_i=1$，
$m_V=\sum\omega_i u_i$，$M_V=\sum\omega_i u_iu_i^T$，则

$$q_L=C_d-\tfrac12\{\operatorname{tr}(A_dM_V)-2\mu_d^TA_dm_V+\mu_d^TA_d\mu_d\}.$$

该恒等式说明评分使用视频局部方向的一、二阶矩。若所有有效向量单位范数、$\mu=0$、$A=\lambda I$，能量为常数；近零向量例外应明确。
完整协方差编码**特征通道间**相关性，不是Patch之间空间相关性；同网格D2不是物理点加速度。CLS不是Patch均值。
有限代数测试支持信息不等价，不支持fake一定异常、D2一定更好或理论AUC保证。

### 4.6 融合和阈值

$$S(V)=\tfrac12G(V)+\tfrac12L(V).$$

在无Local额外非线性拆分的当前定义下，它是GS、GT、Local三个视角约0.25/0.25/0.5的层级融合，不是各占三分之一。
所有分数越高越真实，不是真实概率，不默认0.5阈值。独立阈值视频只在所有评分规则固定后用于确定$\tau$；判fake为$S<\tau$，同分边界保持固定。

## 5. 数据集、真实来源与生成器单元

完整开发评分池21421唯一视频=9382 real+12039 fake。主配对表含13033唯一视频和20870行，20个生成器单元。
主表配对身份继承M31固定CSV；不用各实验重新抽样；已有`fake.head(...)`截取也须原样追溯，后续若更改为新的抽样协议必须另建版本。

### 5.1 真实参考与评测来源

| 大数据集 | real原始来源 | 目标拟合片段 | 拟合源组 | 完整评测real | 主配对real使用 |
| --- | --- | ---: | ---: | ---: | --- |
| ComGenVid | MSVD | 200 | 132 | 898 | 每生成器898，两个生成器复用同真实池 |
| VideoFeedback | DiDeMo | 105 | 105 | 250 | 每生成器150 |
| VideoFeedback | Panda70M | 95 | 95 | 250 | 每生成器150 |
| GenVideo | MSR-VTT | 200 | 200 | 7984 | 各生成器按fake数平衡，总唯一real1400 |

Paper必报VideoFeedback两种真实来源拆分误报率；不能只报合并500-real。拟合105/95不是评测150/150，不人为改成一样。
ComGenVid898评测与原4300行清单、GenVideo13623可用视频与16188行清单存在过滤差异，需保存排除原因及清单哈希，不用行数替代实际可用数。

### 5.2 每个生成器：新主线已有结果

以下数值直接来自`b3_development_full_v1/generator_metrics.csv`的`adapted_Global_plus_Local`。real与fake配对数每列相等。

| 大数据集 | 原始source_model键 | 可用fake | 配对real/fake各数 | 新主线AUC | AP-real |
| --- | --- | ---: | ---: | ---: | ---: |
| ComGenVid | Sora | 1700 | 898 | 0.904343 | 0.908404 |
| ComGenVid | VEO3 | 1700 | 898 | 0.929153 | 0.934799 |
| VideoFeedback | AnimateDiff | 300 | 300 | 0.843611 | 0.866544 |
| VideoFeedback | Fast-SVD | 300 | 300 | 0.912967 | 0.929094 |
| VideoFeedback | LVDM | 300 | 300 | 0.893950 | 0.906403 |
| VideoFeedback | LaVie-base | 300 | 300 | 0.824767 | 0.839257 |
| VideoFeedback | ModelScope | 300 | 300 | 0.817167 | 0.817846 |
| VideoFeedback | Pika | 300 | 300 | 0.881011 | 0.903435 |
| VideoFeedback | SoRA-Clip | 300 | 300 | 0.780750 | 0.781187 |
| VideoFeedback | Text2Video-Zero | 300 | 300 | 0.720539 | 0.737911 |
| VideoFeedback | VideoCrafter2 | 300 | 300 | 0.908728 | 0.919518 |
| VideoFeedback | ZeroScope-576w | 300 | 300 | 0.800422 | 0.821006 |
| GenVideo | Crafter | 188 | 188 | 0.890448 | 0.872908 |
| GenVideo | Gen2 | 1380 | 1380 | 0.951274 | 0.958397 |
| GenVideo | Lavie | 1400 | 1400 | 0.884987 | 0.878712 |
| GenVideo | ModelScope | 700 | 700 | 0.882981 | 0.879902 |
| GenVideo | MorphStudio | 700 | 700 | 0.881308 | 0.876878 |
| GenVideo | Show_1 | 700 | 700 | 0.874710 | 0.851140 |
| GenVideo | Sora | 56 | 56 | 0.949617 | 0.953894 |
| GenVideo | WildScrape | 515 | 515 | 0.798635 | 0.716250 |

保留原始键与展示名称的显式映射，不自动合并`Sora`/`SoRA-Clip`、`Lavie`/`LaVie-base`或不同数据集的`ModelScope`。
它们是20个dataset-generator单元，不是20个不同生成器家族。
原表出现而当前无分数的VideoFeedback Hotshot-XL、GenVideo HotShot-XL/MoonValley只记“未纳入当前可用协议”，不填0、不无依据认定下载缺失或方法失败。

### 5.3 域级新基线与操作点

| 域 | 配对AUC/AP-real | VATEX名义1%阈值下实际real FPR | 全部fake recall |
| --- | --- | ---: | ---: |
| ComGenVid | 0.916748/0.921602 | 0.3341% | 20.9118% |
| VideoFeedback | 0.838391/0.852220 | 7.8000% | 45.0333% |
| GenVideo | 0.889245/0.873510 | 1.3652% | 29.1364% |
| Macro-3 | 0.881462/0.882444 | 不将三域混成统一阈值保证 | 不与配对ROC recall混用 |
| GenVidBench Pair1 | 0.862411/0.863166 | 5.3333% | 29.3333% |
| ViF-Bench | 0.588856/0.591924 | 2.3529% | 4.2456% |

前两列分类指标和后两列完整池操作点来自不同汇总规则，表注必须说明。外部表现不允许省略；当前没有“全面泛化”的证据。

### 5.4 外部数据与修正版身份

GenVidBench当前有效协议：VRIPT real拟合136片段/115源组，评测300 real/231源组 +300 ModelScope fake（键`ms`）。
旧199拟合中63片段与评测real共享43源组，修正版仅排除这些拟合片段，不删评测。
本次已重新核验manifest哈希→参数registry哈希→score_plan身份→evaluation身份→指标CSV哈希，链条全部相符。
所以“136尚未评分”属于旧状态，不能覆盖M30已完成结果。没有fake到真实caption的完整对应关系，不声称语义彻底去重。

ViF拟合80 real/80源组，评测85 real/83源组 +1531 fake。按每生成器对应真实内容配对，不把重复real当独立视频。

| ViF原始生成器键 | fake数 | ViF原始生成器键 | fake数 |
| --- | ---: | --- | ---: |
| CogVideoX1.5-5B-T | 85 | HunyuanVideo | 85 |
| HunyuanVideo-I2V | 85 | LTX-Video-13B-I | 84 |
| LTX-Video-13B-T | 82 | SkyReels-V2 | 85 |
| SkyReels-V2-I2V-14B-540P | 85 | Wan2.1-T2V-1.3B | 85 |
| Wan2.1-VACE-1.3B-T | 85 | Wan2.2-I2V-14B | 85 |
| Wan2.2-T2V-14B | 85 | Wan2.2-TI2V-5B-I | 85 |
| Wan2.2-TI2V-5B-T | 85 | gen4-turbo | 57 |
| hailuo | 70 | kling-v1 | 72 |
| pika-v2 | 76 | pixverse-v4-5 | 78 |
| sora-2 | 77 | 合计 | 1531 |

GenVidBench与ViF已反复观察，不能称新冻结后的untouched确认；不能跨域用200/136/80片段冒充同预算。

### 5.5 VATEX角色

清理后本地保留4200视频：源域同预算fit200、CDF2000、threshold2000，身份分离。源域200清单为data/catalog/vatex_source_fit200.csv。
新主线使用独立CDF与threshold两批；本轮源/目标同预算机制对照使用fit200，不再假设历史fit5000原视频仍齐备。
官方STALL NPZ的大VATEX参考与本地4200不是同一个可替代资产，不按测试表现更换这几批身份。

## 6. 指标、表格与统计契约

1. 主指标AUC、AP-real；逐生成器→域内等权平均→三个开发域等权Macro-3。不得改为按视频数加权以获得更好Macro。
2. 补充AP-fake（若新增计算）、完整池AUC/AP及真实比例。现有0.882444特指AP-real，不能换AP口径仍保留旧数。
3. ROC操作点：Fake Recall@1%及0.1% real FPR、real FPR@95% fake TPR。阈值方向统一为低分fake。
4. 部署阈值：独立阈值real确定$\tau$后，报每域及各真实来源实际FPR、每生成器Recall。与在evaluation ROC选阈值严格分开。
5. 样本少于1000 real不声称可靠验证0.1%；ViF85 real一次误报约1.18%。给出分母、二项区间；同源片段相关时另给源级不确定性。
6. AUC/AP差值沿固定身份源组Poisson配对bootstrap，real跨生成器共享权重，ViF对应real/fake联合权重，至少1000次。
7. 旧区间条件于固定参考且未多重校正；不追认成预注册确认。新三项核心假设可预先约定Holm处理或同时区间，不从四舍五入结果算显著性。
8. 参考抽样不确定性另做fit-source重抽样；25/50/100嵌套5seed，200同池不是5次独立来源。
9. 成本：解码、1FPS粗扫、密集前向、评分、CDF、总时间；唯一帧、补齐计算帧、峰值GPU/RAM和缓存命中。批吞吐不等于单视频延迟，不能只报warm cache与旧冷启动比较。

表格输出统一字段：`protocol_id,run_id,dataset,generator,real_source,split,n_real,n_fake,n_source_groups,auc,ap_real,ap_fake,recall_at_real_fpr_001,recall_at_real_fpr_01,real_fpr_at_fake_tpr95,threshold_origin,actual_real_fpr,fake_recall,seed,scope`。
本轮仅规定schema，旧CSV不原地重写。主表每个配置必须覆盖本节20单元；缺失项显式NA和原因，不从另一run补洞。

### 6.1 已有主对照：排序与低误报分别报告

以下均取自同一`b3_development_full_v1/macro_metrics.csv`，不是原论文公开数字；百分比列按逐生成器、逐域等权宏平均。低FPR两列是evaluation ROC操作点，不是独立VATEX阈值迁移结果。

| 对照 | AUC | AP-real | Fake Recall@0.1% real FPR | Fake Recall@1% real FPR | real FPR@95% fake TPR |
| --- | ---: | ---: | ---: | ---: | ---: |
| 官方Global，同FC K3 | 0.852900 | 0.854311 | 5.10% | 16.57% | 53.47% |
| 目标适配Global，同FC K3 | 0.872422 | 0.876222 | 5.64% | 20.29% | 46.99% |
| Local D2-only，目标参考 | 0.845758 | 0.843662 | 7.07% | 22.09% | 55.69% |
| 旧B3：官方Global＋目标Local | 0.878177 | 0.877191 | 8.22% | 27.80% | 47.12% |
| **新主线：目标Global＋目标Local** | **0.881462** | **0.882444** | **8.09%** | **24.19%** | **44.72%** |

新主线相对同目标预算Global的AUC/AP增量为0.009040/0.006222；相对旧B3为0.003284/0.005253。
但新主线相对旧B3的Recall@1%下降约3.61个百分点，不能用更高AUC/AP替代“所有指标更好”的证据。
采用新主线是统一方法定义的决定，不按某一操作点再为不同数据域切换版本。

指标计算必须记录同分处理：当前实现使用离散ROC中FPR不超过预算的最大TPR，不插值；FPR@95%取达到目标TPR的最小FPR。
AP-real按real为正类计算，若另报AP-fake，使用fake标签及相反方向分数重新计算，而不是`1-AP-real`。

## 7. 围绕新基线的必要实验

### 7.1 优先级与状态

| ID | 问题/变体 | 固定项与数据范围 | 现状/最小工作 |
| --- | --- | --- | --- |
| P00 | 新主线精确复现 | 全三域20单元、当前全部参考与索引 | 本轮重提/重评分完成，Final/G/Local逐视频差为0 |
| P01 | 原版STALL原生K1 | 官方参数/官方窗口规则，同评测身份 | 本轮21421条完成，Macro0.840248/0.844552；工程batch8、官方NumPy评分与上游版本核对 |
| P02 | STALL同FC K3控制 | 仅使用官方Global，同样选窗 | 已完成0.852900/0.854311；含目标信息差异，非纯Local因果归因 |
| P03 | 同预算目标Global vs 新主线 | 同200 fit、同FC窗、同Global CDF | 已完成：Local宏增益0.009040/0.006222，区间为正 |
| P04 | 去Global / 去Local / 去GS / 去GT | 固定新Global与LT参考、窗口 | 本轮五项点表、源组区间及独立阈值表全部完成 |
| P05 | Local D1 vs D2 | **目标Global固定**，各自LT模型与匹配CDF | 本轮Gaussian、CDF、全量评价完成；D1拟合特征经新旧raw/global/索引逐元素核对后复用，其余查询重提 |
| P06 | 未归一化D2、幅度、Global D2 | 新Global固定，每种Local有自身CDF | 本轮重新拟合并重评分完成，分支单独/融合表齐备 |
| P07 | 目标/外部参考，完整/对角，均值/白化 | 同200片段预算，Gaussian变则重评分CDF | 本轮四格、交换与对角控制完成；目标Global适配后Local再适配的额外区间跨零，不搬用旧收益 |
| P08 | K1 / Uniform K3 / FC K3 | 新主线GS/GT/LT固定、每观察规则匹配Local视频CDF | 四种观察全量完成，19项区间包包含观察对照；48视频×4策略×2遍成本测试完成 |
| P09 | 来源与外部确认 | 新主线统一，逐真实来源/生成器报告 | 外部新主线结果已存在；补新baseline对应差值与分组表，不重跑全部视频 |
| P10 | 25/50/100/200双分支样本预算 | 同子集同步拟合GS/GT/LT、外部CDF身份固定 | 旧M29只缩Local，不等价；正文若声称少样本则必补，否则只作历史边界 |
| P11 | 有限后处理与错误案例 | 原始、一种压缩、一种重复帧；先冻结再跑 | 次级优先；固定参考/匹配real适配两个协议分开，案例不能只挑成功 |

P04去GS：Global剩GT；去GT：Global剩GS。顶层仍0.5G+0.5L；去整个Global或Local则剩余分支单独评分，明确不拿除以2的量纲变化包装新方法。
P08的K1定义预先写清是FC排名第一窗还是原版固定单窗，不把两者都记`K1`；可分别命名`fc_top1`、`official_single_window`。
改变selector必须重新产生匹配CDF raw；不将K3参考子窗近似等同于独立Top1校准。若保持旧effective-K规则做单因素对照，必须单列为该规则下的受控K对照。

### 7.2 三个核心假设与表格职责

- H1：同目标real及观察预算下，Local提供额外信息。主证据P03/P04，不只强调较大但混有数据优势的P02差值。
- H2：局部归一化D2优于对应D1及幅度控制。主证据P05/P06，必须换到新Global背景；不保证每域每生成器都提高。
- H3：目标real的主要作用在特征度量而非末端CDF。P07加历史M22/M26解释，历史预算不同的对照保留来源限制。
- 观察策略为效率问题P08；不要求它必须胜出才算成功完成实验，失败即降低选窗创新表述。

正文主表先STALL与新主线，第二表组件，第三表表示与参考度量，第四图观察成本，外部/失败与样本效率另成短表。
当前不新增D3/SPLIT复现实验；已完成证据和近邻工作引用不能删除。最终投稿是否把强比较放正文由作者决定，不能因结果不利撤掉原始数据。
不追加新的语义CDF、匹配、曲率、Tail、448、低秩或收缩sweep。动态覆盖selector本轮非必要，不在重构时悄悄加入主方法。

### 7.3 可直接交给后续实现者的最小实验包

本轮已执行此矩阵，实际重提/缓存复用边界与所有产物见EXPERIMENT_EXECUTION_zh.md和results/paper。下表保留复现设计，不表示当前仍缺这些结果。

所有开发实验逐项输出第5.2节原始键的20行；不能只交Macro。每行使用同一冻结pair_ids，附完整池与真实来源操作点。外部结果独立成表，不增加到Macro-3分母。

| 实验包 | 预先固定的行 | 主要输入/是否重提特征 | 完成标准 |
| --- | --- | --- | --- |
| 主比较 | 原生STALL K1、官方Global FC3、目标Global FC3、新主线 | 原生K1先核验官方资产与索引；其余已有全量分数 | 标明目标real预算、窗口与CDF来源；原生K1缺证据就标待核验，不用FC3代替 |
| 组件移除 | 完整、Local-only、Global-only、GS＋Local、GT＋Local | 当前标准每窗raw＋目标参考包；无需DINO | 每行唯一开关明确，所有20单元、域、Macro和相对完整差值齐备 |
| 表示对照 | 目标Global＋Local D1、归一化D2、raw D2、幅度、Global D2 | 旧候选各自raw与各自CDF，新Global固定；先检查候选资产能否完整导入 | 不使用官方Global背景的历史融合数；若有资产缺口，明确补评分范围 |
| 度量/参考 | 同预算外部Local、目标Local；完整/对角；均值/白化四格 | 保持目标Global和FC3；每个Local模型有匹配CDF | M26旧pilot仅作机制证据；若宣称新主线全量机制成立，需新背景全量控制 |
| 观察预算 | FC top1、Uniform K3、FC K3 | 固定目标GS/GT/LT；缺失窗口与匹配Local CDF需补提 | 明确requested-K/effective-K；冷/热缓存分报，至少记录唯一帧与端到端成本 |
| 外部/来源 | 同窗口官方Global、目标Global、新主线 | 已有GenVidBench和ViF结果，逐生成器/真实源导出 | 保留136/80 fit预算和全部失败域；不称未接触确认集 |

组件中的GS＋Local等于`0.5 GS + 0.5 Local`，GT＋Local同理；它们与完整模型的差异包含移除后重新分配Global内部权重，应这样标注，不声称纯粹“关一项但权重完全不变”。
额外的Global-only GS/GT可用同一raw顺便导出，但不扩展为新的GPU实验。无需为表格整齐重跑已有正确视频分数。

推荐执行顺序：先验证P00和来源身份，再导出主对照及组件移除，然后做新Global背景的表示对照，最后才补原生STALL和预算缺窗。
用这个顺序尽快形成完整论文主表，而不是先清空研究脚本后发现还要恢复它们来取数。

### 7.4 不随重构静默修正的两项敏感性

1. **Global参考选窗不完全匹配。** 当前目标Global的CDF来自VATEX Uniform首窗，测试来自FC3。若检验FC匹配Global窗口CDF，另设敏感性协议、保持Gaussian和Local不变；它不是复现当前基线的默认步骤，也不是恢复Global视频CDF。
2. **200片段不等于200独立源。** ComGenVid拟合200片段来自132源组。若论文主张对独立视频的样本效率，应另建源组抽样协议；不得把历史每片段等权估计改成源组等权后仍引用0.881462。

二者都应披露，但当前不是阻止整理文档的原因。改动前冻结方案，结果不佳也保存；不强制开新的参数扫描。

### 7.5 新主线组件移除结果

历史来源ID：`main_components_v1`，现保存在results/reference/comparison_tables.csv.gz及历史总结中。输入为五域23637条冻结分支分数，开发指标仍限定20单元的Macro-3。没有重新拟合或重新抽取配对。

| 变体 | 实际Final | Macro AUC | AP-real |
| --- | --- | ---: | ---: |
| 完整主线 | 0.5G＋0.5L | 0.881462 | 0.882444 |
| 去Local | G | 0.872422 | 0.876222 |
| 去Global | L | 0.845758 | 0.843662 |
| 去GS | 0.5GT＋0.5L | 0.872380 | 0.869836 |
| 去GT | 0.5GS＋0.5L | 0.875545 | 0.881204 |

完整模型五项点估计中最高，但不能由此声称各项均显著或各域一致。
`components_without_gt_bootstrap_v1`的1000次源组区间显示full减去GT的Macro AUC差0.005917，区间[0.002874,0.008870]；AP差0.001240，区间[-0.001414,0.004003]，跨零。
GenVideo的同一差值为AUC -0.010483、AP -0.017847，主要正向收益来自VideoFeedback。论文必须保留这种域依赖，不为改善单域数字切换模型。
上述历史表当时未包含全部区间/阈值；本轮已在results/paper补齐，正式论文以本轮导出为准。

## 8. 论文段落脉络

1. 引言：不微调/不训练真假分类器的设定；提出局部方向与真实度量的具体问题，承认目标real可用条件。
2. 相关工作：明确继承STALL的部分、与标量时序/局部粗糙度的区别，不声称已有工具首次出现。
3. 方法：先数据职责，再三个方面的动机→公式→算法，最后必要CDF与融合；不把M01–M32按时间抄入正文。
4. 实验设定：原始来源、片段/源组、过滤、固定配对、参数出处、标签用途、成本和指标定义。
5. 结果：主比较→公平组件→局部表示→参考层归因→预算；新主线消融与历史机制分开展示。
6. 外部/限制：ViF低性能、VideoFeedback部分下降、阈值漂移、源组依赖、既有开发选择；不把失败只藏在附录。
7. 结论：局部方向补充及适用条件，不声称统一最优或已解决泛化。

理论推导只作为解释评分保留的信息；实验优势和代数正确性分别验收。正式稿尚未生成，本文件是作者已给定框架的规划，不代表三项均获同等原创性证明。

## 9. 证据入口与后续第一步

- 当前分数与窗口raw：`results/reference/baseline/`，不需要旧run或源码。
- 当前GS/GT/LT与CDF：`precomputed/target_reference/`；原版STALL参数单独保留。
- 精选对照表：`results/reference/comparison_tables.csv.gz`，按history_experiment/history_table筛选。
- 可复用原始评分：`cache/scores/`；旧参数和清单按`data/catalog/path_migration.json`映射定位。
- 外部当前清单：`data/manifests/active/genvidbench`、`vifbench`，源组规则不因清理改变。
- 唯一历史方法、配置和逐子集结果：`docs/All_Branches_Experiments_and_Data_Summary_zh.md`。

后续直接用当前代码和缓存开展新实验，不恢复旧源码/报告/迁移快照。复用缓存必须匹配实际特征、采样和统计合同，不仅看路径名称。

### 9.1 本次核验锚点

以下是清理前核对的历史文件SHA256，仅作来源标识；不要求已删除的旧路径继续存在。原数值已写入统一总结和精选对照表：

| 文件 | SHA256 |
| --- | --- |
| `results/runs/b3_development_full_v1/macro_metrics.csv` | `c1b105bacebeba37b9f36d97d31d4d82476744782a455fe53a9c1f7eaacac822` |
| `results/runs/b3_development_full_v1/generator_metrics.csv` | `7ef1cb560a223a66d080a4fd952fd4236f690b5c876cfbb6dd2eda95f12ffb49` |
| `results/runs/b3_external_validation_v1/dataset_metrics.csv` | `3945d4e5ae6f290632e8dcbdb7ba1191ebfb7b50d0b4e7d0112dd4a31f30d9a1` |

当前baseline数值与清理前一致，工程副本已删除。新的对比应使用精简参考包或重新评分，不能把历史来源ID当成可恢复run目录。
