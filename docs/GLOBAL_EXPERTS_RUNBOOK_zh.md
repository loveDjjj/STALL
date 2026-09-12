# Global统计专家研究运行说明

分支：research/global-statistical-experts。研究目录src/statistical_experts，配置configs/global_experts.yaml；与原论文Local/FC入口分开。

## 数据与算法

- fit：旧VATEX source-fit200＋旧threshold2000，共2200；CDF为另一批2000。
- 每种时长独立建立模型/CDF，使用同样视频身份划分。评价沿results/paper_complete的23单元及配对。
- Global输入为官方单窗8/16帧、224、batch上限32、不补尾批。Global缓存float32；T1差分/归一化按官方NumPy float32规则；路由和统计float64。
- 空间拟合帧在整个fit池中一次性固定，之后簇/邻居只选择视频，不重抽空间帧。
- 全体、离线4专家、在线相似512、在线随机512，固定收缩0.5及ridge1e-5，共同1024维。
- B式整流程CDF，两分支百分位等权；没有Local、FC多窗或独立部署阈值。

## 执行入口

在STALL仓库根运行，使用stall环境。以下是阶段命令，已有完整run不要重覆盖；复用前先核对配置与输入hash。

```bash
export PYTHONPATH=src
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=2
python -m statistical_experts.run prepare
python -m statistical_experts.run cache --pilot --rank 0 --world-size 2
python -m statistical_experts.run cache --pilot --rank 1 --world-size 2
```

全量cache用rank0/1/2/3、world-size4并行运行，rank%2对应GPU0/1。CPU有界解码预取，顺序解码后核对原版摘要锚点，必要时回退索引解码。

```bash
python -m statistical_experts.run cache --rank 0 --world-size 4
python -m statistical_experts.run fit --length 8 --rank 0
python -m statistical_experts.run fit --length 16 --rank 1
python -m statistical_experts.run score --length 16 --rank 0 --world-size 4
python -m statistical_experts.run score --length 8 --rank 0 --world-size 4
```

示例中的cache/score rank0不是完整任务；必须运行全部四个rank。每个rank先16后8，四rank并行；统计查询固定batch4，最后补计算查询后裁回真实输出。检查点异步写入但完成标记须等待全部写入成功。

```bash
python -m statistical_experts.run evaluate
python -m statistical_experts.run analyze
python -m statistical_experts.run diagnose
python -m statistical_experts.run measure
python -m statistical_experts.run verify
python -m statistical_experts.run report
```

analyze为CPU并行源组bootstrap，同时导出仅替换空间/时序的组合。measure需要GPU，计时仅为已有Global后的统计+C​​DF，不含编码和解码。diagnose在fit内部1760/440划分，重新拟合路由与统计，不读取fake。

## 完成判定与恢复

- cache 23969窗口；统计查询19569（4000 CFD窗口＋15569评价片段身份）。
- 每个长度raw目录的所有rank完成后才能evaluate；未完成、hash变化或非法邻居会报错。
- 已有缓存和检查点均按身份核验，禁止改参数继续写同一run。配置变化应新协议/新run，不修改已完成模型文件。
- 缓存身份包含源码与窗口manifest，不能改cache.py后强行沿用旧identity；修改读取/写入方案需做显式兼容审计。
- cache/score再次运行已完成分片可能写新的“复用”完成标记；正式计时以已保存progress与最终manifest为准，不把复用时间当重提时间。

## 已交付资产

- data/manifests/global_experts：新角色、窗口、拟合空间帧及配对。
- cache/global/stall_single_windows：约1.4GiB Global缓存，不含Patch。
- results/runs/global_experts/models_8、models_16：总体/离线模型、簇支持。
- raw_8、raw_16：四算法raw及实际512邻居索引，含原子payload哈希。
- evaluation：23单元、三域、Average、分支、混合分支、CDF数组和区间。
- real_diagnostic、runtime：独立真实留出诊断、60片段×4方法×2遍统计计时。
- verification.json、manifest.json、RESULTS_zh.md：验收与最终报告。

原论文源码/配置/文档/清单和tracked改动在results/runs/global_experts_setup中保存了开启研究前的快照。当前分支继承原工作区未提交改动；本任务未推送远程，不将原论文结果写成新研究成绩。

## 官方空间与时序专家CDF后续对照

使用相同Python环境与`PYTHONPATH=src`，先准备独立run，两个长度可在两卡同时评分；不重提Global、不拟合新Gaussian。

```bash
python -m statistical_experts.run controls-prepare
python -m statistical_experts.run controls-score --length 8 --rank 0 --world-size 2
python -m statistical_experts.run controls-score --length 16 --rank 1 --world-size 2
python -m statistical_experts.run controls-evaluate
python -m statistical_experts.run controls-analyze
python -m statistical_experts.run controls-verify
python -m statistical_experts.run controls-audit
```

配置`configs/global_expert_controls.yaml`；输出`results/runs/global_expert_controls`。`cdf_scores_8/16.npz`保存全部专家对同一2000参考的时序raw，原路由列须与首轮raw逐位一致。`video_scores.csv.gz`含10配置，`generator_metrics.csv`含每配置23单元，`confidence_intervals.csv`含9个源组配对对比。`source_snapshot`仅绑定这次统计分析代码，不复制原视频或大型特征库。

B表示CDF视频各自路由后的整体算法排名；A表示固定查询专家对全库评分后的排名。两者参考视频身份相同，不把它误称为新增一层CDF。

时序收益门槛通过后，用同一2200拟合参数、同一路由完成均值/协方差四格。四组各自重建B式8/16帧CDF；不利用A/B表现重新选择四格的校准方式。

```bash
python -m statistical_experts.run moments-score --length 8 --rank 0 --world-size 2
python -m statistical_experts.run moments-score --length 16 --rank 1 --world-size 2
python -m statistical_experts.run moments-evaluate
python -m statistical_experts.run moments-analyze
python -m statistical_experts.run moments-verify
python -m statistical_experts.run controls-audit
python -m statistical_experts.run controls-report
```

四格位于同run的`moments/`，含总体/专家均值与总体/专家协方差四种组合，输出raw-only、时序校准后及官方空间等权融合；5个配对对比。2条全静态评价片段的raw `+inf`原样保留在`raw_temporal_score`，raw-only排序指标用保序且保同分的有限秩计算，绝不把该秩用于模型融合。原版边界没有排除视频或随意裁剪分数。

当前后续对照全部完成，报告为`results/runs/global_expert_controls/RESULTS_zh.md`。测试及完整产物hash由该run的`tests.xml`与`manifest.json`记录。`moments_numerical_check`只有首次浮点调用形状检查的失败身份，不纳入结果；正式四格恢复首轮广播形状，端点raw/Final逐位一致，未扩大容差。

## 同簇CDF与连续精度组合

配置`configs/global_expert_routing.yaml`，独立目录`results/runs/global_expert_routing`。只有四个新增检测配置：总体/专家的同簇C校准，以及固定平均/连续内容精度。共享均值hard使用既有协方差-only锚点；温度按2200 fit内容的第一、第二相似度差中位数确定，下限1e-6仅处理退化数值。

```bash
python -m statistical_experts.run routing-prepare
python -m statistical_experts.run routing-score --length 8 --rank 0 --world-size 2
python -m statistical_experts.run routing-score --length 16 --rank 1 --world-size 2
python -m statistical_experts.run routing-evaluate
python -m statistical_experts.run routing-analyze
python -m statistical_experts.run routing-verify
python -m statistical_experts.run routing-report
```

两个长度可双卡并行；不重提DINO或重新拟合Gaussian。组合必须在逐转移能量上进行，之后mask零T1并取min；保存的positions只占轻量空间，不能用原窗口min标量直接代替。CDF人口对照与评分器对照独立，不把连续度量同时套入C校准。报告前需tests.xml、独立公式/全量百分位验收与10组1000次配对区间齐备。

测试使用`python -m pytest tests -q --junitxml=results/runs/global_expert_routing/tests.xml`。根目录pytest.ini将默认收集范围限定为tests，避免results中的源码快照被重复计为新测试。

## 原版单窗T1参考来源诊断

配置`configs/global_reference_sources.yaml`，产物`results/runs/global_reference_sources`，新目标Global缓存`cache/global/target_single_windows`。每源一片段：ComGenVid132、VideoFeedback200、GenVideo200；同预算VATEX取旧source-fit200固定随机排列的前132/200，来源与CDF隔离。

```bash
python -m statistical_experts.run source-prepare
python -m statistical_experts.run source-cache --rank 0 --world-size 2
python -m statistical_experts.run source-cache --rank 1 --world-size 2
python -m statistical_experts.run source-fit --length 8 --rank 0 --world-size 2
python -m statistical_experts.run source-fit --length 16 --rank 1 --world-size 2
python -m statistical_experts.run source-score --length 8 --rank 0 --world-size 2
python -m statistical_experts.run source-score --length 16 --rank 1 --world-size 2
python -m statistical_experts.run source-evaluate
python -m statistical_experts.run source-analyze
python -m statistical_experts.run source-verify
python -m pytest -q --junitxml=results/runs/global_reference_sources/tests.xml
python -m statistical_experts.run source-report
```

同阶段两个rank/长度可双卡并行；后续阶段需等上一阶段完整结束。source-fit同时计算源级5折留出NLL，包含logdet，仅诊断、不选参；正式检测使用全拟合集。所有Gaussian分别重评分相同2000独立VATEX参考。官方空间与原版评价窗口不变。

target_sources/target_windows记录原始视频身份、源内片段选择、5折和原版窗口；新旧Global缓存合同严格分开。该实验显式使用目标真实信息，不是zero-target版本，也不是严格理论上界。raw-only排序指标采用保序有限秩处理+inf，不参与融合。

## 冻结外部六行及研究收束

输入为active/genvidbench与active/vifbench的原始配对；每源取一拟合片段，共195拟合源和2216评价片段。独立输出results/runs/global_external，native Global缓存cache/global/external_single_windows。当前结论为停止本轮专家细化，不替换原论文主线。

```bash
python -m statistical_experts.run external-prepare
python -m statistical_experts.run external-cache --rank 0 --world-size 2
python -m statistical_experts.run external-cache --rank 1 --world-size 2
python -m statistical_experts.run external-score
python -m statistical_experts.run external-evaluate
python -m statistical_experts.run external-analyze
python -m statistical_experts.run external-verify
python -m statistical_experts.run external-report
```

同阶段cache两rank可双卡并行。统一CLI的评价入口为external_audit.evaluate，使用显式gt列名，避免与Pandas.gt方法冲突；external_study中的初始评价函数仅随冻结提取/评分源码身份保留，不作为执行入口。这个表格读取修正未改变raw或cache，不需重新提取。

7组对比各1000次源组Poisson区间。ViF按83内容源联合重抽样；两外部域分开报告，不构造新的“有利Average”。完整结论见GLOBAL_EXPERTS_PLAN_zh.md第25节和该run的RESULTS_zh.md。

## 相邻T1条件预测（新假设，非旧专家复活）

配置configs/global_predictive.yaml；独立目录results/runs/global_predictive。拟合VATEX2200，CDF独立2000；每长度1760/440真实内部留出只作诊断。三组为匹配预测位置的边际T1、真实相邻条件模型、跨视频源打乱的同复杂度模型。1024维、固定ridge1e-5，不新增PCA、专家或正则搜索。

```bash
python -m statistical_experts.run predictive-prepare
python -m statistical_experts.run predictive-fit --length 8 --rank 0 --world-size 2
python -m statistical_experts.run predictive-fit --length 16 --rank 1 --world-size 2
python -m statistical_experts.run predictive-score --length 8 --rank 0 --world-size 2
python -m statistical_experts.run predictive-score --length 16 --rank 1 --world-size 2
python -m statistical_experts.run predictive-evaluate
python -m statistical_experts.run predictive-analyze
python -m statistical_experts.run predictive-verify
python -m pytest -q --junitxml=results/runs/global_predictive/tests.xml
python -m statistical_experts.run predictive-report
```

同阶段两种长度可双卡并行。16帧阶段还复用全部外部native Global，因此一次获得三个开发域23单元和两个外部域20单元。全部三模型在同样的有效前后T1对上评分，均先min后各自CDF再与官方空间等权；不能让边际用15次极值、预测只用14次而称单因素归因。拟合与NLL保留零向量；查询前后T1任一为零则该对不评分，全无有效对返回+inf。边际是B=0的同位置控制；打乱保留x/y边际、禁止同源自配对。

## 联合评分的单次信息保留检查

独立目录results/runs/global_joint，复用global_predictive的全部模型和查询，只拟合一个共同x边际。对三组分别逐对相加后再min；不平均已有窗口标量，不改变mask，不再增加预测专家。

```bash
python -m statistical_experts.run joint-prepare
python -m statistical_experts.run joint-score --length 8 --rank 0 --world-size 2
python -m statistical_experts.run joint-score --length 16 --rank 1 --world-size 2
python -m statistical_experts.run joint-evaluate
python -m statistical_experts.run joint-analyze
python -m statistical_experts.run joint-verify
python -m pytest -q --junitxml=results/runs/global_joint/tests.xml
python -m statistical_experts.run joint-report
```

两个长度可并行。评分时重算原有条件raw及mask并逐位核对，官方空间始终固定。三组各自重建相同2000真实参考CDF，开发Average与外部域分别报告。
