# 循环时空证据检测：实现与实验合同

**2026-09-11状态：用户已停止本路线。** 调度器状态为`stopped`，训练进程已退出，检查点、分数和Patch保留。下文启动命令与120模型矩阵是历史执行合同，不代表当前应继续训练。当前研究返回[真实参考Local D2](REAL_REFERENCE_D2_RESEARCH_PLAN_zh.md)，不得自动续跑Looped。

## 当前执行矩阵与无人值守命令（十配置）

最新用户决定：扩展为十配置，并全部从epoch1/update0重新训练。旧五配置中断记录位于interrupted/before_ten_models，未混入本轮结果。17785份FP32 Patch以及视频、源组、配对、窗口清单hash保持不变；ten_model_restart.json记录核对证据。原五模型实现/流水线描述在下文仅解释演进，当前规模为120模型、12组。

| 配置键 | 宽度/头数 | 更新次数 | 参数共享 | 可训练参数 | 唯一变化 |
| --- | --- | ---: | --- | ---: | --- |
| looped | 256/4 | 4 | 是 | 1189121 | 完整模型 |
| untied | 256/4 | 4 | 否 | 3565313 | 四个独立更新块 |
| no_refresh | 256/4 | 4 | 是 | 1189121 | 每轮读当前状态，不重读固定观察 |
| spatial_only | 256/4 | 4 | 是 | 1189121 | 取消跨帧交互 |
| state_only | 256/4 | 4 | 是 | 1189121 | 最终只读状态，读出参数量不变 |
| loop1 | 256/4 | 1 | 单块 | 1189121 | 训练/测试均1次 |
| loop2 | 256/4 | 2 | 是 | 1189121 | 训练/测试均2次 |
| loop8 | 256/4 | 8 | 是 | 1189121 | 训练/测试均8次 |
| wide512 | 512/8 | 4 | 是 | 4211201 | 扩大宽度，保持每头64维 |
| no_time_position | 256/4 | 4 | 是 | 1189121 | 保留空间位置，时间正弦/余弦分量置零；跨帧注意力保留 |

循环次数是一次前向的更新深度，全部模型训练预算仍20epoch。模型配置来自configs/looped_video.yaml的experiment_overrides，未知覆盖项拒绝；不再把所有变体笼统写作R4或118.9万参数。

执行顺序为seed17的comgenvid→videofeedback→genvideo→pooled，再seed29同序、seed43同序。每组十模型同时驻留、依次计算，共用供数及H2D；参数/optimizer独立。三个LODO分别只测试对应留出域；pooled训练模型在GenVidBench/ViF冻结测试，不提供额外目标拟合。

每组流程：20epoch→各自最佳验证checkpoint→共用测试供数→重新组批的六视频前向探针→每模型逐生成器/域级AUC、AP-real、AP-fake和ROC操作点→源组单seed配对bootstrap→代码/数据/模型/选模/覆盖验收→组级MD报告和全局部分汇总→下一组。

完整三开发域齐备后才生成该seed的Average；不足的域/seed不填零、不冒充完整结果。组级区间条件于单seed，全部120模型完成后补三个seed平均指标差的区间。组总wall time单列，不相加模型重叠wall time。更多循环或宽模型增加计算，不能承诺120模型一夜跑完。

```bash
bash scripts/run_looped.sh start   # 后台启动或断点续跑；检测现有父/子进程，防止重复占卡
bash scripts/run_looped.sh status  # 实际PID、组、训练/验证/测试阶段与进度
bash scripts/run_looped.sh plan    # 12组执行顺序
bash scripts/run_looped.sh stop    # 停止精确父进程，保留各模型最近25更新检查点
bash scripts/run_looped.sh report  # 未运行时重新汇总已有结果；运行中由调度器自动执行
```

start使用脱离终端的后台进程，stdin关闭、stdout/stderr追加group_training.log；系统盘不存数据副本。不做无限失败自动重启，异常记录pipeline.json并停止，需要修正后同start命令恢复。postprocess未验收通过就不切下一组，不把训练进程退出0当全部实验完成。

结果入口：results/runs/looped_video/RESULTS_zh.md（部分/完整总表），groups/<fold>__s<seed>/RESULTS_zh.md（每组十配置），generator_metrics.csv，seed_summary.csv，average_summary.csv，group_costs.csv；各组confidence_intervals.csv、postprocess.json保留全精度证据。

本轮直接实现用户选定的独立有监督Looped Transformer方案。冻结DINOv3，训练投影、共享关系块与读出；不依赖Gaussian/CDF、MoE或Local D2。旧论文及完成研究只作为已有参照，不覆盖旧结果。

## 当前优化：同批输入供给五个独立模型

最新执行决定：用户明确要求五模型全部从头开始。旧Looped已完成更新不接入本次正式运行，移入interrupted/before_shared_batch/training。所有变体从epoch1、update0开始，从第一批就五模型活跃；共享同一批已完成的Patch，不重提特征。下文cursor接续仅为中断恢复能力，不代表本次采用旧Looped领先进度。

启动入口为`NUMPY_MADVISE_HUGEPAGE=0 PYTHONPATH=src python -m looped_video.group_train`。按相同fold、相同seed组成五变体组，每批Patch只读取、拼批和传到GPU一次，随后分别对各模型执行前向、反向和独立AdamW更新。五模型继续分别使用两卡DDP，不是一个五头集成，也不共享可训练投影或参数。没有dropout等训练中跨模型随机状态；每模型初始化前重置相同指定seed，保持与单独运行一致。

每模型训练数据、全局抽样顺序、有效batch32、更新次数、预热/余弦调度、尾批权重和20epoch预算不变。已经完成的更新由各自cursor跳过，其他模型先补齐；不能拿较后epoch的模型验证较早epoch。每模型仍独立选择最佳验证epoch。训练结束后，各自加载最佳模型，共用一次测试供数但分别保存分数；测试不参与选模。

原单模型代码和Patch保持；新group_train有明确执行身份。迁移已有未完成checkpoint时，只更新身份字段，模型、optimizer、统计计数、epoch/cursor全部逐张量hash一致；原小checkpoint备份为*.pre_group，收据group_migration.json。已完成模型禁止换身份。

组级状态/进度/真实wall time位于results/runs/looped_video/groups/<fold>__s<seed>。每模型active wall time包含共用I/O及其他成员计算，彼此重叠，不能相加当总硬件成本。样本/指标仍按原60模型矩阵输出，不因共享读取合并seed或计算集成分数。

验证包括：模型参数存储互不重叠；真实双GPU下，独立训练与共享批次的模型参数、AdamW状态和验证指标一致（浮点容差固定）；已有成员跳过完成更新、其余四变体正常补齐；所有模型独立测试输出。部分catch-up阶段仅4模型活跃，是保留旧Looped更新的预期行为。

## 当前执行方式：双卡同模型

传输优化已改为每rank两个预分配共享槽及两个锁页槽；直接把原始帧按冻结窗口索引写入槽，队列只传小型元数据。锁页复制完成才归还共享槽，下一次复用锁页槽前等待其H2D事件；GPU张量用record_stream确保生命周期。无效填充内容在模型输入处按掩码归零，不参与观察。纯内存同批次拼批/共享探针从约0.32秒降到约0.02秒，不代表冷盘端到端训练同比加速。

27项测试通过，包含固定槽复用、陈旧/NaN填充屏蔽、越界拒绝和真实两GPU训练。第1epoch全部322次更新的模型/optimizer/统计计数通过逐张量hash保持，迁移只改变生产代码身份和传输实现，收据位于对应训练目录transport_migration.json。原检查点小副本保留，可回溯。中途恢复后的epoch_seconds/train_seconds是当前执行段，整体耗时应使用累计seconds，不能把恢复后只跑验证的短时段当整轮训练耗时。

后续硬件排查：在进程供数而GPU和磁盘同时空闲的时段，观察到大量minor faults、内核CPU占用及透明大页整理失败。纯内存同操作探针启用NumPy大页约0.42–1.88秒，关闭约0.32–0.33秒（各3次，不能当全程提速比）。当前命令增加`NUMPY_MADVISE_HUGEPAGE=0`，从已有优化器更新检查点续跑；不改系统级THP、不改数据或模型。详细记录results/runs/looped_video/runtime_memory_tuning.json。

用户要求停止I/O受限的两模型并行并优化重跑。两个早期单卡seed尚无完整epoch/checkpoint，记录移入results/runs/looped_video/interrupted，不作为正式训练结果。Patch缓存和原准备清单不修改。

当前入口：`PYTHONPATH=src python -m looped_video.ddp_train`。两张卡DDP训练同一模型，完成后测试并运行下一seed/变体。NCCL禁用不可用的P2P，单机文件握手避免主机名反查延迟。旧looped_video.pipeline不再是当前训练入口。

全局batch仍为32：每卡8视频、两次microbatch，DDP梯度均值通过`world_size/global_actual_count`校正。尾部无真实样本的rank采用零损失占位，不增加训练权重；同一个全局EpochSampler序列分片，数据抽样和学习率更新次数不变。

单个CPU供数进程使用最多48GiB常驻FP32热数据，两GPU通过有界共享队列消费，避免两套随机读盘。原始文件按唯一视频去重读取，训练采样重复次数不变。热集合按训练访问概率/字节预算选择，不根据真假检测结果选择；无额外系统盘副本、无精度压缩。两读取线程配合内存拼批，异步锁页传输与下一批预取重叠。热集合惰性建立，不先等待48GiB整库复制。

每25个优化器更新保存model/optimizer/epoch/next_update及分rank累计loss/count，epoch结束另存恢复点。中断恢复重建同一抽样序列并跳过已完成更新；模型没有训练时dropout等额外随机操作。记录当前实际DDP父/子PID、更新进度、RAM命中及请求读取字节数（不是物理硬盘I/O字节）。每个完整模型仍固定20epoch，验证只选epoch，所有60个预定配置保留。

25项测试覆盖原始观察、循环展开、掩码、RAM数值等价、DDP样本分配及尾批梯度归一化，并用真实两GPU完成两epoch/检查点测试。生产训练吞吐需按真实数据日志持续更新，不能把内存热命中的短时速度当冷启动全程速度。

最新核验调整：用户要求足够时直接进入后续阶段。每个Patch已经在生产时检查有限值、写入后完整读取计算SHA256并记录size/mtime；开发Global完成逐元素回归。停止阻塞训练的第二次整库正文重读，保留已完成前缀的日志证据。训练前仍完整检查17785条的生产者、源视频stat、缓存stat、NPY头部、索引及形状。状态明确为ready_for_training，不称二次全盘hash已完成；记录verification_transition.json。训练后不要用旧verify_cache入口覆盖被模型引用的不可变manifest。

## 方法

输入为现有FC最多3个窗口，每窗口8或16帧、224输入、14×14 Patch、1024维。窗口分别处理，不将FC排名顺序当连续视频时间。短帧补零只服务批计算，掩码排除其注意力键和最终聚合；每视频窗口logit均值后计算一次BCE。

固定本次前向的观察 E=Projection(LayerNorm(X))+时间/行/列正弦位置，H0=E。每轮共享同一个空间注意力、原始观察读取和FFN：

1. 当前状态在每帧196个位置进行空间交互；
2. 当前状态查询原始观察：2×2空间组内的所有有效窗口帧；
3. Pre-Norm和初始化0.1的可训练通道残差门更新状态。

该小空间组允许跨位置时间交互；空间层提供跨组交互。它不是物体轨迹或显式物理对应。原始观察的共享K/V在一次前向中计算一次，四轮梯度共同回传；不跨优化步骤缓存可训练投影。

固定4轮、宽256、4头、FFN扩展2。最终在每个位置连接原始观察与最终状态，经同一个小MLP得到局部logit，先有效位置平均、再窗口平均得到视频fake logit。分类只监督最终输出；不强制逐轮收敛、置信提高或real/fake产生不同收敛轨迹。高为real的评价分数为负fake logit，避免sigmoid极端饱和造成额外同分。

## 固定矩阵

| 名称 | 唯一结构变化 |
| --- | --- |
| looped | 完整四轮共享、每轮读取原始观察、联合读出 |
| untied | 四轮参数独立，仍每轮读取原始观察 |
| no_refresh | 读取当前状态，不反复读取固定观察 |
| spatial_only | 相同空间组只读本帧，没有跨帧交互 |
| state_only | 最终连接两份最终状态，替代观察/状态联合读取；读出参数量相同 |

五变体×三个种子17/29/43×三个留域fold及pooled，共60模型。模型全部保留，不按测试域挑变体。相同深度不等于相同参数预算；头参数和DINO总参数分别报告。后续若需要同参数非共享对照，明确另列，不把当前untied冒充两种公平条件同时满足。

AdamW lr=3e-4、weight decay=.01；20epoch，2epoch预热后余弦调度。microbatch8视频、累积4、有效batch32；部分尾批按实际视频数归一化。梯度裁剪5。训练头BF16，DINO提取FP32且TF32关闭。完整数据及参数先冻结，训练侧验证的域均值AUC/AP-real平均仅选择epoch，不使用测试选epoch。

局部logit、窗口及视频的平均采用FP32，避免BF16累加输出进一步量化排序。KV复用只在一次前向内发生，不跨优化步骤复用可训练输出。

第一阶段直接训练完整Looped，再完成固定对照。时间错位辅助任务和可变循环预算暂不接入第一版，以免结构和训练目标同时变化。

## 数据

复用results/runs/discriminative_moe的length_matched_supervision_v2源级用途：开发15569身份、外部2216身份。三个留域fold的测试只来自留出域；pooled仅做训练/验证并用于外部。所有同源短长片段沿既有split_group成组隔离。训练保持域/真假/fake生成器等权，real有效长度质量匹配fake，长度内源等权。所有变体共享采样规则和验证配对。

开发主评价仍使用23个dataset-generator固定配对单元。指标报告AUC、AP-real、AP-fake、ROC低FPR召回、每域及Average；另报Brier/NLL用于概率诊断。负logit不是校准真实性概率。外部两域单列；LODO不自动等于未知生成器家族，已观察外部也不称untouched。

## 存储与执行

新代码src/looped_video，配置configs/looped_video.yaml，结果results/runs/looped_video，缓存cache/patch/looped_video。唯一帧FP32缓存约282.67GiB（开发+外部），所有模型复用。身份绑定原视频stat、帧索引、编码器hash、batch8/尾批及源码；开发Global逐元素回归旧缓存。完整提取后验收全部文件SHA256，训练读取时核对size/mtime，最终再验收。

按用户授权回收旧GenVideo Uniform评价分片约396GiB，清单data/catalog/looped_cache_reclamation.json。它们不被新FC训练读取；原视频、既有Global、D2拟合资产、模型及结果保留。旧packed索引保留来源身份，但其中退役条目不再可读，不宣称旧630GiB缓存完整。恢复删除的Patch需要从可用原视频重新提取。

两张GPU各自执行独立任务，尽量相邻seed共享相同数据的OS页缓存；每卡6解码线程/8视频预取/6GiB解码预算，训练4加载进程/进程预取2批。无任务时不能凭旧progress认定活跃，检查实际PID。每epoch保存完整optimizer续跑点；相同输入/配置/源码才可续跑。

性能核验发现/data为机械盘。24条跨域/真假/长度样本顺序解码与原路径逐像素一致，合计9.99秒→2.19秒。现采用顺序解码，开发Global逐元素检查；外部Global摘要逐元素检查；不一致回退原路径。早期约2600条随机解码缓存经显式producer_compatibility链保留，旧分片不伪造新producer。顺序优化不改变FP32数值合同。

用户随后明确系统盘空间紧张：NVMe副本计划已取消，未执行任何复制，也未创建相应缓存目录。所有持久缓存、模型和结果留在/data；自动流程及CLI已移除系统盘缓存步骤。训练使用有界CPU加载/预取及操作系统页缓存，不改变FP32输入精度。此项只取消存储副本，不改变模型、数据或实验矩阵。

自动执行器`PYTHONPATH=src python -m looped_video.pipeline`在每模型完成并冻结后立即做对应留域/外部测试，再启动下一项。只有全部60项完成才运行完整配对区间；部分结果明确标记。pipeline.json记录父PID及实际worker PID，异常保留epoch续跑点。

```bash
PYTHONPATH=src python -m looped_video.run prepare
PYTHONPATH=src python -m looped_video.run extract --rank 0 --world-size 2
PYTHONPATH=src python -m looped_video.run extract --rank 1 --world-size 2
PYTHONPATH=src python -m looped_video.run verify_cache
PYTHONPATH=src python -m looped_video.run train --rank 0 --world-size 2
PYTHONPATH=src python -m looped_video.run train --rank 1 --world-size 2
```

提取、模型训练、测试评分、bootstrap与最终报告是不同状态。不得把工程探针或部分epoch写成最终检测结果。完成要求包含所有预定模型、同身份配对指标、组件差值及源组区间、外部冻结测试、输入/模型验收和失败结果说明。
