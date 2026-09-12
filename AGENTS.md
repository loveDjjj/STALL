# 后续AI工作入口

## 最新写作：精简IEEE论文正文（2026-09-12）

paper/ieee_alpha_stalled/main.tex已按当前23单元的目标Global＋Local D2更新为中文科研论文稿，三张主表由export_tables.py从冻结CSV导出，bash paper/ieee_alpha_stalled/build.sh编译。主成绩仍为三域Average 0.874472/0.877075。正文围绕局部方向增量与匹配控制，不逐项叙述历史实验、运行恢复或缓存清理；关键结果边界仍保留。新流程图按用户要求留空；旧图、旧表不在当前输入链中。作者信息及最终公开地址由作者填写，不视作已经投稿。完整证据稿保留在docs/MANUSCRIPT_REVISION_PLAN_zh.md，不能再将其全部细节直接拼入正文。Looped保持停止。

## 最新执行：独立新真实池与重编码验证（2026-09-11）

两项现已完成并写入论文。新池5次平均Full=0.876140/0.878616，Global=0.866419/0.874679；Full对Global仅AUC区间正、AP跨零，D2对D1两项正；源等权没有重现原池收益，默认主线不变。原池位置数匹配D2=0.874697/0.877485，仍优于pooled宏平均。重编码固定1534身份：clean Full=0.874042/0.878552，CRF23=0.873217/0.873741，CRF35=0.859838/0.858934；重压缩损失明确，Local交互正但直接增量区间跨零。新池/编码独立审计分别690/483行指标通过；全部进程退出。不要根据下面历史中断记录再次启动或重跑。

编码曾在约82%因WildScrape/D467默认编码时间基偏差>1ms停止；现已验证保留输入时间基回退，未放宽检查、未删除难例。原评分/评价函数AST不变，兼容记录在`encoding/timebase_compatibility.json`，旧identity及已完成块不重写。继续状态以`encoding/status.json`的修复调度PID为准，不恢复已失败的旧父PID；rank0已补完，rank1从1200/1514断点续跑。新真实池本身已完成。

用户已授权实现结果分析后的两项验证。合同见`docs/REFERENCE_CONFIRMATION_zh.md`，配置`configs/reference_confirmation.yaml`。新池与旧fit/evaluation/CDF/threshold已知源隔离，每域5次200片段、彼此可重叠，共2357唯一新真实视频；同池配对Global/full/D1/Local源等权/位置数量匹配。新参考调度器`evaluation.confirmation_run`，结果`results/runs/reference_confirmation`；按源组和独立身份检验，不能把不同ComGenVid源数差异当算法单因素。

重编码按当前23单元最多50对/单元、1534唯一身份预选，固定原窗口和干净参考，H264 CRF23/35、yuv444p、不缩放/补帧、时间戳核验。它是固定观察位置下的处理敏感性，不声称重选窗端到端鲁棒性。`evaluation.encoding_confirmation run`等待新参考完成后执行，不能重复启动。新结果完成并验收后写正文，旧主线和上轮结果保持。

## 最新执行：补局部方向证据并更新论文（2026-09-11）

首批评分、区间及独立验收已完成，进程已退出，数据已写入正文表3a/3b/5a/7、图2及阈值段。D2=0.874472/0.877075，pooled=0.854733/0.856729，TTR=0.776643/0.816149，SPLIT统计=0.739068/0.789162，二维(TTR,LSMI)Gaussian=0.778564/0.787450。源等权候选=0.875883/0.878766，但主线未自动替换。原池5次源重拟合保留宏平均增量，完整/对角域间反转仍稳定；不能据此默认开展更多收缩搜索。新独立真实库与当前版本压缩验证仍是后续范围，不把原池bootstrap当作已完成新库确认。

用户已授权实施上一轮方案并将数据写入`docs/MANUSCRIPT_REVISION_PLAN_zh.md`。当前执行配置为`configs/local_direction.yaml`，入口`PYTHONPATH=src python -m evaluation.direction_study pipeline`，输出`results/runs/local_direction_evidence`。新增meanPatch-D2、TTR/完整SPLIT与固定目标Global的匹配融合，并共享一次Patch读取计算原拟合池五次源组bootstrap及源等权控制；每个新模型重建自身VATEX CDF，Global逐视频固定。原池bootstrap不称独立新参考库。

`status.json`记录真实父/子PID及阶段，逐视频检查点绑定输入与代码；不得重复启动、修改数值代码后混合恢复。评价源组区间和四格交互在`evaluation.direction_analysis`。Looped保持停止，不恢复监督训练；新FC Patch只作为冻结特征复用。本节实施授权覆盖下文上一轮“尚未启动新实验”的历史状态。

本轮另补同目标real预算的二维(TTR,LSMI)Gaussian控制，入口`evaluation.direction_scalar_control`，只消费本轮已有fit小统计与窗口raw，无新GPU提取，结果在`local_direction_evidence/scalar_control`。它用于区分高维方向与“增加真实统计拟合”的作用，不叫原生SPLIT、不替换主线。独立完整验收入口为`evaluation.direction_audit`；后处理/验收等待器绑定实际pidfd，不因进度暂未更新重启评分。

## 最新指示：停止Looped，回到真实参考Local D2论文主线（2026-09-11）

用户已明确取消继续运行Looped，本次通过`bash scripts/run_looped.sh stop`停止；`results/runs/looped_video/pipeline.json`为`stopped`，父进程及两GPU工作进程均已退出。已有检查点、结果、原视频和Patch缓存保留。不要根据下文历史启动命令、未完成训练矩阵或旧目标自动续跑。

当前分支为`paper/alpha-stalled`，HEAD为`b3576cae12498078429f1671e3781cc25fdbb527`。从`research/global-statistical-experts`切换时保留全部未提交成果；两个分支原本指向同一提交，因此这次不是恢复旧文件快照。`main`及研究分支未删除、未合并。

最新研究方案见[真实参考局部D2：证据复核与补实验计划](docs/REAL_REFERENCE_D2_RESEARCH_PLAN_zh.md)。本次完成停止、分支切换、只读审计和方案文档，未启动新的GPU实验。冻结方法仍为目标Global＋目标Local D2；当前23单元论文Average（三域等权）为0.874472/0.877075。研究组织收紧为局部高维方向的独立价值，不再预设三个同等强度创新或目标Local适配必然有额外收益。

新FC缓存`cache/patch/looped_video`约283GiB，含17785个开发/外部评价身份，可在核对帧索引和数值合同后复用于real-only实验；目录名不表示只能用于监督训练。不要删除、搬移或重提整库。它不含本轮新控制所需的全部fit/CDF特征。旧GenVideo Uniform评价426分片已退役，不能按旧packed索引假定仍可读。

以下Looped/MoE内容是历史授权和研究记录，不能覆盖本节最新停止指示。

## 历史执行记录：Looped视频检测（已由用户停止）

停止前的方法与创新性阶段说明见docs/LOOPED_VIDEO_METHOD_AND_EVIDENCE_zh.md（2026-09-11，seed17三个开发域）。补算的新旧同身份配对指标在results/runs/looped_video/method_review；随后训练已按顶部最新指示停止。该说明明确区分Patch整体增益、循环机制和参数效率，不替代未完成的多seed/外部结果。

最新执行覆盖：用户批准十配置，并要求全部从头训练。当前variants为原五组＋loop1/loop2/loop8/wide512/no_time_position，共120模型/12组。旧五组检查点/日志移到interrupted/before_ten_models，不接续旧epoch；17785条Patch及划分完全复用。配置覆盖在configs/looped_video.yaml，未知项拒绝。启动/续跑统一`bash scripts/run_looped.sh start`，状态/计划/停止分别status/plan/stop；后台脱离终端、防重复锁、失败明确停止。每组完成训练→最佳验证模型测试→六视频重组探针→逐域/生成器表→配对CI→postprocess验收及MD汇总后，才进入下一组。当前文档顶部十配置节覆盖下文历史五配置/60模型表述。不要自动减少20epoch或seed数以承诺一夜完成。

最新用户指示：五个变体全部从头训练，不再保留Looped领先轮次去等待其他模型补齐。旧Looped检查点已移入results/runs/looped_video/interrupted/before_shared_batch/training，仅留作中断记录。新group_train从epoch1开始五模型同时活跃；Patch及准备清单不重建。

用户最新授权同一批数据服务多个独立消融模型。当前入口改为`NUMPY_MADVISE_HUGEPAGE=0 PYTHONPATH=src python -m looped_video.group_train`。按同fold/seed分组五变体，Patch读取/H2D复用，参数、optimizer、梯度、验证epoch选择完全独立；全局batch32及抽样序列不变。已有Looped进度保留，其他变体先补齐，随后同步推进。group_migration.json逐张量检查迁移；groups/<fold>__s<seed>记录整体成本，不能把重叠的各模型wall time相加。旧ddp_train仅作独立训练参照与复用工具，不再作为当前流水线启动入口。

当前DDP传输已升级固定共享环形缓冲＋复用锁页缓冲，队列只传槽编号，禁止恢复逐batch共享大张量旧路径。模型/optimizer/epoch1更新322的数值逐张量保持一致，迁移收据在training/comgenvid__looped__s17/transport_migration.json；原检查点小副本保留为last.pt.pre_transport。NUMPY_MADVISE_HUGEPAGE=0仍必须保留。读取/拼批/槽等待在io_progress.json，GPU供数等待在progress.json。不要再凭瞬时GPU利用率宣称瓶颈已消除。

当前训练启动必须设置`NUMPY_MADVISE_HUGEPAGE=0`（在NumPy导入前）。已定位透明大页整理失败导致供数进程大量内核等待；这是仅本进程的内存分配优化，不修改全机THP、不改变FP32或采样。启动：`NUMPY_MADVISE_HUGEPAGE=0 PYTHONPATH=src python -m looped_video.ddp_train`。第1epoch更新125的检查点被保留并用于续跑，运行证据见runtime_memory_tuning.json。

用户已要求停止旧单卡并行、优化重跑。当前执行入口为`python -m looped_video.ddp_train`，两卡同模型、单CPU供数、48GiB惰性FP32热缓存、有效global batch32不变、每25更新保存恢复点。旧两个未完成epoch的训练移到results/runs/looped_video/interrupted，不混作结果；旧looped_video.pipeline不再用于启动训练。模型、Patch、源级划分和指标不变，见LOOPED_VIDEO_PLAN_zh.md顶部更新。先查实际PID和进度再判断是否需要续跑。

最新存储约束：不要使用系统盘/或/home上的NVMe持久缓存；用户指出空间不足。NVMe复制计划已在执行前取消，目录未创建；所有新缓存/模型/结果留在/data。不要恢复stage_fast_cache或后台复制到/home。训练提速优先有界内存预取，不能偷偷更换特征精度。

最新核验要求：用户要求已足够时直接进入训练。以逐文件生产时完整SHA256读回收据＋全部stat/头部/索引检查作为训练准入，状态ready_for_training。第二次全盘重读已停止，完成前缀见verification_transition.json；不能宣称二次整库核验完成，也不能在训练后用旧verify_cache覆盖已绑定模型的manifest。

用户已直接选定冻结DINO＋四轮共享时空Transformer，授权监督训练和回收新路线不用的缓存。按docs/LOOPED_VIDEO_PLAN_zh.md、configs/looped_video.yaml、src/looped_video执行，结果results/runs/looped_video。此研究的监督训练/完整Patch缓存授权优先于下文旧paper的real-only和禁止新大缓存规则；旧结果不改写。先直接训练完整Looped，普通独立层仅作消融，不重新讨论Gaussian/MoE路线。

已按精确清单回收旧GenVideo Uniform评价Patch 426分片约396GiB（data/catalog/looped_cache_reclamation.json）。原视频/Global/fit/结果保留。下文“630GiB完整保留”是旧状态，packed索引中的退役条目已不可读取。新FC唯一帧缓存约283GiB、开发15569＋外部2216身份；禁止从summary恢复伪Patch。进程/输出以实际状态为准，尚未完成60模型/测试时不可标记研究完成。

## 历史研究分支与已完成结果

Global输入的MoE容量/预热诊断已完成：configs/moe_training.yaml，src/moe_training，results/runs/moe_training，说明docs/MOE_TRAINING_DIAGNOSTIC_zh.md。48模型、88项tests、开发/外部评价和独立验收通过。宽256 MLP/2×128均匀/联合/预热开发AUC为0.965796/0.966954/0.967168/0.966818；joint相对旧窄MoE提高，但未稳定超过同容量uniform，预热无明确新增收益。pooled warm最佳epoch6/7/4尚未解冻gate，与uniform外部完全相同。保留旧参照，不继续扫描预热/容量/路由；旧完成研究和论文主线不覆盖。

冻结DINO的监督判别MoE实验已完成：docs/DISCRIMINATIVE_MOE_PLAN_zh.md、configs/discriminative_moe.yaml、src/discriminative_moe，产物results/runs/discriminative_moe。使用开发real/fake、源级训练验证划分和留一数据域测试，2/4专家，比较线性/单MLP/均匀组合/学习路由，分别G/G+T/G+T+Local。不得把该监督实验称training-free；原paper主线及参数未替换。本轮停止继续扫描专家/路由，结论及外部边界见完成报告。

判别MoE正式训练身份为length_matched_supervision_v2：验证真假匹配8/16帧；训练保持fake生成器等权、real长度质量匹配fake，长度层内源等权。初始140个长度未匹配模型已隔离到training_unmatched_lengths，不得加入选模；修正在任何测试输出前完成。216正式模型已完成，原特征全部复用。模型/代码/epoch规范见DISCRIMINATIVE_MOE_PLAN_zh.md。

216模型与开发/外部均已独立验收：G线性/MLP/均匀组合/MoE的留域Average AUC约0.962115/0.965899/0.967776/0.963824；GT/GTL没有提高总体AUC，MoE三个输入均未过门槛。冻结pooled外部G均匀双头在GenVidBench/ViF AUC为0.958243/0.748688，G MoE为0.954556/0.739100。监督判别有信号，学习路由没有稳定排序优势；ViF低FPR召回的MoE点估计较高但不改主门槛。G的2头由验证选择，不按域拼模型。86项tests、开发432和外部360个独立前向探针通过。静态Infinity序列化修复有559摘要逐元素不变恢复记录，不删除静态视频。不要重新启动已结束任务。

原论文主线两协方差专家四格已完成并停止：configs/mainline_experts.yaml，src/mainline_experts，结果results/runs/mainline_experts。R0=最新23单元0.874472/0.877075；R1仅GT=0.870729/0.875437，R2仅Local=0.874814/0.878930，R3双专家=0.866651/0.874063。R2两个Average增益区间均跨零，VideoFeedback明显下降；未通过预定开发门槛，不触发后续打乱/外部确认，不升级主线或继续扫参。15569评价身份、23单元、8组配对区间、76项tests及独立验收通过，R0原始/最终回归误差均0。方案及停止原因见MAINLINE_EXPERTS_PLAN_zh.md和结果报告。不要将native单窗结果或旧20单元数字作为这轮基线。

用户已授权新分支research/global-statistical-experts，验证原版单窗Global的总体、离线4专家、在线相似512、在线随机512。配置configs/global_experts.yaml，代码src/statistical_experts，统一入口PYTHONPATH=src python -m statistical_experts.run。研究不含Local D2/FC K3，使用独立VATEX2200 fit＋2000整流程CDF；旧threshold2000已在此新协议中改作fit，不再作为其独立阈值。原论文方法和结果保留，下面“当前主线”指原论文主线，不覆盖新研究的显式定义。新结果位于results/runs/global_experts。

后续官方空间＋时序专家、A/B参考校准与Global T1均值/协方差四格已完成，位于results/runs/global_expert_controls，配置configs/global_expert_controls.yaml。在线B为0.835912/0.842062，离线A为0.836669/0.841792；A相对B有AP-real损失，不能按域挑选版本。时序收益几乎跟随协方差，不重新打开均值中心搜索。运行和验收入口见docs/GLOBAL_EXPERTS_RUNBOOK_zh.md；原论文主线未替换。

同簇CDF/连续精度实验也已完成：results/runs/global_expert_routing，configs/global_expert_routing.yaml。专家C为0.835306/0.842086、连续精度为0.835245/0.842246，均未稳定超过既有控制；四个新增配置不采纳，停止这轮CDF/温度/专家数搜索。完整原因与停止条件在GLOBAL_EXPERTS_PLAN_zh.md第23节。测试以根tests目录为准，pytest.ini排除结果源码快照的重复收集。

单窗T1参考来源诊断已完成：results/runs/global_reference_sources，配置configs/global_reference_sources.yaml。官方空间固定，同预算132/200/200独立目标源的Final0.846156/0.850704，对应VATEX0.827195/0.832416，VATEX2200为0.832329/0.837316；Average原始T1/Final增益区间均为正，真实留出NLL改善。说明参考来源确有限制，但使用目标信息，不称zero-target或严格上界；不自动替换原论文主线。新目标native Global在cache/global/target_single_windows，不能与旧Uniform特征混用；后续优先参考覆盖/适配而非继续专家细化。

最新冻结外部六行已完成：results/runs/global_external，2216评价片段、195拟合源、20单元。专家在GenVidBench/ViF均未稳定超过同库总体；在线相对官方的GenVidBench AUC出现负区间。目标适配在ViF也未确认收益。按用户禁止炼丹及停止条件，结束本轮纯Global专家扩展，不升级新主线、不继续搜索CDF/温度/簇数。全过程和边界见GLOBAL_EXPERTS_PLAN_zh.md第25节；原论文MANUSCRIPT_REVISION_PLAN_zh.md保持原已验证主线。

用户随后授权的相邻T1条件预测也已全量验证：results/runs/global_predictive，configs/global_predictive.yaml。边际/预测/跨源打乱的开发Final分别0.831399/0.836461、0.819762/0.825288、0.829847/0.835609；预测对两控制区间为负，外部无稳定增强。该总体预测不采纳、不扩展成专家，不据fake结果反向评分或扫正则/PCA。见研究方案第26节。原主线不变。

最后一次输入边际信息保留检查也已完成：results/runs/global_joint。独立/真实联合/打乱联合Final为0.825020/0.830906、0.821093/0.828160、0.824351/0.830637，真实联合对两个控制的开发AUC/AP区间为负。结束当前预测修补，不扩展专家；原论文主线保持。见研究方案第27节。

## 当前主线

论文baseline已由用户确认改为 **目标域适配Global + Local D2**，Feature-change K3、0.5/0.5融合。
历史结果键`adapted_Global_plus_Local`，完整开发配对Macro AUC/AP-real为0.881462/0.882444。
该数字是旧20单元回归锚点。最新主实验及全部消融已补齐23单元：原文动态筛选＋官方真实配对协议Macro-3为0.874472/0.877075，结果在results/paper_complete；仅补覆盖对照为0.877264/0.880074。模型定义未变，不混用评价范围。8帧manifest在data/manifests/short_video，不删除新增三个生成器。
旧B3（0.878177/0.877191）只作历史对照。不要把旧`predict_b3.sh`或`run_alpha_stall.sh`当成新主线已验收入口。

## 必须先读

1. [论文主线与实验协议](docs/PAPER_MAINLINE_zh.md)：公式、数据角色、逐生成器矩阵与证据边界。
2. [数据和缓存](docs/DATA_zh.md)：活跃清单、精选结果与缓存合同。
3. [仓库规范](docs/REPOSITORY_RULES_zh.md)：配置、运行、数据、指标、测试和删除契约。

旧源码、报告、run和archive副本已清理。历史研究只维护docs/All_Branches_Experiments_and_Data_Summary_zh.md，精选分数位于results/reference；未来实验用新入口重跑。
最新缓存/数据范围按论文P00–P11收敛，见docs/DATA_zh.md及data/catalog/paper_asset_pruning.json。主线输入与必要Patch/粗扫/拟合资产保留；停止方向、重复资产和池外视频已按授权清理。不要恢复此前“所有缓存都永远保留”的旧规则，也不要仅凭容量大就删630GiB必要Patch库。
本地分支为main及paper/alpha-stalled；main不改。不要恢复旧迁移计划或建立另一套永久archive。远程删除和发布单独确认。

## 执行底线

- 活跃目录按用途/数据集命名，不新增paper_v1等代码版本目录；测试直接放tests。最新路径映射为data/catalog/path_migration.json。协议/schema版本及legacy_path不可随物理改名而改变。

- 先核对工作区、Git、活动进程与输入hash；不覆盖未提交成果。
- 新Global/Local都用目标real拟合，但CDF仍独立VATEX，Global窗口与Local视频参考不可混用。
- 科学实验必须保留负结果；不得为取得有利结果挑域、挑seed、翻转分数或用evaluation调权重/阈值。
- 中文注释/文档/YAML解释/SH开头示例；结构化配置，不用脚本拼接或sed改基础配置。
- 主线不import历史脚本。删除前明确哪些是可复用资产，必要结论先汇总到唯一历史文档；不为旧复现保留全套代码。
- 新主线回归目标及范围见论文文档；参数准备、GPU评分、指标、bootstrap和论文完成是不同状态。

<!-- CODEGRAPH_START -->
## CodeGraph

如仓库根存在`.codegraph/`，理解或定位代码时先使用`codegraph_explore`或`codegraph explore`，再用文本搜索。
如不存在，跳过CodeGraph，不自行索引；索引由用户决定。
<!-- CODEGRAPH_END -->
