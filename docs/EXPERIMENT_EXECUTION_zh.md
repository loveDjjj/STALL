# 论文前五组实验执行账

**状态：前五组已完成并通过全范围验收。** 正式结果见[论文实验报告](../results/paper/RESULTS_zh.md)，验收见[verification.json](../results/runs/paper_completion_audit/verification.json)。

覆盖21421条开发视频、20个生成器单元；137105份原始记录已验证，19项1000次源组区间、独立阈值和384次成本测试完成。full/G/Local分数与旧基线逐条一致。main未修改，GPU任务已结束。

主线Macro AUC/AP-real为0.881462/0.882444；原版STALL单窗为0.840248/0.844552；Local D1融合为0.876485/0.875981；Uniform K3为0.874540/0.875159。负结果及域依赖完整保留。

本轮只运行论文主比较、组件、表示、真实统计适配、观察预算五组。所有结果由当前主线重新生成；历史分数仅用于身份和数值核对，不填充新run缺项。基线为目标Global＋目标Local D2，FC K3，等权融合。

## 冻结范围

三个开发域21421条评价视频，固定20个生成器配对单元。各实验同时交付逐视频、逐生成器、逐域和Macro-3，AUC/AP-real/AP-fake、低FPR操作点及源组配对bootstrap。独立VATEX阈值迁移与评价ROC分开。外部和后处理不属于本轮前五组。

| 组 | 必须运行的配置 |
| --- | --- |
| 主比较 | 原生STALL；官方Global FC3；目标Global FC3；目标Global＋Local D2 FC3 |
| 组件 | 完整、Global-only、Local-only、GS＋Local、GT＋Local |
| 表示 | 目标Global固定；Local单位D1、单位D2、raw D2、D2范数、Global单位D2 |
| 适配 | 同预算源/目标Global×源/目标Local四格；Local均值/白化四格；目标对角Local |
| 观察预算 | Uniform K1、FC K1、Uniform K3、FC K3；准确率与实际解码/粗扫/前向/评分成本 |

D2范数定义为float32二阶向量的L2范数，用真实拟合样本的一维Gaussian、位置均值及独立视频CDF评分。Global D2为逐帧Global向量归一化二阶差分，以独立真实Gaussian和视频CDF构成候选分支。它们不复用Local D2的CDF。

源/目标适配四格均使用相同200片段预算；原生STALL单列官方大参考。源域输入为data/catalog/vatex_source_fit200.csv。目标200为现有各域池，ComGenVid包含132源组，不能称200独立源。每个Gaussian须重评分同一VATEX CDF身份。

观察实验固定主线Gaussian；Global保留Uniform首窗CDF。K1与Uniform K3建立各自匹配的Local CDF，不把K3中间子窗CDF冒充独立选择器。原生STALL帧规则另行核对，不默认等于Uniform K1。

## 工程执行

1. 从目标real原视频重新生成Uniform K3拟合资产与GS/GT/LT参数；先与原参考比较均值、评分矩阵WWᵀ、相同视频raw，不直接比较存在符号不唯一性的特征向量。
2. 扩展共享评分流程，让同一次前向服务多个统计模型/表示，保存轻量窗口raw与计时，避免每个变体单独提取DINO。
3. 两张5090按视频身份分片，CPU有界解码预取，固定batch8及fp32特征/fp64评分。当前每卡4解码线程、4预取，不用增加GPU批量改变数值协议。
4. 先完成独立CDF及小批数值验证，再全量评价。需要同GPU争用的任务不同时盲目启动；CPU汇总/拟合与GPU评分可并行。
5. 新结果不覆盖results/reference；输出到results/runs。比较报告必须区分原视频重提、缓存特征重评分、标量重算三种来源。

## 执行记录（按时间追加）

以下保留执行与修正过程；早期“尚未完成”不代表最新状态。当前状态须结合末尾记录、对应run完成文件和原始分数校验。

- `results/runs/paper_fit_comgenvid`、`paper_fit_videofeedback`、`paper_fit_genvideo`：600条原视频拟合资产及三域Gaussian全部完成。GS/GT均值完全等于原包，Local均值差小于4e-19；ComGenVid评分矩阵最大差4.75e-9，须继续检查实际raw。
- VideoFeedback与GenVideo的独立2000条VATEX CDF已分别在GPU0/1运行，原run按相同配置恢复。ComGenVid CDF等待GPU。
- `src/evaluation/representations.py`已定义五种表示和共享多Gaussian评分，通过逐位置直接白化交叉校验；尚未接入全量实验CLI，不能标记表示消融完成。
- 三域`paper_representation_fit_<dataset>`已完成五种表示及对角D2的Gaussian拟合。D1缓存复用前逐视频校验本轮raw D2、Global及帧索引完全一致，来源写入各自manifest；这是拟合完成，不是CDF/检测结果完成。
- `src/evaluation/evidence.py`新增共享前向评分与分片检查点，保留每视频batch8和选择器的独立前向定义；尚需真实视频基线一致性核验和CLI接入。它用于避免逐模型重复提取，不通过增大batch改变协议。
- 已排队：两个在跑CDF任务完成且生成完整参考后，双卡恢复ComGenVid CDF；失败则队列停止，不跳过失败条件。
- VideoFeedback/GenVideo标准CDF均已完成；ComGenVid双卡CDF在跑。源域VATEX200已从原视频重新提取并完成GS/GT/LT拟合，见paper_fit_source。
- 共享评分实测8条VATEX、24窗：选择与Global raw逐元素相同，Local D2最大raw差2.28e-13。验证目录paper_shared_check；这是工程核对，不作论文样本结果。
- 已启动paper_evidence_comgenvid_fc3，4298条全量评价按行号分片到两张卡；每卡6解码线程、8视频预取、8192MiB预算、OMP/BLAS各2线程。它与ComGenVid CDF短暂重叠，后续比较总吞吐判断多进程收益，不能由利用率单次峰值宣称加速。
- 三域标准CDF均完成。共享FC3任务使用tmux paper_shared_queue，每卡最多两个进程，依次运行三个开发域完整evaluation和2000条匹配CDF；ComGenVid评价两个分片已完成，CDF及VideoFeedback评价在运行。
- 原生STALL核对官方src/video_index.py与src/stall.py：seed42依次抽1秒、2秒窗；保留NumPy归一化/评分，与当前Uniform首窗分开。已加入official_baseline.py及两个测试，尚未完成官方全量评分。
- study_tables.py只接收完整、模型与清单匹配的共享证据，输出各组件/表示/统计四格逐视频和逐生成器表；K1参考单独构造，不要求不存在的K2/K3。尚待真实完整产物验收。
- ComGenVid共享evaluation4298与CDF2000均完成，paper_tables_comgenvid_fc3已生成完整点表；full AUC/AP=0.916748/0.921602。对角Local该域更高，必须保留，不能据此按域换方法。其他域尚在队列中，不生成虚假Macro-3。
- paper_observation_queue已在tmux等待前置FC3队列全部成功，随后运行官方单窗和uniform1/fc1/uniform3的三域evaluation/CDF。三域21421条原2_sec_idxs与官方seed42窗口规则核对全部一致。
- 为填补空闲，ComGenVid官方单窗已提前在每卡一个额外进程运行（短期每卡3任务）；队列以后会校验复用其检查点。当前测得官方约3.7视频/秒/卡，属于并发条件吞吐，不是独立端到端基准。
- 增加独立AP-fake字段，正类取fake、方向为负真实性；原AP-real字段保持不变。跨域合并阶段重新计算完整指标，旧生成的单域点表未原地覆盖。
- ComGenVid full对global_only完成1000次源组配对区间：AUC差0.016936，CI[0.010619,0.023290]；AP-real差0.012841，CI[0.006842,0.019553]。仅单域，不是Macro结论。见paper_interval_comgenvid_local_contribution。
- table_queue在tmux自动等待完整证据并汇总五种观察/原生协议；合并Macro仅使用开发三域配对，active/pairs.csv还包含两个外部域，不能无过滤地要求本轮开发评分覆盖外部。
- hardware_monitor每15秒记录GPU利用率/功率/显存及CPU/RAM。加到每卡三个任务后CPU接近95%–96%，不再盲目加进程；瞬时GPU空闲不代表CPU仍有可用解码预算。
- runtime_benchmark使用实际单模型RawVideoScorer，按固定视频身份哈希每subset/source_model取至多2条，四种观察各测两遍。无特征缓存，OS页缓存不清空，不能将第一遍称严格冷磁盘。paper_runtime_queue等待观察任务结束后在独立GPU上执行，记录阶段耗时、唯一/补齐帧及显存；尚未运行，不把共享消融吞吐当作该结果。
- 只读解码线程对照：三域各real/fake1条、两次，单线程与默认像素完全相同，但除一条GenVideo real外均更慢；不采用强制单线程。原始耗时见hardware_monitor/decode_thread_check.json，仅工程小样本，不能当论文速度结果。
- ComGenVid官方单窗4298条已完成：域级AUC/AP-real=0.855047/0.860620；当前full为0.916748/0.921602。full减官方单窗的1000次源组CI：AUC[0.048564,0.075221]、AP[0.047271,0.076888]。该比较包含目标数据和观察差异，不等于Local独立贡献。
- VideoFeedback官方单窗已提前接续双卡运行；GPU队列仍保留每卡最多两个共享任务，额外官方Global任务构成短期每卡三进程。缺行/错误video_id/清单次序改变的汇总拒绝测试通过。
- 三域完整表后的19个预定源组配对对照已排入paper_interval_queue，每项1000次、最多2个CPU任务。原始分数重算、point表、CI及独立成本分别验收，不能只凭队列启动认定完成。
- VideoFeedback全量3500和CDF2000完成；full AUC/AP-real=0.838391/0.852220，与旧主线逐视频Final差为0。Global-only为0.836988/0.856073：Local增加后AUC小幅上升但AP下降，需保留这一边界。D1融合为0.833172/0.843633。单域源组CI正在计算。
- 表示拟合预算：各域Local均为51200位置；Global D2使用同视频Uniform窗口全部时间差分，ComGenVid/VideoFeedback/GenVideo分别8400/6776/8400，不伪称与Local位置数相同。
- VideoFeedback官方单窗完成，AUC/AP-real=0.857156/0.869419，高于本域full；D2对D1源组CI为AUC[0.001815,0.008270]、AP[0.005458,0.011857]。Local对同预算Global的区间跨零。GenVideo官方单窗已提前接续。
- 独立阈值表所需paper_threshold_queue已排队：等待GenVideo匹配CDF完成后，每卡一个任务重算各目标模型下的VATEX threshold2000；ROC操作点不能冒充这一迁移结果。该任务只补预定指标，不新增算法。
- 顺序解码优化通过：24个来源/生成器视频Uniform窗口像素完全一致；同24个实际FC窗口（13个非零起点）也完全一致。8条VATEX完整共享评分的Global/窗口完全一致，Local最大差2.28e-13。多数样本解码显著更快，但这些是背景并发小样本，不能直接当端到端提速倍数。
- data/sequential_decode.py只定位一次并顺序读取，失败回退原严格解码。后续观察/阈值任务显式启用sequential_decode，config记录实现文件hash；已运行FC3/官方任务不改。独立成本四种观察均统一使用此解码器。
- 已重建尚未启动的观察/阈值队列：GenVideo匹配CDF完成后即可启动，不再等待全部FC3评价；官方三域已独立启动，不重复同rank。快解码任务4线程/8预取/8GiB，每卡一个观察和一个阈值任务。runtime队列改为等待所有必要完成文件，不依赖旧队列PID退出。
- GenVideo匹配CDF已完成，观察uniform1和ComGenVid独立阈值队列已实际启动。GenVideo FC3评价与官方单窗仍使用原解码继续，不混改已生成检查点。
- 独立阈值汇总共用calibrated_scores，与评价完全相同；tau取排序real分数floor(n*alpha)位置，判fake为S<tau，同分不拆分。输出来源FPR/生成器Recall及描述性片段Wilson区间，不宣称处理同源相关性或保证跨域FPR。
- read_evidence新增跨rank模型/代码/配置一致性检查（设备号允许不同）；阈值并列与零误报区间测试通过。
- 顺序解码后CPU约40%，观察队列从2提升到4进程（每卡最多2个观察任务），短暂重启后按原身份复用了2124条Uniform K1检查点；旧进程已退出再接续。随后采样GPU均98%、CPU47%、约10GiB显存/卡。瞬时值仅作调度依据，正式成本以独立benchmark为准。
- CPU汇总修复了CSV默认浮点解析及1-score的末位舍入问题：score读取统一round_trip，ROC异常分数改为精确取负。GPU分数不变；已有表移至precision_diagnostic供审计后从原raw重建，最终区间使用修复后的三域表，不引用临时旧解析区间作为最终结果。排序/ROC往返回归测试通过，全套43测试通过。
- 官方STALL三域完整新Macro AUC/AP-real=0.8402476187194191/0.8445520524378721。不是原论文公开表格数字，而是本轮固定20单元与官方单窗/评分控制的结果。主线GenVideo全量未结束，暂不将旧锚点冒充新Macro。
- ComGenVid Uniform K1评价/CDF与独立阈值已完成并自动生成表；其余观察协议和域继续排队。
- 三域FC3全部21421条重评分完成，full/global_only/local_only与原冻结分数逐视频最大差均为0；最新Macro仍0.8814615038969483/0.8824439675246946。组件、表示、统计四格的三域点表在paper_tables_all_fc3，15项源组区间已逐项计算。
- 新Global背景D1融合Macro=0.876485/0.875981，对角Local=0.869914/0.871640。四格：source/source=0.853962/0.850651，source/target=0.876073/0.874291，target/source=0.880766/0.879346，target/target=0.881462/0.882444。需解释适配的条件收益与交互，不将旧Local收益直接搬过来。
- FC1/Uniform3只需固定主线，已用BudgetEvidenceScorer省去无关表示计算；8视频24窗核对Global/窗口一致，Local最大差4.55e-13。当前Uniform1/FC3不改。CLI可显式--all-evidence或--baseline-only，默认实验函数对FC1/Uniform3采用后者，作用范围写入config身份。
- 观察调度改为每GPU独立3线程池，消除跨GPU排队占位；停止前验证父子PID，保留检查点。已完成分片先纯CPU核验，不重复加载DINO或重写完成计时。
- Uniform K1三域Macro=0.8736245579972737/0.8776993882134931；full减Uniform1的Macro AUC CI[0.003897,0.012481]、AP CI[0.001446,0.008390]。
- GenVideo Uniform K3改为6个固定全局行号分片，每卡3个；i的GPU仍为i%2，因此既有奇数行检查点身份可直接验证复用。GenVideo此协议在重新分片时尚无完成标记。read_evidence与等待队列按实际world_size验收，6分片复用/完整性测试通过。
- 统一CLI目前仍只运行标准D2；同预算四格及匹配K1参考需继续实现。旧结果存在不代表新实验已完成。
- 机器检查：两张RTX5090各32GB、约114GiB可用RAM、磁盘约116GiB空闲。禁止新增数百GB Patch缓存。

## 验收

新旧结果差异须逐视频、逐生成器报告：窗口身份、raw最大/分位误差、分数差、AUC/AP差。任何窗口或协议差异先定位，不能只凭Macro接近就通过。最终使用核验后的新run，不择优选新旧结果。
每组完整覆盖20单元，缺失明确失败；所有指定变体、置信区间、成本与新旧对照完成后，才算本目标完成。
