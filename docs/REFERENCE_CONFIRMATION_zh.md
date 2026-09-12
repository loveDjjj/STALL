# 独立真实池与重编码对照：执行合同

完成状态：新真实池及重编码对照均已完成并通过独立指标审计，结果已写入`MANUSCRIPT_REVISION_PLAN_zh.md`的拟合位置匹配段、5.3.1/表5b及5.6/表8。新参考产物在`results/runs/reference_confirmation/results`，重编码在`encoding`；对应`independent_audit.json`均为passed。以下保留执行前冻结的设计与实际时间基修复记录，不代表仍需重新跑这些任务。

用户于2026-09-11授权实施上一轮结果分析后的两项验证。本文件记录冻结范围，结果完成前不修改原主线参数或填入预测数字。

## 1. 独立新真实池

从保留原视频中排除旧fit、当前23单元evaluation、VATEX CDF/threshold的源组及物理路径；ComGenVid与VATEX增加已知YouTube媒体ID交叉检查。该独立性是已知来源规则下的隔离，不声称完成跨库语义去重。

每域每次200片段，seed17/29/43/59/71，共15个拟合清单。五次新池可相互重叠，完整交集在`fit_overlap.csv`；不能称五套互斥数据库。所有新fit与旧fit互斥。共享提取2357条唯一真实视频，仍采用Uniform K3、224、8FPS、batch8尾批、FP32差分和FP64评分。

ComGenVid五池独立源数181/182/179/183/184；VideoFeedback和GenVideo均每池200源。新池比原ComGenVid200片段/132源更分散，不能把新旧池结果差全归因于算法。关键比较必须在同一个新池内部进行：

| 比较 | 冻结条件 | 回答问题 |
| --- | --- | --- |
| Global vs Global＋D2 | 同一新池、同窗口 | 新参考下Local仍有增量吗 |
| 片段等权 vs Local源等权 | 同一新池及同一Global | 源等权收益能否重现 |
| D1 vs D2 | 同一新池及同一Global | 二阶增量是否依赖旧池 |
| D2 video256 vs D2位置数匹配 | 同一视频、同Global、匹配pooled实际观察数 | 位置样本数量解释多少差异 |

位置匹配取每条视频原256个确定性随机位置中的前`K×14`个（实际Uniform窗口数K），不是复制pooled向量。另补原fit池的位置匹配D2，便于与已完成的原池pooled-D2作最直接对照。每个Gaussian重新评分同一批VATEX2000，8/16帧和effective-K分别匹配；Global参考仍使用Uniform首窗。

每次新池同时拟合Global和Local。五次均值是五个模型各自指标的均值，不平均预测当成集成模型。评价源组区间以同次重采样权重计算各模型指标差，再跨五池求均值；其条件为五个已固定新池，参考池间波动另报，不能解释为对所有可能新池的精确置信区间。

## 2. 固定输入窗口的重编码

当前23单元每单元最多50对，由身份hash和seed20260911预选，共1534个唯一评价身份。真假均经过同样处理，原始对照严格使用同一子集。首轮不扩大到全量，也不按结果挑选子集。

处理固定为libx264、medium、CRF23/35、yuv444p、每编码进程2线程。不改变图像大小/帧率，不复制帧；保留奇数图像尺寸，因此使用4:4:4并明确此结果包含编码及像素格式转换，不泛化为所有常见4:2:0平台转码。逐视频检查帧数和图像尺寸完全相同、归零后的时间戳偏差不超过1ms。

时间基修复：WildScrape/D467是约250秒变帧率视频，默认编码时间基在前200帧产生约16.645ms偏差；显式`-enc_time_base -1`后该探针误差为0。对已知此类失败输入直接保留输入时间基，其他输入仍走原路径、必要时才回退；完整输出继续逐视频满足原1ms门槛，不放宽、不丢样本。已完成且满足合同的旧结果保留；`encoding/timebase_compatibility.json`记录原/新生产代码、未变评分函数和探针依据，每个新收据记录实际时间基与生产代码。

本轮固定原来的FC窗口及干净Gaussian/CDF，只改变实际编码后的观测像素，比较Global、Global＋D2、TTR/SPLIT统计及Local原始排序。它检验原观察位置下的评分敏感性，**不等于在重编码视频上重新粗扫选窗的完整端到端鲁棒性**。不重新适配参考、不按压缩强度调阈值。临时编码文件在`/data`的新run目录，处理后释放，原视频不改动，不保存新全量Patch。

## 3. 实现、速度与状态

基础配置`configs/reference_confirmation.yaml`；独立数据清单、拟合、评分、导出分别为：

```text
evaluation.confirmation_data
evaluation.confirmation_fit
evaluation.confirmation_engine
evaluation.confirmation_results
```

统一新参考调度器`evaluation.confirmation_run`，结果`results/runs/reference_confirmation`。编码对照`evaluation.encoding_confirmation`，结果子目录`encoding`。

评价Patch采用一个主进程单流读取、4个固定FP32共享内存槽、两个独立GPU评分进程。窗口不足16帧时仅消费实际长度，不读取未初始化尾部。CDF三个域及同表示多个模型共用一次方向矩；每32条结果原子写一个块，减少机械盘逐文件fsync。每个完整域可先导出partial域表，三域未齐不生成Average。运行期间不改变数值代码、不重提/搬移已缓存Patch。

新参考先执行CDF与评价分层探针，原Global/Local回归通过才全量执行；编码先跑跨域/长度/标签探针，检查原路径回归与帧时间合同，再进入固定子集。

状态入口`status.json`、`evaluation_progress.json`、`cdf_progress_*.json`及`encoding/status.json`。最终需要：完整逐视频分数、逐生成器/域/Average、配对区间、新池交集与源数、原锚点回归、重编码时间戳收据和独立指标核对。只有真实完成数据进入论文正文；本合同不作为完成报告。
