# 仓库规范与AI修改契约

最新用户范围：只维护当前主线，不维护旧复现入口或源码archive。历史结果、配置和做法汇总到唯一历史文档，后续实验用主线重跑。

## 2026-09-12代码整理

清理前完整代码与论文材料已提交为Git恢复点 **efede85**。当前树退役5个研究包（72个Python文件）、9个配置、7个脚本和15个专用测试，并删除33个旧表文件及3张旧流程图。已有视频、Patch、模型和实验结果没有删除；停止方向的源码可从该提交查看。

当前仅维护三个配置、统一CLI、主线src及论文所需evaluation。数据或结果目录中的旧研究名是资产身份，不按代码包名做递归清理。主线仍读取mainline_experts的既有plans/global参数和looped_video的Patch缓存。

代码统一采用Python 3.10、Ruff 0.12.12、100列和LF；配置见pyproject.toml，开发依赖见requirements-dev.txt。Ruff当前检查语法、非法比较和未定义名称等高信号问题，格式检查独立执行。pytest.ini集中设置src路径，单文件测试不依赖收集顺序。

本轮合并JSON/检查点标准化、提取CLI参数构建、移除11个已无定义的公共导出名。其余数值与解码函数保持AST一致；格式变化不代表允许绕过已有run的源码hash校验。CSV等冻结数据由.gitattributes保留原字节。

最新跟踪规则：results整个目录仅保留本地，不纳入Git，包括轻量CSV和验收文件；paper源码、生成TeX表格及最终PDF纳入Git，编译临时文件忽略。新检出直接编译已提交表格；重新导出和--check须先取得本地结果CSV。该规则覆盖此前精确纳入results轻表的做法。

## 目录命名

活跃目录用小写snake_case，按“资产类型/数据集或用途”组织，不用paper_v1、iclr、confirmation或旧实验编号标记当前代码。测试直接放tests；正式参数放precomputed/target_reference；缓存按global、patch、fit、models、scores、contracts分类。
时间尺度、样本预算等有实际区分意义的数字可以保留，例如coarse_1fps、vatex_200。数据集原始split/生成器名保留，不为了美观合并不同身份。
协议ID、缓存格式schema和不可变run身份仍须版本化；目录不带版本不等于允许覆盖旧实验。冻结NPZ、Patch、raw和历史记录中的旧路径是来源身份，不做全文替换。物理路径迁移查data/catalog/path_migration.json，当前运行只读活跃清单。
父目录datasets/cache是补充数据资源，STALL/datasets/cache是当前论文资产；补充数据未完成新主线源级划分前，不自动并入正式评价，也不称未见确认集。

科学定义以[PAPER_MAINLINE_zh.md](PAPER_MAINLINE_zh.md)为准，数据和缓存位置见[DATA_zh.md](DATA_zh.md)。当前只维护一份历史实验总结，不再建立永久源码/报告archive。
缓存按当前论文实验需要保留，具体范围见DATA_zh.md和paper_asset_pruning.json。年代旧或容量大不是删除依据，但已停止方向和没有实际用途的副本不永久保留。原视频删减必须保护active身份和明确的真实对照池，记录清单，不根据AUC挑选删除。

## 1. 方法身份与变更原则

1. 论文主线是目标域适配Global + Local D2。当前23单元三域Average为0.874472/0.877075；历史结果键`adapted_Global_plus_Local`的0.881462/0.882444属于旧20单元。
2. 旧B3=0.878177/0.877191，不再是后续开发基线。旧名字、文件和hash保留历史身份，不能原地重新解释。
3. “主线已确认”不等于“YAML/入口已迁移”。完成入口回归之前，不把`predict_b3.sh`、`run_alpha_stall.sh`宣传为新主线命令。
4. 不微调DINO，不训练真假分类器；只用fit real估计Gaussian。开发阶段曾观察fake指标，应公开，不能宣称研究从未使用fake做设计选择。
5. 任何影响特征、采样、dtype、归一化、正则、CDF、聚合和权重的更改都是科学协议变更，必须版本化；不要作为普通“提速”混入。
6. 同条件失败结果也写入run registry，不筛选有利域或生成器；不得按标签翻转分数、根据eval选择权重/源模型/阈值。

## 2. 配置与命令

目标仅保留一个`configs/paper.yaml`基础配置；历史benchmark.yaml归档后不再作为active默认。
SH顶部中文注释必须说明问题、与主线相比唯一变化、参数和示例；参数透传给统一CLI，先打印resolved差异。
不在SH实现采样、评分、bootstrap，不创建每个实验一份Python算法脚本。

建议基础配置的必要字段（名称实施时统一，不是现有CLI已支持）：

| 配置组 | 必要内容 |
| --- | --- |
| protocol | id/version、是否固定、seed、数据预算和任务范围 |
| encoder | model、权重hash、源码revision、224、fp32前向、batch8、pad-tail |
| data | catalog、fit/CDF/threshold/eval/pairs清单及hash，真实源分组字段 |
| fit | 每片段权重、Uniform K3、Local256抽样、epsilon/ridge/零向量规则 |
| method.global | GS/GT启用及0.5/0.5、target_real参数源、max/min聚合 |
| method.local | D2/归一化、region1、mean、无Local Spatial、无窗口CDF |
| selection | FC K3、1FPS、2秒/16帧、4位置步长、稳定tie-break、短视频策略 |
| calibration | Global独立VATEX Uniform K1、Local effective-K策略、右包含、不插值 |
| fusion | global0.5/local0.5；移除分支的明确重新归一化规则 |
| runtime | device/devices、线程/有界预取、cache合同、日志间隔、空闲空间阈值 |
| evaluation | AP正类、配对来源、源组bootstrap、操作点与阈值数据职责 |
| output | 新run目录、禁止覆盖、版本和恢复模式 |

每个YAML叶子配置都必须中文说明意义、单位、合法范围及是否影响协议；默认值不能只引用“原论文”而不给来源。
unknown配置字段报错，不能静默忽略拼写错误；解析用YAML/JSON结构化API，不依赖字符串replace。
`--dry-run`不得加载GPU或创建会被误认为完成的结果；恢复只允许相同协议hash，不允许`--resume`同时变参数。

## 3. 数据与模型合同

逻辑数据ID与物理存储路径分离，但迁移时保留`legacy_video_id/legacy_path`，不得使旧抽样seed失效。
必要manifest字段：`dataset,split,video_id,source_group,real_source,generator,video_path,content_sha256,fps,num_frames,downsample_idxs,exclusion_reason`。
真实来源和生成器分字段，`subset=annotated`历史键映射为fake时登记，不改旧raw。
评测real、生成器fake的数目以及每个过滤原因都必须报告，不能把缺失帧默认为零特征。

三类real资产严格分工：

- fit：拟合GS/GT/LT，不用evaluation real。
- CDF：独立VATEX参考视频，对每个Gaussian自己评分。
- threshold：冻结完整评分器后生成操作点；不能当作CDF带宽/模型选择验证集后仍宣称独立。

同源片段跨fit/eval不允许；跨fit/CDF/threshold也检查源组。仅路径不同不证明独立，源前缀匹配是近似审计，语义去重另论。
一个新域必须显式提供/选择真实参考，不根据测试内容猜数据域或按fake AUC挑模型。
参数包至少保存GS/GT/LT mean/W、Global两个窗口CDF、Local K1/2/3视频CDF、权重/代码/fit/CDF清单hash、dtype和零值规则。
阈值可独立文件，不硬编码0.5；导出的分数不是概率。不同统计包不能通过文件名相似而复用。

## 4. 缓存、精度和性能

1. 最小缓存身份包括视频内容或受控文件stat、采样索引、模型权重/源码、输入分辨率/预处理、batch/尾批、dtype和cache schema。
2. 缓存目录名不等于dataset边界，shard中的source/model分组不替代manifest。一个shard可以服务多表，不创建重复视频身份。
3. 新主线不生成全量巨大Patch缓存；默认原视频→有界预取→特征→小向量统计/窗口分数→释放。
4. 630GiB旧Uniform库不能直接拿来声称任意FC窗口已有特征；命中索引和合同都要检查。
5. 批量优化先同一视频、同一帧、同一权重逐元素对照；速度随域变化不能当性能优化证据。
6. fp32特征差分/归一化→fp64评分是合同；尾批补齐后必须裁去补齐帧，不能增加真实观察数量。
7. Global零T1与Local零D2不同规则不得合并；所有NaN/不允许的inf显式停止并标明video/window。
8. 哈希对不变大计划缓存一次，验收前后复核；不在每个视频重新扫描50MiB同文件。不允许为提速跳过raw身份验证。

## 5. 运行与产物

建议run名称：`<study>__<variant>__<scope>__s<seed>__<UTC时间>`；不要只用`latest`或覆盖`alpha_stall`。
`latest`最多是带版本索引链接，正式表必须引用不可变run_id。

每次run必须生成或引用：

```text
resolved_config.yaml
command.txt
run_manifest.json
status.json / progress.json
logs/launcher.log / worker_*.log
reference_manifest.json
selection_manifest.jsonl
window_scores.parquet或.csv.gz
video_scores.csv.gz
pair_ids.csv
generator_metrics.csv
dataset_metrics.csv / macro_metrics.csv
real_source_metrics.csv
operating_points.csv
comparison_deltas.csv / comparison_bootstrap.csv
artifact_manifest.json
```

不是每阶段都必须重复复制参考；可通过内容hash引用已有只读包。没有输出的阶段在状态中注明，不用空CSV伪装完成。
run_manifest记录git commit及dirty diff hash/未跟踪源码快照hash、环境、硬件、输入清单、参数、seed、范围、过滤和成本。
命令及stdout/stderr全部记录；token/凭据和私人URL不写日志。视频绝对路径可留本地映射，不直接公开。
状态至少区分preparing/fit/cdf/score/evaluate/bootstrap/export/completed/failed；CLI退出0不证明所有阶段已完成。
原子写结果，持有单run锁；worker结束后主进程可能仍验收，等待主PID和状态而非将worker消失当失败。
不按状态文件单独认定活跃，检查进程/锁/完成资产。停止任务前解析精确PID/工作目录/命令，不使用广泛pkill GPU/Python。

## 6. 指标与证据不变式

- 当前S高为real；AUC/AP-real固定方向。AP-fake另列，不能改正类后对比旧0.882444。
- 当前主表23单元使用paper_complete固定pair_ids，不重新抽real或fake；shared real跨generator同一源权重；旧20单元单独保留历史身份。
- 样本数写清片段、唯一视频、源组、窗口、配对行，不用其中之一代替其他。
- 预注册/冻结指向实际时间点，不允许看到结果后回填“预注册”。本轮新baseline是在旧结果已看过后选择。
- 新Global下的消融必须固定新Global，不能拿旧B3融合结果直接减新baseline。
- 原版STALL原生K1、官方评分器FC K3、目标适配Global三个对照标记独立protocol。
- 参考Gaussian改变时，CDF身份可同但raw数组必须重算；选窗改变匹配参考也须相应处理。
- 阈值在独立real确定，实际FPR按源和域报告；ROC最佳阈值不是部署阈值。
- 参考抽样与评价抽样的CI不同；指标CSV还未写完、bootstrap没结束、hash不符时不能导出正式表。

## 7. 测试与迁移验收

最低覆盖：

1. 主线配置未知项/非法项拒绝、SH透传、dry-run不执行实验。
2. 原始视频dense索引、FC稳定排序、窗口去重、K1/2/3及短视频/缺帧。
3. 固定batch8、边界D2、零T1/零D2、fp64均值能量及CDF同分/+inf。
4. 片段权重与真实源隔离、GS/GT/LT参数和对应CDF完整性。
5. 新主线多个短/长real/fake原视频端到端回归，再从已有全量raw重算0.8814615038969483/0.8824439675246946。
6. 新Global下组件重算、配对身份、分数方向、源组bootstrap、指标/文档一致。
7. 中断续跑不重算已完成视频、父进程收尾不误停止、坏分片拒绝、迁移前后hash及路径映射。
8. 原版STALL通过官方资产/代码兼容检查，协议差异不被默认消失。

浮点容差沿现有验证起点：window raw1e-8、Global/final1e-10；任何扩大须解释并重新验收。不能用“接近0.8815”替代逐视频回归。
旧测试数157仅历史状态，重构中测试可能合并/拆分；以覆盖行为和实际结果为准，不为保持测试数复制测试。

## 8. 文件、Git与删除规则

主src不得import scripts或archive，scripts不得作为算法库；不依赖运行时修改模块全局OUT/DOMAINS。
所有新注释和说明优先中文，标识符/字段/协议ID用稳定ASCII；公式和论文术语统一。
数据读写使用结构化格式；注释解释边界与原因，不逐行复述代码。

Git最终main＋一个paper分支；main不改。新文件/工作树脏状态归用户，不能reset/checkout覆盖。
每次迁移前保护tracked、untracked和ignored必要资产；Git bundle只保护已纳入Git对象的内容。
分支删除、远程删除、数据删除分别列清单；代码归档不自动授权清空数据缓存或远程分支。
先快照→hash→恢复试验→改引用→回归→删活跃冗余；绝不能反序“先删再看哪里报错”。
历史run文件只读；迁移另存映射，不重新编号或改写旧协议键以追求整齐。共享原视频不复制为多个实验目录。

## 9. 后续AI每次开始时

读AGENTS与三份主文档，核对用户最新指令、分支/dirty状态、活动进程和待执行阶段。
先读代码/manifest/参数hash，再引用历史报告；报告状态冲突不等于结果无效，应追踪身份链。
若`.codegraph/`存在，按用户规则先用CodeGraph定位；不存在不自行索引。
先说明本轮是否仅分析、是否编辑、是否会移动/删除；不把“继续”理解为无限新实验。
按计划完成一个可验收阶段，不重复输出运行中状态或新增一份近义方案文档来冒充进展。
总结时说明实际修改、测试结果、未完成事项和风险。不得虚构已跑实验、编造指标、承诺录用或声称未验证的新颖性。

当前继续实现；main、原视频与大缓存保护规则仍有效。远程删分支和数据回收依计划单独确认，不因“开始实现”扩大为任意删除。
