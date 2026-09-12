# 活跃数据与历史来源

2026-09-11：Looped训练已按用户指示停止，当前返回real-only论文研究。下述新FC Patch缓存保留并可复用，停止训练不等于删除数据；本次没有新的数据移动或删除。后续缺失fit/CDF表示优先流式计算轻资产，见[最新研究方案](REAL_REFERENCE_D2_RESEARCH_PLAN_zh.md)。

最新Looped监督路线已授权回收旧GenVideo Uniform评价Patch：426分片约396GiB，收据`data/catalog/looped_cache_reclamation.json`。下文630GiB为回收前状态；旧packed索引保留身份但对应退役分片不能读取。新缓存`cache/patch/looped_video`保存新FC唯一帧FP32，身份与用途见`LOOPED_VIDEO_PLAN_zh.md`。原视频、Global、拟合资产、参数和完成结果没有因本次回收删除。

最新23单元补充资产见data/manifests/short_video，以及datasets/genvideo/fake/HotShot、MoonValley和datasets/videofeedback/fake/Hotshot-XL。已恢复4063个视频，其中VF仅恢复动态等级3–4。元数据及CRC/SHA256收据位于datasets/recovery。主评价采用results/paper_complete的pairs.csv/evaluation.csv，旧active/pairs.csv仍是20单元。详见[协议核验](MANUSCRIPT_PROTOCOL_NOTES_zh.md)；旧清理清单反映过去状态，不能据其再次删除已恢复的短视频。

当前论文需要的原视频保留；缓存已按用途迁移，内容不变。池外视频及停止方向缓存已按P00–P11清理；活跃清单在`data/manifests/active`，内容没有因清理而改变。
旧报告/run/快照已汇总清理；可复用资产与合同已保留，旧到新路径映射在data/catalog/retention_cleanup.json。

## 清单职责

- `<dataset>/fit.csv`：目标real拟合，包含原video_id、路径、源组、Global/D2轻量feature_asset及hash。
- `<dataset>/evaluation.csv`：严格继承当前冻结评价范围，不从旧全清单无条件添加视频。
- `<dataset>/pairs.csv`及`pairs.csv`：原有固定生成器配对；旧reference/selector字段保留为legacy字段，不冒充当前方法名称。
- `vatex/cdf.csv`、`vatex/threshold.csv`：各2000独立真实视频，职责分开；并非目标域Gaussian拟合集。

新增统一字段：dataset、split、video_id、legacy_video_id、legacy_path、source_group、real_source、generator、content_sha256、exclusion_reason。
原数据没有内容hash时content_sha256留空，不把stat或路径hash冒充视频内容hash。
source_group沿已审计源前缀组织；ViF对应real/fake联合分组，其他fake无真实源对应证据时保持视频身份分组。
这不是完整语义去重。跨角色独立性仍应结合原协议审计，不能仅因两条路径不同就判断独立。

## 范围与排除

开发评价21421视频，主平衡配对13033唯一视频、20870行、20生成器单元；外部GenVidBench600、ViF1616。
各原始真实来源与生成器数量见[PAPER_MAINLINE_zh.md](PAPER_MAINLINE_zh.md)，不在多份文档重复手工维护表格。
`data/catalog/excluded_from_frozen_scope.csv`记录原manifest中未进入当前冻结评价的行。
排除理由是“未进入既定范围，查历史审计”，不伪造每个视频的具体解码/时长失败原因。

## 标准raw输入

`results/reference/baseline/window_scores.csv.gz`保留当前主线窗口排名、原帧索引和raw分数。
其GS、GT、LT raw与对应target_reference参考包hash绑定，供融合/组件重算，不代表可以用于另一套Gaussian。
`results/reference/baseline`是轻量冻结分数包，没有历史源码或执行检查点。可供components/bootstrap读取，不能以--resume假装恢复一次旧运行。
`results/reference/comparison_tables.csv.gz`按history_experiment/history_table保留精选对照表，详细配置和其余结果在唯一历史总结中。
更新此输入或活跃清单必须新建版本并记录来源，不直接覆盖以改变历史统计。

## 统一fit当前需要的轻量资产

`fit.csv`中的`feature_asset`必须有对应`feature_asset_sha256`。NPZ包含video_id、Uniform K3的frame_indices、float32 global_windows（K×16×1024）和已按原规则抽样的float32 raw_d2（256×1024）。
新fit从raw_d2在float32中重新归一化，保留零D2；不读取旧existing_d2关联文件，也不依赖prototype实验模块。
资产内容hash确保复用明确的已有特征，不能仅凭同形状就把其他分辨率、权重或数值协议的资产改名混用。
现已支持`fit.feature_source=video`从原视频生成上述小资产，不再要求新域先运行历史准备脚本；仍需显式fit/CDF及用于隔离审计的evaluation/threshold清单。
拟合视频必须有固定dense索引；先前缺少这些字段的数据仍需正规manifest准备，而不是靠文件夹名推断角色。
新的`prepared_fit.csv`仅在当前run中登记生成路径及hash，原816条小资产与原始视频继续保留。

## 路径与抽样身份

每视频256位置使用`SHA256("17:" + legacy_path)`的前8字节小端整数作为NumPy随机种子，无放回抽样。
legacy_path缺失时使用video_path。两个字段都写入清单；物理路径映射只改变读取位置，不应改变抽样身份。
原视频fit检查输入size/mtime、生成NPZ身份、窗口索引和内容hash；这是受控输入合同，不是完整视频语义去重。

## 可复用缓存保留策略

最新规则：以P00–P11确实要做的实验决定保留，不再因“可能复用”永久保留所有中间产物。

| 保留缓存 | 对应实验与依据 |
| --- | --- |
| patch/benchmarks，约630GiB | P01/P05/P06/P08；索引覆盖开发三域及GenVidBench所需22757个fit/eval键，所有分片保留 |
| patch/vifbench，约2.5GiB | P09/P10；覆盖ViF全部80条fit键 |
| global/coarse_1fps，约652MiB | P08选窗；使用前核验批次/采样合同 |
| fit/target，约1.9GiB | P00/P05/P06/P10；816个Global/D1/D2拟合资产及相关索引保留 |
| fit/vatex_200，约187MiB | P07同预算源域/目标域统计对照 |
| models，约603MiB | 43个主/外部raw对应模型，加45个带明确video_ids的普通少样本Local模型；P05–P10 |
| scores，约220MiB | M31/M30原始评分及定义；P02–P07/P09，不保留已停止All/旧选择器raw |
| contracts，约54MiB | 解释保留缓存的旧采样身份；不是当前运行清单 |

已删除Tail位置场、语义CDF、第三方XCLIP环境/权重、轨迹/条件/收缩等专用产物，以及不在保留模型集合内的1971个文件。
b3_prototype_fit_v1的600份unit D2在删除前与当前raw D2重新归一化逐元素核对一致，没有删除唯一拟合特征。
630GiB库的少量旧GenVidBench calibration条目混在保留分片中，不为这63项重写巨大分片；新实验只按active身份取样，不能自动把这些旧条目纳入fit。
当前保留模型的精确路径和hash在`data/catalog/path_migration.json`的current_models中；paper_asset_pruning.json是迁移前清理记录。旧搬移映射retention_cleanup.json仅是历史记录，其中部分资产已被这次清理淘汰。
contracts中的旧完整数据清单可能包含已删除的视频；不得把它们当当前可用清单，当前使用active及以下源域200清单。
这些缓存不能无条件用于新实验：必须匹配编码器、输入尺寸、采样索引、batch/精度和统计模型。旧Uniform Patch不等于FC任意窗口都有命中；旧似然也不能用于更换Gaussian后的评分。
当前入口自动复用的是active fit清单引用的轻量资产；其他大缓存的读取按新实验需要接入，不宣称已自动命中全部历史缓存。

## 原视频精简范围

保留全部当前fit/evaluation/CDF/threshold、P07固定VATEX源域200，以及三个开发域的真实校准候选池。未根据检测分数筛选样本。
VATEX保留4200条：2000 CDF、2000 threshold、200 source-fit。后者清单为`data/catalog/vatex_source_fit200.csv`，不与CDF/threshold重叠。
删除40572个池外视频：34708个额外fake、5800个非必要VATEX、64个旧GenVidBench拟合候选；ViF压缩包在逐成员CRC核对后删除。
这些删除不改变冻结20个开发生成器单元和两个外部域的评价身份。真实候选池后续用于P10前仍须做源级隔离，不自动视作独立样本。
逐视频删除路径和理由在`data/catalog/deleted_video_inventory.csv`；这些原文件未留备份，若重新扩展数据范围需重新获取，不保证可恢复原下载集。

## 当前目录命名与补充资源

代码只有当前主线：测试直接位于`tests/`，五域正式参考位于`precomputed/target_reference/`。不再用论文会议名、实验阶段或paper_v1给活跃目录命名。协议ID和缓存schema的版本仍保留在内容中，用于兼容性检查。

```text
STALL/cache/
  global/coarse_1fps/       按数据集组织的粗扫全局特征
  patch/benchmarks/         开发域与GenVidBench的可复用Patch
  patch/vifbench/           ViF拟合Patch
  fit/target/              816份目标域拟合特征
  fit/vatex_200/           源域200视频统计对照
  models/development/      开发域对照模型
  models/external/         外部域对照模型
  models/sample_efficiency/ 少样本普通Gaussian
  models/reference_source/ 源域与目标域参数对照
  scores/development/      开发域轻量窗口/视频分数
  scores/external/         外部域轻量窗口/视频分数
  contracts/               保留资产的历史抽样和格式合同
```

`packed_v1`等内部名称表示存储格式，不是第二套主线代码。已有NPZ、Patch、冻结raw中的身份字符串不批量改写。旧路径通过`data/catalog/path_migration.json`逐次映射；新实验读取active清单，不能读取历史合同里的旧路径并假定仍存在。

### 父目录datasets：补充资源，不自动加入论文主表

父目录现在只按`aegis/`和`aigvdbench/`组织这些数据。它们曾用于历史探索，不能称为从未查看过的新确认集；要用于当前主线，需另建源级隔离、固定帧索引与明确fit/evaluation预算的清单。

| 当前位置（相对父目录datasets） | 实际内容 | 保留判断 |
| --- | --- | --- |
| aegis/videos | real：dvf109、youtube109；fake：kling111、sora107，共436视频 | 保留唯一原视频，可作后续外部对照 |
| aegis/metadata | 原元数据及Kling/Sora提示文本 | 保留，解释来源与内容 |
| aegis/archives | 下载归档及failed_videos.zip | 暂留；未完成全部归档成员与可用视频的覆盖核验，不能当作全部重复 |
| aigvdbench/views/pools | real test300、val200；fake CogVideoX1.5/Pika/Vidu各300，Open-Sora test300、val200 | 保留基础池；这些是历史划分名称，不代表当前主线角色 |
| aigvdbench/views/test_*、val_* | Open-Sora/Pika历史真假组合 | 保留硬链接视图，非多份独立样本 |
| aigvdbench/views/*_prefixed | 同一批视频采用来源前缀的身份视图 | 保留以解释旧身份，不能与无前缀版合并计数 |
| aigvdbench/views/resplit_cogvideox15、resplit_vidu | 各300条fake的重划分视图 | 保留，与基础池共享视频，不是新增生成器样本 |
| aigvdbench/views/validation_real_200 | 原val200真实校准视图 | 保留；是否能作为新fit须核对评价源组 |
| aigvdbench/real_reserve | 额外300条real | 保留，后续真实参考预算候选 |
| aigvdbench/metadata | 官方Split等元数据 | 保留，重新划分需要 |
| aigvdbench/archives | 已下载的生成器/真实视频压缩包 | 暂留；可能含未提取成员，当前不按已提取小样本推断全包冗余 |

上述视频数量为各视图的文件数，不能相加得到独立视频数或磁盘容量。大量文件存在硬链接；同inode的视频只占一份内容空间。精确逐目录数量和硬链接检查见path_migration.json的workspace_video_inventory。

已清理：AEGIS临时提取目录中436个经inode或SHA256验证与videos库相同的视频入口；唯一的两份提示文本移到metadata，失败包移到archives，extracted目录已移除。重复内容仍可从videos恢复；删除入口的逻辑字节数约1.90GB，不等于实际释放空间。空empty_fake及空父级Patch目录也已移除。

### 父目录cache：按资产类型保留

| 当前位置（相对父目录cache） | 用途与决定 |
| --- | --- |
| frames/d3 | 约611MiB旧解码帧；是可复用输入，保留，不等于保留旧D3代码 |
| global | 约134MiB历史AEGIS/AIGVDBench整体特征；保留，使用前核对骨干、帧索引及预处理，不能自动用于当前DINO协议 |
| manifests | 旧数据划分和帧索引CSV；物理video_path已更新，并保留legacy_path；6997行路径检查全部可访问 |
| models/aegis/full | 历史AEGIS参数，作为已有缓存保留，非当前五域正式参考 |
| models/aigvdbench/real_300 | 300-real旧参数控制 |
| models/aigvdbench/real_300_controls | 同预算旧变体；仅供匹配协议时复用 |
| models/aigvdbench/vidu_real_300_region_3_mean | 300-real、region3均值的旧参数 |
| models/aigvdbench/vidu_real_600_region_3_mean | 600-real、region3均值的旧参数 |
| models/calibration_budget | 历史真实校准数量参数；非本轮B3/双适配的新曲线 |

这些父级模型合计约305MiB，当前仅保留已有可复用缓存，不将旧region控制引入主方法，也不承诺与新协议兼容。参数名中的预算与region有统计含义，因此保留；删除的是冗长的数据集前缀拼接，不删除含义。后续只在真正需要的对照中显式验证并引用，正式运行默认不会扫描父级cache。

迁移验收：34项测试通过；20个active清单文件hash、816份fit资产hash、5个参考包hash均通过；大型Patch索引hash不变，保留88个主仓库对照模型。冻结分数组件重算仍为Macro AUC/AP-real 0.881462/0.882444。此次为路径整理，不是新增检测实验。
