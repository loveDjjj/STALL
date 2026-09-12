# Alpha-STALLED 中文论文稿

本目录维护 IEEE 双栏论文正文。当前方法为**目标 Global＋局部归一化 D2，FC K3，0.5/0.5 融合**；当前23单元、三域等权 Average AUC/AP-real 为 **0.874/0.877**。它不是旧20单元的0.881/0.882，也不是监督 Looped 的结果。

正文按“问题—方法—比较—核心消融—结论”组织。局部方向、真实度量和固定预算观察服务同一个研究问题，不将每个组成都包装成独立新算法。新流程图按作者要求留空，旧图没有参与编译。

## 文件与写作职责

近期原文阅读与本次表格组织见[投稿组织方案第11节](../../docs/ALPHA_STALLED_SUBMISSION_STRATEGY_zh.md#11-近期免训练论文的价值定位与本稿修订)。正文突出先测量局部方向、再汇总证据，并区分已有工作的任务条件。

| 文件 | 职责 |
| --- | --- |
| main.tex / main.pdf | 中文工作稿入口与最近一次成功编译的PDF |
| sections/00–07 | 摘要、引言、相关工作、方法、实验、消融、讨论、结论 |
| sections/08_declarations.tex | 简短AI辅助声明；作者确认最终措辞 |
| tables/main_results.tex | 23个生成器的发表值背景比较，保留STALL原表结构 |
| tables/component_ablation.tex | A分支互补；B测量/聚合与位置数匹配；C阶次及标量控制 |
| tables/reference_stability.tex | 五个新真实池的逐域均值、Average标准差及配对增量区间 |
| export_tables.py | 从冻结结果生成上述表格；支持只读 --check |
| references.bib | 书目；文中实际引用的条目进入PDF |
| build.sh / build/ | 编译入口及可再生成的中间文件；成功后更新main.pdf |

旧表和旧方法图已从当前树移除，可从Git恢复点efede85查看；这里只维护三张当前表。历史研究记录见仓库唯一[实验总账](../../docs/All_Branches_Experiments_and_Data_Summary_zh.md)。

## 编译与校验

在仓库根运行：

~~~bash
python3 paper/ieee_alpha_stalled/export_tables.py --check
bash paper/ieee_alpha_stalled/build.sh
~~~

重新导出相同冻结输入的表格时，去掉 --check。本地支持XeLaTeX或[Tectonic](https://tectonic-typesetting.github.io/en-US/install.html)。本工作区的便携工具和字体/宏包缓存均位于父目录 .paper_tools，在 /data 下。编译不读取视频/Patch，不启动GPU实验。其他机器可设置 TECTONIC_BIN 或使用已安装的 xelatex。

Overleaf上传本目录，选择 **XeLaTeX**、入口 **main.tex**；已生成的表格无需上传视频和CSV，也无需执行导出脚本。

## 指标与证据

| 内容 | 仓库内权威输入 |
| --- | --- |
| 当前23单元主结果及Global/D1 | results/paper_complete/ 的 main.csv、main_generators.csv、components.csv、representation.csv |
| pooled及标量控制 | results/runs/local_direction_evidence/ 及 scalar_control/ |
| 位置数匹配、新真实池 | results/runs/reference_confirmation/results/ |
| 固定窗口重编码 | results/runs/reference_confirmation/encoding/ |
| 发表主表 | STALL arXiv:2603.15026v2 Table 1；核对转录在 docs/MANUSCRIPT_REVISION_PLAN_zh.md |

正文 **Average 是三域等权**，对应paper_complete CSV中的 Macro-3；CSV中名为Average的另一行是23单元等权，不能误取。AP为真实正类。表格用三位小数，差值从未舍入数据计算；三位显示并列共同标粗。

发表组与本次组分开，前者的AP及Average照录。两组的信息预算、视频身份和聚合口径未完全一致，主要归因由同协议消融支持。右组总体深红只强调本文结果。

导出脚本校验12份CSV的SHA256、发表转录身份、23生成器映射、三域均值和五次均值/SD。改变科学协议时不能仅更新哈希绕过检查，须完成相应评价并说明来源。

## 复现范围

基础方法配置在 configs/paper.yaml，其默认active清单仍是旧20单元。完整23单元使用 results/paper_complete/evaluation.csv、pairs.csv，并包含 data/manifests/short_video 的8帧扩展；默认 run_main.sh 不是23单元的一键完整复现命令。

位置数匹配、新真实池与CRF23/35重编码均已完成。本轮是取数、写作与排版，不重跑已完成实验。数据角色、短视频、零差分、CDF、运行环境及未舍入结果见[协议说明](../../docs/MANUSCRIPT_PROTOCOL_NOTES_zh.md)、[主线定义](../../docs/PAPER_MAINLINE_zh.md)和对应run。

## 后续修改规范

- 主文围绕局部方向主张，运行恢复、缓存迁移与历史分支不逐项叙述。
- 阶次、位置数和新真实池承担主要论证；目标适配与FC是方法条件。
- 新池结果为五次模型指标均值，不是集成、新默认模型或五个全新数据域。
- 表格用booktabs、caption在上、无竖线；只强调可比组内最佳值，不删除同表不利域或改变正类。
- 必要源级划分、目标真实预算和结果边界留在设置/结果中；工程规范放独立文档。
- 修改后校验取数、编译并逐页检查，不修改IEEEtran.cls或页边距来压缩篇幅。

## 作者最终填写

当前是中文工作稿，未实际投稿。作者姓名、单位、邮箱仍保留占位；基金/致谢、AI声明及目标会议要求需作者最终确认。公开代码URL、许可证和发布版本尚未提供，不声称本地资产已公开。流程图按本轮要求留空；正式投稿前应补绘，转英文后重新检查篇幅。
