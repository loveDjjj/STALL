# 后续AI工作入口

## 当前范围

唯一活跃方法是Alpha-STALLED：冻结DINOv3、目标Global＋同网格归一化Local D2、FC K3、0.5/0.5融合。当前23单元三域Average AUC/AP-real为 **0.874472/0.877075**，结果在results/paper_complete。旧20单元0.881462/0.882444只作历史回归，不是当前论文主成绩。

先读README.md、docs/PAPER_MAINLINE_zh.md、docs/DATA_zh.md和docs/REPOSITORY_RULES_zh.md。长文中的旧20单元表和早期“待运行”段落须按本入口及最新结果判定，不自动重启已完成实验。

局部强控制、新真实拟合池、位置数匹配和CRF23/35重编码均已完成。论文在paper/ieee_alpha_stalled，三张表由export_tables.py校验生成，流程图按用户要求留空。科学论证与近期原文比较见docs/ALPHA_STALLED_SUBMISSION_STRATEGY_zh.md第11节。

## 已退役研究

统计专家、主线专家、监督MoE与Looped已停止。源码、配置和专用测试从当前树移除前已提交在 **efede85**；查看历史用git show或单独检出该提交，不在当前树恢复旧入口。当前仅保留paper/local_direction/reference_confirmation三个配置及论文所需evaluation工具。

已有研究结果、模型检查点、视频和Patch没有因代码清理删除。下列仍是活跃资产路径，不能随停用包清理：
- cache/patch/looped_video：可复用的FP32 FC Patch。
- results/runs/mainline_experts/plans.json、data_manifest.json及global/*.npz。
- results/runs/paper_fit_*、paper_representation_fit_*及参考模型。
- cache/contracts/official_stall中的已核验上游实现。

历史做法、结果和边界见docs/All_Branches_Experiments_and_Data_Summary_zh.md；其他研究说明仅作参考，不是活跃命令清单。

## 科学与数据合同

- 不更新DINO、不训练真假分类器；Gaussian只用指定fit real。方法开发曾查看fake指标，不声称全过程未使用fake。
- Global与Local共享目标真实拟合片段；每域200片段，独立源数另计。拟合为Uniform K3，Local每片段256向量、片段总权重相同。
- 224输入，batch8尾批填充后裁去补齐输出；FP32特征/差分，FP64评分，ridge=1e-5。Global零T1排除、全静态为+Infinity；Local零D2保留。
- Global使用独立VATEX均匀首窗CDF；Local无窗口CDF，视频原始均值按effective-K映射一次。不同Gaussian须分别重评相同参考身份。
- 默认data/manifests/active是旧20单元；完整23单元使用results/paper_complete/evaluation.csv、pairs.csv及short_video扩展。不要把默认CLI称为23单元一键复现。
- 论文Average为三域等权，在paper_complete CSV中对应Macro-3；CSV的Average另一行是23单元等权。AP-real与AP-fake分开。
- 按源组隔离fit/CDF/threshold/evaluation，保留legacy_path、帧索引及固定配对。路径名不同不证明独立。
- 原视频、可复用Patch和模型参数保留。禁止为代码整理重写大型缓存、转换数据dtype或更改冻结清单换行。
- /data保存大资产；不向系统盘或/home上的NVMe复制数据。

## 开发与运行

- 算法在src，scripts/run.py负责CLI；不得从src反向import脚本或恢复旧实验实现。
- 用pyproject.toml规定的Ruff版本格式化；pytest.ini固定testpaths和src导入。开发依赖见requirements-dev.txt。
- 代码格式改变也会改变run的源码hash；不修改旧manifest去绕过恢复校验。新运行使用新输出，旧断点需原源码快照。
- 当前数值函数经AST对照保留；更改预处理、归一化、聚合、CDF、dtype或权重属于科学协议变更，须做对应回归。
- 测试按风险覆盖：配置/CLI、数据角色、评分边界、恢复/原子写、指标和论文表。已结束的实验不因测试通过而自动再跑。
- 新run禁止覆盖、并发持锁；进度以真实PID和完成产物核对，不凭旧status或瞬时GPU利用率重启。
- 中文说明和配置注释，稳定ASCII标识符；不输出凭据，不自动发送消息、发布或推送。
- 保留用户未提交修改，main分支不改，不使用reset --hard或checkout覆盖。提交前核对范围；重构前保护未跟踪源码。
- .gitattributes保护冻结数据原字节，.gitignore排除大权重/raw并精确纳入论文轻表。
- 论文表格校验：python3 paper/ieee_alpha_stalled/export_tables.py --check。
- 完成时说明实际修改、验证、提交及仍存在的限制，不以格式整理宣称新检测收益。

<!-- CODEGRAPH_START -->
## CodeGraph

如仓库根存在.codegraph/，理解或定位代码时先使用codegraph_explore或codegraph explore，再用文本搜索。
如不存在，跳过CodeGraph，不自行索引；索引由用户决定。
<!-- CODEGRAPH_END -->
