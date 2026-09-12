# 结果区

| 路径 | 用途 |
| --- | --- |
| paper_complete/ | 当前23单元主表、消融、区间和评价身份；论文Average对应CSV的Macro-3 |
| paper/ | 旧20单元当前方法回归，不能替代23单元主结果 |
| reference/baseline/ | 五域冻结分数与来源；不是可直接resume的旧run |
| reference/comparison_tables.csv.gz | 精选历史对照，按history_experiment/history_table区分协议 |
| runs/local_direction_evidence/ | 当前局部表示和统计归因 |
| runs/reference_confirmation/ | 新真实拟合池、位置数匹配及重编码验证 |
| runs/其他目录 | 已有运行产物；目录名不自动表示仍维护对应源码 |

Git仅纳入主表和必要验收的轻量CSV/JSON。逐视频raw、训练检查点、NPZ和源码快照仍留本地；通过.gitignore逐文件放行论文导出依赖，不开放整个run。

清理停止方向的代码不删除其结果。历史方法与逐子集记录见[实验总账](../docs/All_Branches_Experiments_and_Data_Summary_zh.md)，本轮代码恢复点为efede85。数据身份、模型及源码hash不随目录说明更新而改写。
