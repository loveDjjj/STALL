# 结果区

- reference/baseline：当前五域冻结分数、窗口raw、指标及最小身份；可由components/bootstrap读取，不是可resume的旧run。
- reference/comparison_tables.csv.gz：精选对照表，history_experiment/history_table区分实验和口径。
- reference/manifest.json：来源和内容hash。
- runs：后续新实验输出。不要把无用工程smoke永久堆积在这里。

历史配置、做法、逐子集结果只维护docs/All_Branches_Experiments_and_Data_Summary_zh.md。
可复用位置场、选窗、核心raw和参数在cache中，路径映射见data/catalog/retention_cleanup.json。旧报告和源码不再保留。
