# 配置说明

活跃主干只保留 `benchmark.yaml`。它包含我们方法的 Global/Local 分支、采样、校准、
数据范围和指标的全部默认参数，并在每一段前用中文说明用途。

不要为 D1、D2、K=1、K=3 或单独数据集复制 YAML。对应 `run_*.sh` 在脚本开头
解释实验目的，并通过 `--set key=value` 覆盖该实验唯一需要改变的字段。每次运行
都会把最终合并结果写入 `results/runs/<run-name>/resolved_config.yaml`。

原文 STALL 的官方基线不使用本配置，也不在本仓库复现。
