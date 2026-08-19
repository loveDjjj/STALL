# 文件结构约定

活跃代码只保留以下层次：

```text
configs/benchmark.yaml     唯一基础配置
scripts/                   Conda 启动入口
src/                       Alpha STALL 的活跃源码
src/branches/              Global 与 Local 分支
src/data/                  数据、视频、采样和缓存
src/evaluation/            指标、bootstrap 和结果表
results/runs/              每一次运行的可追溯结果
release/                   最终冻结版本
```

原文 STALL 不属于活跃源码树，应从官方仓库单独运行。禁止使用“某个实验编号”或
“某个历史版本号”作为活跃 Python 模块、配置文件或工具文件名。实验差异必须在
`run_*.sh` 开头说明，并以 `--set` 传给唯一 runner。
