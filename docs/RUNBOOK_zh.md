# 新主线运行手册

从STALL根运行，conda环境`stall`。唯一基础配置`configs/paper.yaml`。全局`--config`、`--set`放在子命令之前，未知字段报错。

## 评分和评价

```bash
bash scripts/run_main.sh --help
bash scripts/run_main.sh score --dataset comgenvid --limit 2 --dry-run
bash scripts/run_main.sh --set 'runtime.devices=[cuda:0,cuda:1]' score \
  --dataset comgenvid --output results/runs/new_scores
bash scripts/run_main.sh evaluate --run-dir results/runs/new_scores \
  --pairs data/manifests/active/comgenvid/pairs.csv
```

score默认使用活跃evaluation清单，可用`--manifest`显式提供；limit仅用于小样本检查。
单视频使用`predict --video PATH --dataset NAME`，可用`--indices-json`给固定dense索引。参考域必须指定，不自动推断。
新输出目录禁止覆盖，分数越高越真实，不是真实概率，也没有自动验证的真假阈值。

## 重算与组件

```bash
bash scripts/run_main.sh replay \
  --window-scores results/reference/baseline/window_scores.csv.gz \
  --output results/runs/new_replay
bash scripts/run_main.sh components --run-dir results/runs/new_replay \
  --pairs data/manifests/active/pairs.csv --output results/runs/new_components
bash scripts/run_main.sh bootstrap --run-dir results/runs/new_components \
  --variant full --baseline-run results/runs/new_components \
  --baseline-variant without_gt --pairs data/manifests/active/pairs.csv \
  --output results/runs/new_interval
```

components固定full/global_only/local_only/without_gs/without_gt，不扫描权重。普通score/replay的bootstrap不传variant，组件run必须指定。
现有精选baseline也可直接读取：`components --run-dir results/reference/baseline --pairs data/manifests/active/pairs.csv --dry-run`，无需旧run或源码；冻结参考包不支持resume。
配对身份固定，不重新抽样。三开发域齐备才产生Macro-3。AP-real、完整池指标和独立阈值迁移分开。

## 真实参考拟合

```bash
bash scripts/run_main.sh --set fit.feature_source=video \
  --set 'runtime.devices=[cuda:0,cuda:1]' fit --dataset comgenvid \
  --stop-after gaussian --output results/runs/new_reference
bash scripts/run_main.sh --set fit.feature_source=video \
  --set 'runtime.devices=[cuda:0,cuda:1]' fit --dataset comgenvid \
  --resume --stop-after cdf --output results/runs/new_reference
bash scripts/run_main.sh export --run-dir results/runs/new_reference \
  --output results/runs/new_export
```

默认cache模式复用有hash的小资产，video模式从原视频生成。fit默认读取目标fit、VATEX cdf、目标evaluation、VATEX threshold四份清单，后两者仅检查隔离；可用四个`--ROLE-manifest`参数显式提供。
Gaussian拟合使用Uniform K3/video256；Global CDF用独立Uniform首窗，Local CDF用选择器K3及原effective-K规则。
仅Gaussian完成时paused，不能导出；CDF全部完成后才有完整包。新包不替换默认参数，使用时显式覆盖reference.directory/manifest。
当前fit生成requested-K3参考。K1、其他表示等新实验需实现匹配参考，不是改配置就已完成。

## 论文实验共享入口

前五组实验的拟合、共享证据、官方单窗和表格使用同一个CLI的`study`子命令：

```bash
bash scripts/run_main.sh --set runtime.device=cuda:0 study evidence \
  --dataset comgenvid --manifest data/manifests/active/comgenvid/evaluation.csv \
  --rank 0 --world-size 2 --output results/runs/new_shared_evidence
bash scripts/run_main.sh study tables --dataset comgenvid \
  --evidence-run results/runs/new_shared_evidence \
  --cdf-run results/runs/new_shared_cdf --output results/runs/new_tables
```

另一个GPU进程传rank1，world-size一致；不能两个进程写同一rank。CDF输入使用独立VATEX清单并加`--include-uniform`。这些实验接口显式读取本轮paper_fit和paper_representation_fit参数，尚不是任意参数库的通用实验管理器。
`study official`按官方seed42单窗、官方参数与NumPy评分；`source-fit`、`representation-fit`生成Gaussian而非完整CDF。`--dry-run`仅打印规划，不代表模型/结果已经完成。
已有通过python函数调用启动的队列保持原命令恢复，因为相对/绝对路径是运行身份的一部分；不要换成新CLI后绕过身份检查。新run统一使用CLI。

## 设备、恢复与日志（主线命令）

runtime.devices为空时沿用device；双卡按固定视频行号分片，Gaussian由父进程在完整fit池拟合。
每卡默认4线程、4视频预取、4096MiB估计预算，不含GPU模型和FFmpeg内部内存。score/CDF的prefetch_enabled可关闭以运行同步对照，GPU仍逐视频batch8。
score/replay/fit恢复在原命令及原output上加`--resume`；evaluate在原run-dir/pairs上加`--resume`。输入、配置、源码改变时拒绝恢复，旧run须使用原源码快照或新建运行。
父进程等worker退出且检查点完整才完成；某卡失败则回收本次worker，已完成视频保留。不因短暂无输出自动重启。
评价先在stages/evaluate准备表再发布，持有run锁并保存评价源码；不同内容不覆盖。

run保存配置、命令、源码、输入manifest快照、状态和产物hash。日志在logs/console.log；双卡在workers/STAGE/rank_N/logs/console.log。
completed_steps仅说明完成的阶段，不表示整篇论文已完成。NaN、缺帧、角色交叉、hash错误须修正来源或新建run，不用零值补洞。
科学协议见[PAPER_MAINLINE_zh.md](PAPER_MAINLINE_zh.md)，数据见[DATA_zh.md](DATA_zh.md)。历史实验只维护统一总结，可复用缓存按DATA说明保留。
