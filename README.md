# Alpha-STALLED

使用冻结DINOv3表示、真实统计度量和局部归一化二阶方向检测生成视频。主线采用目标Global＋Local D2、Feature-change K3及0.5/0.5融合；不训练真假分类器，需独立目标真实拟合集。

当前论文覆盖23个生成器单元，三域等权Average AUC/AP-real为 **0.874472 / 0.877075**。旧20单元的0.881462/0.882444仅作历史回归锚点，不能与当前范围混用。位置数匹配、新真实拟合池和重编码验证均已完成。

- [IEEE论文与构建说明](paper/ieee_alpha_stalled/README.md)
- [完整指标](results/paper_complete/RESULTS_zh.md)
- [方法与数据协议](docs/PAPER_MAINLINE_zh.md)
- [数据和缓存](docs/DATA_zh.md) · [运行手册](docs/RUNBOOK_zh.md)

## 安装与运行

服务器使用conda环境stall，新环境按environment.yml安装。DINOv3源码/权重、视频、特征缓存和目标参考NPZ不随Git提交；具体资产见数据说明。参考包manifest和冻结评价清单纳入版本控制。

~~~bash
bash scripts/run_main.sh --help
bash scripts/run_main.sh score --dataset comgenvid --limit 2 --dry-run
bash scripts/run_main.sh --set 'runtime.devices=[cuda:0,cuda:1]' score \
  --dataset comgenvid --output results/runs/new_scores
bash scripts/run_main.sh evaluate --run-dir results/runs/new_scores \
  --pairs data/manifests/active/comgenvid/pairs.csv
~~~

默认active清单沿用旧20单元。完整23单元身份为results/paper_complete/evaluation.csv及pairs.csv，8帧扩展在data/manifests/short_video；上述默认命令不等于23单元的一键完整复现。新run禁止覆盖，代码或输入改变后不混用旧断点。

## 代码结构

~~~text
configs/
  paper.yaml                   主方法
  local_direction.yaml         局部表示与归因对照
  reference_confirmation.yaml  新真实池与重编码验证
scripts/
  run.py / run_main.sh          统一CLI与环境入口
src/
  features.py / selection.py    冻结特征与窗口
  reference.py / reference_fit.py
  math_utils.py / workflow.py   数值评分与视频流程
  artifacts.py / execution.py   产物、恢复及调度
  data/                        视频解码、预取、清单
  branches/                    Global评分
  evaluation/                  当前论文评价及复现工具
tests/                         活跃功能、数值与恢复测试
paper/ieee_alpha_stalled/       论文与三张自动导出表
data/manifests/                 冻结身份，不做格式化或换行转换
precomputed/target_reference/   本地NPZ；Git只记录清单与验收
results/                       轻量指标版本化，大raw/检查点留本地
~~~

已退役的专家、MoE与Looped源码保存在Git恢复点 **efede85**，当前分支只维护论文主线及必要验证。已有研究结果、检查点和可复用Patch仍在原位置；资产目录中出现旧研究名，不代表依赖已删除的Python包。

## 开发检查

~~~bash
conda run -n stall python -m pip install -r requirements-dev.txt
conda run -n stall python -m ruff check src scripts tests paper/ieee_alpha_stalled/export_tables.py
conda run -n stall python -m ruff format --check src scripts tests paper/ieee_alpha_stalled/export_tables.py
conda run -n stall python -m pytest -q
python3 paper/ieee_alpha_stalled/export_tables.py --check
~~~

Ruff版本及规则固定在pyproject.toml；pytest.ini显式设置src导入路径。格式化只针对代码，不处理模型、结果、清单或缓存。论文所需轻量CSV已纳入Git，新检出仓库可直接执行表格校验，无需GPU。

修改约束见[仓库规范](docs/REPOSITORY_RULES_zh.md)和[AGENTS.md](AGENTS.md)。历史方法、结果及边界见[实验总账](docs/All_Branches_Experiments_and_Data_Summary_zh.md)。
