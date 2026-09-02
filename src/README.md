# 源码结构

主干直接放在 `src/`，不再套一层方法名目录：

| 包 | 职责 |
|---|---|
| `branches/` | 唯一的算法差异：Global Spatial/T1 与 Local Spatial/D1/D2 分支 |
| `correspondence/` | Stage 1 的 same-grid、hard local、soft local 与熵置信度 |
| `dynamics/` | correspondence 后的统一 Local D1/D2 证据构造 |
| `data/` | 视频解码、数据清单、采样和特征缓存 |
| `evaluation/` | 指标、bootstrap 和结果表 |
| `src/` 根目录 | 配置、DINO 特征、统计参数、校准、聚合、评分、运行与发布 |

真实依赖方向是：`runner -> pipeline -> 数据/分支/动力学/统计/评测`。Global 与
Local 可独立开关，并在 `pipeline.py` 中完成窗口校准、视频聚合与融合；不存在
backend、方法选择器或原文 STALL 方法包。根目录的 `scoring.py`、`calibration.py`
与 `aggregation.py` 是待归档历史模块，不在当前 runner 主路径中。

`scripts/build_manifest.py` 负责生成数据 manifest；源码中的 `data/` 只保留读取、
采样、视频解码和 Global+patch 特征缓存，不再维护原版 STALL 的 Global-only cache。

没有按实验编号命名的源码。D1、D2、K=1、K=3 和校准数量等差异均由启动脚本中的
显式配置覆盖实现。

Local 参数拟合和 evaluation 评分必须共同调用 `dynamics.build_local_dynamics`，
禁止为某个实验另建平行评分脚本。Global STALL、CDF、effective-K 和结果表继续
复用唯一的 `pipeline.py` 主链。
