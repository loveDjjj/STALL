# 源码结构

主干直接放在 `src/`，不再套一层方法名目录：

| 包 | 职责 |
|---|---|
| `branches/` | 唯一的算法差异：Global Spatial/T1 与 Local Spatial/D1/D2 分支 |
| `data/` | 视频解码、数据清单、采样和特征缓存 |
| `evaluation/` | 指标、bootstrap 和结果表 |
| `src/` 根目录 | 配置、DINO 特征、统计参数、校准、聚合、评分、运行与发布 |

依赖方向是：`runner -> 数据/特征/分支/校准/评测`。Global 与 Local 可独立开关，
但只在 `scoring.py` 和 `calibration.py` 汇合；不存在 backend、方法选择器或原文
STALL 方法包。

`scripts/build_manifest.py` 负责生成数据 manifest；源码中的 `data/` 只保留读取、
采样、视频解码和 Global+patch 特征缓存，不再维护原版 STALL 的 Global-only cache。

没有按实验编号命名的源码。D1、D2、K=1、K=3 和校准数量等差异均由启动脚本中的
显式配置覆盖实现。
