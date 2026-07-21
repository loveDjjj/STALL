# 全局 temporal derivative order 是否应补实验

## 当前代码状态

- patch 分支在 `src/patch_matching.py` 和 `src/eval_patch_fast.py` 中显式支持 `same_grid_lag1`、`same_grid_second_order`、`same_grid_third_order` 和 `same_grid_fourth_order`。
- 全局分支当前通过 `src/create_params.py` 和 `src/eval.py` 使用原 STALL 式全局 embedding 校准参数；release 中没有命令行参数可以直接把全局 temporal likelihood 切换到 D=2/D=3/D=4。
- 因此，全局不同阶数不是现有 score CSV 的重汇总问题，而是需要重定义全局时序特征、重建真实视频校准参数并重跑三数据集 score 的新实验。

## 写作建议

当前主文不应伪造全局 D sweep，也不应把 patch D=2 的机制结论外推到 global 分支。合理表述是：

1. 原 STALL global-only 作为固定主干和主要 baseline 保留；
2. 本文的 temporal derivative order 消融限定在新增 patch 分支，因为创新点是局部同网格二阶证据；
3. 若审稿人要求验证“全局高阶导数是否同样有效”，应作为后续 P2 实验：在 `create_params.py` 中增加全局 order 参数，重建 real-calibration `.npz`，再对 ComGenVid、VideoFeedback 和 GenVideo 统一重跑。

## 最小实验计划

1. 在 `src/create_params.py` 增加 global temporal order 参数，默认保持 D=1 以兼容原 STALL。
2. 在 `STALL._scores_from_embs` 对应全局 temporal likelihood 处接收相同 order。
3. 对 VATEX/目标真实校准集分别构建 D=1/2/3 参数。
4. 三数据集重跑 global-only，并与当前 patch D=2 和 Alpha-STALLED 对齐。

当前阶段结论：可以分析其必要性，但不建议在主文中声称已有 global D=2/D=3/D=4 结果。
