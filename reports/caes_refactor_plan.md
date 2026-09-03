# CAES 最小代码重构计划

> 原则：不删除旧K=3 sampler，不复制630GB Patch cache，不把negative exploration带入默认方法，不新增FS0-FS5五套Python脚本。

## 1. 推荐分支基点

从`explore/local-dynamics-likelihood@4949f0c`创建：

```text
feature/calibration-aware-evidence-search
```

理由：默认配置仍是数值一致的Global+Local D2；可靠基础设施已经完整。`005f4d3`之后的`4949f0c`只新增Stage 4结果报告，没有侵入性方法变化。重新从`718f780`cherry-pick会使single-pass、参数追踪和测试产生高冲突风险。

## 2. 目标依赖方向

```text
manifest + raw video
  -> coarse_global_cache (1 FPS, Global fp16)
  -> selector_reference (calibration real only)
  -> temporal_selector
  -> WindowManifest
  -> selected-window on-demand DINO extraction
  -> EvidenceManifest / score-only artifacts
  -> existing calibration + aggregation + fusion
  -> existing metrics/bootstrap
```

selector与detector只通过WindowManifest交互。后续Tail/Fusion不得重新运行selector或DINO。

## 3. 最小新增模块

```text
src/temporal_selection/
  __init__.py
  models.py            # CandidateWindow, SelectedWindow, WindowManifest
  candidates.py        # 8 FPS轴上2秒/0.5秒候选
  selectors.py         # uniform/random/change/anomaly/NMS/stratified
  reference.py         # 1 FPS real-only whitening/CDF
  manifest.py          # schema、hash、JSONL读写与验证

src/data/
  coarse_global_cache.py

scripts/
  build_coarse_global_cache.py
  run_coarse_global_cache.sh
  run_window_selection.py
  run_window_selection.sh
```

第一轮不创建`aggregation/`和`fusion/`新包；Tail/Fusion过门槛后再按真实重复度拆分。

## 4. 现有模块改动

| 文件 | 最小改动 |
|---|---|
| `configs/benchmark.yaml` | 新增`temporal_selection`段，默认`uniform` |
| `src/config.py` | 严格校验coarse_fps=1、stride=0.5、K=3、NMS=0.5、seed |
| `src/data/cache_contract.py` | 支持`coarse_global` cache kind和float16 descriptor |
| `src/features.py` | 暴露Global-only batched extraction，复用相同DINO/preprocess |
| `src/pipeline.py` | 接受显式WindowManifest；uniform缺省路径保持原样 |
| `src/artifacts.py` | 将selector config/reference/window manifest hashes写入run manifest |
| `src/evaluation/*` | 不改公式，只补selector runtime/dense frames统计 |

## 5. Coarse cache schema

每视频一个原子写入条目，第一版不急于packed：

```text
{
  schema_version,
  video_id,
  video_path,
  source_video_sha256,
  frame_indices,          # 从8 FPS downsample轴每8个位置选取
  downsample_positions,
  timestamps_seconds,
  global: float16 [Tc,1024],
  contract_sha256
}
```

路径使用`dataset/split/<sha256(video_id)>.pt`，不再使用可能跨数据集碰撞的stem。根contract绑定encoder repo/checkpoint/layer/resize/normalize、coarse_fps、8 FPS base grid、extractor源码hash和dtype。

预计25,387文件可能带来inode/I/O开销，但总量小于1.2GB；构建稳定后可复用现有pack机制按128-256视频打包。不得先写一份大临时cache再复制。

## 6. WindowManifest schema

每视频一条JSONL：

```json
{
  "schema_version": "caes_window_manifest_v1",
  "video_id": "dataset:path",
  "duration_seconds": 8.2,
  "base_fps": 8,
  "candidate_stride_seconds": 0.5,
  "window_seconds": 2.0,
  "effective_k": 3,
  "candidates": [
    {
      "candidate_id": 0,
      "start_position": 0,
      "frame_indices": [0, 4, 7],
      "start": 0.0,
      "end": 2.0,
      "center": 1.0,
      "feature_change_mean": 0.1,
      "real_anomaly_mean": 0.3,
      "real_anomaly_max": 0.7
    }
  ],
  "selected": [
    {
      "candidate_id": 4,
      "rank": 1,
      "score": 0.72,
      "reason": "stratum_1_real_anomaly_max"
    }
  ],
  "selector": {
    "name": "stratified_real_anomaly",
    "seed": 17,
    "reference_sha256": "...",
    "config_sha256": "..."
  }
}
```

实际`frame_indices`必须完整16个；上例仅为结构示意。manifest写完后执行schema、bounds、distinct frames、distinct windows和hash验证。

## 7. Selector统一接口

```python
select_windows(
    video_id: str,
    downsample_indices: list[int],
    coarse: CoarseSequence,
    reference: SelectorReference | None,
    config: SelectorConfig,
) -> WindowManifest
```

- selector不接收`subset`、`source_model`、fake label或evaluation metrics。
- FS0允许`reference=None`；FS2只用coarse features；FS3-FS5必须有selector reference。
- 输出始终按时间排序，同时保留selection rank。

## 8. FS0基线保护

必须有两层验证：

1. 索引级：`selector=uniform`返回值与`uniform_windows(...,K=3)`完全相同。
2. 特征级：对固定真实/生成样本，用on-demand selected-window extractor与现有strict cache比较Global/Patch token；目标`rtol=0, atol=0`，若底层batch顺序导致非逐位一致，先定位并给出不高于`1e-6`的有依据tolerance。

FS0正式评测直接复用现有C0分数，不重新跑；on-demand路径仅做等价测试。

## 9. Selector reference

每数据集只用选定200 calibration real：

1. 读取1 FPS Global sequence；
2. 计算normalized adjacent difference；
3. reservoir上限不高于全部real transitions；
4. 拟合empirical whitening；
5. 计算全部calibration transition likelihood并排序为CDF；
6. 保存`mu/W/CDF/calibration IDs/coarse contract hash`。

不能复用8 FPS STALL Global T1，也不能读取evaluation real或fake。

## 10. Matched calibration实现

对每个FS：

```text
calibration real coarse -> same selector -> selected dense windows -> branch scores
evaluation video coarse -> same selector -> selected dense windows -> branch scores
```

窗口CDF：Global继续官方VATEX component CDF；Local D2用该selector选出的calibration real窗口raw分数。视频CDF：按selector和effective-K分别从calibration real建立Global/Local reference。Random FS的calibration/test均按video-hashed固定seed。

## 11. Dense selected-window执行

按WindowManifest聚合每视频最多48个原始frame indices，去重解码并一次DINO前向；随后按窗口映射回`[16]`序列，计算现有Global与Local D2。只落盘：

- `window_scores_raw.csv/parquet`；
- selected timestamps/indices；
- 可选Local likelihood field；
- extraction runtime、dense unique frames和失败原因。

不保存Global/Patch token。对于FS1-FS5可将所有selector窗口并集一次提取，减少重复DINO；按selector分别评分/校准。

## 12. Cross-fitting接口

现在实现fold assignment、reference identity和OOF manifest字段，但不在首轮全量执行。FS过Go后：5fold按stable video hash分组；每fold用另4foldfit reference并选择held-out real，拼接OOF video scores；test用全部200real reference。

## 13. Tests

新增不少于以下测试：

- FS0 exact uniform；
- K=3与effective-K；
- 短视频空选择；
- candidate 0.5秒步长与尾对齐；
- frame/window去重；
- bounds；
- random deterministic且不同video seed不同；
- IoU/NMS；
- strata每层最多一个；
- calibration/test selector parity；
- selector不接受label字段；
- coarse cache contract key invalidation；
- 1 FPS/8 FPS reference mismatch拒绝；
- WindowManifest round-trip/hash；
- on-demand与strict cache特征等价；
- matched calibration独立reference。

现有34 tests必须全部保留。

## 14. 实现顺序

1. 创建CAES分支并记录base。
2. models/candidates/selectors纯函数 + tests。
3. WindowManifest schema/hash + tests。
4. coarse cache contract/extractor + small real/fake smoke。
5. selector reference + no-leakage tests。
6. selected-window on-demand extraction + FS0 feature equivalence。
7. FS0-FS5小样本端到端。
8. 三benchmark全量FS single-pass。
9. Go/No-Go后才进入Tail。

## 15. 保留/归档

- 保留探索分支所有negative报告和结果。
- CAES默认不暴露matching/geometry/conditional选项到run脚本。
- 历史Stage1-4脚本不删除，后续可移到`historical/`，但不与FS实现同一提交重构。
- `paper/`本阶段不改，直到FS Go/No-Go后冻结故事。
