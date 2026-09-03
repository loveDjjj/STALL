# CAES Design Review

> 日期：2026-09-03  
> 审查基线：`explore/local-dynamics-likelihood@4949f0c`  
> 目标：在不改变 DINO、Global/Local evidence、K、窗口长度、calibration IDs和融合的前提下，只比较窗口位置。

## 1. 当前 K=3 sampler 在哪里？

主实现是 `src/data/sampling.py::uniform_windows`：对 manifest 中的 8 FPS `downsample_idxs`，令 `max_start=L-16`，使用 `rint(linspace(0,max_start,K))` 生成起点，切取连续 16 帧并删除完全重复窗口。`src/pipeline.py::_window_features` 在 calibration/evaluation共用该函数；`src/data/patch_cache.py::cached_uniform_frame_indices` 将 K=3窗口并集写入现有严格缓存。

FS0 必须直接调用这一函数，不能重新实现“近似 uniform”。

## 2. Candidate window 如何定义？

候选在现有 8 FPS离散轴上定义，而不是按容器浮点 timestamp直接索引：

```text
downsample positions = 0 ... L-1
window length        = 16 positions
stride               = 4 positions = 0.5 s
candidate starts     = 0,4,8,... <= L-16
```

若 `L-16` 不可被4整除，追加尾对齐起点 `L-16`，保证末尾可被检查；随后按完整16帧索引去重。每个候选保存 start position、原视频 frame indices、start/end/center seconds和合法性原因。

## 3. duration/FPS/effective-K 如何处理？

- manifest由 `scripts/build_manifest.py::downsample_frames` 将原 FPS降到8 FPS，不上采样；原 FPS<8的视频构建阶段已排除。
- `L<16` 继续使用现有 short-video policy；不为 CAES 重复帧补齐。
- 合法候选少于3时，selector最多返回候选数，形成 effective-K=1/2。
- Random/NMS/stratified均不得复制窗口凑到3。
- 视频级 CDF继续按 effective-K分别建立，但每个 selector独立建立自己的 reference。

## 4. 当前 cache 是否包含 arbitrary temporal positions？

不能。`cache/patch_embeddings_k3_2s_8fps/.stall_cache_contract.json` 明确记录 `uniform_window_union, window_count=3, window_frames=16`。每条最多保存FS0首/中/尾48帧；示例 payload为：

```text
global [48,1024] float32
patch  [48,196,1024] float32
```

因此现有630GB cache只能精确复现FS0，不能评分FS1-FS5任意0.5秒候选位置。

## 5. 630GB cache 到底存什么？

共有22,820 entries、718 packed shards。每帧同时保存DINOv3 ViT-L/16最终归一化CLS token和14x14 patch token，float32；另存grid size、原视频frame indices和strict contract。体积几乎全部来自`patch [T,196,1024]`，不是Global。

## 6. 能否从现有 cache 得到完整视频 coarse Global？

不能。只覆盖FS0窗口，长视频中间绝大多数位置不存在。用现有48帧插值会把selector限制在baseline已观察位置，无法公平检验adaptive search。

## 7. 新 coarse Global cache 多大？

对当前development+external manifests去重后共25,387视频、54.51小时。按1 FPS等间隔位置约218,781帧（不强行追加不足1秒的尾间隔）：

```text
218,781 * 1024 * 2 bytes = 0.417 GiB
```

考虑torch/metadata/文件系统开销，预计0.7-1.2 GiB。只跑三个development benchmark约0.405 GiB张量。即使保留逐条strict metadata，也远低于69GB剩余空间。

## 8. 如何避免复制大 cache？

1. 现有630GB cache只读，FS0直接复用。
2. 新建`cache/coarse_global_1fps/`，只存float16 Global与frame indices。
3. WindowManifest单独存JSONL/CSV，预计几十至数百MB以内。
4. adaptive dense窗口按需解码、DINO提取、立即计算Global+Local D2 raw score；不落盘Patch token。
5. Tail阶段最多保存`(T-2)x14x14` float16/float32 likelihood field，三个窗口/视频约0.5-1.5GB量级。

## 9. adaptive selected windows 的Patch token是否已有？

通常没有。只有当某个adaptive窗口恰好与FS0窗口重合时已有；不能假设或为个别视频混用不同提取路径，第一版adaptive统一按需提取selected windows。

## 10. on-demand / selected-window cache / score-only cache如何选？

推荐：**on-demand extraction + score-only cache**。

- selected-window Patch cache每视频仍最多48帧，体积接近另一份630GB，禁止。
- 纯on-demand每个后续tail/fusion实验会重复DINO，浪费时间。
- score-only保存每窗口Global component raw/percentile、Local D2 raw和可选likelihood field；后续aggregation/fusion/calibration不再提取Patch。

## 11. Global temporal statistics时间间隔是什么？

正式Global T1使用8 FPS窗口内相邻帧，间隔约0.125秒。Coarse scan使用1 FPS相邻帧，间隔约1秒，两者分布不一致。

## 12. selector是否必须独立1 FPS reference？

必须。禁止复用官方VATEX 8 FPS Global T1 reference。对1 FPS normalized Global differences，按每数据集calibration real重新拟合selector专用`mu/W/window-free transition CDF`，并在contract记录coarse FPS、frame selection、encoder和样本IDs。

## 13. calibration/evaluation sampling目前在哪里分开？

`src/pipeline.py::resolve_dataset_manifests/run_from_cache`分别加载`*_calibration.csv`和`*_evaluation.csv`，`_ensure_disjoint`无条件检查路径互斥，`_choose_calibration`只从real选择200条。两者最终都调用同一`_score_windows`，但输入rows不同。

CAES应在manifest阶段保持相同模式：selector接口完全不接收subset label；pipeline分别传入calibration rows和evaluation rows，输出相同schema的WindowManifest。

## 14. 如何保证selector parity？

- selector只接收video identity、8 FPS frame index轴、coarse features、selector reference和固定config；不接收`subset/source_model/fake label`。
- 对FS1 random，seed由`global_seed + stable video_id hash`派生，calibration/test同规则。
- 每个selector的calibration real也执行candidate generation、ranking、NMS/strata和effective-K。
- 每个selector建立独立window CDF和video-level effective-K CDF。
- run保存所有calibration/test selected windows及selector reference hash。
- 单元测试用同一无标签输入验证calibration/test返回完全相同选择。

## 15. 5-fold cross-fitting成本如何？

算法实现成本中等，计算成本低于dense detection但不是零：

1. coarse Global只提取一次；
2. 每fold重新拟合1024维selector whitening；
3. held-out 40个real运行selector；
4. test使用全部200real reference。

主要成本是5次1024维covariance/eigh，每数据集预计数分钟；不增加Patch提取。建议接口现在预留fold/reference identity，但Stage FS第一轮先用standard matched calibration。只有最优selector过Go或出现明显calibration shift时，再在9/17后补5-fold strict protocol；否则deadline前收益不足。

## 16. 是否有更自然的selector design？

有两点修改：

1. **coarse signal对齐8 FPS grid。** 1 FPS帧从现有`downsample_idxs[::8]`选择，而不是重新按原FPS round；不强行追加不足1秒的尾帧，保证所有selector transition具有相同时间间隔。
2. **FS5优先于FS4。** K=3下Temporal NMS只防局部重叠，仍可能把三个窗口放在同一半视频；三strata直接保证全局coverage，无需relevance/coverage权重，是更适合本仓库和deadline的主候选。

另外，FS3 primary固定使用窗口内coarse anomaly mean；max作为诊断列保存但不作为第二主候选，避免隐性二选一。

## 17. Branch基点审查

`refactor/...@718f780`到`explore/...@4949f0c`新增65个文件/约4,893行。虽然包含negative method code，但核心baseline保持逐位一致，并新增：

- 0.1%/1% FPR与FPR@95TPR；
- 显式calibration IDs和Local参数SHA；
- per-window/per-video标准产物；
- single-pass cache scan；
- pairwise AUC/AP bootstrap；
- 34个tests；
- modular dynamics/correspondence/likelihood。

negative方法位于显式config和独立模块，不改变默认`finite_difference/same_grid/conditional=false/empirical`。从release重新cherry-pick会有较高冲突和遗漏风险。因此推荐从**`4949f0c`**创建CAES分支，而不是任务文本已过时的`005f4d3`。CAES默认配置继续复现baseline，negative入口保留作证据但不进入新主方法。

## 18. KEEP / MODIFY / REJECT

### KEEP

- `uniform_windows`原实现与FS0 exact reproduction；
- 严格manifest split和disjoint检查；
- DINOv3 extractor与cache contract；
- single-pass评分、完整metrics/bootstrap/artifacts；
- Local D2-only，Local Spatial继续关闭。

### MODIFY

- 新增小型coarse Global cache kind与独立contract；
- 将窗口位置抽象为WindowManifest输入，`_window_features`不再只内部生成uniform；
- 新增`temporal_selection/`最小模块；
- adaptive路径使用on-demand dense extraction + score-only artifacts；
- 每selector独立matched calibration。

### REJECT

- 删除或重写旧sampler；
- 第二份selected-window Patch cache；
- 直接复用8 FPS Global T1 reference；
- test adaptive/calibration uniform；
- 第一轮wavelet/DPP/bandit；
- fake-guided selector参数选择；
- 重启Local Spatial、matching、geometry或conditional D2。

## 19. Blocking issue检查

没有代码层blocking issue。真正约束是现有cache不含arbitrary positions，因此FS1-FS5完整评分必然需要一次coarse Global构建和一次selected dense DINO提取。该代价可控，但FS0与adaptive走不同feature来源时必须做数值等价测试：对FS0窗口，on-demand extractor与现有strict cache的Global/Patch token应在明确tolerance内一致，否则停止实验。

真实4视频smoke显示：coarse `batch=64,float16` 与旧cache重合Global的cosine最低
`0.99999982`、max absolute difference约`4.87e-4`，可用于同contract selector；
dense `batch=8`重新提取Global相对旧cache的max absolute difference约`2.63e-6`。
因此coarse不与8 FPS参考混用，dense FS0等价测试暂定`atol=3e-6`，并进一步要求
最终窗口/视频分数回归，而不能只靠feature tolerance证明baseline一致。
