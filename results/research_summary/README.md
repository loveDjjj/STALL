# Alpha-STALLED 研究结果统一索引

本目录保存能够进入论文结论的轻量结果索引。完整逐窗口、逐视频、bootstrap
抽样和特征缓存保留在本地 `results/`，不作为 Git release 资产。

## 正式主方法

锁定 Alpha-STALLED U0 使用两个分支和统一配置：

```text
G_k = 0.5 * GlobalSpatial_k + 0.5 * GlobalT1_k
L_k = 0.1 * PatchSpatial_k + 0.9 * same-grid PatchD2_k
G_raw, L_raw = K=3 uniform 2-second window means
G, L = effective-K-matched real-only CDF calibration
S = 0.6 * G + 0.4 * L
```

Local 固定 DINOv3 ViT-L/16 final layer、原始 14x14 patch、region1、mean
aggregation；whitening、Gaussian likelihood 和 CDF 使用 float64。每个数据集
200 条独立真实视频用于校准。生成视频和评测真实视频不参与拟合或校准，locked
run 不再调参；但 `alpha/beta/K` 和结构曾在这三个开发基准上查看生成结果后冻结。

## 运行 provenance

`run_manifests/alpha_stalled_u0_locked.json` 是 locked U0 的可校验运行索引，直接
记录 28 个 input/intermediate/output artifact 的字节数和 SHA-256。由于原始运行
没有保留开始时间与逐进程 exact invocation，它诚实标记为 `reconstructed`；对应
YAML spec 和 canonical commands 可以确定性重建该 JSON。

```bash
conda run --no-capture-output -n stall \
  python tools/build_run_manifest.py \
  --spec configs/run_manifests/alpha_stalled_u0_locked.yaml \
  --output results/research_summary/run_manifests/alpha_stalled_u0_locked.json \
  --check

conda run --no-capture-output -n stall \
  python tools/verify_run_manifest.py
```

## 缓存库存

`cache_inventory.json` 登记本机 `cache/` 的全部 13 个互不重叠组；当前快照覆盖
249,764 个文件和 386.08 GiB 逻辑容量。其政策源为
`configs/cache_inventory.yaml`，人工决策表为 `reports/cache_inventory.md`。
大型特征缓存只做路径/大小布局指纹，索引文件额外做内容哈希。

```bash
conda run --no-capture-output -n stall \
  python tools/build_cache_inventory.py --check

conda run --no-capture-output -n stall \
  python tools/verify_cache_inventory.py
```

清单验证和 release 验证相互独立：locked U0 release 不依赖这些缓存，清单工具
也不具备删除能力。

当前快照中的十个 global/patch feature roots 全部是 legacy，不能因为 shape 与当前
DINOv3 一致就升级为 strict evidence。新空目录由
`src/alpha_stalled/cache_contract.py` 写 root contract 和逐文件 sidecar；协议、验证
成本和历史 direct-loader 边界见 `reports/feature_cache_contract.md`。

正式 Local D1/D2、raw mean-likelihood 和固定 beta 融合已集中到
`src/alpha_stalled/local_branch.py`；`patch_matching.py` 的历史实验边界见
`reports/local_branch_code_boundary.md`。

正式 Global Spatial/T1 max/min raw scoring 和固定等权融合已集中到
`src/alpha_stalled/global_branch.py`；原版 `stall.py` 与 D3/volatility 的边界见
`reports/global_branch_code_boundary.md`。

DINOv3 路径、标准预处理、模型加载和 STALL/PatchSTALL 进程内复用已集中到
`src/alpha_stalled/backbone.py`；它与 strict feature-cache 持久身份的区别见
`reports/backbone_code_boundary.md`。

`data_catalog.json` 是由 `configs/data_catalog.yaml`、三个 canonical index 和 locked
release manifests 确定性生成的数据身份总账。它登记 60,949 个 canonical 视频身份，
其中 22,021 个属于 U0 calibration/evaluation，38,928 个当前未进入 locked U0；
来源级时长与成员覆盖见 `reports/data_catalog.md`。该清单验证身份、index hash 和本机
文件存在性，不声称对 115 GiB 源视频逐文件做内容哈希。

正式特征生产的解码入口集中在 `src/alpha_stalled/video_io.py`。strict cache 要求每个
sidecar 中的帧索引都实际解码成功；legacy 省略缺帧的兼容行为和专用 metadata 工具
边界见 `reports/video_io_code_boundary.md`。

locked manifest/raw shard/K1 reference 读取集中到 `src/alpha_stalled/u0_protocol.py`，
严格解码与 raw scoring 集中到 `src/alpha_stalled/u0_scoring.py`，窗口/视频两级校准
集中到 `src/alpha_stalled/u0_analysis.py`，
generator-pair、dataset/Macro-3 指标和 paired bootstrap 集中到
`src/alpha_stalled/metrics.py`。正式 scorer、analyzer 和 verifier 不再相互导入
`tools/` CLI；兼容出口和剩余 D3 历史边界见 `reports/u0_protocol_code_boundary.md`。

`tool_dependency_inventory.json` 由 `configs/tool_dependencies.yaml` 和当前 `tools/*.py`
AST 确定性生成。当前登记 126 个工具、8 条历史内部边、50 个隔离入口和 0 个循环；
已无待迁移边，8 条均作为紧耦合证据保留，并归入4个retained family。每个family
登记完整工具端点、关键证据文件大小/SHA-256、归档目标和移动门槛。可读清单见
`reports/tool_dependency_inventory.md`，由 `tools/verify_tool_dependencies.py` 检查。
清单还覆盖全部126个入口的唯一生命周期：formal release 10、paper evidence 45、
governance 16、historical frozen 38、research utility 1、compatibility 16；正式发布与
治理入口不得依赖其他CLI，现存内部依赖不得跨生命周期。
其中research utility和compatibility共17个入口进一步按实际文本引用分类：14个
`compatibility_wrapper`、3个`retain_referenced`。十三个journal utility实现已移入
`research_archive/tools/journal_experiments/`，旧pre-U0 verifier移入
`research_archive/tools/pre_release_assets/`；原命令
继续可用；机器清单记录引用位置以及原始、包装器、归档实现和结果证据哈希。
历史 pre-release multi-window 的身份、窗口、解码、resume 和 calibrated scoring
基础设施已集中到 `src/alpha_stalled/legacy_window_scoring.py`；正式 U0 不导入它。
duration-aware 23-source 的协议任务、联合窗口解码和full-coverage cohort/指标已分别
集中到 `duration_aware_protocol.py`、`duration_aware_scoring.py` 和
`duration_aware_metrics.py`，相关CLI均已加入零内部工具依赖的隔离清单。
U0真实校准扩展的reserve设计、size/seed候选、cross-domain/OAS两级校准集中到
`u0_calibration_experiments.py`；locked与外部K1缓存评分共同调用
`u0_scoring.score_raw_components`，不再从评分CLI反向导入。
历史multiscale、clean-universal与Joint Typicality的窗口参考集中到
`historical_window_analysis.py`；Local-D2 residual和multi-window feasibility共用
`legacy_local_d2_protocol.py`的数据集与strict交集定义。
历史multi-window分析同时支持完整分片和release保留的合并窗口CSV，并拒绝部分
分片与合并CSV的隐式混用，因此清理运行中间分片后仍能从发布产物复算统计结果。

| 数据集 | U0 AUC/real-positive AP |
|---|---:|
| ComGenVid | 0.8968/0.9064 |
| VideoFeedback | 0.8597/0.8689 |
| GenVideo | 0.8657/0.8416 |
| **Macro-3** | **0.8741/0.8723** |

相对 Original STALL Macro AUC/AP `0.8388/0.8428`，U0 提高
`+0.0353/+0.0295`；相对统一 K1 `0.8632/0.8636`，K=3 提高
`+0.0109/+0.0087`，AP 的 95% paired bootstrap CI 为
`[+0.0057,+0.0120]`。加入 Local 相对 K3 Global-only 提高 `+0.0237 AP`；
加入 Global 相对 K3 Local-only 提高 `+0.0431 AP`。

## 已确认边界

| 方向 | 结果 | 决策 |
|---|---:|---|
| K=5 / all-window | Macro AP 0.8672 / 0.8695 | 成本更高且不超过 K=3，拒绝 |
| bottom-2 / hybrid | 0.8684 / 0.8691 | lower-tail 过度惩罚，拒绝 |
| Joint Typicality J2/J3 | 0.8437 / 0.8602 | 低于固定线性融合，拒绝 |
| Spatial-mean residual | 0.8609，delta -0.0088 | CI 全负，4/20 生成器不下降，拒绝 |
| Fine+coarse | 0.8687 | 未超过 fine-only，拒绝 |
| late+final layer | 0.8698，delta 约 +0.0002 | CI 跨 0 且两个数据集下降，拒绝 |
| cross-layer min / motion gate | 0.8695 / 0.8692 | Macro AP 均下降，拒绝 |
| D3/D4、multi-lag、hard/soft | 均未超过 same-grid D2 | 不重复搜索 |

PatchSpatial 不是主要信号：统一 K1 中从 PatchD2-only 加入 0.1 PatchSpatial
使 Macro AP 下降约 0.0070。U0 保留 beta=0.1 是因为配置在最终审计前已冻结，
而不是根据敏感性结果事后选择。

## 23-source 与全覆盖协议审计

严格 2 秒主表之外，duration-aware 协议恢复 VideoFeedback Hotshot-XL、
GenVideo HotShot 和 MoonValley 的 1 秒/8 帧特例，覆盖 23 个生成器和全部
45,185 条生成视频分数。其目标域 Local 校准规模由 held-out real 误差冻结为
ComGenVid 600、VideoFeedback 400、GenVideo 1,500；测试身份与校准身份互斥。

冻结 seed-42 的 37,993 个生成器级平衡 pairs 上，K3 Global 与 Alpha 的
Macro-3 AUC/AP_fake/AP_real 分别为 `0.8404/0.8200/0.8454` 与
`0.8721/0.8608/0.8741`。全 45,185 fake 覆盖并将 AP 类别先验固定为 50% 时，
Macro-3 分别为 `0.8438/0.8260/0.8476` 与 `0.8733/0.8629/0.8748`；
All-23 分别为 `0.8373/0.8194/0.8401` 与 `0.8619/0.8500/0.8618`。

原论文固定随机窗口的公平因子审计也已完成。全 45,185 fake 覆盖下，
K1 STALL 与当前 K3 Alpha 的 Macro-3 AUC/AP_fake/AP_real 分别为
`0.8335/0.8188/0.8377` 和 `0.8733/0.8629/0.8748`；All-23 分别为
`0.8281/0.8117/0.8330` 和 `0.8619/0.8500/0.8618`。因此 All-23 增益为
`+0.0338/+0.0383/+0.0288`。平衡 seed-42 cohort 的 1,000 次配对 bootstrap
中，K3 Alpha 相对 K1 STALL 的 AP_fake/AP_real 平均增益为
`+0.0457/+0.0392`，95% CI 分别为 `[+0.0404,+0.0511]` 和
`[+0.0354,+0.0432]`。

因子分解表明：保留原 K1 窗口只加入 Local，AP_fake 提高 `+0.0377`
（CI `[+0.0334,+0.0425]`）；K1 Alpha 再改为 K3 Alpha，提高 `+0.0085`
（CI `[+0.0057,+0.0112]`）。主要增益来自 Local D2，三窗口覆盖贡献较小但
独立为正。完整结果见 `reports/original_k1_full23_factorial.md`。

100 个平衡身份 seed 的 Alpha-minus-Global AP_real 增益范围为
`[+0.0262,+0.0294]`，AP_fake 范围为 `[+0.0350,+0.0425]`。全覆盖 AP_fake
在 20/23 个生成器上提高；主要负迁移为 Text2Video-Zero `-0.0535`，
VideoCrafter2 和 GenVideo Lavie 的下降分别为 `-0.0039/-0.0015`。

原 STALL 论文 Table 1 明确以生成视频为 AP 正类，而本项目冻结主表历史上以
真实视频为正类。两者必须分别写为 AP_fake/AP_real；AUC 可反向保持不变，
AP 不能只改标签名后直接比较。`Macro-3` 是三个数据集等权，`All-23` 是 23 个
生成器等权，也不能互换。完整审计见
`reports/original_stall_protocol_and_coverage_audit.md` 和
`reports/full_coverage_paper_protocol.md`。

## 校准与数值稳定性

- 25/50/100/200 条真实视频的五 seed Macro AP 均值为
  `0.8237/0.8487/0.8628/0.8680`；200 条的 seed 标准差为 `0.0022`。
- Local 比 Global 更依赖校准规模。方法定位是目标域真实视频无监督校准，
  不是 target-data-free universal detector。
- FP32 rank-1023 whitening 会产生 batch-shape 数值漂移；float64 score stage
  在 batch 1/4/8/16 下最大误差不超过 `2.27e-13`。
- Release 包含 600 条校准视频、21,421 条评测视频、58,496 个窗口，
  calibration/evaluation overlap 为 0；61 项 release validation 全部通过。
- 三份锁定 200-real Local 参数已随 `release/u0/params/` 发布；完整 release
  约 `57 MB`，不包含 DINO 特征缓存或原始视频。
- 跨域校准中 target-domain 对角线最终 AP 为 `0.8723`，单一源 off-domain
  均值为 `0.8489`；Local target/off-domain AP `0.8292/0.7425`，显著比 Global
  `0.8486/0.8475` 更依赖目标域。
- Pooled-200/600 Macro AP 为 `0.8628/0.8636`，没有替代目标域 200-real。
- OAS 将 Macro AP 只提高 `+0.000043`，CI 跨 0 且 seed 标准差略增，正式拒绝。
- 锁定后 GenVidBench 上 STALL/Clean K1/U0 AP 为 `0.8043/0.8369/0.8495`；
  U0 相对 STALL/K1 的 AP CI 分别为 `[+0.0297,+0.0634]` 和
  `[+0.0017,+0.0247]`，确认主增益可迁移到新生成器。
- 300-real 锁定注入并未证明通用异常定位：K3 对局部冻结的最终 score drop
  仅 `+0.0023`，对全帧冻结为 `-0.1667`，对合成切镜为 `-0.0264`；局部编辑
  patch-time AUPRC 仅 `0.0670--0.1558`。Local D2 是真实分布典型性证据，不能
  解释成对任意合成伪影都单调响应的语义定位器。
- 1,600-video 锁定鲁棒性子集中，Scenario-A 的 CRF23/drop10 AP 均下降约
  `0.0024`，其 95% CI 分别为 `[-0.0071,+0.0032]` 和
  `[-0.0057,+0.0006]`；这是有限观测变化而非等效性证明。CRF35、resize、
  drop25、repeat25 和 4 FPS 的 AP 区间全负；matched CDF 未恢复严重退化。
  完整结果见 `u0_robustness_bootstrap_deltas.csv`。

## 历史结果分类

- `0.8737/0.8750` 及早期逐数据集高分结果存在 sample calibration leakage，
  只能作为审计记录，不能进入有效主表。
- `0.8694/0.8697` 没有样本泄漏，但 region/aggregation 使用目标 fake 指标选择，
  命名为 Historical dataset-specific tuned baseline。
- `0.8725/0.8722` 只统一了 temporal 分支，PatchSpatial 仍继承历史聚合，
  属于 pre-release audit result。
- 历史 clean single-window `0.8570/0.8600` 用于展示方法演进；统一 K1
  `0.8632/0.8636` 才是 K=3 覆盖的直接因果对照。

## 主要入口

- 锁定配置：`configs/alpha_stalled_u0_locked.yaml`
- 发布验证：`tools/verify_u0_locked_release.py`
- 最终逐视频分数：`release/u0/final_video_scores.csv`
- 协议与结果报告：`reports/u0_*.md`
- 当前实验注册表：`reports/u0_experiment_registry.csv`
