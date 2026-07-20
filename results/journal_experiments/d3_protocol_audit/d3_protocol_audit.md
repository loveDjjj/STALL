# D3 baseline protocol 审计

本审计不运行 D3，不新增外部 baseline 分数；它只把 `2603.15026v2` 附录 A.3/A.4 中明确影响 D3 数值的协议因素，映射到当前 Alpha-STALLED release。
用途是避免把不同采样、不同类别比例或不同 encoder 的 D3 结果直接放入同一表格。

## 结论

- 当前 Alpha-STALLED 主线指标与参考文献的公平比较原则一致：2 秒、8 FPS 窗口；pairwise balanced real-vs-generator AUC/AP；不使用生成样本训练。
- 尚未完成的是外部 D3 实跑，因此论文中不能声称已经重跑 D3。可以声称已经完成 protocol audit，并列出若审稿要求时的重跑条件。
- GenVideo 是最高风险数据集：参考文献指出 D3 官方协议曾对 3 FPS MSR-VTT 真实视频做 3→8 FPS 帧复制，这会影响依赖帧间差分的 detector。任何 D3 对照都必须避免该路径。

## 协议对照矩阵

| 审计项 | 参考协议 / 风险来源 | Alpha-STALLED 当前协议 | 状态 | 下一步 |
|---|---|---|---|---|
| Input duration and FPS | Reference STALL standardizes evaluated videos to 8 FPS and 2 s, yielding 16 frames, except 1 s generators. Temporal detectors can gain or lose performance from frame-rate artifacts rather than generation artifacts. | Current release indexes store 1/2/3/4 s windows; main baseline uses 2 s compact cache where available, and short-source coverage gaps are documented separately. | aligned_for_main_2s_release | If reporting external D3 numbers, run D3 only on the same index rows and same frame indices used by Alpha-STALLED. |
| Balanced pairwise evaluation | Reference STALL uses pairwise real-vs-generator comparisons with equal real and generated counts. AP is sensitive to class imbalance; using many more real videos can inflate or deflate AP depending on the score distribution. | All committed Alpha-STALLED metrics use src/metrics.py pairwise balanced AUC/AP. | aligned | Do not compare against D3 numbers produced with an unbalanced test set. |
| GenVideo real-video FPS source | Reference STALL reports that D3's official GenVideo protocol upsampled 3 FPS MSR-VTT real videos to 8 FPS by frame duplication. Duplicating real frames creates artificial temporal redundancy and can inflate temporal-difference detectors. | The current GenVideo index includes explicit fps and uniform window indices; this audit does not find or endorse duplicated-frame upsampling as a release protocol. | must_verify_before_external_d3_rerun | Before any D3 rerun, audit duplicate-frame rate on the exact GenVideo real videos used for scoring. |
| Encoder / feature backbone | Reference STALL reports D3 with DINOv3 for consistency and X-CLIP16 for completeness. Changing D3's encoder changes the measured detector rather than only the protocol. | Alpha-STALLED uses DINOv3 for global and patch branches. | requires_explicit_labeling | If D3 is rerun, report D3-DINOv3 and optionally D3-X-CLIP16 as separate rows. |
| Training / calibration data | D3 is training-free; STALL/Alpha-STALLED use real-video calibration statistics and no generated samples for fitting. Using generated samples or test-batch labels for calibration would invalidate the zero-shot claim. | Global STALL uses VATEX-derived precomputed parameters; patch branch uses real-video percentile calibration from the target benchmark's real subset in current release assets. | aligned_zero_shot_but_calibration_source_differs | State calibration source explicitly when comparing to D3; do not describe target-real patch calibration as synthetic training. |
| Metric direction and positive class | Reference reports AUC and AP; AP is defined for generated videos as the positive class. A sign error can invert D3 or Alpha-STALLED results. | Repository score convention is higher-is-real; metric wrappers convert direction consistently for AUC/AP. | aligned_with_direction_check | Keep raw score direction and AP positive class documented in external-baseline tables. |

## 当前 index 覆盖审计

| 数据集 | subset | sources | videos | fps min/median | duration min/median | 有 2s window |
|---|---|---:|---:|---:|---:|---:|
| ComGenVid | annotated | 2 | 3400 | 24.00 / 27.00 | 4.00 / 8.00 | 3400 |
| ComGenVid | real | 1 | 1700 | 10.00 / 29.97 | 1.74 / 8.02 | 1698 |
| VideoFeedback | annotated | 11 | 33581 | 8.00 / 8.00 | 1.00 / 2.00 | 30330 |
| VideoFeedback | real | 2 | 4080 | 8.00 / 8.00 | 2.00 / 3.00 | 4080 |
| GenVideo | annotated | 10 | 8204 | 8.00 / 10.00 | 1.00 / 2.00 | 5639 |
| GenVideo | real | 1 | 9984 | 8.33 / 29.97 | 9.56 / 13.00 | 9984 |

## 若审稿要求补 D3，应执行的最小协议

1. 使用当前 `cache/indexes/<dataset>.csv` 的同一批视频和同一窗口索引；短视频来源继续按 `results/paper_tables/patch_coverage_gaps.md` 单独说明。
2. 每个生成器与真实视频做 pairwise balanced AUC/AP；不要使用 D3 官方 unbalanced AP 作为主表数值。
3. GenVideo 真实视频不得通过帧复制从 3 FPS 补到 8 FPS；若发现低 FPS 原文件，应剔除或回到高帧率来源。
4. D3-DINOv3 与 D3-X-CLIP16 必须作为两个不同 baseline 行报告。
5. 表格脚注必须写清 score direction、AP positive class、输入帧数、FPS、duration 和 encoder。

## 可写入论文的边界表述

我们完成了 D3 protocol audit，而不是外部 D3 重跑。该审计显示，当前 Alpha-STALLED 主线采用与参考文献公平比较原则一致的平衡评测和固定窗口采样；但是，D3 数值对类别比例、真实视频 FPS 处理和 encoder 选择敏感。因此，除非在完全相同的视频索引、帧索引和 pairwise balanced 指标下重跑，否则不应把外部 D3 数字作为直接主表对照。

机器可读文件：

- `d3_protocol_audit.csv`：逐协议项审计矩阵。
- `d3_protocol_dataset_summary.csv`：当前三个 index 的 FPS、duration 和 2s window 覆盖摘要。
