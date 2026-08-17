# 初投稿材料草案

目标期刊尚未确定，因此以下内容只作为占位模板。未知事实均保留为待补充项，不能直接投稿。

## 投稿信草案

Dear [待补充：Editor name or Editors],

Please consider our manuscript, “Alpha-STALLED: 基于全局--局部真实视频校准的免训练生成视频检测,” for publication as a [待补充：article type] in [待补充：target journal].

This manuscript addresses the detection of fully generated videos under a generator-zero-shot setting. We introduce Alpha-STALLED, a training-free detector that extends the global real-video likelihood framework of STALL to DINOv3 patch tokens and models same-grid second-order local temporal evidence. The method keeps the inference path independent of generated-video training, test-batch ranking, generator identity, and real/fake labels.

Across ComGenVid, VideoFeedback and GenVideo, the current draft reports consistent gains over global-only STALL and the single-window baseline. Component ablations, paired bootstrap analysis, temporal-derivative controls, cross-dataset frozen-hyperparameter analysis and keyframe explanations support the main conclusion that local second-order patch temporal evidence complements the global STALL branch while exposing concrete boundaries such as VideoFeedback negative transfer and region-size sensitivity.

The manuscript is original, has not been published elsewhere, and is not under consideration elsewhere. [待补充：confirm or revise.] All authors have approved the submitted version. [待补充：confirm.] Conflicts of interest, funding, data availability and code availability are stated in the manuscript. [待补充：complete before submission.]

Thank you for your consideration.

Sincerely,

[待补充：Corresponding author name, affiliation and email]

## Highlights 草案

- Alpha-STALLED extends STALL from global frame likelihoods to DINOv3 patch-token local likelihoods.
- Same-grid second-order patch temporal evidence captures local dynamic inconsistencies without generated-video training.
- Three-benchmark results and paired bootstrap analysis show positive macro-average gains over global-only STALL.
- Ablations identify local second-order temporal evidence as the main patch contribution and region size as a key boundary.
- Protocol audits clarify why external D3 numbers should not be directly compared without matched sampling and balanced metrics.

## 必需作者输入

- Target journal and article type.
- Final title and short title.
- Author order, affiliations, corresponding author, ORCID.
- Funding, conflicts, acknowledgements and author contributions.
- Repository URL, release tag, code license and data access conditions.
- Whether this version has a preprint, related manuscript, prior submission or concurrent submission.
