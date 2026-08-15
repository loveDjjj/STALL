# Duration-aware 23-source full-data evaluation

This confirmation protocol was frozen before generated-video scoring. Calibration uses only disjoint real videos; both the maximum pools and the held-out-real custom-size rule are fixed without generated-video performance.

## Protocol

- A video with a valid 2 s span uses three deterministic uniformly spaced windows. Each window contains 16 distinct frames sampled at 8 FPS.
- A video without a valid 2 s span uses one deterministic 1 s window containing 8 distinct frames. There is no frame duplication or partial-window fallback.
- A listed undecodable tail window is removed without dropping the video; this leaves `GenVideo/D325.mp4` with effective K=2.
- Window scores are `G_k=0.5*global_spatial+0.5*global_T1`, `L_k=0.1*patch_spatial+0.9*same-grid_D2`; video scores average each branch over available windows and use `S=0.6*G+0.4*L`.
- The 1 s and 2 s streams use independent real-only whitening/CDF calibration. Video-level CDFs are additionally separated by effective K.
- Uniform calibration controls are N=200 per dataset and the predeclared maxima (ComGenVid 800, VideoFeedback 500, GenVideo 2,000).
- The dataset-specific candidate was frozen before generated scoring using held-out-real quantile calibration error: ComGenVid 600, VideoFeedback 400, and GenVideo 1,500. The unused real validation blocks contain 200, 100, and 500 videos, respectively.

## Data coverage

All 45,185 indexed generated videos from 23 sources were scored. Physical evaluation collections contain 4,298 ComGenVid, 37,161 VideoFeedback, and 16,188 GenVideo videos. Calibration and evaluation real-video identities have zero overlap.

Primary AUC/AP uses 37,993 full-23 matched pairs and 32,206 strict-20 matched pairs. It is computed per generator with equal real/fake counts and then macro-averaged. This prevents class prevalence and large generators from changing AP or dominating the macro result. It does not discard generated scoring: every generated raw score is retained, while a deterministic balanced subset is used where the disjoint real pool is smaller.

## Macro results

| Protocol | Calibration | Branch | Generators | AUC | AP |
|---|---|---|---:|---:|---:|
| full23 | n200 | G | 23 | 0.8414 | 0.8451 |
| full23 | n200 | L | 23 | 0.8307 | 0.8292 |
| full23 | n200 | S | 23 | 0.8681 | 0.8721 |
| full23 | ncustom | G | 23 | 0.8404 | 0.8454 |
| full23 | ncustom | L | 23 | 0.8361 | 0.8384 |
| full23 | ncustom | S | 23 | 0.8721 | 0.8741 |
| full23 | nmax | G | 23 | 0.8401 | 0.8453 |
| full23 | nmax | L | 23 | 0.8349 | 0.8380 |
| full23 | nmax | S | 23 | 0.8719 | 0.8738 |
| strict20_2s | n200 | G | 20 | 0.8483 | 0.8480 |
| strict20_2s | n200 | L | 20 | 0.8358 | 0.8325 |
| strict20_2s | n200 | S | 20 | 0.8737 | 0.8738 |
| strict20_2s | ncustom | G | 20 | 0.8468 | 0.8475 |
| strict20_2s | ncustom | L | 20 | 0.8387 | 0.8378 |
| strict20_2s | ncustom | S | 20 | 0.8758 | 0.8743 |
| strict20_2s | nmax | G | 20 | 0.8465 | 0.8474 |
| strict20_2s | nmax | L | 20 | 0.8373 | 0.8373 |
| strict20_2s | nmax | S | 20 | 0.8756 | 0.8739 |

## Full-23 dataset results

| Dataset | N=200 Alpha AUC/AP | N=max Alpha AUC/AP | N=custom Alpha AUC/AP | Custom minus N=200/max AP | Custom Global AUC/AP | Alpha minus Global AP |
|---|---:|---:|---:|---:|---:|---:|
| comgenvid | 0.8963/0.9055 | 0.9050/0.9118 | 0.9035/0.9109 | +0.0054/-0.0009 | 0.8620/0.8697 | +0.0411 |
| videofeedback | 0.8644/0.8757 | 0.8594/0.8714 | 0.8623/0.8736 | -0.0021/+0.0022 | 0.8481/0.8621 | +0.0115 |
| genvideo | 0.8435/0.8351 | 0.8511/0.8381 | 0.8504/0.8378 | +0.0027/-0.0002 | 0.8109/0.8043 | +0.0335 |

## Restored one-second sources

The fair causal comparison is Alpha (`S`) versus Global STALL (`G`) under the same duration-aware protocol and the same paired videos.

| Dataset/source | Pairs | N=custom Global AUC/AP | N=custom Alpha AUC/AP | Delta AUC/AP |
|---|---:|---:|---:|---:|
| genvideo/HotShot | 700 | 0.7743/0.7617 | 0.8050/0.7880 | +0.0308/+0.0263 |
| genvideo/MoonValley | 626 | 0.7773/0.8204 | 0.8552/0.8828 | +0.0778/+0.0625 |
| videofeedback/Hotshot-XL | 3222 | 0.8106/0.8230 | 0.8247/0.8402 | +0.0141/+0.0172 |

## Breadth and calibration effect

- N=max Alpha improves AP over N=200 Alpha on 11/23 generators and decreases it on 12/23.
- N=max Alpha improves AP over same-protocol Global STALL on 21/23 generators and decreases it on 2/23.
- N=custom improves AP over N=200 on 11/23 generators and over N=max on 12/23.
- N=custom Alpha improves AP over same-protocol Global STALL on 21/23 generators.
- The expanded calibration gain is therefore small and not generator-wide; it must not be described as a universal scaling benefit.

## Paired cluster bootstrap

| Protocol | Contrast | Mean delta AP | 95% CI |
|---|---|---:|---:|
| full23 | nmax_S_minus_n200_S | +0.0017 | [+0.0008, +0.0025] |
| full23 | nmax_S_minus_nmax_G | +0.0283 | [+0.0255, +0.0312] |
| full23 | ncustom_S_minus_n200_S | +0.0020 | [+0.0012, +0.0028] |
| full23 | ncustom_S_minus_nmax_S | +0.0003 | [+0.0001, +0.0005] |
| full23 | ncustom_S_minus_ncustom_G | +0.0286 | [+0.0258, +0.0314] |
| strict20_2s | nmax_S_minus_n200_S | +0.0001 | [-0.0009, +0.0010] |
| strict20_2s | nmax_S_minus_nmax_G | +0.0264 | [+0.0232, +0.0294] |
| strict20_2s | ncustom_S_minus_n200_S | +0.0005 | [-0.0005, +0.0013] |
| strict20_2s | ncustom_S_minus_nmax_S | +0.0004 | [+0.0002, +0.0006] |
| strict20_2s | ncustom_S_minus_ncustom_G | +0.0267 | [+0.0235, +0.0297] |

## Interpretation guardrails

- `full23` includes every indexed generated video and uses 1s only when 2s is unavailable.
- `strict20_2s` excludes the three entirely short generators and all sub-2s clips, isolating the long-video comparison.
- Each generator is pairwise balanced. Selected real videos use exactly the same 1s/2s counts as selected fake videos.
- AP in this report treats real video as the positive class, matching the locked U0 release. The original STALL Table 1 instead states generated-positive AP, so direct Table-1 comparison requires the separately reported `AP_fake` orientation.
- The old paper numbers and this experiment are not a causal comparison because the evaluated identities, sample counts, and short-video protocol differ.
- The size-selection artifact declares `generated_videos_used=0` and status `frozen_before_generated_candidate_scoring`.

## Decision

Adopt the real-only custom upper limits for the duration-aware full-23 protocol: they use 2,500 calibration videos in total instead of N=max's 3,300, reach 0.8721/0.8741, and improve AP over N=200 by +0.0020 (95% CI +0.0012 to +0.0028) and over N=max by +0.0003 (+0.0001 to +0.0005). Strict-20 custom versus N=200 remains inconclusive, so this is a small full-coverage calibration refinement, not a new detector component or evidence that more real data always helps. It adds no inference-time branch or DINO forward.
