# Original STALL protocol and coverage audit

## Questions separated by this audit

Three quantities must not be conflated:

1. **Calibration real videos** fit real-only whitening and empirical CDFs. They are not test examples.
2. **Unique evaluation real videos** are the physical authentic clips available for evaluation.
3. **Pairwise real occurrences** count a real clip again when it is reused against another generator. This is the denominator used by the generator-wise balanced protocol, not the number of unique files.

Bootstrap changes none of these counts. It only resamples identities already present in a frozen evaluation cohort to estimate finite-sample uncertainty.

## Original STALL paper protocol

The source is `2603.15026v2.pdf`, Sections 5.1, A.3, and C, together with the public preprocessing implementation in `src/video_index.py`.

- Calibration uses about 33,976 disjoint VATEX real videos.
- Evaluation uses 1,700 MSVD real videos for ComGenVid, 3,722 real videos for VideoFeedback (1,861 DiDeMo and 1,861 Panda70M), and a 1,400-video MSR-VTT real pool for GenVideo: 6,822 unique evaluation reals in total.
- For each generator, the paper selects the same number of real and generated videos, limited by the smaller side. Real clips may therefore be reused across different generator-wise comparisons.
- The supplementary generator counts sum to 36,641 generated videos: 3,400 ComGenVid, 24,262 VideoFeedback, and 8,979 GenVideo.
- Dataset and all-benchmark numbers are averages of generator-wise metrics. Consequently, a generator contributes one unit regardless of whether it has 56 or 3,722 clips. The paper's `All Benchmarks` row is generator-weighted over 23 sources; it is not identical to an equal-weight mean of three dataset averages (`Macro-3`).
- The paper states that generated video is the positive class for AP. This repository's historical Alpha-STALLED tables instead use real-positive AP. AUC is invariant to reversing both labels and score direction, but AP is not. Both AP orientations must therefore be named explicitly.

### Original temporal sampling

The public preprocessing first approximately downsamples each video to 8 FPS using indices `round((original_fps / 8) * j)`. It then uses `numpy.random.RandomState(42)` to choose one contiguous window from all valid starts. The RNG is initialized per video and is advanced while the 1, 2, 3, and 4 second windows are built. The stored `1_sec_idxs` and `2_sec_idxs` are therefore the authoritative original windows.

- Normal sources: one contiguous 2 second window, 16 distinct frames at 8 FPS.
- VideoFeedback Hotshot-XL, GenVideo HotShot, and GenVideo MoonValley: one 1 second window, 8 distinct frames, because no valid 2 second span exists.
- There is no frame duplication in this paper protocol. The paper specifically identifies D3's 3-to-8 FPS duplication of GenVideo real clips as a protocol artifact and instead uses high-frame-rate MSR-VTT sources.
- Sampling at 8 FPS does not make a 2 second interval shorter. It selects 16 timestamps spread across approximately 2 seconds; it does not play those frames back at a higher speed.

The manuscript text says `truncated to 2 seconds` but does not specify the start position. The fixed-seed random start is established by the released code and stored index columns, not by prose alone.

## Current duration-aware Alpha-STALLED protocol

The current full-23 confirmation keeps target-domain calibration reals disjoint from evaluation reals. The selected Local calibration sizes are 600 ComGenVid, 400 VideoFeedback, and 1,500 GenVideo. The Global branch retains the original VATEX calibration. The remaining unique evaluation pools are:

| Dataset | Evaluation real | Generated | Generators |
|---|---:|---:|---:|
| ComGenVid | 898 | 3,400 | 2 |
| VideoFeedback | 3,580 | 33,581 | 11 |
| GenVideo | 7,984 | 8,204 | 10 |
| Total | 12,462 | 45,185 | 23 |

For valid 2 second videos, Alpha-STALLED uses deterministic start/middle/end windows over the full valid start range. Each window still contains 16 distinct frames at 8 FPS. Duplicate windows are removed, so an exactly 2 second video has effective K=1 rather than three copies. A video without a valid 2 second span uses its stored 1 second/8-frame window. One known undecodable tail leaves a single video with effective K=2.

The frozen seed-42 pairwise evaluation contains 37,993 real/fake pairs. It evaluates every small generator in full, but a generator with more generated videos than the eligible real-side capacity is deterministically subsampled. This affects metric coverage, not score coverage: scores for all 45,185 generated videos are retained.

## Full generated-video coverage

To determine whether balanced subsampling hides behavior, `tools/analyze_full_coverage_protocol.py` evaluates all 45,185 generated scores. For each generator it uses every eligible physical real once and reports:

- AUC, which is prevalence invariant;
- `AP_fake_std50`, with generated video as positive and total real/fake weight fixed to 0.5/0.5;
- `AP_real_std50`, the corresponding repository-historical orientation;
- raw AP only as a prevalence diagnostic.

Fixed-prior AP prevents a generator with many more fake than real clips from receiving an artificial AP shift solely because of class prevalence. It does not manufacture real videos, and it does not make unseen videos covered.

The full-coverage K3 same-protocol comparison is:

| Scope | Global AUC/AP_fake/AP_real | Alpha AUC/AP_fake/AP_real | Delta AUC/AP_fake/AP_real |
|---|---:|---:|---:|
| ComGenVid | 0.8628/0.8449/0.8709 | 0.9045/0.8973/0.9117 | +0.0417/+0.0525/+0.0408 |
| VideoFeedback | 0.8478/0.8248/0.8608 | 0.8605/0.8418/0.8710 | +0.0127/+0.0170/+0.0102 |
| GenVideo | 0.8208/0.8085/0.8111 | 0.8550/0.8495/0.8417 | +0.0342/+0.0411/+0.0306 |
| Macro-3 | 0.8438/0.8260/0.8476 | 0.8733/0.8629/0.8748 | +0.0295/+0.0368/+0.0272 |
| All-23 | 0.8373/0.8194/0.8401 | 0.8619/0.8500/0.8618 | +0.0246/+0.0305/+0.0217 |

These are causal same-score-pipeline comparisons between the K3 Global branch and K3 Alpha fusion. They are not yet the final original-window factorial comparison.

At the calibrated diagnostic threshold 0.5, 41,942 generated videos are detected by both branches, 1,543 are rescued by Alpha after Global misses them, 281 are harmed by fusion after Global detects them, and 1,419 are missed by both. Thus the number of fake misses falls from 2,962 to 1,700 on the full scored pool. This threshold analysis is diagnostic rather than a replacement for ranking metrics. Full-coverage `AP_fake` improves on 20/23 generators; the three negative transfers are VideoFeedback Text2Video-Zero (-0.0535), VideoFeedback VideoCrafter2 (-0.0039), and GenVideo Lavie (-0.0015).

## What bootstrap does and does not do

The 1,000 paired bootstrap runs repeatedly draw video identities with replacement from the same frozen cohort. On a particular draw, some videos appear multiple times and some do not appear. Global and Alpha scores for an identity always receive the same multiplicity, so the distribution estimates uncertainty in their difference rather than adding independent selection noise.

It answers: if a comparable finite sample had been drawn from the same population, how variable would the measured gain be? It does not:

- score an omitted generated video;
- enlarge the real pool;
- correct a biased manifest;
- replace full-coverage evaluation.

Selection-seed sensitivity is a separate check. Across 100 deterministic balanced subsets, the existing K3 Alpha-minus-Global `AP_real` gain has mean +0.0278, standard deviation 0.0007, and range +0.0262 to +0.0294. The corresponding `AP_fake` gain has mean +0.0378, standard deviation 0.0014, and range +0.0350 to +0.0425. This shows that the gain is not specific to seed 42, but only the all-generated analysis establishes coverage of the full fake pool.

## Original-window factorial results

All 75,011 requested video/configuration scores were assembled and validated: 47,899 could be reused exactly, 17,570 were recomputed from compatible caches, and 9,542 required video/DINO scoring. There are no duplicate, missing, or extra task keys. The fair decomposition uses the same video identities and metric implementation. `K1_STALL` retains the original VATEX component CDFs; the four causal `K1_G/K1_S/K3_G/K3_S` cells share the duration-aware real-only calibration, so Local and window effects can be isolated without changing calibration between the four cells.

Full generated-video coverage, with AP standardized to a 50/50 class prior, gives:

| Config | Macro-3 AUC/AP_fake/AP_real | All-23 AUC/AP_fake/AP_real |
|---|---:|---:|
| K1_STALL | 0.8335/0.8188/0.8377 | 0.8281/0.8117/0.8330 |
| K1_G | 0.8360/0.8181/0.8402 | 0.8314/0.8112/0.8364 |
| K1_S | 0.8663/0.8542/0.8688 | 0.8577/0.8441/0.8593 |
| K3_G | 0.8438/0.8260/0.8476 | 0.8373/0.8194/0.8401 |
| K3_S | 0.8733/0.8629/0.8748 | 0.8619/0.8500/0.8618 |

Thus, over all 23 generators and all 45,185 scored fake videos, current K3 Alpha improves over original-window STALL by `+0.0338/+0.0383/+0.0288` All-23 AUC/AP_fake/AP_real. On Macro-3 the corresponding deltas are `+0.0398/+0.0441/+0.0371`.

Full-coverage AP_fake improves for 20/23 generators relative to K1 STALL. The three losses are VideoFeedback Text2Video-Zero (`-0.0542`), Hotshot-XL (`-0.0072`), and VideoCrafter2 (`-0.0058`). Thus the aggregate conclusion is broad but not generator-universal.

On the frozen seed-42 balanced cohort, K3 Alpha versus K1 STALL improves Macro-3 AUC/AP_fake/AP_real from `0.8306/0.8149/0.8347` to `0.8721/0.8608/0.8741`. The 1,000 paired video-ID cluster bootstrap estimates mean AP deltas of `+0.0457` for AP_fake (95% CI `[+0.0404,+0.0511]`) and `+0.0392` for AP_real (`[+0.0354,+0.0432]`). Across 100 balanced identity-selection seeds, the corresponding gain ranges are `[+0.0410,+0.0486]` and `[+0.0350,+0.0399]`.

The factorial attribution on that balanced cohort is:

- adding Local while retaining the original K1 window: AP_fake `+0.0377`, 95% CI `[+0.0334,+0.0425]`;
- changing K1 Alpha to K3 Alpha: AP_fake `+0.0085`, 95% CI `[+0.0057,+0.0112]`;
- changing duration calibration alone at K1: AP_fake `-0.0005`, 95% CI `[-0.0013,+0.0002]`.

Therefore most of the improvement comes from Local D2, with a smaller but independently positive contribution from wider temporal coverage. Duration-aware calibration is necessary to include the three 1-second generators correctly, but it does not explain the aggregate AP_fake gain.

For a standalone reproduction using the paper's generator counts and real-pool caps, K1 STALL obtains ComGenVid `0.8532/0.8459`, VideoFeedback `0.8459/0.8260`, GenVideo `0.8119/0.7996`, and All-23 `0.8318/0.8163` in AUC/AP_fake, compared with published rounded values `0.85/0.86`, `0.83/0.85`, `0.80/0.80`, and `0.82/0.82`. The local ComGenVid index exposes 1,698 rather than 1,700 usable real files, so this reproduction has four fewer pairwise occurrences than the paper. It is a protocol reproduction, not the causal Alpha comparison, because target reals reserved for Alpha calibration can still appear in this Global-only standalone row.

Detailed per-dataset, per-generator, bootstrap, and seed results are in `reports/original_k1_full23_factorial.md` and `results/full_coverage_paper_protocol/`.
