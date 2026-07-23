# Global-local multi-order final report

Date: 2026-07-23

## Executive decision

The experiments do **not** support the originally proposed default, "global calibrated
second-order likelihood + local second-order residual." Every calibrated global D3
candidate fails the predeclared admission criteria, and both residual local D2 variants
underperform plain same-grid D2. The evidence-backed two-branch method is instead:

```text
Global = 0.5 * global_spatial + 0.5 * global_first_order_temporal
Local  = 0.1 * local_spatial + 0.9 * local_same_grid_D2
Final  = 0.6 * Global + 0.4 * Local
```

This fixed method reaches Macro-3 AUC/AP `0.8570/0.8600`, improving original STALL by
`+0.0183/+0.0173`. It improves AP on 17/20 generators. Raw and calibrated D3 remain
ablations; reliability fusion is not admitted because one scalar window per video cannot
provide a valid per-video variance estimate.

## Formulas and implementation

Global DINOv3 ViT-L/16 uses the normalized CLS token `g[t]` (`src/stall.py:244-276`),
not register tokens or mean patch tokens. With whitening parameters `(mu, W)` and the
standard-normal log likelihood `ell(x) = -0.5 * (D log(2*pi) + ||x||^2)`:

```text
Gs  = ECDF_real(max_t ell((g[t] - mu_s) W_s))
Gt1 = ECDF_real(min_t ell(((normalize(g[t+1]-g[t])) - mu_t) W_t))
G   = 0.5 * (Gs + Gt1)
```

The implementation is at `src/stall.py:99-136,280-322`; repeated adjacent embeddings
are excluded from the temporal minimum by assigning their likelihood `+inf`.

Operator-controlled D3 is:

```text
d[t] = ||g[t+1] - g[t]||_2
a[t] = d[t+1] - d[t]
V = std(a, ddof=0)
```

It is implemented at `tools/build_multi_order_baselines.py:200-214`. One-sided D3 is
`F_real(V)`, two-sided realness is `2 min(F_real(V), 1-F_real(V))`, and conditional
calibration applies the same two-sided CDF within five real-motion quantile bins
(`tools/build_multi_order_baselines.py:307-363`). Time-normalized D3 uses native frame
timestamps for velocity and acceleration.

Patch D2 uses normalized vector acceleration, option A from the audit:

```text
a[t,p] = z[t+2,p] - 2 z[t+1,p] + z[t,p]
L0 feature = normalize(a[t,p])
L1 feature = normalize(a[t,p] - a_global[t])
L2 feature = normalize(a[t,p] - median_q a[t,q])
```

NumPy definitions are at `src/patch_matching.py:113-162`; the GPU path, including the
even-patch median matching NumPy semantics, is at `src/eval_patch_fast.py:135-180`.
L3 changes only aggregation to a length-2 temporal-run bottom-20% mean
(`src/eval_patch_fast.py:212-227`). Full audit details and tensor shapes are in
`global_local_d3_audit.md`.

## Unified protocol

- Operator-controlled results use the same DINOv3 cache, exact indexed 2 s window,
  16 frames at nominal 8 FPS, no short-video fallback, and one-to-one video keys.
- Evaluation has 4,298 ComGenVid, 3,500 VideoFeedback, and 13,623 GenVideo videos,
  totaling 21,421 unique rows with no missing scores or duplicate keys.
- D3 and leakage-free local models use each dataset's disjoint 200-real calibration split.
- Metrics are higher-is-real AUC/AP, macro-averaged over pairwise generator comparisons;
  each comparison deterministically balances real and fake counts.
- Paired bootstrap resamples paired video rows within generators for 1,000 iterations.
- B5-B8 in the historical baseline table reuse release scores calibrated on all available
  real videos and are therefore marked legacy/leakage-affected. Stage-3 P0 and the final
  method are the leakage-free replacements.
- Official D3 is separate: XCLIP-16, L2, JPEG frame folders, crop/resize, random-start
  3 s at 8 FPS, up to 1,000 real and 1,000 fake per generator. It is not mixed with the
  DINOv3 operator-controlled table.

## Baselines

| ID | Configuration | ComGenVid AUC/AP | VideoFeedback AUC/AP | GenVideo AUC/AP | Macro-3 AUC/AP |
|---|---|---:|---:|---:|---:|
| B0 | Global spatial | 0.8364/0.8648 | 0.8006/0.8250 | 0.7958/0.8064 | 0.8109/0.8321 |
| B1 | Global first-order | 0.8234/0.8048 | 0.8492/0.8536 | 0.7655/0.7558 | 0.8127/0.8047 |
| B2 | Original STALL | 0.8550/0.8606 | 0.8528/0.8640 | 0.8085/0.8036 | 0.8388/0.8428 |
| B3 | Raw D3 percentile | 0.7421/0.7152 | 0.5245/0.5637 | 0.7218/0.7480 | 0.6628/0.6756 |
| B4 | STALL + raw D3 | 0.8279/0.8260 | 0.6630/0.7153 | 0.7950/0.8161 | 0.7620/0.7858 |
| B5 | Legacy patch spatial | 0.8286/0.8451 | 0.7374/0.7455 | 0.7651/0.7694 | 0.7770/0.7866 |
| B6 | Legacy patch D2 | 0.9281/0.9298 | 0.8060/0.8153 | 0.8157/0.8049 | 0.8499/0.8500 |
| B7 | Legacy patch combined | 0.9288/0.9315 | 0.8079/0.8206 | 0.8178/0.8103 | 0.8515/0.8541 |
| B8 | Legacy global + patch | 0.9199/0.9231 | 0.8561/0.8687 | 0.8452/0.8332 | 0.8737/0.8750 |

Component AP win counts are: add global first-order 14/20, add global spatial 17/20,
add raw D3 6/20, add local D2 to spatial 15/20, add local spatial to D2 13/20,
and add legacy patch to global 17/20. Full per-video, dataset, generator, and bootstrap
tables are in `results/multi_order_baselines/`.

## Global D3 calibration

| ID | Configuration | ComGenVid AUC/AP | VideoFeedback AUC/AP | GenVideo AUC/AP | Macro-3 AUC/AP | AP wins |
|---|---|---:|---:|---:|---:|---:|
| G0 | Original STALL | 0.8550/0.8606 | 0.8528/0.8640 | 0.8085/0.8036 | 0.8388/0.8428 | - |
| G1 | STALL + raw D3 | 0.8671/0.8720 | 0.7537/0.7974 | 0.8190/0.8277 | 0.8132/0.8324 | 10/20 |
| G2 | Two-sided D3 | 0.8486/0.8506 | 0.7838/0.8160 | 0.7425/0.7569 | 0.7916/0.8078 | 3/20 |
| G3 | Motion-conditioned | 0.8271/0.8397 | 0.7829/0.8166 | 0.7300/0.7583 | 0.7800/0.8049 | 1/20 |
| G4 | Time-normalized conditional | 0.8238/0.8383 | 0.7829/0.8166 | 0.7487/0.7710 | 0.7852/0.8086 | 0/20 |

Raw G1 gains `+0.0241` GenVideo AP (95% CI `[+0.0167,+0.0304]`) but loses
`-0.0666` VideoFeedback AP (`[-0.0737,-0.0596]`). The best calibrated GenVideo result,
G4, is `-0.0326` below G0, so calibrated retention of the positive raw gain is 0% after
clipping negative retention. G2-G4 still lose 0.0475-0.0481 AP on VideoFeedback. None
meets the 70%-retention, <=0.01 transfer-loss, Macro-3-improvement, and broad-generator
admission rule.

Raw GenVideo gains concentrate in Gen2 (+0.0591 AP), Sora (+0.0463), Crafter (+0.0447),
Show_1 (+0.0284), and WildScrape (+0.0242); Lavie and MorphStudio reverse. On
VideoFeedback, raw D3 improves only Pika (+0.0663) and Fast-SVD (+0.0057), with the
largest losses on Text2Video-Zero (-0.1772), ZeroScope-576w (-0.1402), and LVDM
(-0.1029).

## Leakage-free local models

| ID | Configuration | ComGenVid AUC/AP | VideoFeedback AUC/AP | GenVideo AUC/AP | Macro-3 AUC/AP | AP wins vs L0 |
|---|---|---:|---:|---:|---:|---:|
| Ls | Local spatial | 0.7273/0.7473 | 0.6487/0.6444 | 0.7204/0.7095 | 0.6988/0.7004 | - |
| L0 | Same-grid D2 | 0.8386/0.8278 | 0.7807/0.7897 | 0.8101/0.7946 | 0.8098/0.8040 | - |
| L1 | Global residual D2 | 0.5290/0.5288 | 0.5080/0.5077 | 0.5193/0.5193 | 0.5188/0.5186 | 0/20 |
| L2 | Spatial-median residual | 0.7807/0.7846 | 0.7193/0.7376 | 0.7881/0.7743 | 0.7627/0.7655 | 2/20 |
| L3 | Persistent L0 | 0.8379/0.8271 | 0.7506/0.7663 | 0.7745/0.7520 | 0.7877/0.7818 | 1/20 |
| P0 | 0.1 spatial + 0.9 L0 | 0.8427/0.8554 | 0.7820/0.7926 | 0.8131/0.8006 | 0.8126/0.8162 | 20/20 |

L1 collapses near chance, showing that CLS-vector acceleration is not a valid common-mode
reference for patch-token acceleration despite equal feature dimension. L2 removes shared
dynamics that are themselves discriminative; L3 hurts GenVideo and VideoFeedback. Patch
spatial remains useful only as a 0.1 auxiliary term: P0 adds `+0.0122` Macro-3 AP over L0,
improves all 20 generators, and has no dataset AP decline. It therefore passes the stated
retention rule even though its standalone score is weak.

## Final fusion

| Method | ComGenVid AUC/AP | VideoFeedback AUC/AP | GenVideo AUC/AP | Macro-3 AUC/AP |
|---|---:|---:|---:|---:|
| Original STALL B2 | 0.8550/0.8606 | 0.8528/0.8640 | 0.8085/0.8036 | 0.8388/0.8428 |
| Legacy Alpha-STALLED B8 | 0.9199/0.9231 | 0.8561/0.8687 | 0.8452/0.8332 | 0.8737/0.8750 |
| Final leakage-free, alpha=0.6 | 0.8699/0.8819 | 0.8631/0.8708 | 0.8382/0.8274 | 0.8570/0.8600 |

Worst-case selection chooses P0/alpha 0.5; fixed alpha 0.6 and LODO choose P0/0.6 for
ComGenVid and GenVideo, while VideoFeedback LODO chooses P0/0.2 and transfers poorly.
Target oracles are diagnostics only. Reliability fusion is excluded: the compact protocol
has one 2 s window, so inverse-variance weights cannot be estimated without new independent
windows; the prohibited `|score-0.5|` surrogate was not used.

Paired-bootstrap deltas for the fixed final method:

| Dataset | Comparison | Delta AUC [95% CI] | Delta AP [95% CI] |
|---|---|---:|---:|
| ComGenVid | vs B2 | +0.0148 [+0.0126,+0.0173] | +0.0213 [+0.0178,+0.0249] |
| VideoFeedback | vs B2 | +0.0103 [+0.0045,+0.0161] | +0.0068 [+0.0018,+0.0122] |
| GenVideo | vs B2 | +0.0296 [+0.0242,+0.0356] | +0.0237 [+0.0183,+0.0295] |
| ComGenVid | vs legacy B8 | -0.0500 [-0.0562,-0.0439] | -0.0411 [-0.0461,-0.0361] |
| VideoFeedback | vs legacy B8 | +0.0070 [+0.0042,+0.0098] | +0.0021 [-0.0003,+0.0046] |
| GenVideo | vs legacy B8 | -0.0071 [-0.0115,-0.0029] | -0.0059 [-0.0095,-0.0025] |

### Per-generator final results

| Dataset | Generator | B2 AUC/AP | legacy B8 AUC/AP | Final AUC/AP |
|---|---|---:|---:|---:|
| ComGenVid | Sora | 0.8429/0.8505 | 0.9099/0.9122 | 0.8584/0.8723 |
| ComGenVid | VEO3 | 0.8672/0.8707 | 0.9299/0.9339 | 0.8814/0.8915 |
| GenVideo | Crafter | 0.7990/0.7887 | 0.8234/0.7992 | 0.8247/0.8001 |
| GenVideo | Gen2 | 0.8783/0.8885 | 0.9114/0.9255 | 0.9080/0.9197 |
| GenVideo | Lavie | 0.8524/0.8474 | 0.8639/0.8638 | 0.8674/0.8642 |
| GenVideo | ModelScope | 0.7744/0.7783 | 0.8366/0.8391 | 0.8190/0.8275 |
| GenVideo | MorphStudio | 0.8246/0.8355 | 0.8362/0.8387 | 0.8386/0.8403 |
| GenVideo | Show_1 | 0.8169/0.8025 | 0.8466/0.8291 | 0.8430/0.8261 |
| GenVideo | Sora | 0.7972/0.8123 | 0.8776/0.8823 | 0.8524/0.8610 |
| GenVideo | WildScrape | 0.7256/0.6758 | 0.7662/0.6882 | 0.7522/0.6799 |
| VideoFeedback | AnimateDiff | 0.8538/0.8721 | 0.8575/0.8765 | 0.8679/0.8809 |
| VideoFeedback | Fast-SVD | 0.8918/0.8922 | 0.8999/0.9107 | 0.9130/0.9198 |
| VideoFeedback | LVDM | 0.8920/0.9122 | 0.9025/0.9136 | 0.9142/0.9228 |
| VideoFeedback | LaVie-base | 0.8378/0.8492 | 0.8250/0.8402 | 0.8393/0.8493 |
| VideoFeedback | ModelScope | 0.8239/0.8404 | 0.8526/0.8582 | 0.8422/0.8438 |
| VideoFeedback | Pika | 0.8431/0.8577 | 0.8917/0.9113 | 0.8944/0.9073 |
| VideoFeedback | SoRA-Clip | 0.8163/0.8185 | 0.8112/0.8221 | 0.8110/0.8161 |
| VideoFeedback | Text2Video-Zero | 0.8228/0.8192 | 0.7325/0.7517 | 0.7490/0.7577 |
| VideoFeedback | VideoCrafter2 | 0.9438/0.9486 | 0.9015/0.9137 | 0.9303/0.9321 |
| VideoFeedback | ZeroScope-576w | 0.8022/0.8304 | 0.8868/0.8893 | 0.8696/0.8786 |

## Robustness

Embedding-cache perturbations cover 4 FPS/2 s, 8 FPS/1 s, frame drop, two duplicated
frames, and native-timestamp exact-deduplication. Macro-3 AP is respectively `0.8350`,
`0.8264`, `0.8321`, `0.8299`, and `0.8305`, versus reference `0.8324`. Frame duplication
changes raw D3 most (mean absolute change `1.1136`); condition-matched calibration reduces
the percentile change to `0.0983`. Exact deduplication has median change zero.

Motion quartile results are in `d3_robustness_motion_groups.csv`. Because no ground-truth
scene-cut annotations exist, `d3_robustness_scene_cut_proxy.csv` uses a declared proxy:
maximum adjacent embedding distance above the disjoint-real 95th percentile. It must not
be cited as annotated scene-cut performance.

Pixel robustness reruns DINOv3 on 700 evaluation videos plus the disjoint calibration
sets. JPEG quality 30 preserves raw-D3 correlation `0.9801/0.9648/0.9827`; half-resolution
preserves `0.9958/0.9828/0.9949` for ComGenVid/VideoFeedback/GenVideo. All 2,100 scored
condition rows succeed. This bounded subset is robustness evidence, not a replacement for
the full protocol.

## Official D3 baseline

The corrected official path loads `XCLIPModel(...).vision_model`; loading an XCLIP
checkpoint directly into `XCLIPVisionModel` had left the vision tower randomly initialized.
The corrected XCLIP-16/L2 batch completes 10/10 generators, 8,026 unique videos and 17,026
pairwise rows, with zero failures. GenVideo macro AUC/AP is `0.8011/0.8529`.

| Generator | Official D3 AUC/AP |
|---|---:|
| Crafter | 0.8428/0.8274 |
| Gen2 | 0.9111/0.9151 |
| HotShot | 0.7762/0.8287 |
| Lavie | 0.7812/0.7647 |
| ModelScope | 0.7163/0.7596 |
| MoonValley | 0.8971/0.9263 |
| MorphStudio | 0.7650/0.8285 |
| Show_1 | 0.8350/0.8828 |
| Sora | 0.7733/0.9812 |
| WildScrape | 0.7132/0.8151 |

The final method's GenVideo AUC/AP is `0.8382/0.8274`, a descriptive AUC gap of +0.0370
and AP gap of -0.0256 against official D3. This is **not a valid improvement estimate**:
the encoders, windows, generator set, and sampling caps differ. A paired delta would require
running both methods on one shared official manifest.

## Keep, remove, and publication placement

Keep in the default method: original global STALL, leakage-free same-grid local D2, the
0.1 local-spatial auxiliary, and fixed alpha 0.6. Keep raw/two-sided/conditional D3 only as
ablations and official XCLIP D3 only as an external baseline.

Remove from the default: global-residual D2, spatial-median residual D2, persistent D2,
raw/calibrated global D3, reliability fusion, volatility as a third branch, hard/soft
matching, D3/D4 patch derivatives, multi-lag expansion, and score-distance gates.

Recommended paper layout:

1. Main table: B2, leakage-free P0, fixed final method, plus official D3 in a visibly
   separate protocol block.
2. Ablation table: B0-B4, G0-G4, Ls/L0-L3/P0, alpha fixed/LODO/worst-case/oracle.
3. Supplement: all generator tables, paired-bootstrap CIs, sampling/dedup/window and
   pixel robustness, scene-transition proxy definition, historical B5-B8 leakage audit,
   negative matching/higher-order results, and official command/protocol integrity.

## Answers to the six questions

1. **How much GenVideo D3 gain survives calibration?** None. Raw G1 gains +0.0241 AP;
   the best calibrated candidate is -0.0326 below G0, so clipped retention is 0%.
2. **Does calibration solve VideoFeedback negative transfer?** No. Calibrated variants
   still lose 0.0475-0.0481 AP, with bootstrap intervals wholly below zero.
3. **Are global D3 and local D2 genuinely complementary?** Not under the admission rule.
   Both can help GenVideo separately, but D3 transfer failures prevent a robust combined
   default. The supported complement is original global STALL plus local P0.
4. **Is patch spatial still necessary?** Yes as a small auxiliary. P0 adds +0.0122
   Macro-3 AP over L0, improves 20/20 generators, and harms no dataset in AP.
5. **Can the final method use only two branches?** Yes: original global STALL and local
   spatial+D2. It cannot honestly be described as calibrated-global-D3 plus residual-D2.
6. **Improvement over STALL, current Alpha-STALLED, and official D3?** Versus B2, the
   fixed method gains +0.0183 AUC/+0.0173 AP Macro-3. Versus legacy B8 it is -0.0167/
   -0.0150, but B8 is leakage-affected. Against official D3, only the non-comparable
   GenVideo descriptive gap (+0.0370 AUC/-0.0256 AP) is reported; no paired improvement
   claim is valid across the different protocols.
