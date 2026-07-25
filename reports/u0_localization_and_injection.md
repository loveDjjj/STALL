# U0 localization and synthetic injection

The 300 real videos, target K3 windows, native affected frame indices, center-25% region, scene-cut donors, and anomaly definitions were locked before scoring.

## Score response

Positive values are realness-score drops, so larger values indicate stronger anomaly response.

| Condition | K1 Global/Local/Final drop | K3 Global/Local/Final drop |
|---|---:|---:|
| L1_local_freeze4 | +0.0011/+0.0028/+0.0018 | +0.0021/+0.0027/+0.0023 |
| L2_local_repeat1 | +0.0024/-0.0005/+0.0013 | +0.0031/-0.0007/+0.0016 |
| L3_local_flicker | +0.0089/-0.0109/+0.0010 | +0.0089/-0.0153/-0.0008 |
| L4_local_affine_jitter | -0.0030/-0.0077/-0.0049 | -0.0029/-0.0106/-0.0060 |
| L5_local_temporal_shift | -0.0039/-0.0008/-0.0027 | -0.0045/-0.0023/-0.0036 |
| L6_full_frame_freeze4 | -0.0038/-0.2905/-0.1185 | -0.0055/-0.4084/-0.1667 |
| L7_scene_cut | -0.0154/-0.0382/-0.0245 | -0.0182/-0.0387/-0.0264 |

## PatchD2 localization

| Condition | Patch-time AUPRC | Inside-outside anomaly | Top-k patch hit | Temporal hit |
|---|---:|---:|---:|---:|
| L1_local_freeze4 | 0.1558 | 13.7346 | 0.3612 | 0.4467 |
| L2_local_repeat1 | 0.0670 | 5.6492 | 0.3255 | 0.1733 |
| L3_local_flicker | 0.1032 | -4.4118 | 0.3133 | 0.2233 |
| L4_local_affine_jitter | 0.1030 | -9.3124 | 0.3000 | 0.3500 |
| L5_local_temporal_shift | 0.1245 | 3.1016 | 0.3146 | 0.3600 |
| L6_full_frame_freeze4 | 0.3424 | -265.8437 | 1.0000 | 0.2200 |

## Coverage and difficult negative

- K1 has zero native-frame overlap for 165/300 locked injections. Across L1-L6 in that subset, mean K1/K3 final-score drops are `+0.0000/-0.0195`.
- Scene-cut K1/K3 final-score drops are `-0.0245/-0.0264`; fractions below 0.5 are `0.5900/0.5367`.
- Maximum R0 reproduction error across raw and final scores is `0`.
- Per-video scores, localization rows, coverage groups, and compressed heatmap arrays are retained.
