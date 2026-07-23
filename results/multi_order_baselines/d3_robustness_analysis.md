# D3 robustness analysis

## Scope

This is an operator-controlled DINOv3 embedding-cache study. Each perturbation is recalibrated with the matching transform on the disjoint 200-real calibration set. The fused score is the fixed Stage-2 structure `0.5 spatial + 0.25 first-order temporal + 0.25 D3`.

## Dataset macro metrics

| Condition | ComGenVid AUC/AP | VideoFeedback AUC/AP | GenVideo AUC/AP | Macro-3 AUC/AP |
|---|---:|---:|---:|---:|
| reference_8fps_2s | 0.8671/0.8720 | 0.7537/0.7974 | 0.8190/0.8277 | 0.8132/0.8324 |
| uniform_4fps_2s | 0.8476/0.8597 | 0.7894/0.8184 | 0.8183/0.8270 | 0.8184/0.8350 |
| central_8fps_1s | 0.8654/0.8721 | 0.7540/0.7992 | 0.7953/0.8080 | 0.8049/0.8264 |
| frame_drop_one | 0.8641/0.8701 | 0.7573/0.7999 | 0.8168/0.8262 | 0.8127/0.8321 |
| frame_duplicate_two | 0.8693/0.8733 | 0.7692/0.8043 | 0.8009/0.8119 | 0.8132/0.8299 |
| remove_exact_duplicates | 0.8671/0.8720 | 0.7541/0.7975 | 0.8227/0.8219 | 0.8146/0.8305 |

## Score stability

| Condition | Score | Pearson r | Mean absolute change | Median absolute change |
|---|---|---:|---:|---:|
| uniform_4fps_2s | d3_raw | 0.9162 | 0.5016 | 0.2515 |
| uniform_4fps_2s | d3_one_sided | 0.8642 | 0.1104 | 0.0750 |
| central_8fps_1s | d3_raw | 0.7535 | 0.5127 | 0.1695 |
| central_8fps_1s | d3_one_sided | 0.8026 | 0.1336 | 0.0900 |
| frame_drop_one | d3_raw | 0.9939 | 0.1047 | 0.0630 |
| frame_drop_one | d3_one_sided | 0.9846 | 0.0321 | 0.0150 |
| frame_duplicate_two | d3_raw | 0.8949 | 1.1136 | 0.9885 |
| frame_duplicate_two | d3_one_sided | 0.9035 | 0.0983 | 0.0650 |
| remove_exact_duplicates | d3_raw | 0.9936 | 0.0089 | 0.0000 |
| remove_exact_duplicates | d3_one_sided | 0.9997 | 0.0007 | 0.0000 |

## Interpretation boundary

- `uniform_4fps_2s`, `central_8fps_1s`, frame drop, duplication, and exact-duplicate removal isolate temporal-operator sensitivity after the encoder. The last condition is the native timestamp sequence after removing adjacent duplicate embeddings.
- JPEG compression and resize cannot be reconstructed from cached embeddings. They require decoding perturbed pixels and rerunning DINOv3, so they are not claimed as completed here.
- Scene-cut annotations are absent. `d3_robustness_scene_cut_proxy.csv` therefore reports a declared proxy: maximum adjacent embedding distance above the 95th percentile of the disjoint real calibration set. It is not treated as ground truth.
