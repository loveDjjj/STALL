# Parameter asset inventory

This report is generated from `configs/parameter_assets.yaml` and the current
parameter files. Governed assets are content-addressed; local ignored sweeps are
inventory only and cannot become release inputs without explicit registration.

## Summary

- Governed assets: 8
- Current locked-U0 assets: 4
- External-confirmation assets: 1
- Historical frozen assets: 3
- Local unregistered sweep assets: 73
- Local unregistered sweep bytes: 615643012

## Governed assets

| Path | Family | Lifecycle | Protocol | Dataset | Calibration | SHA-256 |
|---|---|---|---|---|---:|---|
| `precomputed/stall_params_vatex_dino_v3.npz` | `global_stall` | `current_release` | `u0_locked_v1` | `vatex_real_reference` | 33976 | `beede546ca4385242c2a26075b8087ab51bee709549fc4ddc79d0eef5acce240` |
| `release/u0/params/comgenvid_region1_mean.npz` | `local_patch` | `current_release` | `u0_locked_v1` | `comgenvid` | 200 | `5716abb1e865f7e36552ecec72973e90040b9affe415c34e1807bc6fe08f134f` |
| `release/u0/params/videofeedback_region1_mean.npz` | `local_patch` | `current_release` | `u0_locked_v1` | `videofeedback` | 200 | `66e46032199bbe9ed58bb24e4682c06a6534ee80e025b0854fd40d139b7266ff` |
| `release/u0/params/genvideo_region1_mean.npz` | `local_patch` | `current_release` | `u0_locked_v1` | `genvideo` | 200 | `880d0e376cd0d9727806d243396cb1c6e1364369494bf9888d488da3e2340a05` |
| `release/u0_external_genvidbench/params/region1_mean.npz` | `local_patch` | `external_confirmation` | `u0_external_genvidbench_v1` | `genvidbench` | 199 | `7875778f4713e3c477c4614f37d20a1eb4c2d22bab6854dc7b96ed5d32df7c48` |
| `precomputed/patch_params_comgenvid_real_second_order_region3_bottomk0p20_v2.npz` | `local_patch` | `historical_frozen` | `legacy_k1_paper_scores_v1` | `comgenvid` | 1698 | `8eca37a94f944306fe75e4095a855254d1530574fd2be6e3364e7ba9f078a057` |
| `precomputed/patch_params_videofeedback_real_same_grid_second_order_region1_mean_v2.npz` | `local_patch` | `historical_frozen` | `legacy_k1_paper_scores_v1` | `videofeedback` | 4080 | `57e48de4474bbad206e42d38d3d20eb2a4ade62c2e5e29f5a37a620a7d089f7c` |
| `precomputed/patch_params_genvideo_real_same_grid_second_order_region2_mean_v2.npz` | `local_patch` | `historical_frozen` | `legacy_k1_paper_scores_v1` | `genvideo` | 9984 | `be22f42dfecd65c1c9cf95ec2f134e7707af13b1527eca0edaf5841cc1a97882` |

## Local sweep boundary

The local root is `precomputed`. Files not
listed above are permitted only when their names match one declared local-sweep
patterns; they remain ignored, rebuildable research assets and are not protocol
authorities.
