# Parameter asset inventory

This report is generated from `configs/parameter_assets.yaml` and the current
parameter files. Governed assets are content-addressed; local ignored sweeps are
inventory only and cannot become release inputs without explicit registration.

## Summary

- Governed assets: 5
- Current locked-U0 assets: 4
- External-confirmation assets: 1
- Historical frozen assets: 0
- Local unregistered sweep assets: 0
- Local unregistered sweep bytes: 0

## Governed assets

| Path | Family | Lifecycle | Protocol | Dataset | Calibration | SHA-256 |
|---|---|---|---|---|---:|---|
| `precomputed/stall_params_vatex_dino_v3.npz` | `global_stall` | `current_release` | `u0_locked_v1` | `vatex_real_reference` | 33976 | `beede546ca4385242c2a26075b8087ab51bee709549fc4ddc79d0eef5acce240` |
| `release/u0/params/comgenvid_region1_mean.npz` | `local_patch` | `current_release` | `u0_locked_v1` | `comgenvid` | 200 | `5716abb1e865f7e36552ecec72973e90040b9affe415c34e1807bc6fe08f134f` |
| `release/u0/params/videofeedback_region1_mean.npz` | `local_patch` | `current_release` | `u0_locked_v1` | `videofeedback` | 200 | `66e46032199bbe9ed58bb24e4682c06a6534ee80e025b0854fd40d139b7266ff` |
| `release/u0/params/genvideo_region1_mean.npz` | `local_patch` | `current_release` | `u0_locked_v1` | `genvideo` | 200 | `880d0e376cd0d9727806d243396cb1c6e1364369494bf9888d488da3e2340a05` |
| `release/u0_external_genvidbench/params/region1_mean.npz` | `local_patch` | `external_confirmation` | `u0_external_genvidbench_v1` | `genvidbench` | 199 | `7875778f4713e3c477c4614f37d20a1eb4c2d22bab6854dc7b96ed5d32df7c48` |

## Local sweep boundary

The local root is `precomputed`. Files not
listed above are permitted only when their names match one declared local-sweep
patterns; they remain ignored, rebuildable research assets and are not protocol
authorities.
