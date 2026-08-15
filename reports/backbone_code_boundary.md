# DINOv3 backbone code boundary

## Decision

`src/alpha_stalled/backbone.py` is the canonical implementation for:

- default/environment-overridden DINOv3 repository and checkpoint paths;
- canonical path resolution and process-cache identity;
- the `224x224` ImageNet evaluation transform;
- local ViT-L/16 loading without `torch.hub`;
- one process-local model handle shared by `STALL` and `PatchSTALL`.

`src/stall.py` re-exports the historical function and constant names so original STALL tools
and external callers remain compatible. Current project tools import backbone symbols directly
from `alpha_stalled.backbone`.

## Process reuse identity

The in-process key is:

```text
resolved DINO repo path
+ resolved checkpoint path
+ device
+ checkpoint byte size
+ checkpoint mtime_ns
```

This key prevents a second load when Global and Patch extraction use the same model, and forces
a reload when the checkpoint file version or device changes. It is a runtime reuse mechanism,
not release-grade provenance.

## Persistent cache identity

Strict feature-cache roots remain stronger. Their contract records DINO Git commit,
checkpoint bytes and content SHA-256, output layer/tokens, preprocessing, batching, and extractor
source hashes. `backbone.py` is now included in those source hashes for both Global and Patch
caches. Existing ten feature-cache roots remain legacy and were not backfilled.

## Extraction boundary

`STALL` retains original global CLS extraction and original-detector compatibility.
`PatchSTALL` retains the single-forward global/patch-token split and intermediate-layer patch
extraction. Both receive the same shared model/transform handle. Scoring formulas remain in
`global_branch.py`, `local_branch.py`, and `whitening.py`; the backbone module does not implement
detector fusion or calibration.

## Verification

`tests/test_backbone.py`, `tests/test_patch_single_forward.py`, and
`tests/test_cache_contract.py` verify:

- old `stall` names re-export the canonical functions;
- model keys change with checkpoint identity;
- `STALL` and `PatchSTALL` load one shared model for the same key;
- changing checkpoint reloads the handle;
- global and patch tokens still share one DINO forward per frame batch;
- strict contracts include `backbone.py` source identity.

The locked U0 release verifier must remain at Macro AUC/AP
`0.8740724396350226/0.8722992273320395`.
