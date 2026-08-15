# Feature cache contract and legacy boundary

## Decision

Existing feature caches remain historical `legacy` assets. They are not modified, backfilled,
or promoted to provenance-safe evidence. New cache roots created through contract-aware
producers use two immutable identity layers:

1. `.alpha_stalled_cache_contract.json` at the cache root;
2. `<video>.pt.meta.json` beside every tensor payload.

The implementation is in `src/alpha_stalled/cache_contract.py`. This change affects cache
governance only; it does not alter locked U0 scores, parameters, or release artifacts.

## Current snapshot

The cache inventory snapshot `cache_inventory_20260814` contains ten global/patch feature
roots and 97,900 feature payload files. None of those roots contains a strict root contract.
They therefore remain legacy even when their tensor shapes match the current encoder.

The small patch demo root was checked with:

```bash
conda run --no-capture-output -n stall \
  python tools/verify_feature_cache_contract.py \
  --cache-root cache/patch_embeddings/demo_dataset_duration2_filtered \
  --allow-legacy --json
```

It reports 34 cache files, `mode=legacy`, and `strict_evidence=false`. `--allow-legacy` is an
inventory diagnostic, not evidence that checkpoint, layer, preprocessing, or source videos
match a new experiment.

## Root identity

The strict root contract binds:

- cache kind: global or global+patch;
- DINOv3 implementation and tracked Git commit;
- checkpoint filename, bytes, and SHA-256 (`checkpoint_sha256`);
- final output layer, token types, feature dimension, dtype, and CLS/register policy;
- BGR-to-RGB conversion, resize, antialiasing, and normalization;
- frame/video batch sizes and cross-video frame grouping;
- hashes of the shared backbone, video decoder, and local extractor source files;
- PyTorch and torchvision versions;
- path-key and per-entry metadata conventions.

Tracked modifications to the DINOv3 source tree are rejected in strict mode. Untracked local
checkpoint files are allowed because checkpoint content is independently identified by SHA-256.

## Entry identity

Each strict entry binds:

- root-contract SHA-256;
- cache-relative path, bytes, and SHA-256;
- source-video resolved path, bytes, mtime, and SHA-256;
- the exact ordered native frame-index list and its SHA-256;
- tensor format, shapes, dtypes, and patch grid.

Strict producers decode the exact recorded index sequence with
`decode_indexed_frames(..., require_all=True)`. An undecodable requested frame aborts cache
production instead of producing a shorter tensor with a misleading sidecar. Historical legacy
paths retain the original omission behavior only for reproduction compatibility. The complete
decoder boundary is documented in `video_io_code_boundary.md`.

Normal reads verify contract, path, file size, source identity, frame indices, and payload
descriptor. Full cache-content hashing is optional because reading every patch tensor can add
hundreds of GiB of I/O; release-grade cache audits should add `--verify-cache-hashes`.

Patch whitening/CDF parameter files created by `src/create_patch_params.py` additionally store
`feature_cache_contract_sha256`. `src/eval_patch_fast.py` requires that value to equal the patch
cache root contract before scoring. Historical parameter files without the field are treated as
`legacy_uncontracted` and can only be used with legacy caches. Thus a tensor cache and statistics
fitted from that cache form one provenance chain rather than two independently interchangeable
assets.

## Policy modes

| Policy | Empty/new root | Existing root without contract | Root with contract |
|---|---|---|---|
| `auto` | creates strict contract | legacy with warning | strict comparison |
| `strict` | creates/requires contract | refuses to bless old `.pt` files | strict comparison |
| `legacy` | writes/reads legacy payloads | explicitly allowed | rejected |

Changing checkpoint, DINO commit, layer, preprocessing, batch grouping, or implementation
requires a new cache root. A mismatched contract is never overwritten.

## Contract-aware paths

- `src/eval.py` and `src/dataset_utils.py` for global cache production/readback;
- `tools/prefill_patch_cache.py` and `src/dataset_utils_patch.py` for patch production;
- `src/create_patch_params.py` for contract-bound whitening/CDF fitting;
- `src/eval_patch_fast.py` for strict patch score loading;
- `tools/verify_feature_cache_contract.py` for root and entry audits.

Historical research tools that directly call `torch.load` remain legacy consumers. They may
reproduce historical results, but they cannot establish strict cache compatibility for new
evidence until migrated to the shared loader.

## New-cache commands

```bash
conda run --no-capture-output -n stall \
  python tools/prefill_patch_cache.py \
  --csv cache/indexes/<dataset>.csv \
  --patch-emb-cache cache/patch_embeddings/<new_contract_root> \
  --duration 2 --compact --cache-policy strict --execute \
  --output-summary-csv results/runs/<experiment_id>/patch_prefill.csv

conda run --no-capture-output -n stall \
  python tools/verify_feature_cache_contract.py \
  --cache-root cache/patch_embeddings/<new_contract_root> \
  --verify-cache-hashes
```

Do not point `--cache-policy strict` at an existing legacy root. Strict mode intentionally
refuses to infer missing provenance from filenames, shapes, or current local defaults.
