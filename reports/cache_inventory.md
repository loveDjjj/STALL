# Cache inventory and retention audit

Snapshot: `cache_inventory_20260814`. This report is non-destructive; deletion requires explicit manual approval.

Registered groups/files: 13/249764. Logical size: 386.08 GiB. Filesystem `du` may differ because it reports allocated blocks.

| Cache group | Path | Files | GiB | Lifecycle | Priority | Metadata |
|---|---|---:|---:|---|---|---|
| `protocol_indexes` | `cache/indexes` | 40 | 0.05 | `protocol_index` | `P0_keep` | `external_only` |
| `global_demo_compact` | `cache/embeddings/demo_dataset_duration2_filtered` | 34 | 0.00 | `feature_cache` | `P0_keep` | `partial` |
| `global_comgenvid_compact` | `cache/embeddings/comgenvid` | 5,098 | 0.32 | `feature_cache` | `P1_keep_active` | `partial` |
| `global_videofeedback_legacy` | `cache/embeddings/videofeedback` | 37,661 | 2.60 | `feature_cache` | `P2_review` | `partial` |
| `global_genvideo_legacy` | `cache/embeddings/genvideo` | 18,188 | 5.31 | `feature_cache` | `P1_keep_active` | `partial` |
| `global_genvidbench_external` | `cache/embeddings/genvidbench_pair1_ms_vript_eval` | 900 | 0.06 | `feature_cache` | `P2_review` | `partial` |
| `patch_demo_compact` | `cache/patch_embeddings/demo_dataset_duration2_filtered` | 34 | 0.41 | `feature_cache` | `P0_keep` | `partial` |
| `patch_comgenvid_compact` | `cache/patch_embeddings/comgenvid` | 10,198 | 91.98 | `feature_cache` | `P1_keep_active` | `partial` |
| `patch_videofeedback_compact` | `cache/patch_embeddings/videofeedback` | 4,500 | 51.11 | `feature_cache` | `P1_keep_active` | `partial` |
| `patch_genvideo_compact` | `cache/patch_embeddings/genvideo` | 20,188 | 215.33 | `feature_cache` | `P1_keep_active` | `partial` |
| `patch_genvidbench_external` | `cache/patch_embeddings/genvidbench_pair1_ms_vript` | 1,099 | 13.22 | `feature_cache` | `P1_keep_active` | `partial` |
| `patch_shards_debug` | `cache/patch_embedding_shards_debug` | 7 | 1.92 | `debug_benchmark` | `P3_safe_delete_candidate` | `external_only` |
| `d3_extracted_frames` | `cache/d3_frames` | 151,817 | 3.77 | `external_method_frames` | `P2_review` | `external_only` |

## Cleanup classes

- `P0_keep`: 108 files, 0.46 GiB.
- `P1_keep_active`: 59,271 files, 377.27 GiB.
- `P2_review`: 190,378 files, 6.43 GiB.
- `P3_safe_delete_candidate`: 7 files, 1.92 GiB.

## Key findings

- No registered cache is required to validate the immutable locked U0 release; the release retains manifests, raw-shard hashes, parameters, and final scores.
- Existing feature cache filenames do not encode checkpoint SHA, DINO commit, layer, preprocessing, frame grouping, or numerical implementation.
- The current snapshot predates the strict feature-cache contract and remains legacy. New empty roots created by contract-aware producers receive immutable root and per-entry metadata; existing roots are never backfilled automatically.
- Patch payloads retain frame indices and grid size, but that is not a complete cache key. Shape compatibility alone does not authorize reuse.
- `P3_safe_delete_candidate` means protocol-independent evidence says the group is disposable; this report does not delete it.

## Group decisions

### `protocol_indexes`

- Purpose: Enriched video indexes, calibration subsets, and evaluation holdout manifests.
- Producer: `src/video_index.py and protocol-specific manifest builders`
- Retention: `keep`; release dependency: `false`.
- Missing cache-key fields: source_video_sha256, index_builder_git_commit, ffprobe_version.
- Decision note: Small protocol-bearing assets; base indexes are reproducible, but no single retained command rebuilds every derived holdout and calibration CSV exactly.

### `global_demo_compact`

- Purpose: Small two-second global feature cache used for smoke tests and examples.
- Producer: `src/eval.py via src/dataset_utils.py`
- Retention: `keep`; release dependency: `false`.
- Missing cache-key fields: checkpoint_sha256, dinov3_commit, output_layer, preprocessing, frame_grouping, numerical_implementation, source_video_fingerprint.
- Decision note: Retain because it is small and useful for fast integration checks.

### `global_comgenvid_compact`

- Purpose: Compact global DINO features for ComGenVid experiments.
- Producer: `src/eval.py via src/dataset_utils.py`
- Retention: `keep_active`; release dependency: `false`.
- Missing cache-key fields: checkpoint_sha256, dinov3_commit, output_layer, preprocessing, frame_grouping, numerical_implementation, source_video_fingerprint.
- Decision note: Active research cache; reuse only after checking the missing model and preprocessing identity fields.

### `global_videofeedback_legacy`

- Purpose: Historical full or variably sampled global embeddings for VideoFeedback.
- Producer: `src/eval.py via src/dataset_utils.py`
- Retention: `review_before_delete`; release dependency: `false`.
- Missing cache-key fields: requested_duration, frame_indices, checkpoint_sha256, dinov3_commit, output_layer, preprocessing, frame_grouping, numerical_implementation, source_video_fingerprint.
- Decision note: Review before reuse or deletion because unsuffixed files do not identify their sampling protocol.

### `global_genvideo_legacy`

- Purpose: Historical full or variably sampled global embeddings used by D3 and baseline studies.
- Producer: `src/eval.py via src/dataset_utils.py`
- Retention: `keep_active`; release dependency: `false`.
- Missing cache-key fields: requested_duration, frame_indices, checkpoint_sha256, dinov3_commit, output_layer, preprocessing, frame_grouping, numerical_implementation, source_video_fingerprint.
- Decision note: Still consumed by historical D3 comparisons; isolate from compact two-second caches.

### `global_genvidbench_external`

- Purpose: Compact global features for the locked external GenVidBench evaluation pair.
- Producer: `src/eval.py via src/dataset_utils.py`
- Retention: `review_before_delete`; release dependency: `false`.
- Missing cache-key fields: checkpoint_sha256, dinov3_commit, output_layer, preprocessing, frame_grouping, numerical_implementation, source_video_fingerprint.
- Decision note: No current code consumer was found; review before deletion because the cache is small and tied to external-validation history.

### `patch_demo_compact`

- Purpose: Small two-second patch-token cache used for smoke tests and examples.
- Producer: `tools/prefill_patch_cache.py via src/dataset_utils_patch.py`
- Retention: `keep`; release dependency: `false`.
- Missing cache-key fields: checkpoint_sha256, dinov3_commit, output_layer, preprocessing, frame_grouping, numerical_implementation, source_video_fingerprint.
- Decision note: Retain because it is small and supports cheap end-to-end checks.

### `patch_comgenvid_compact`

- Purpose: Compact one- and two-second patch-token features for ComGenVid research.
- Producer: `tools/prefill_patch_cache.py via src/dataset_utils_patch.py`
- Retention: `keep_active`; release dependency: `false`.
- Missing cache-key fields: checkpoint_sha256, dinov3_commit, output_layer, preprocessing, frame_grouping, numerical_implementation, source_video_fingerprint.
- Decision note: Largest active ComGenVid research asset; do not infer compatibility from tensor shape alone.

### `patch_videofeedback_compact`

- Purpose: Compact one- and two-second patch-token features for VideoFeedback research.
- Producer: `tools/prefill_patch_cache.py via src/dataset_utils_patch.py`
- Retention: `keep_active`; release dependency: `false`.
- Missing cache-key fields: checkpoint_sha256, dinov3_commit, output_layer, preprocessing, frame_grouping, numerical_implementation, source_video_fingerprint.
- Decision note: Active research cache covering both strict two-second and duration-aware one-second protocols.

### `patch_genvideo_compact`

- Purpose: Compact one- and two-second patch-token features for GenVideo research.
- Producer: `tools/prefill_patch_cache.py via src/dataset_utils_patch.py`
- Retention: `keep_active`; release dependency: `false`.
- Missing cache-key fields: checkpoint_sha256, dinov3_commit, output_layer, preprocessing, frame_grouping, numerical_implementation, source_video_fingerprint.
- Decision note: Largest cache group and still actively reused; retain until research is archived.

### `patch_genvidbench_external`

- Purpose: Compact patch-token features for external GenVidBench calibration and evaluation.
- Producer: `tools/prefill_patch_cache.py via src/dataset_utils_patch.py`
- Retention: `keep_active`; release dependency: `false`.
- Missing cache-key fields: checkpoint_sha256, dinov3_commit, output_layer, preprocessing, frame_grouping, numerical_implementation, source_video_fingerprint.
- Decision note: Retain while the external validation result remains in the paper evidence set.

### `patch_shards_debug`

- Purpose: Historical storage-throughput benchmark shards and statistics.
- Producer: `historical producer not retained in current code`
- Retention: `safe_delete_candidate`; release dependency: `false`.
- Missing cache-key fields: source_cache_hashes, checkpoint_sha256, producer_git_commit, exact_producer_command.
- Decision note: No current consumer or release dependency was found; deletion still requires explicit manual approval.

### `d3_extracted_frames`

- Purpose: Three-second 8 FPS JPEG frame sequences for the external D3 comparison protocol.
- Producer: `tools/extract_d3_frames_from_runlist.py`
- Retention: `review_before_delete`; release dependency: `false`.
- Missing cache-key fields: extraction_runlist_sha256, source_video_fingerprint, ffmpeg_version, decoder_configuration.
- Decision note: Rebuildable but costly; review only after the D3 comparison evidence is fully archived.
