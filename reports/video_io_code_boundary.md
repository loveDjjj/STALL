# Video I/O code boundary

## Decision

Feature-producing video decode semantics are centralized in
`src/alpha_stalled/video_io.py`. The module exposes three distinct contracts instead of using
one ambiguous loader for every protocol:

| API | Ordering | Missing-frame behavior | Intended use |
|---|---|---|---|
| `decode_all_frames` | native sequential order | optional open failure | original full-video compatibility |
| `decode_indexed_frames` | preserves requested order and duplicates | legacy omission or `require_all=True` failure | cache producers and original indexed compatibility |
| `decode_selected_frames` | sorted unique indices | always fails | locked multi-window scoring |

`src/stall.py` reexports the canonical `stall.load_video_frames` compatibility object. It no
longer owns a second OpenCV implementation. `src/stall_patch.py`, `src/create_params.py`, and
the D3 pixel robustness evaluator also import the shared implementation.

## Strict cache invariant

Contract-aware producers in `src/dataset_utils.py` and `src/dataset_utils_patch.py` call:

```python
decode_indexed_frames(video_path, frame_indices, require_all=True)
```

when cache policy is strict. A missing frame or unreadable source therefore aborts production.
The tensor length can no longer be shorter than the ordered frame-index list recorded in its
`.pt.meta.json` sidecar. Legacy cache mode keeps the historical omission behavior so old
experiments remain reproducible, but those caches are not strict evidence.

The strict root contract hashes `video_io.py` together with the backbone and extractor source.
Changing decode semantics therefore requires a new cache root rather than silently reusing
features produced by different code.

## Specialized direct OpenCV users

Some tools still open videos directly for metadata, duration audits, keyframe rendering, or
single-frame diagnostics. Those operations do not produce the Global/Patch feature caches and
have task-specific seek/output contracts. They should migrate only when a shared metadata or
visualization API is defined; replacing them mechanically would conflate different behavior.

## Verification

`tests/test_video_io.py` covers full decode, request-order preservation, duplicate indices,
legacy omission, strict missing-frame failure, invalid indices, unopenable videos, and locked
sorted-unique decoding. `tests/test_cache_contract.py` verifies that both Global and Patch
strict producers use the strict indexed entry and that `video_io.py` is part of cache identity.

The locked release verifier remains unchanged at Macro AUC/AP
`0.8740724396350226/0.8722992273320395`; video I/O consolidation does not rewrite released
scores or artifacts.
