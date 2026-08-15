# Dataset catalog

This report is generated from `configs/data_catalog.yaml`, the three canonical index
CSVs, and the locked U0 calibration/evaluation manifests. It inventories identities
and protocol membership; it is not a content-hash manifest for 115 GiB of source video.

## Summary

- Snapshot: `data_catalog_20260814`
- Canonical video identities: 60,949
- Locked U0 identities: 22,021 (600 calibration + 21,421 evaluation)
- Canonical identities outside locked U0: 38,928
- Missing source files on this workspace: 0

## Canonical datasets

| Dataset | Index rows | Real | Generated | Sources | >=2 s | U0 calib | U0 eval | Outside U0 | Missing |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| comgenvid | 5,100 | 1,700 | 3,400 | 3 | 5,098 | 200 | 4,298 | 602 | 0 |
| videofeedback | 37,661 | 4,080 | 33,581 | 13 | 34,410 | 200 | 3,500 | 33,961 | 0 |
| genvideo | 18,188 | 9,984 | 8,204 | 11 | 15,601 | 200 | 13,623 | 4,365 | 0 |

## Source coverage

| Dataset | Subset | Source | Videos | >=2 s | U0 calib | U0 eval | Outside U0 |
|---|---|---|---:|---:|---:|---:|---:|
| comgenvid | annotated | Sora | 1,700 | 1,700 | 0 | 1,700 | 0 |
| comgenvid | annotated | VEO3 | 1,700 | 1,700 | 0 | 1,700 | 0 |
| comgenvid | real | MSVD | 1,700 | 1,698 | 200 | 898 | 602 |
| videofeedback | annotated | AnimateDiff | 1,399 | 1,399 | 0 | 300 | 1,099 |
| videofeedback | annotated | Fast-SVD | 1,000 | 1,000 | 0 | 300 | 700 |
| videofeedback | annotated | Hotshot-XL | 3,251 | 0 | 0 | 0 | 3,251 |
| videofeedback | annotated | LVDM | 3,172 | 3,172 | 0 | 300 | 2,872 |
| videofeedback | annotated | LaVie-base | 3,214 | 3,214 | 0 | 300 | 2,914 |
| videofeedback | annotated | ModelScope | 4,565 | 4,565 | 0 | 300 | 4,265 |
| videofeedback | annotated | Pika | 4,644 | 4,644 | 0 | 300 | 4,344 |
| videofeedback | annotated | SoRA-Clip | 920 | 920 | 0 | 300 | 620 |
| videofeedback | annotated | Text2Video-Zero | 4,642 | 4,642 | 0 | 300 | 4,342 |
| videofeedback | annotated | VideoCrafter2 | 4,596 | 4,596 | 0 | 300 | 4,296 |
| videofeedback | annotated | ZeroScope-576w | 2,178 | 2,178 | 0 | 300 | 1,878 |
| videofeedback | real | DiDeMo | 1,861 | 1,861 | 105 | 250 | 1,506 |
| videofeedback | real | Panda70M | 2,219 | 2,219 | 95 | 250 | 1,874 |
| genvideo | annotated | Crafter | 1,398 | 188 | 0 | 188 | 1,210 |
| genvideo | annotated | Gen2 | 1,380 | 1,380 | 0 | 1,380 | 0 |
| genvideo | annotated | HotShot | 700 | 0 | 0 | 0 | 700 |
| genvideo | annotated | Lavie | 1,400 | 1,400 | 0 | 1,400 | 0 |
| genvideo | annotated | ModelScope | 700 | 700 | 0 | 700 | 0 |
| genvideo | annotated | MoonValley | 626 | 0 | 0 | 0 | 626 |
| genvideo | annotated | MorphStudio | 700 | 700 | 0 | 700 | 0 |
| genvideo | annotated | Show_1 | 700 | 700 | 0 | 700 | 0 |
| genvideo | annotated | Sora | 56 | 56 | 0 | 56 | 0 |
| genvideo | annotated | WildScrape | 544 | 493 | 0 | 515 | 29 |
| genvideo | real | MSR-VTT | 9,984 | 9,984 | 200 | 7,984 | 1,800 |

## Identity and evidence boundary

- A canonical `video_id` is SHA-256 of `dataset|subset|source_model|filename`.
- Every locked calibration/evaluation identity must occur exactly once in a canonical index.
- Index and locked-manifest SHA-256 values are stored in the machine-readable catalog.
- Source-file presence is audited, but source video bytes are not hashed by this catalog.
- Derived calibration/holdout CSVs remain experiment artifacts, not additional canonical datasets.
- `not_in_locked_release` means available in a canonical index but unused by locked U0; it does not mean rejected or invalid.

Rebuild/check with:

```bash
conda run --no-capture-output -n stall python tools/build_data_catalog.py --check
conda run --no-capture-output -n stall python tools/verify_data_catalog.py
```
