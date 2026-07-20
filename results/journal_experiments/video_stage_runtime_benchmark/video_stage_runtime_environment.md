# Video-stage runtime benchmark environment

- platform: `Linux-6.8.0-110-generic-x86_64-with-glibc2.35`
- python: `3.10.20`
- executable: `/home/ubuntu/anaconda3/envs/stall/bin/python`
- repo root: `/data/OneDay/STALL_project/STALL`
- subprocess cwd: `/data/OneDay/STALL_project`
- dataset: `comgenvid`
- duration_sec: `2`
- debug_n per `(subset, source_model)`: `2`
- num_workers: `2`
- video_batch: `2`
- frame_batch: `16`
- score_batch: `8`
- device: `cuda`

## GPU

```text
0, NVIDIA GeForce RTX 5090, 32607 MiB, 580.82.07
1, NVIDIA GeForce RTX 5090, 32607 MiB, 580.82.07
```

## key packages

```text
torch 2.11.0+cu128
pandas 2.3.3
numpy 2.2.6
cuda_available True
cuda_device_count 2
```
