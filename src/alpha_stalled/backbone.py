"""DINOv3 identity, preprocessing, loading, and process-local reuse."""

from __future__ import annotations

import os
import sys
import threading
from dataclasses import dataclass
from pathlib import Path

import torch
from torchvision import transforms


DINO_V3_MODEL_NAME = "dinov3_vitl16"
DINOV3_GITHUB_URL = "https://github.com/facebookresearch/dinov3"

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
DINO_V3_REPO_DIR = str(
    Path(
        os.environ.get("DINO_V3_REPO_DIR", REPOSITORY_ROOT / "dinov3")
    ).resolve()
)
DINO_V3_WEIGHTS = str(
    Path(
        os.environ.get(
            "DINO_V3_WEIGHTS",
            REPOSITORY_ROOT
            / "dinov3/weights/dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth",
        )
    ).resolve()
)


@dataclass(frozen=True)
class DinoBackboneHandle:
    model: torch.nn.Module
    transform: object
    cache_key: tuple[str, str, str, int, int]


_CACHE_LOCK = threading.RLock()
_CACHED_HANDLE: DinoBackboneHandle | None = None


def create_dinov3_transform(resize_size: int = 224):
    """Return the DINOv3 LVD-1689M ImageNet evaluation transform."""

    return transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Resize((resize_size, resize_size), antialias=True),
            transforms.Normalize(
                mean=(0.485, 0.456, 0.406),
                std=(0.229, 0.224, 0.225),
            ),
        ]
    )


def resolve_dinov3_paths(
    repo_dir: str | None = None,
    weights: str | None = None,
) -> tuple[str, str]:
    """Return canonical DINOv3 code and checkpoint paths."""

    return (
        str(Path(repo_dir or DINO_V3_REPO_DIR).resolve()),
        str(Path(weights or DINO_V3_WEIGHTS).resolve()),
    )


def dinov3_model_cache_key(
    device: str | torch.device,
    repo_dir: str | None = None,
    weights: str | None = None,
) -> tuple[str, str, str, int, int]:
    """Bind process reuse to code path, checkpoint path/version, and device."""

    resolved_repo, resolved_weights = resolve_dinov3_paths(repo_dir, weights)
    stat = Path(resolved_weights).stat()
    return (
        resolved_repo,
        resolved_weights,
        str(device),
        stat.st_size,
        stat.st_mtime_ns,
    )


def load_dinov3_model(
    device: str | torch.device,
    repo_dir: str | None = None,
    weights: str | None = None,
):
    """Load the local DINOv3 ViT-L/16 checkpoint without torch.hub."""

    repo_dir, weights = resolve_dinov3_paths(repo_dir, weights)
    if not Path(repo_dir).is_dir():
        raise ValueError(
            f"未在 '{repo_dir}' 找到 DINOv3 repo。\n"
            f"请从 {DINOV3_GITHUB_URL} clone，并把权重放到 weights/。\n"
            "可通过 --dino-repo 或 DINO_V3_REPO_DIR 环境变量覆盖路径。"
        )
    if not Path(weights).is_file():
        raise ValueError(
            f"未在 '{weights}' 找到 DINOv3 权重。\n"
            f"请从 {DINOV3_GITHUB_URL} 下载 dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth。\n"
            "可通过 --dino-weights 或 DINO_V3_WEIGHTS 环境变量覆盖路径。"
        )

    sys.path.insert(0, repo_dir)
    try:
        from dinov3.hub.backbones import dinov3_vitl16

        model = dinov3_vitl16(weights=weights)
    finally:
        if repo_dir in sys.path:
            sys.path.remove(repo_dir)
    model = model.to(device).eval()
    if str(device) == "cuda" and torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model)
    transform = create_dinov3_transform()
    if str(device) == "cuda":
        print(f"DINOv3 模型已加载到 {device} ({torch.cuda.device_count()} 个可见 GPU)")
    else:
        print(f"DINOv3 模型已加载到 {device}")
    return model, transform


def get_shared_dinov3_model(
    device: str | torch.device,
    repo_dir: str | None = None,
    weights: str | None = None,
) -> DinoBackboneHandle:
    """Return one process-local model shared by STALL and PatchSTALL."""

    global _CACHED_HANDLE
    key = dinov3_model_cache_key(device, repo_dir, weights)
    with _CACHE_LOCK:
        if _CACHED_HANDLE is None or _CACHED_HANDLE.cache_key != key:
            resolved_repo, resolved_weights = key[:2]
            model, transform = load_dinov3_model(
                device,
                repo_dir=resolved_repo,
                weights=resolved_weights,
            )
            _CACHED_HANDLE = DinoBackboneHandle(model, transform, key)
        return _CACHED_HANDLE


def clear_shared_dinov3_model() -> None:
    """Clear the process-local handle; intended for tests and explicit teardown."""

    global _CACHED_HANDLE
    with _CACHE_LOCK:
        _CACHED_HANDLE = None


__all__ = [
    "DINOV3_GITHUB_URL",
    "DINO_V3_MODEL_NAME",
    "DINO_V3_REPO_DIR",
    "DINO_V3_WEIGHTS",
    "DinoBackboneHandle",
    "clear_shared_dinov3_model",
    "create_dinov3_transform",
    "dinov3_model_cache_key",
    "get_shared_dinov3_model",
    "load_dinov3_model",
    "resolve_dinov3_paths",
]
