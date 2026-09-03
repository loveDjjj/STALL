from __future__ import annotations

import os
import sys
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import numpy as np
import torch
from PIL import Image
from torchvision import transforms

from data.video import load_video_frames


DINO_V3_MODEL_NAME = "dinov3_vitl16"
DINOV3_GITHUB_URL = "https://github.com/facebookresearch/dinov3"
REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
DINO_V3_REPO_DIR = str(Path(os.environ.get("DINO_V3_REPO_DIR", REPOSITORY_ROOT / "dinov3")).resolve())
DINO_V3_WEIGHTS = str(Path(os.environ.get(
    "DINO_V3_WEIGHTS", REPOSITORY_ROOT / "dinov3/weights/dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth",
)).resolve())


@dataclass(frozen=True)
class DinoBackboneHandle:
    """已加载 DINOv3 模型及其预处理变换。"""

    model: torch.nn.Module
    transform: object
    cache_key: tuple[str, str, str, int, int]


_CACHE_LOCK = threading.RLock()
_CACHED_HANDLE: DinoBackboneHandle | None = None


def create_dinov3_transform(resize_size: int = 224):
    """创建 DINOv3 的固定图像预处理。"""

    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((resize_size, resize_size), antialias=True),
        transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ])


def resolve_dinov3_paths(
    repo_dir: str | None = None, weights: str | None = None
) -> tuple[str, str]:
    """解析 DINOv3 代码与权重的绝对路径。"""

    return str(Path(repo_dir or DINO_V3_REPO_DIR).resolve()), str(Path(weights or DINO_V3_WEIGHTS).resolve())


def _model_cache_key(
    device: str | torch.device, repo_dir: str | None, weights: str | None
) -> tuple[str, str, str, int, int]:
    repo_path, weight_path = resolve_dinov3_paths(repo_dir, weights)
    stat = Path(weight_path).stat()
    return repo_path, weight_path, str(device), stat.st_size, stat.st_mtime_ns


def _load_dinov3_model(device: str | torch.device, repo_dir: str, weights: str):
    if not Path(repo_dir).is_dir():
        raise ValueError(f"未在 '{repo_dir}' 找到 DINOv3 repo，请从 {DINOV3_GITHUB_URL} 获取代码。")
    if not Path(weights).is_file():
        raise ValueError(f"未在 '{weights}' 找到 DINOv3 权重。")
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
    return model, create_dinov3_transform()


def get_shared_dinov3_model(
    device: str | torch.device, repo_dir: str | None = None, weights: str | None = None
) -> DinoBackboneHandle:
    """加载或复用本进程中的 DINOv3 模型。"""

    global _CACHED_HANDLE
    key = _model_cache_key(device, repo_dir, weights)
    with _CACHE_LOCK:
        if _CACHED_HANDLE is None or _CACHED_HANDLE.cache_key != key:
            model, transform = _load_dinov3_model(device, key[0], key[1])
            _CACHED_HANDLE = DinoBackboneHandle(model, transform, key)
        return _CACHED_HANDLE


class AlphaStallFeatureExtractor:
    """Alpha STALL 的 DINOv3 全局与 patch 特征提取器。

    本类只负责 DINOv3 global + patch token 提取。打分、参数拟合和 patch 时序
    逻辑由其他模块实现。
    """

    def __init__(
        self,
        device: str,
        dino_repo: str | None = None,
        dino_weights: str | None = None,
        load_dino: bool = True,
    ):
        self.dino_repo_path, self.dino_weights_path = resolve_dinov3_paths(
            dino_repo, dino_weights
        )
        if load_dino:
            handle = get_shared_dinov3_model(
                device,
                self.dino_repo_path,
                self.dino_weights_path,
            )
            self.model = handle.model
            self.transform = handle.transform
        else:
            self.model = None
            self.transform = None

        self.device = device

    def _forward_features_dict(
        self, x: torch.Tensor, *, require_patch_tokens: bool = True
    ) -> dict:
        if hasattr(self.model, "module"):
            core_model = self.model.module
        else:
            core_model = self.model

        if not hasattr(core_model, "forward_features"):
            raise AttributeError("DINOv3 模型没有暴露 forward_features()")

        features = core_model.forward_features(x)
        if not isinstance(features, dict):
            raise TypeError(f"期望 forward_features() 返回 dict，实际为 {type(features).__name__}")
        if require_patch_tokens and "x_norm_patchtokens" not in features:
            raise KeyError("forward_features() 输出中缺少 'x_norm_patchtokens'")
        return features

    def _embed_flat_frames_global(
        self, flat_frames: List[np.ndarray], batch_size: int = 32
    ) -> np.ndarray:
        """只返回最终归一化Global token，不把Patch token搬回CPU。"""

        if not flat_frames:
            raise ValueError("flat_frames 为空")
        device = next(self.model.parameters()).device
        outputs = []
        with torch.no_grad():
            for start in range(0, len(flat_frames), batch_size):
                batch = flat_frames[start : start + batch_size]
                tensors = [
                    self.transform(
                        Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                    )
                    for frame in batch
                ]
                features = self._forward_features_dict(
                    torch.stack(tensors).to(device), require_patch_tokens=False
                )
                if "x_norm_clstoken" not in features:
                    raise KeyError("forward_features() 输出中缺少 'x_norm_clstoken'")
                core_model = (
                    self.model.module if hasattr(self.model, "module") else self.model
                )
                outputs.append(
                    core_model.head(features["x_norm_clstoken"]).detach().cpu()
                )
        return torch.cat(outputs, dim=0).numpy()

    def _embed_flat_frames_with_patches(
        self, flat_frames: List[np.ndarray], batch_size: int = 32
    ) -> Tuple[np.ndarray, np.ndarray, Tuple[int, int]]:
        """在展平帧列表上运行 DINOv3，并返回 global + patch embeddings。"""
        if len(flat_frames) == 0:
            raise ValueError("flat_frames 为空")

        device = next(self.model.parameters()).device
        global_embs = []
        patch_embs = []
        patch_grid_size = None

        with torch.no_grad():
            for start in range(0, len(flat_frames), batch_size):
                batch = flat_frames[start : start + batch_size]
                tensors = [
                    self.transform(Image.fromarray(cv2.cvtColor(fr, cv2.COLOR_BGR2RGB)))
                    for fr in batch
                ]
                x = torch.stack(tensors).to(device)

                feature_dict = self._forward_features_dict(x)
                if "x_norm_clstoken" not in feature_dict:
                    raise KeyError("forward_features() 输出中缺少 'x_norm_clstoken'")
                if hasattr(self.model, "module"):
                    core_model = self.model.module
                else:
                    core_model = self.model
                global_batch = core_model.head(feature_dict["x_norm_clstoken"])
                patch_batch = feature_dict["x_norm_patchtokens"]

                num_patches = patch_batch.shape[1]
                side = int(round(num_patches ** 0.5))
                if side * side != num_patches:
                    raise ValueError(f"Patch token 数量 {num_patches} 不是平方网格")
                batch_grid_size = (side, side)

                if patch_grid_size is None:
                    patch_grid_size = batch_grid_size
                elif patch_grid_size != batch_grid_size:
                    raise ValueError(
                        f"Patch 网格尺寸不一致: {patch_grid_size} vs {batch_grid_size}"
                    )

                global_embs.append(global_batch.detach().cpu())
                patch_embs.append(patch_batch.detach().cpu())

        global_out = torch.cat(global_embs, dim=0).numpy()
        patch_out = torch.cat(patch_embs, dim=0).numpy()
        return global_out, patch_out, patch_grid_size

    def _embed_flat_frames_with_layers(
        self,
        flat_frames: List[np.ndarray],
        layers: tuple[int, ...],
        batch_size: int = 32,
    ) -> Tuple[Dict[int, np.ndarray], Tuple[int, int]]:
        """Return normalized patch tokens for several blocks in one traversal."""
        if not flat_frames:
            raise ValueError("flat_frames 为空")
        if not layers or len(set(layers)) != len(layers):
            raise ValueError("layers 必须是非空且不重复的层号")

        core_model = self.model.module if hasattr(self.model, "module") else self.model
        if not hasattr(core_model, "get_intermediate_layers"):
            raise AttributeError("DINOv3 模型没有暴露 get_intermediate_layers()")

        device = next(self.model.parameters()).device
        collected: Dict[int, list[torch.Tensor]] = {layer: [] for layer in layers}
        patch_grid_size = None
        with torch.no_grad():
            for start in range(0, len(flat_frames), batch_size):
                batch = flat_frames[start : start + batch_size]
                tensors = [
                    self.transform(Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)))
                    for frame in batch
                ]
                x = torch.stack(tensors).to(device)
                outputs = core_model.get_intermediate_layers(
                    x,
                    n=list(layers),
                    reshape=False,
                    return_class_token=False,
                    norm=True,
                )
                if len(outputs) != len(layers):
                    raise ValueError(
                        f"中间层输出数量不匹配: expected={len(layers)} actual={len(outputs)}"
                    )
                for layer, patch_batch in zip(layers, outputs):
                    if isinstance(patch_batch, tuple):
                        patch_batch = patch_batch[0]
                    num_patches = patch_batch.shape[1]
                    side = int(round(num_patches**0.5))
                    if side * side != num_patches:
                        raise ValueError(f"Layer {layer} patch token 数量不是平方网格: {num_patches}")
                    current_grid = (side, side)
                    if patch_grid_size is None:
                        patch_grid_size = current_grid
                    elif patch_grid_size != current_grid:
                        raise ValueError(
                            f"Patch 网格尺寸不一致: {patch_grid_size} vs {current_grid}"
                        )
                    collected[layer].append(patch_batch.detach().cpu())

        return {
            layer: torch.cat(parts, dim=0).numpy() for layer, parts in collected.items()
        }, patch_grid_size

    def frames_to_layer_patch_embeddings(
        self,
        video_arrays: List[np.ndarray],
        layers: tuple[int, ...] = (11, 17, 23),
        batch_size: int = 32,
    ) -> List[Dict[str, object]]:
        """Extract requested patch layers and split the flat output by video."""
        if not video_arrays:
            return []
        lengths = [len(video) for video in video_arrays]
        if any(length == 0 for length in lengths):
            raise ValueError("video_arrays 至少包含一个空视频")
        flat_frames = [frame for video in video_arrays for frame in video]
        layer_flat, grid_size = self._embed_flat_frames_with_layers(
            flat_frames, layers=layers, batch_size=batch_size
        )
        outputs: List[Dict[str, object]] = []
        cursor = 0
        for length in lengths:
            outputs.append(
                {
                    "layers": {
                        layer: values[cursor : cursor + length]
                        for layer, values in layer_flat.items()
                    },
                    "grid_size": grid_size,
                }
            )
            cursor += length
        return outputs

    def frames_to_global_patch_embeddings(
        self, video_arrays: List[np.ndarray], batch_size: int = 32
    ) -> List[Dict[str, np.ndarray | List[int]]]:
        """为一组视频提取 global + patch embeddings。"""
        if len(video_arrays) == 0:
            return []

        lengths = [len(v) for v in video_arrays]
        if any(length == 0 for length in lengths):
            raise ValueError("video_arrays 至少包含一个空视频")

        flat_frames = [frame for video in video_arrays for frame in video]
        global_flat, patch_flat, grid_size = self._embed_flat_frames_with_patches(
            flat_frames, batch_size=batch_size
        )

        cursor = 0
        outputs = []
        for length in lengths:
            outputs.append(
                {
                    "global": global_flat[cursor : cursor + length],
                    "patch": patch_flat[cursor : cursor + length],
                    "grid_size": list(grid_size),
                }
            )
            cursor += length
        return outputs

    def frames_to_global_embeddings(
        self, video_arrays: List[np.ndarray], batch_size: int = 32
    ) -> List[np.ndarray]:
        """为一组视频只提取Global token，供CAES低成本粗扫描使用。"""

        if not video_arrays:
            return []
        lengths = [len(video) for video in video_arrays]
        if any(length == 0 for length in lengths):
            raise ValueError("video_arrays 至少包含一个空视频")
        flat_frames = [frame for video in video_arrays for frame in video]
        flat = self._embed_flat_frames_global(flat_frames, batch_size=batch_size)
        outputs, cursor = [], 0
        for length in lengths:
            outputs.append(flat[cursor : cursor + length])
            cursor += length
        return outputs

    def video_to_global_patch_embeddings(
        self, video_path: str, frame_indices=None, batch_size: int = 32
    ) -> Dict[str, np.ndarray | List[int]]:
        """单个视频路径的便捷包装。"""
        frames = load_video_frames(video_path, frame_indices)
        if len(frames) == 0:
            raise ValueError(f"未能从 {video_path} 解码出帧")
        return self.frames_to_global_patch_embeddings([frames], batch_size=batch_size)[0]
