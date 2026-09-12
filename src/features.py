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


DINO_V3_MODEL_NAME = "dinov3_vitl16"
DINOV3_GITHUB_URL = "https://github.com/facebookresearch/dinov3"
REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
DINO_V3_REPO_DIR = str(
    Path(os.environ.get("DINO_V3_REPO_DIR", REPOSITORY_ROOT / "dinov3")).resolve()
)
DINO_V3_WEIGHTS = str(
    Path(
        os.environ.get(
            "DINO_V3_WEIGHTS",
            REPOSITORY_ROOT / "dinov3/weights/dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth",
        )
    ).resolve()
)


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

    return transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Resize((resize_size, resize_size), antialias=True),
            transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ]
    )


def resolve_dinov3_paths(
    repo_dir: str | None = None, weights: str | None = None
) -> tuple[str, str]:
    """解析 DINOv3 代码与权重的绝对路径。"""

    return str(Path(repo_dir or DINO_V3_REPO_DIR).resolve()), str(
        Path(weights or DINO_V3_WEIGHTS).resolve()
    )


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
        pad_tail_batch: bool = False,
    ):
        self.dino_repo_path, self.dino_weights_path = resolve_dinov3_paths(dino_repo, dino_weights)
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
        # 新协议必须显式启用，历史缓存和参考统计不自动迁移。
        self.pad_tail_batch = bool(pad_tail_batch)

    def _forward_features_dict(self, x: torch.Tensor, *, require_patch_tokens: bool = True) -> dict:
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

    def _frame_tensor(self, frame):
        """允许预取线程提供已完成相同变换的CPU张量；普通BGR输入路径不变。"""
        if isinstance(frame, torch.Tensor):
            if (
                frame.device.type != "cpu"
                or frame.dtype != torch.float32
                or tuple(frame.shape) != (3, 224, 224)
            ):
                raise ValueError("预处理帧必须是CPU float32 [3,224,224]")
            return frame
        return self.transform(Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)))

    def prepare_frames(self, frames):
        """固定的CPU预处理，无随机变换、无模型前向，可由解码线程调用。"""
        with torch.no_grad():
            return torch.stack([self._frame_tensor(frame) for frame in frames])

    def _batch_tensor(self, tensors, batch_size):
        """重复末帧补齐计算batch；调用者必须裁去补齐输出。"""
        if batch_size < 1 or not tensors or len(tensors) > batch_size:
            raise ValueError("帧batch为空或尺寸非法")
        if getattr(self, "pad_tail_batch", False) and len(tensors) < batch_size:
            tensors = [*tensors, *([tensors[-1]] * (batch_size - len(tensors)))]
        return torch.stack(tensors)

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
                tensors = [self._frame_tensor(frame) for frame in batch]
                features = self._forward_features_dict(
                    self._batch_tensor(tensors, batch_size).to(device), require_patch_tokens=False
                )
                if "x_norm_clstoken" not in features:
                    raise KeyError("forward_features() 输出中缺少 'x_norm_clstoken'")
                core_model = self.model.module if hasattr(self.model, "module") else self.model
                outputs.append(
                    core_model.head(features["x_norm_clstoken"])[: len(batch)].detach().cpu()
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
                tensors = [self._frame_tensor(fr) for fr in batch]
                x = self._batch_tensor(tensors, batch_size).to(device)

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
                side = int(round(num_patches**0.5))
                if side * side != num_patches:
                    raise ValueError(f"Patch token 数量 {num_patches} 不是平方网格")
                batch_grid_size = (side, side)

                if patch_grid_size is None:
                    patch_grid_size = batch_grid_size
                elif patch_grid_size != batch_grid_size:
                    raise ValueError(
                        f"Patch 网格尺寸不一致: {patch_grid_size} vs {batch_grid_size}"
                    )

                global_embs.append(global_batch[: len(batch)].detach().cpu())
                patch_embs.append(patch_batch[: len(batch)].detach().cpu())

        global_out = torch.cat(global_embs, dim=0).numpy()
        patch_out = torch.cat(patch_embs, dim=0).numpy()
        return global_out, patch_out, patch_grid_size

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
