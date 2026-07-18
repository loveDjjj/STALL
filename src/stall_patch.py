from __future__ import annotations

from typing import Dict, List, Tuple

import cv2
import numpy as np
import torch
from PIL import Image

from stall import (
    STALL,
    create_dinov3_transform,
    load_dinov3_model,
    load_video_frames,
)


class PatchSTALL(STALL):
    """STALL 的最小 patch 级扩展。

    本类只负责 DINOv3 global + patch token 提取。打分、参数拟合和 patch 时序
    逻辑由其他模块实现。
    """

    _shared_patch_model = None
    _shared_patch_transform = None

    def __init__(
        self,
        device,
        data_dict: dict | None = None,
        spat_agg: str = "max",
        temp_agg: str = "min",
        dino_repo: str | None = None,
        dino_weights: str | None = None,
        load_dino: bool = True,
    ):
        if load_dino:
            if PatchSTALL._shared_patch_model is None:
                PatchSTALL._shared_patch_model, PatchSTALL._shared_patch_transform = load_dinov3_model(
                    device, repo_dir=dino_repo, weights=dino_weights
                )
            self.model = PatchSTALL._shared_patch_model
            self.transform = PatchSTALL._shared_patch_transform
        else:
            self.model = None
            self.transform = None

        self.device = device
        self.spat_agg = spat_agg
        self.temp_agg = temp_agg

        if data_dict is not None:
            self.w_spat = data_dict["W_spat"]
            self.mu_spat = data_dict["mu_spat"]
            self.w_temp = data_dict["W_temp"]
            self.mu_temp = data_dict["mu_temp"]
            calib_spat = data_dict["calib_ll_spat"]
            calib_temp = data_dict["calib_ll_temp"]
            self.calib_spat_sorted = np.sort(np.max(calib_spat, axis=1))
            self.calib_temp_sorted = np.sort(np.min(calib_temp, axis=1))

    def _forward_features_dict(self, x: torch.Tensor) -> dict:
        if hasattr(self.model, "module"):
            core_model = self.model.module
        else:
            core_model = self.model

        if not hasattr(core_model, "forward_features"):
            raise AttributeError("DINOv3 模型没有暴露 forward_features()")

        features = core_model.forward_features(x)
        if not isinstance(features, dict):
            raise TypeError(f"期望 forward_features() 返回 dict，实际为 {type(features).__name__}")
        if "x_norm_patchtokens" not in features:
            raise KeyError("forward_features() 输出中缺少 'x_norm_patchtokens'")
        return features

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

                global_batch = self.model(x)
                feature_dict = self._forward_features_dict(x)
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

    def video_to_global_patch_embeddings(
        self, video_path: str, frame_indices=None, batch_size: int = 32
    ) -> Dict[str, np.ndarray | List[int]]:
        """单个视频路径的便捷包装。"""
        frames = load_video_frames(video_path, frame_indices)
        if len(frames) == 0:
            raise ValueError(f"未能从 {video_path} 解码出帧")
        return self.frames_to_global_patch_embeddings([frames], batch_size=batch_size)[0]
