"""CAES selector专用1 FPS真实时序参考与候选窗口取证信号。"""

from __future__ import annotations

from dataclasses import dataclass, replace
import hashlib
import json
from pathlib import Path
from typing import Sequence

import numpy as np
import torch

from math_utils import (
    WhiteningTransform,
    apply_whitening,
    empirical_cdf_right_inclusive,
    l2_normalized_first_order,
    log_likelihood,
    stable_sorted,
)
from .models import CandidateWindow


@dataclass(frozen=True)
class SelectorReference:
    """仅由calibration real的1 FPS Global transitions构建。"""

    mean: np.ndarray
    whitening: np.ndarray
    calibration_likelihood: np.ndarray
    calibration_ids: tuple[str, ...]
    coarse_contract_sha256: str
    base_fps: int = 8
    coarse_fps: int = 1
    schema_version: str = "caes_selector_reference_v1"

    def validate(self) -> None:
        if self.schema_version != "caes_selector_reference_v1":
            raise ValueError("selector reference schema不支持")
        if self.base_fps != 8 or self.coarse_fps != 1:
            raise ValueError("selector reference FPS与CAES合同不一致")
        if len(self.coarse_contract_sha256) != 64:
            raise ValueError("selector reference缺少coarse contract SHA")
        if self.mean.ndim != 1 or self.whitening.ndim != 2:
            raise ValueError("selector reference mean/W shape无效")
        if self.whitening.shape[0] != len(self.mean):
            raise ValueError("selector reference mean/W维度不一致")
        if not self.calibration_ids:
            raise ValueError("selector reference缺少calibration IDs")
        stable_sorted(self.calibration_likelihood)

    def digest(self) -> str:
        self.validate()
        identity = {
            "schema_version": self.schema_version,
            "base_fps": self.base_fps,
            "coarse_fps": self.coarse_fps,
            "coarse_contract_sha256": self.coarse_contract_sha256,
            "calibration_ids": list(self.calibration_ids),
            "mean_sha256": hashlib.sha256(np.asarray(self.mean).tobytes()).hexdigest(),
            "whitening_sha256": hashlib.sha256(np.asarray(self.whitening).tobytes()).hexdigest(),
            "cdf_sha256": hashlib.sha256(
                np.asarray(self.calibration_likelihood).tobytes()
            ).hexdigest(),
        }
        return hashlib.sha256(
            json.dumps(identity, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest()


def normalized_transitions(global_features: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """返回normalized 1 FPS transitions与exact-zero mask。"""

    values = torch.as_tensor(global_features, dtype=torch.float32).unsqueeze(0)
    if values.shape[1] < 2:
        return np.empty((0, values.shape[-1]), dtype=np.float32), np.empty(0, dtype=bool)
    normalized, zero = l2_normalized_first_order(values)
    return normalized.squeeze(0).numpy(), zero.squeeze(0).numpy()


def fit_selector_reference(
    sequences: Sequence[np.ndarray],
    calibration_ids: Sequence[str],
    *,
    coarse_contract_sha256: str,
    device: str = "cpu",
) -> SelectorReference:
    """只用显式calibration real序列拟合whitening和transition CDF。"""

    if len(sequences) != len(calibration_ids) or not sequences:
        raise ValueError("selector reference序列与calibration IDs不一致")
    transitions = [normalized_transitions(item) for item in sequences]
    usable = [features[~zero] for features, zero in transitions if np.any(~zero)]
    if not usable:
        raise ValueError("selector reference没有可用1 FPS transition")
    matrix = np.concatenate(usable, axis=0).astype(np.float32, copy=False)
    transform = WhiteningTransform(matrix, device=device, covariance_estimator="empirical")
    mean = transform.mean_.detach().cpu().numpy().astype(np.float64)
    whitening = transform.whitening_matrix_.detach().cpu().numpy().astype(np.float64)
    likelihoods = []
    for features, zero in transitions:
        if not len(features):
            continue
        values = log_likelihood(apply_whitening(features, mean, whitening))
        # 与STALL Global T1一致：exact-zero不提供方向信息，映射为最高真实似然。
        values = np.asarray(values, dtype=np.float64)
        values[zero] = np.inf
        likelihoods.extend(values[np.isfinite(values)].tolist())
    reference = SelectorReference(
        mean=mean,
        whitening=whitening,
        calibration_likelihood=stable_sorted(np.asarray(likelihoods, dtype=np.float64)),
        calibration_ids=tuple(str(item) for item in calibration_ids),
        coarse_contract_sha256=coarse_contract_sha256,
    )
    reference.validate()
    return reference


def score_coarse_sequence(
    global_features: np.ndarray,
    downsample_positions: Sequence[int],
    reference: SelectorReference,
) -> dict[str, np.ndarray]:
    """输出transition midpoint、feature change、likelihood、percentile和anomaly。"""

    reference.validate()
    values = np.asarray(global_features, dtype=np.float32)
    positions = np.asarray(downsample_positions, dtype=np.int64)
    if values.ndim != 2 or len(values) != len(positions):
        raise ValueError("coarse features与positions shape不一致")
    if len(values) < 2:
        empty = np.empty(0, dtype=np.float64)
        return {
            "midpoint_position": empty,
            "feature_change": empty,
            "likelihood": empty,
            "percentile": empty,
            "anomaly": empty,
        }
    raw_difference = values[1:] - values[:-1]
    feature_change = np.linalg.norm(raw_difference, axis=-1).astype(np.float64)
    normalized, zero = normalized_transitions(values)
    likelihood = log_likelihood(
        apply_whitening(normalized, reference.mean, reference.whitening)
    ).astype(np.float64)
    percentile = np.ones(len(likelihood), dtype=np.float64)
    finite = ~zero
    percentile[finite] = empirical_cdf_right_inclusive(
        likelihood[finite], reference.calibration_likelihood
    )
    return {
        "midpoint_position": 0.5 * (positions[:-1] + positions[1:]),
        "feature_change": feature_change,
        "likelihood": likelihood,
        "percentile": percentile,
        "anomaly": 1.0 - percentile,
    }


def attach_candidate_scores(
    candidates: Sequence[CandidateWindow],
    coarse_scores: dict[str, np.ndarray],
    *,
    window_frames: int = 16,
) -> list[CandidateWindow]:
    """按transition midpoint落入窗口的规则附加mean/max coarse信号。"""

    midpoints = np.asarray(coarse_scores["midpoint_position"], dtype=np.float64)
    change = np.asarray(coarse_scores["feature_change"], dtype=np.float64)
    anomaly = np.asarray(coarse_scores["anomaly"], dtype=np.float64)
    if not (len(midpoints) == len(change) == len(anomaly)):
        raise ValueError("coarse score字段长度不一致")
    output = []
    for candidate in candidates:
        mask = (
            (midpoints >= candidate.start_position)
            & (midpoints < candidate.start_position + window_frames)
        )
        if not np.any(mask):
            # dense-eligible视频至少应有一个1 FPS transition；若浮点边界导致空窗，
            # 使用最近transition并保留确定性，而不是读取标签回退。
            nearest = int(
                np.argmin(np.abs(midpoints - (candidate.start_position + window_frames / 2)))
            ) if len(midpoints) else None
            mask = np.zeros(len(midpoints), dtype=bool)
            if nearest is not None:
                mask[nearest] = True
        scores = {
            "feature_change_mean": float(change[mask].mean()) if np.any(mask) else 0.0,
            "feature_change_max": float(change[mask].max()) if np.any(mask) else 0.0,
            "real_anomaly_mean": float(anomaly[mask].mean()) if np.any(mask) else 0.0,
            "real_anomaly_max": float(anomaly[mask].max()) if np.any(mask) else 0.0,
        }
        output.append(replace(candidate, scores=scores))
    return output


def save_selector_reference(path: Path, reference: SelectorReference) -> str:
    reference.validate()
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp.npz")
    np.savez_compressed(
        temporary,
        mean=reference.mean,
        whitening=reference.whitening,
        calibration_likelihood=reference.calibration_likelihood,
        calibration_ids=np.asarray(reference.calibration_ids),
        coarse_contract_sha256=np.asarray([reference.coarse_contract_sha256]),
        base_fps=np.asarray([reference.base_fps]),
        coarse_fps=np.asarray([reference.coarse_fps]),
    )
    temporary.replace(path)
    return hashlib.sha256(path.read_bytes()).hexdigest()


__all__ = [
    "SelectorReference",
    "attach_candidate_scores",
    "fit_selector_reference",
    "normalized_transitions",
    "save_selector_reference",
    "score_coarse_sequence",
]
