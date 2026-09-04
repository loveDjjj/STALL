"""用selector匹配的真实位置分布将likelihood field转换为Tail nonconformity。"""

from __future__ import annotations

import numpy as np

from math_utils import stable_sorted


def fit_position_reference(fields: list[np.ndarray]) -> np.ndarray:
    """拼接calibration real位置似然，并返回稳定排序的float64 reference。"""

    if not fields:
        raise ValueError("位置级真实reference不能为空")
    flattened = [np.asarray(field, dtype=np.float64).reshape(-1) for field in fields]
    if any(not np.isfinite(field).all() for field in flattened):
        raise ValueError("位置级真实reference包含非有限likelihood")
    return stable_sorted(np.concatenate(flattened))


def conformal_tail_authenticity(
    likelihood_field: np.ndarray,
    reference_sorted: np.ndarray,
    ratio: float,
) -> float:
    """返回负的top-anomaly均值，使输出越高仍表示越接近真实视频。"""

    if not 0.0 < ratio <= 1.0:
        raise ValueError("Tail ratio必须位于(0,1]")
    likelihood = np.asarray(likelihood_field, dtype=np.float64).reshape(-1)
    reference = np.asarray(reference_sorted, dtype=np.float64)
    # reference只能由fit_position_reference构造；排序与有限性在那里一次验证。
    # 此函数按窗口调用，不能在每次二分查找前线性扫描百万级reference。
    if reference.ndim != 1 or not len(reference):
        raise ValueError("位置reference必须是非空一维已排序数组")
    if not len(likelihood) or not np.isfinite(likelihood).all():
        raise ValueError("likelihood field必须非空且有限")
    percentile = np.searchsorted(reference, likelihood, side="right") / float(
        len(reference)
    )
    anomaly = 1.0 - percentile
    count = max(1, int(np.ceil(len(anomaly) * ratio)))
    top_anomaly = np.partition(anomaly, len(anomaly) - count)[-count:]
    return -float(top_anomaly.mean(dtype=np.float64))


def conformal_tail_authenticity_batch(
    likelihood_fields: np.ndarray,
    reference_sorted: np.ndarray,
    ratio: float,
) -> np.ndarray:
    """批量计算多个field的Tail真实性，定义与单窗口函数完全一致。"""

    if not 0.0 < ratio <= 1.0:
        raise ValueError("Tail ratio必须位于(0,1]")
    fields = np.asarray(likelihood_fields, dtype=np.float64)
    reference = np.asarray(reference_sorted, dtype=np.float64)
    if fields.ndim < 2 or not len(fields) or not np.isfinite(fields).all():
        raise ValueError("likelihood fields必须是非空、有限的batch")
    if reference.ndim != 1 or not len(reference):
        raise ValueError("位置reference必须是非空一维已排序数组")
    flattened = fields.reshape(len(fields), -1)
    percentile = np.searchsorted(reference, flattened, side="right") / float(
        len(reference)
    )
    anomaly = 1.0 - percentile
    count = max(1, int(np.ceil(anomaly.shape[1] * ratio)))
    split = anomaly.shape[1] - count
    top = np.partition(anomaly, split, axis=1)[:, split:]
    return -top.mean(axis=1, dtype=np.float64)


def conformal_tail_authenticity_multi(
    likelihood_fields: np.ndarray,
    reference_sorted: np.ndarray,
    ratios: dict[str, float],
) -> dict[str, np.ndarray]:
    """一次CDF与多k partition同时计算多个预注册Tail比例。"""

    if not ratios or any(not 0.0 < ratio <= 1.0 for ratio in ratios.values()):
        raise ValueError("Tail ratios必须是非空且全部位于(0,1]")
    fields = np.asarray(likelihood_fields, dtype=np.float64)
    reference = np.asarray(reference_sorted, dtype=np.float64)
    if fields.ndim < 2 or not len(fields) or not np.isfinite(fields).all():
        raise ValueError("likelihood fields必须是非空、有限的batch")
    if reference.ndim != 1 or not len(reference):
        raise ValueError("位置reference必须是非空一维已排序数组")
    flattened = fields.reshape(len(fields), -1)
    anomaly = 1.0 - np.searchsorted(
        reference, flattened, side="right"
    ) / float(len(reference))
    counts = {
        name: max(1, int(np.ceil(anomaly.shape[1] * ratio)))
        for name, ratio in ratios.items()
    }
    splits = sorted({anomaly.shape[1] - count for count in counts.values()})
    partitioned = np.partition(anomaly, splits, axis=1)
    return {
        name: -partitioned[:, anomaly.shape[1] - count :].mean(
            axis=1, dtype=np.float64
        )
        for name, count in counts.items()
    }


__all__ = [
    "conformal_tail_authenticity",
    "conformal_tail_authenticity_batch",
    "conformal_tail_authenticity_multi",
    "fit_position_reference",
]
