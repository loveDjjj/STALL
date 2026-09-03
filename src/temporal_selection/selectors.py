"""无标签、固定预算的FS0-FS5窗口选择纯函数。"""

from __future__ import annotations

import hashlib
from typing import Sequence

import numpy as np

from .candidates import generate_candidate_windows, uniform_candidate_windows
from .models import CandidateWindow, SelectedWindow, WindowManifest


SELECTORS = {
    "uniform", "random", "feature_change", "real_anomaly",
    "real_anomaly_nms", "stratified_real_anomaly",
}


def _video_seed(seed: int, video_id: str) -> int:
    digest = hashlib.sha256(f"{seed}\0{video_id}".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "little") % (2**32)


def temporal_iou(left: CandidateWindow, right: CandidateWindow) -> float:
    intersection = max(
        0.0,
        min(left.end_seconds, right.end_seconds)
        - max(left.start_seconds, right.start_seconds),
    )
    union = max(left.end_seconds, right.end_seconds) - min(
        left.start_seconds, right.start_seconds
    )
    return intersection / union if union > 0 else 0.0


def _ranked(candidates: Sequence[CandidateWindow], score_name: str) -> list[CandidateWindow]:
    if any(score_name not in item.scores for item in candidates):
        raise ValueError(f"候选窗口缺少selector score：{score_name}")
    return sorted(
        candidates,
        key=lambda item: (-float(item.scores[score_name]), item.start_position),
    )


def _selected(
    ranked: Sequence[CandidateWindow], score_name: str, reason: str
) -> tuple[SelectedWindow, ...]:
    decisions = [
        SelectedWindow(
            candidate_id=item.candidate_id,
            rank=rank,
            score=float(item.scores.get(score_name, 0.0)),
            reason=reason,
        )
        for rank, item in enumerate(ranked, start=1)
    ]
    return tuple(sorted(decisions, key=lambda item: item.candidate_id))


def select_windows(
    *,
    video_id: str,
    duration_seconds: float,
    downsample_indices: list[int],
    selector_name: str,
    requested_k: int = 3,
    base_fps: float = 8.0,
    window_seconds: float = 2.0,
    stride_seconds: float = 0.5,
    seed: int = 17,
    nms_iou_threshold: float = 0.5,
    scored_candidates: Sequence[CandidateWindow] | None = None,
    selector_reference_sha256: str | None = None,
) -> WindowManifest:
    """运行一个selector；接口不接受subset、source_model或真假标签。"""

    if selector_name not in SELECTORS:
        raise ValueError(f"不支持的temporal selector：{selector_name}")
    if requested_k < 1:
        raise ValueError("requested_k必须为正数")
    if selector_name == "uniform":
        candidates = uniform_candidate_windows(
            downsample_indices,
            requested_k=requested_k,
            base_fps=base_fps,
            window_seconds=window_seconds,
        )
        ranked = candidates
        score_name = "uniform"
        reason = "exact_uniform_baseline"
    else:
        candidates = list(scored_candidates) if scored_candidates is not None else generate_candidate_windows(
            downsample_indices,
            base_fps=base_fps,
            window_seconds=window_seconds,
            stride_seconds=stride_seconds,
        )
        if selector_name == "random":
            rng = np.random.default_rng(_video_seed(seed, video_id))
            chosen = rng.choice(
                len(candidates), size=min(requested_k, len(candidates)), replace=False
            ) if candidates else np.array([], dtype=int)
            ranked = [candidates[int(index)] for index in chosen]
            score_name = "random"
            reason = "deterministic_random"
        else:
            score_name = (
                "feature_change_mean"
                if selector_name == "feature_change"
                else "real_anomaly_mean"
            )
            ranked_all = _ranked(candidates, score_name)
            if selector_name in {"feature_change", "real_anomaly"}:
                ranked = ranked_all[:requested_k]
                reason = f"top_{score_name}"
            elif selector_name == "real_anomaly_nms":
                ranked = []
                for candidate in ranked_all:
                    if all(
                        temporal_iou(candidate, selected) <= nms_iou_threshold
                        for selected in ranked
                    ):
                        ranked.append(candidate)
                    if len(ranked) == requested_k:
                        break
                reason = "top_real_anomaly_after_temporal_nms"
            else:
                ranked = []
                if candidates:
                    centers = np.asarray([item.center_seconds for item in candidates])
                    edges = np.linspace(centers.min(), centers.max(), requested_k + 1)
                    used = set()
                    for stratum in range(requested_k):
                        eligible = [
                            item for item in ranked_all
                            if item.candidate_id not in used
                            and (
                                edges[stratum] <= item.center_seconds < edges[stratum + 1]
                                or (stratum == requested_k - 1 and item.center_seconds == edges[-1])
                            )
                        ]
                        if eligible:
                            ranked.append(eligible[0])
                            used.add(eligible[0].candidate_id)
                    for item in ranked_all:
                        if len(ranked) == requested_k:
                            break
                        if item.candidate_id not in used:
                            ranked.append(item)
                            used.add(item.candidate_id)
                reason = "best_real_anomaly_per_temporal_stratum"
    ranked = list(ranked[:requested_k])
    selected = tuple(
        sorted(
            [
                SelectedWindow(
                    candidate_id=item.candidate_id,
                    rank=rank,
                    score=float(item.scores.get(score_name, 0.0)),
                    reason=reason,
                )
                for rank, item in enumerate(ranked, start=1)
            ],
            key=lambda item: next(
                candidate.start_position
                for candidate in candidates
                if candidate.candidate_id == item.candidate_id
            ),
        )
    )
    # 时间排序不能改变selection rank；rank表示原始score/stratum选择顺序。
    result = WindowManifest(
        video_id=video_id,
        duration_seconds=float(duration_seconds),
        base_fps=float(base_fps),
        candidate_stride_seconds=float(stride_seconds),
        window_seconds=float(window_seconds),
        requested_k=requested_k,
        candidates=tuple(candidates),
        selected=selected,
        selector={
            "name": selector_name,
            "seed": seed,
            "nms_iou_threshold": nms_iou_threshold,
            "reference_sha256": selector_reference_sha256,
        },
    )
    result.validate()
    return result


__all__ = ["SELECTORS", "select_windows", "temporal_iou"]
