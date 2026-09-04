"""CAES selector 5-fold cross-fitting 的确定性分折与泄漏检查。"""

from __future__ import annotations

import hashlib


def balanced_crossfit_assignments(
    video_ids: list[str], *, folds: int = 5, seed: int = 17
) -> dict[str, int]:
    """按稳定哈希排序后轮转分折，使每折样本数最多相差1。"""

    if folds < 2:
        raise ValueError("crossfit folds必须至少为2")
    if len(video_ids) < folds:
        raise ValueError("crossfit视频数必须不少于fold数")
    if len(video_ids) != len(set(video_ids)):
        raise ValueError("crossfit video_id不能重复")

    def key(video_id: str) -> tuple[bytes, str]:
        digest = hashlib.sha256(
            f"{seed}\0{video_id}".encode("utf-8")
        ).digest()
        return digest, video_id

    ordered = sorted(video_ids, key=key)
    assignments = {
        video_id: position % folds for position, video_id in enumerate(ordered)
    }
    counts = [sum(fold == index for fold in assignments.values()) for index in range(folds)]
    if max(counts) - min(counts) > 1 or min(counts) < 1:
        raise AssertionError("crossfit分折未保持平衡")
    return assignments


def validate_crossfit_partition(
    assignments: dict[str, int], *, folds: int
) -> None:
    """确认每条视频恰好held out一次且训练/held-out集合互斥。"""

    if not assignments:
        raise ValueError("crossfit assignments为空")
    if set(assignments.values()) != set(range(folds)):
        raise ValueError("crossfit assignments未覆盖全部fold")
    all_ids = set(assignments)
    heldout_union = set()
    for fold in range(folds):
        heldout = {video_id for video_id, value in assignments.items() if value == fold}
        training = all_ids - heldout
        if not heldout or not training or heldout & training:
            raise ValueError(f"crossfit fold={fold}发生空集或泄漏")
        heldout_union.update(heldout)
    if heldout_union != all_ids:
        raise ValueError("crossfit并非每条视频恰好held out一次")


__all__ = ["balanced_crossfit_assignments", "validate_crossfit_partition"]
