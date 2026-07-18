from __future__ import annotations

import numpy as np


def l2_normalize_last_dim(arr: np.ndarray) -> np.ndarray:
    arr = arr.astype(np.float32)
    norm = np.linalg.norm(arr, axis=-1, keepdims=True)
    norm = np.where(norm == 0, 1.0, norm)
    return arr / norm


def patch_id_to_rc(patch_id: int, grid_size: tuple[int, int]) -> tuple[int, int]:
    gh, gw = grid_size
    return patch_id // gw, patch_id % gw


def rc_to_patch_id(row: int, col: int, grid_size: tuple[int, int]) -> int:
    gh, gw = grid_size
    return row * gw + col


def local_window_indices(
    patch_id: int,
    grid_size: tuple[int, int],
    radius: int,
) -> np.ndarray:
    gh, gw = grid_size
    row, col = patch_id_to_rc(patch_id, grid_size)
    rows = range(max(0, row - radius), min(gh, row + radius + 1))
    cols = range(max(0, col - radius), min(gw, col + radius + 1))
    ids = [rc_to_patch_id(r, c, grid_size) for r in rows for c in cols]
    return np.array(ids, dtype=np.int64)


def normalized_spatial_distance(
    patch_id_a: int,
    patch_ids_b: np.ndarray,
    grid_size: tuple[int, int],
    radius: int,
) -> np.ndarray:
    row_a, col_a = patch_id_to_rc(patch_id_a, grid_size)
    coords_b = np.array([patch_id_to_rc(int(pid), grid_size) for pid in patch_ids_b], dtype=np.float32)
    dr = coords_b[:, 0] - float(row_a)
    dc = coords_b[:, 1] - float(col_a)
    dist = np.sqrt(dr * dr + dc * dc)
    max_dist = max(1.0, np.sqrt(2.0) * max(radius, 1))
    return dist / max_dist


def cosine_scores(query: np.ndarray, candidates: np.ndarray) -> np.ndarray:
    q = query.astype(np.float32)
    c = candidates.astype(np.float32)
    q_norm = np.linalg.norm(q)
    c_norm = np.linalg.norm(c, axis=1)
    denom = np.maximum(q_norm * c_norm, 1e-8)
    return (c @ q) / denom


def softmax_np(x: np.ndarray) -> np.ndarray:
    z = x - np.max(x)
    exp = np.exp(z)
    denom = np.maximum(exp.sum(), 1e-8)
    return exp / denom


def pool_patch_regions(
    patch_seq: np.ndarray,
    grid_size: tuple[int, int],
    region_size: int = 1,
) -> tuple[np.ndarray, tuple[int, int]]:
    """对不重叠空间 patch 区域求平均。

    输入形状为 [T, P, D]。当 region_size=2 且网格为 14x14 时，输出为
    [T, 7*7, D]。如果网格不能被 region_size 整除，会裁掉底部/右侧边界，
    保留最大可整除网格。
    """
    if region_size <= 1:
        return patch_seq.astype(np.float32), grid_size
    if patch_seq.ndim != 3:
        raise ValueError(f"期望 patch_seq [T,P,D]，实际为 {patch_seq.shape}")

    gh, gw = grid_size
    expected_patches = gh * gw
    if patch_seq.shape[1] != expected_patches:
        raise ValueError(
            f"Patch 数量 {patch_seq.shape[1]} 与网格 {grid_size} 不匹配"
        )

    pooled_h = gh // region_size
    pooled_w = gw // region_size
    if pooled_h < 1 or pooled_w < 1:
        raise ValueError(f"region_size={region_size} 对网格 {grid_size} 过大")

    crop_h = pooled_h * region_size
    crop_w = pooled_w * region_size
    arr = patch_seq.reshape(patch_seq.shape[0], gh, gw, patch_seq.shape[-1])
    arr = arr[:, :crop_h, :crop_w, :]
    arr = arr.reshape(
        patch_seq.shape[0],
        pooled_h,
        region_size,
        pooled_w,
        region_size,
        patch_seq.shape[-1],
    )
    pooled = arr.mean(axis=(2, 4)).reshape(patch_seq.shape[0], pooled_h * pooled_w, patch_seq.shape[-1])
    return pooled.astype(np.float32), (pooled_h, pooled_w)


def match_patches_local_window(
    patch_t: np.ndarray,        # [P, D]
    patch_t1: np.ndarray,       # [P, D]
    grid_size: tuple[int, int],
    radius: int = 2,
    top_m: int = 4,
    temperature: float = 0.07,
    lambda_dist: float = 0.01,
    mode: str = "soft",         # "hard" 或 "soft"
) -> np.ndarray:
    """返回形状为 [P, D] 的运动对齐差分特征。"""
    if patch_t.shape != patch_t1.shape:
        raise ValueError(f"patch_t 和 patch_t1 形状必须一致，实际为 {patch_t.shape} vs {patch_t1.shape}")

    num_patches, dim = patch_t.shape
    matched = np.empty((num_patches, dim), dtype=np.float32)

    for patch_id in range(num_patches):
        candidate_ids = local_window_indices(patch_id, grid_size, radius)
        candidate_vecs = patch_t1[candidate_ids]  # [C, D]
        sim = cosine_scores(patch_t[patch_id], candidate_vecs)
        dist_penalty = normalized_spatial_distance(patch_id, candidate_ids, grid_size, radius)
        scores = sim - lambda_dist * dist_penalty

        if mode == "hard":
            best_idx = int(np.argmax(scores))
            matched[patch_id] = candidate_vecs[best_idx]
            continue

        if mode != "soft":
            raise ValueError(f"不支持的 matching mode: {mode}")

        m = min(top_m, len(candidate_ids))
        top_idx = np.argpartition(scores, -m)[-m:]
        top_scores = scores[top_idx] / max(temperature, 1e-8)
        weights = softmax_np(top_scores).astype(np.float32)
        top_vecs = candidate_vecs[top_idx]
        matched[patch_id] = (weights[:, None] * top_vecs).sum(axis=0)

    return matched - patch_t.astype(np.float32)


def matching_diagnostics_for_pair(
    patch_t: np.ndarray,
    patch_t1: np.ndarray,
    grid_size: tuple[int, int],
    radius: int = 2,
    top_m: int = 4,
    temperature: float = 0.07,
    lambda_dist: float = 0.01,
    mode: str = "soft",
) -> dict[str, np.ndarray]:
    """返回单个帧对的逐 patch matching 诊断量。"""
    if patch_t.shape != patch_t1.shape:
        raise ValueError(f"patch_t 和 patch_t1 形状必须一致，实际为 {patch_t.shape} vs {patch_t1.shape}")

    num_patches = patch_t.shape[0]
    matched_ids = np.empty(num_patches, dtype=np.int64)
    top1_scores = np.empty(num_patches, dtype=np.float32)
    top2_scores = np.empty(num_patches, dtype=np.float32)
    top1_cosines = np.empty(num_patches, dtype=np.float32)
    displacements = np.empty(num_patches, dtype=np.float32)
    weights_sum = np.full(num_patches, np.nan, dtype=np.float32)
    weights_max = np.full(num_patches, np.nan, dtype=np.float32)
    entropy = np.full(num_patches, np.nan, dtype=np.float32)
    cosine_min = []
    cosine_mean = []
    cosine_max = []
    dist_min = []
    dist_mean = []
    dist_max = []
    penalty_min = []
    penalty_mean = []
    penalty_max = []
    score_min = []
    score_mean = []
    score_max = []

    for patch_id in range(num_patches):
        candidate_ids = local_window_indices(patch_id, grid_size, radius)
        candidate_vecs = patch_t1[candidate_ids]
        sim = cosine_scores(patch_t[patch_id], candidate_vecs)
        dist = normalized_spatial_distance(patch_id, candidate_ids, grid_size, radius)
        penalty = lambda_dist * dist
        scores = sim - penalty

        order = np.argsort(scores)[::-1]
        best_local = int(order[0])
        second_local = int(order[1]) if len(order) > 1 else best_local
        matched_id = int(candidate_ids[best_local])

        if mode == "soft":
            m = min(top_m, len(candidate_ids))
            top_idx = order[:m]
            w = softmax_np(scores[top_idx] / max(temperature, 1e-8)).astype(np.float32)
            weights_sum[patch_id] = float(w.sum())
            weights_max[patch_id] = float(w.max())
            entropy[patch_id] = float(-(w * np.log(np.maximum(w, 1e-8))).sum())

        row, col = patch_id_to_rc(patch_id, grid_size)
        mrow, mcol = patch_id_to_rc(matched_id, grid_size)
        disp = np.sqrt(float((mrow - row) ** 2 + (mcol - col) ** 2))

        matched_ids[patch_id] = matched_id
        top1_scores[patch_id] = float(scores[best_local])
        top2_scores[patch_id] = float(scores[second_local])
        top1_cosines[patch_id] = float(sim[best_local])
        displacements[patch_id] = disp

        cosine_min.append(float(sim.min()))
        cosine_mean.append(float(sim.mean()))
        cosine_max.append(float(sim.max()))
        dist_min.append(float(dist.min()))
        dist_mean.append(float(dist.mean()))
        dist_max.append(float(dist.max()))
        penalty_min.append(float(penalty.min()))
        penalty_mean.append(float(penalty.mean()))
        penalty_max.append(float(penalty.max()))
        score_min.append(float(scores.min()))
        score_mean.append(float(scores.mean()))
        score_max.append(float(scores.max()))

    return {
        "matched_ids": matched_ids,
        "top1_scores": top1_scores,
        "top2_scores": top2_scores,
        "top1_cosines": top1_cosines,
        "displacements": displacements,
        "weights_sum": weights_sum,
        "weights_max": weights_max,
        "entropy": entropy,
        "cosine_min": np.array(cosine_min, dtype=np.float32),
        "cosine_mean": np.array(cosine_mean, dtype=np.float32),
        "cosine_max": np.array(cosine_max, dtype=np.float32),
        "dist_min": np.array(dist_min, dtype=np.float32),
        "dist_mean": np.array(dist_mean, dtype=np.float32),
        "dist_max": np.array(dist_max, dtype=np.float32),
        "penalty_min": np.array(penalty_min, dtype=np.float32),
        "penalty_mean": np.array(penalty_mean, dtype=np.float32),
        "penalty_max": np.array(penalty_max, dtype=np.float32),
        "score_min": np.array(score_min, dtype=np.float32),
        "score_mean": np.array(score_mean, dtype=np.float32),
        "score_max": np.array(score_max, dtype=np.float32),
    }


def patch_temporal_delta(
    patch_seq: np.ndarray,      # [T, P, D]
    grid_size: tuple[int, int],
    mode: str = "same_grid",
    radius: int = 2,
    top_m: int = 4,
    temperature: float = 0.07,
    lambda_dist: float = 0.01,
    region_size: int = 1,
) -> np.ndarray:
    if patch_seq.ndim != 3:
        raise ValueError(f"期望 patch_seq [T, P, D]，实际为 {patch_seq.shape}")

    if region_size > 1:
        if mode not in {
            "same_grid",
            "same_grid_lag1",
            "same_grid_multilag",
            "same_grid_second_order",
            "same_grid_multilag_second_order",
        }:
            raise ValueError("patch region pooling 只支持 same-grid 模式")
        patch_seq, grid_size = pool_patch_regions(patch_seq, grid_size, region_size)

    if mode in {"same_grid", "same_grid_lag1"}:
        return l2_normalize_last_dim(patch_seq[1:] - patch_seq[:-1])

    if mode == "same_grid_multilag":
        deltas = []
        for lag in (1, 2, 4):
            if len(patch_seq) > lag:
                deltas.append(l2_normalize_last_dim(patch_seq[lag:] - patch_seq[:-lag]))
        if not deltas:
            raise ValueError(f"视频帧数过少，无法使用 {mode}: T={len(patch_seq)}")
        return np.concatenate(deltas, axis=0).astype(np.float32)

    if mode == "same_grid_second_order":
        if len(patch_seq) < 3:
            raise ValueError(f"视频帧数过少，无法使用 {mode}: T={len(patch_seq)}")
        accel = patch_seq[2:] - 2.0 * patch_seq[1:-1] + patch_seq[:-2]
        return l2_normalize_last_dim(accel)

    if mode == "same_grid_multilag_second_order":
        deltas = []
        for lag in (1, 2, 4):
            if len(patch_seq) > lag:
                deltas.append(l2_normalize_last_dim(patch_seq[lag:] - patch_seq[:-lag]))
        if len(patch_seq) >= 3:
            accel = patch_seq[2:] - 2.0 * patch_seq[1:-1] + patch_seq[:-2]
            deltas.append(l2_normalize_last_dim(accel))
        if not deltas:
            raise ValueError(f"视频帧数过少，无法使用 {mode}: T={len(patch_seq)}")
        return np.concatenate(deltas, axis=0).astype(np.float32)

    matching_mode = {
        "motion_hard": "hard",
        "motion_soft": "soft",
    }.get(mode)
    if matching_mode is None:
        raise ValueError(f"不支持的 patch temporal mode: {mode}")

    deltas = []
    for t in range(len(patch_seq) - 1):
        delta_t = match_patches_local_window(
            patch_t=patch_seq[t],
            patch_t1=patch_seq[t + 1],
            grid_size=grid_size,
            radius=radius,
            top_m=top_m,
            temperature=temperature,
            lambda_dist=lambda_dist,
            mode=matching_mode,
        )
        deltas.append(delta_t)

    return l2_normalize_last_dim(np.stack(deltas, axis=0))  # [T-1, P, D]
