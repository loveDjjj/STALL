"""跨真实域校准矩阵的纯统计计算。"""

from __future__ import annotations

import numpy as np
import pandas as pd

from evaluation.metrics import binary_metrics
from math_utils import stable_sorted


OPERATING_FPRS = (0.001, 0.01, 0.05)


def _cdf(target: pd.Series, reference: pd.Series) -> np.ndarray:
    values = target.to_numpy(dtype=np.float64)
    ref = stable_sorted(reference.to_numpy(dtype=np.float64))
    return np.searchsorted(ref, values, side="right") / float(len(ref))


def _calibrate_windows_and_videos(
    calibration: pd.DataFrame, evaluation: pd.DataFrame, config: dict
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """复现正式 Global/Local D2 的窗口和 effective-K 视频校准。"""

    calibration = calibration.copy()
    evaluation = evaluation.copy()
    reference = calibration[calibration["subset"].eq("real")]
    if reference.empty:
        raise ValueError("跨域calibration缺少真实窗口")
    for frame in (calibration, evaluation):
        frame["patch_temporal"] = _cdf(
            frame["patch_temporal_raw"], reference["patch_temporal_raw"]
        )
        frame["global_score_window"] = (
            float(config["method"]["global"]["spatial_weight"])
            * frame["global_spatial"].to_numpy(dtype=np.float64)
            + float(config["method"]["global"]["temporal_weight"])
            * frame["global_t1"].to_numpy(dtype=np.float64)
        )
        frame["local_score_window"] = frame["patch_temporal"]

    keys = ["video_id", "dataset", "subset", "source_model", "video_path"]

    def aggregate(frame: pd.DataFrame) -> pd.DataFrame:
        return frame.groupby(keys, as_index=False).agg(
            effective_k=("window_id", "size"),
            global_raw=("global_score_window", "mean"),
            local_raw=("local_score_window", "mean"),
        )

    def reference_for_k(target_k: int) -> pd.DataFrame:
        rows = []
        for values, windows in calibration.groupby(keys, sort=False):
            ordered = windows.sort_values("window_id").reset_index(drop=True)
            if len(ordered) < target_k:
                continue
            positions = np.unique(
                np.rint(np.linspace(0, len(ordered) - 1, target_k)).astype(int)
            )
            if len(positions) != target_k:
                continue
            selected = ordered.iloc[positions]
            row = dict(zip(keys, values))
            row.update({
                "effective_k": target_k,
                "global_raw": float(selected["global_score_window"].mean()),
                "local_raw": float(selected["local_score_window"].mean()),
            })
            rows.append(row)
        return pd.DataFrame(rows)

    evaluation_videos = aggregate(evaluation)
    outputs = []
    for effective_k, target in evaluation_videos.groupby("effective_k", sort=False):
        reference_video = reference_for_k(int(effective_k))
        if len(reference_video) < 2:
            raise ValueError(f"跨域bank在effective-K={effective_k}下不足两个real")
        output = target.copy()
        output["global_score"] = _cdf(
            output["global_raw"], reference_video["global_raw"]
        )
        output["local_score"] = _cdf(
            output["local_raw"], reference_video["local_raw"]
        )
        global_weight = float(config["method"]["fusion"]["global_weight"])
        local_weight = float(config["method"]["fusion"]["local_weight"])
        total = global_weight + local_weight
        output["final_score"] = (
            global_weight * output["global_score"]
            + local_weight * output["local_score"]
        ) / total
        outputs.append(output)
    return calibration, pd.concat(outputs, ignore_index=True)


def real_only_threshold(scores: np.ndarray, target_fpr: float) -> float:
    """返回只由真实分数确定的保守下尾阈值。

    分数低于阈值时判为生成。使用严格小于号，因此 calibration real 的经验
    FPR 不超过目标值；当 real 数量不足时，0.1% 等操作点会自然退化为 0。
    """

    values = np.sort(np.asarray(scores, dtype=np.float64), kind="mergesort")
    if values.ndim != 1 or not len(values) or not np.isfinite(values).all():
        raise ValueError("真实阈值需要非空的一维有限分数")
    if not 0.0 < target_fpr < 1.0:
        raise ValueError("target_fpr必须位于(0,1)")
    index = min(int(np.floor(target_fpr * len(values))), len(values) - 1)
    return float(values[index])


def evaluate_cross_domain_cell(
    calibration_windows: pd.DataFrame,
    evaluation_windows: pd.DataFrame,
    config: dict,
    *,
    calibration_bank: str,
    evaluation_domain: str,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """对一个 calibration bank -> evaluation domain 单元完成校准与评价。"""

    calibration = calibration_windows.copy()
    evaluation = evaluation_windows.copy()
    if calibration.empty or evaluation.empty:
        raise ValueError("跨域单元的calibration/evaluation窗口不能为空")
    if not calibration["subset"].eq("real").all():
        raise ValueError("跨域calibration bank只能包含真实视频")

    # pipeline按dataset分组构造effective-K参考；这里把bank行映射为目标域标签，
    # 原始来源保存在calibration_bank字段，避免因标签不同错误得到空参考。
    calibration["calibration_bank"] = calibration_bank
    calibration["dataset"] = evaluation_domain
    evaluation["dataset"] = evaluation_domain
    _, videos = _calibrate_windows_and_videos(calibration, evaluation, config)
    _, calibration_videos = _calibrate_windows_and_videos(
        calibration, calibration, config
    )
    if not np.isfinite(videos["final_score"].to_numpy(dtype=np.float64)).all():
        raise ValueError("跨域视频分数包含非有限值")

    metric = binary_metrics(videos, "final_score")
    summary = pd.DataFrame([{
        "calibration_bank": calibration_bank,
        "evaluation_domain": evaluation_domain,
        "auc": metric["auc"],
        "ap_real": metric["real_positive_ap"],
        "n_calibration_real": int(calibration_videos["video_id"].nunique()),
        "n_evaluation_real": int(videos["subset"].eq("real").sum()),
        "n_evaluation_fake": int(videos["subset"].eq("annotated").sum()),
        "n_generators": int(
            videos.loc[videos["subset"].eq("annotated"), "source_model"].nunique()
        ),
        "calibration_score_mean": float(calibration_videos["final_score"].mean()),
        "evaluation_real_score_mean": float(
            videos.loc[videos["subset"].eq("real"), "final_score"].mean()
        ),
        "evaluation_fake_score_mean": float(
            videos.loc[videos["subset"].eq("annotated"), "final_score"].mean()
        ),
    }])

    operating_rows = []
    calibration_scores = calibration_videos["final_score"].to_numpy(dtype=np.float64)
    for target_fpr in OPERATING_FPRS:
        threshold = real_only_threshold(calibration_scores, target_fpr)
        calibration_fake = calibration_videos["final_score"].to_numpy() < threshold
        real = videos[videos["subset"].eq("real")]
        fake = videos[videos["subset"].eq("annotated")]
        operating_rows.append({
            "calibration_bank": calibration_bank,
            "evaluation_domain": evaluation_domain,
            "target_real_fpr": target_fpr,
            "authenticity_threshold": threshold,
            "calibration_empirical_fpr": float(calibration_fake.mean()),
            "evaluation_actual_real_fpr": float(
                (real["final_score"].to_numpy() < threshold).mean()
            ),
            "evaluation_fake_recall": float(
                (fake["final_score"].to_numpy() < threshold).mean()
            ),
            "n_calibration_real": len(calibration_videos),
            "n_evaluation_real": len(real),
            "n_evaluation_fake": len(fake),
        })

    generator_rows = []
    real = videos[videos["subset"].eq("real")]
    for generator, fake in videos[videos["subset"].eq("annotated")].groupby(
        "source_model", sort=True
    ):
        value = binary_metrics(pd.concat([real, fake], ignore_index=True))
        generator_rows.append({
            "calibration_bank": calibration_bank,
            "evaluation_domain": evaluation_domain,
            "generator": generator,
            "auc": value["auc"],
            "ap_real": value["real_positive_ap"],
            "n_real": len(real),
            "n_fake": len(fake),
        })
    return summary, pd.DataFrame(operating_rows), pd.DataFrame(generator_rows)


__all__ = [
    "OPERATING_FPRS",
    "evaluate_cross_domain_cell",
    "real_only_threshold",
]
