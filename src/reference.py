"""目标Global/Local参考包及平衡Gaussian统计，不导入任何研究脚本。"""
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import tempfile

import numpy as np

from math_utils import StableGaussianParams

SCHEMA = 'alpha_stalled_target_reference_v1'


def file_digest(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def percentile(values, reference, *, allow_positive_infinity=False):
    values = np.asarray(values, dtype=np.float64)
    reference = np.asarray(reference, dtype=np.float64)
    if reference.ndim != 1 or len(reference) < 2:
        raise ValueError('经验CDF至少需要两个真实参考')
    for x in (values, reference):
        if np.isnan(x).any() or np.isneginf(x).any() or (not allow_positive_infinity and np.isposinf(x).any()):
            raise ValueError('CDF输入非有限或不允许的无穷')
    return np.searchsorted(np.sort(reference), values, side='right') / len(reference)


def window_mean(values):
    """兼容旧视频groupby.mean归约，防止1ULP微差打破百分位融合的并列分数。"""
    import pandas as pd
    values=np.asarray(values,dtype=np.float64)
    if values.ndim!=1 or not len(values) or not np.isfinite(values).all():raise ValueError('窗口均值需要非空有限向量')
    return float(pd.Series(values).groupby(np.zeros(len(values),dtype=np.int8),sort=False).mean().iloc[0])


def local_video_cdfs(ranked_window_scores):
    """按旧协议的选择排名取linspace子窗，不将其改为独立Top-k选择。"""
    windows = [np.asarray(values, dtype=np.float64) for values in ranked_window_scores]
    if any(x.ndim != 1 or not 1 <= len(x) <= 3 or not np.isfinite(x).all() for x in windows):
        raise ValueError('Local参考必须为每视频1–3个有限窗口分数')
    result = {}
    for k in (1, 2, 3):
        values = []
        for scores in windows:
            if len(scores) >= k:
                ids = np.unique(np.rint(np.linspace(0, len(scores)-1, k)).astype(int))
                values.append(float(scores[ids].mean()))
        if len(values) < 2:
            raise ValueError(f'Local K={k}参考不足')
        result[k] = np.sort(np.asarray(values))
    return result


def fit_gaussian(videos, *, ridge=1e-5, diagonal=False):
    """每片段总权重相同；与已验证的np.cov(aweights,ddof=1)规则一致。"""
    videos = [np.asarray(x, dtype=np.float64) for x in videos if len(x)]
    if len(videos) < 2 or ridge <= 0:
        raise ValueError('有效拟合片段不足或正则无效')
    if any(x.ndim != 2 or not np.isfinite(x).all() for x in videos):
        raise ValueError('Gaussian拟合向量无效')
    values = np.concatenate(videos)
    weights = np.concatenate([np.full(len(x), 1./len(x)) for x in videos])
    mean = np.average(values, axis=0, weights=weights)
    covariance = np.atleast_2d(np.cov(values.T, aweights=weights, ddof=1))
    if diagonal:
        covariance = np.diag(np.diag(covariance))
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    if eigenvalues.min() < -1e-8:
        raise ValueError('实质非PSD协方差')
    whitening = eigenvectors / np.sqrt(np.maximum(eigenvalues, 0.) + ridge)
    return StableGaussianParams(mean, whitening, np.empty(0, dtype=np.float64))


@dataclass(frozen=True)
class ReferenceBundle:
    global_spatial: StableGaussianParams
    global_temporal: StableGaussianParams
    local_temporal: StableGaussianParams
    local_cdfs: dict[int, np.ndarray]
    metadata: dict

    def validate(self):
        if self.metadata.get('schema') != SCHEMA:
            raise ValueError('参考包schema不符')
        for params in (self.global_spatial, self.global_temporal, self.local_temporal):
            if params.mean.ndim != 1 or params.whitening.ndim != 2 or params.whitening.shape[0] != len(params.mean):
                raise ValueError('Gaussian参数维度不符')
            if not np.isfinite(params.mean).all() or not np.isfinite(params.whitening).all():
                raise ValueError('Gaussian参数非有限')
        percentile([], self.global_spatial.calibration_raw)
        percentile([], self.global_temporal.calibration_raw, allow_positive_infinity=True)
        if set(self.local_cdfs) != {1, 2, 3}:
            raise ValueError('Local有效K参考不完整')
        for values in self.local_cdfs.values():
            percentile([], values)

    def score_video(self, global_spatial_raw, global_temporal_raw, local_raw, *, use_global=True,
                    use_local=True, spatial_weight=.5, global_weight=.5):
        gs, gt, local = (np.asarray(x, dtype=np.float64) for x in (global_spatial_raw, global_temporal_raw, local_raw))
        if gs.ndim != 1 or gs.shape != gt.shape or gs.shape != local.shape or not 1 <= len(gs) <= 3:
            raise ValueError('视频窗口分数形状不一致或effective-K无效')
        if not use_global and not use_local:
            raise ValueError('至少保留一个分支')
        if not 0 <= spatial_weight <= 1 or not 0 <= global_weight <= 1:
            raise ValueError('融合权重不合法')
        s = percentile(gs, self.global_spatial.calibration_raw)
        t = percentile(gt, self.global_temporal.calibration_raw, allow_positive_infinity=True)
        g = window_mean(spatial_weight*s+(1-spatial_weight)*t)
        q = window_mean(local)
        l = float(percentile([q], self.local_cdfs[len(local)])[0])
        final = global_weight*g+(1-global_weight)*l if use_global and use_local else g if use_global else l
        return dict(global_score=g, global_spatial_score=window_mean(s), global_temporal_score=window_mean(t),
                    local_raw=q, local_score=l, final_score=final, effective_k=len(local))


def save_bundle(path, bundle):
    bundle.validate()
    arrays = {'metadata': np.asarray(json.dumps(bundle.metadata, sort_keys=True, ensure_ascii=False))}
    for name, params in [('gs', bundle.global_spatial), ('gt', bundle.global_temporal), ('lt', bundle.local_temporal)]:
        arrays[name+'_mean'] = params.mean
        arrays[name+'_whitening'] = params.whitening
        if name != 'lt':arrays[name+'_cdf'] = params.calibration_raw
    arrays.update({f'local_cdf_k{k}': values for k, values in bundle.local_cdfs.items()})
    publish_arrays(path,arrays)


def publish_arrays(path, arrays):
    """完整临时NPZ独占发布；恢复时只接受逐数组一致的已有文件。"""
    path = Path(path)
    if path.exists():
        with np.load(path, allow_pickle=False) as old:
            if set(old.files) != set(arrays):raise ValueError('拒绝覆盖不同参考包')
            for key, value in arrays.items():np.testing.assert_array_equal(old[key], value)
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    # 同目录临时文件完整写入后独占链接，避免中断留下半个正式npz或覆盖已有包。
    with tempfile.TemporaryDirectory(prefix='.reference-',dir=path.parent) as directory:
        temporary=Path(directory)/'bundle.npz'
        with temporary.open('wb') as stream:
            np.savez_compressed(stream, **arrays)
            stream.flush();os.fsync(stream.fileno())
        os.link(temporary,path)


def load_bundle(path):
    with np.load(path, allow_pickle=False) as z:
        parameters = [StableGaussianParams(z[n+'_mean'].copy(), z[n+'_whitening'].copy(),
                      z[n+'_cdf'].copy() if n != 'lt' else np.empty(0)) for n in ('gs', 'gt', 'lt')]
        result = ReferenceBundle(*parameters, {k: z[f'local_cdf_k{k}'].copy() for k in (1, 2, 3)},
                                 json.loads(str(z['metadata'])))
    result.validate()
    return result
