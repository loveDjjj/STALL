"""官方空间组合与同专家全库CDF对照；复用冻结特征和Gaussian。"""
from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

import numpy as np
import pandas as pd
import torch
import yaml

from artifacts import atomic_csv, paper_json
from config import config_digest
from reference import file_digest
from evaluation.bootstrap import paired_source_contrast
from evaluation.tables import evaluate_fixed_pairs
from statistical_experts.cache import feature_identity, load_feature
from statistical_experts.evaluation import read_raw
from statistical_experts.gaussian import energy, transitions
from statistical_experts.manifests import settings


CONTRASTS = {
    'offline_b_vs_official': ('official_s_offline_t_b', 'official_final'),
    'online_b_vs_official': ('official_s_online_t_b', 'official_final'),
    'offline_b_vs_pooled_t': ('official_s_offline_t_b', 'official_s_pooled_t'),
    'online_b_vs_pooled_t': ('official_s_online_t_b', 'official_s_pooled_t'),
    'online_b_vs_random_t': ('official_s_online_t_b', 'official_s_random_t'),
    'offline_a_vs_b': ('official_s_offline_t_a', 'official_s_offline_t_b'),
    'offline_a_vs_official': ('official_s_offline_t_a', 'official_final'),
    'pooled_space_a_vs_b': ('pooled_s_offline_t_a', 'pooled_s_offline_t_b'),
    'temporal_a_vs_b': ('offline_t_a', 'offline_t_b'),
}


def configuration(root):
    c = yaml.safe_load((Path(root) / 'configs/global_expert_controls.yaml').read_text())
    keys = {'protocol', 'source_directory', 'run_directory', 'query_batch',
            'bootstrap_iterations', 'bootstrap_seed', 'analysis_workers'}
    if set(c) != keys or c['query_batch'] != 4 or c['bootstrap_iterations'] != 1000 or c['bootstrap_seed'] != 17:
        raise ValueError('本轮配置不符，科学协议变更须新建run')
    if c['source_directory'] != settings(root)['run_directory'] or c['run_directory'] == c['source_directory']:
        raise ValueError('输入与输出目录必须分离且绑定首轮')
    return c


def study_identity(root):
    root = Path(root); c = configuration(root); source = root / c['source_directory']
    paths = ['configs/global_experts.yaml', 'configs/global_expert_controls.yaml',
             'src/statistical_experts/controls.py', 'src/statistical_experts/gaussian.py',
             'src/statistical_experts/evaluation.py', 'src/statistical_experts/cache.py',
             'src/evaluation/tables.py', 'src/evaluation/metrics.py', 'src/evaluation/bootstrap.py',
             'data/manifests/global_experts/windows.csv', 'data/manifests/global_experts/pairs.csv']
    paths += [str((source / p).relative_to(root)) for p in
              ['manifest.json', 'verification.json', 'evaluation/video_scores.csv.gz',
               'evaluation/cdf_arrays.npz', 'evaluation/summary.csv',
               'models_8/models.pt', 'models_16/models.pt']]
    return dict(config=c, files={p: file_digest(root / p) for p in paths},
                feature_identity=config_digest(feature_identity(root, settings(root))))


def prepare(root):
    root = Path(root); identity = study_identity(root); out = root / identity['config']['run_directory']
    marker = out / 'identity.json'
    if marker.exists():
        if json.loads(marker.read_text()) != identity:
            raise ValueError('本轮配置/源码/输入已改变，不允许覆盖恢复')
        return identity
    out.mkdir(parents=True, exist_ok=True)
    for p in identity['files']:
        if p.startswith(('src/', 'configs/')):
            target = out / 'source_snapshot' / p
            target.parent.mkdir(parents=True, exist_ok=True); shutil.copy2(root / p, target)
    paper_json(marker, identity)
    paper_json(out / 'status.json', dict(status='prepared',
        scope='官方空间组合＋固定离线时序Gaussian的B整体CDF/A同专家全库CDF；不改变路由/拟合集/评价'))
    return identity


def expert_percentiles(query, routes, reference):
    """每一列为同一专家对全部独立real的评分，右包含且保留+inf质量。"""
    query = np.asarray(query, dtype=np.float64); routes = np.asarray(routes)
    reference = np.asarray(reference, dtype=np.float64)
    if reference.ndim != 2 or not len(reference) or query.ndim != 1 or routes.shape != query.shape:
        raise ValueError('查询或参考维度错误')
    if routes.dtype.kind not in 'iu' or (routes < 0).any() or (routes >= reference.shape[1]).any():
        raise ValueError('专家路由非法')
    if np.isnan(query).any() or np.isneginf(query).any() or np.isnan(reference).any() or np.isneginf(reference).any():
        raise ValueError('分数含非法非有限值')
    result = np.empty_like(query)
    for m in range(reference.shape[1]):
        selected = routes == m
        result[selected] = np.searchsorted(np.sort(reference[:, m]), query[selected], side='right') / len(reference)
    return result


def score_reference(root, length, device):
    root = Path(root); spec = prepare(root); c = spec['config']; base = settings(root)
    out = root / c['run_directory']; target = out / f'cdf_scores_{length}.npz'
    marker = out / f'cdf_scores_{length}.json'
    if marker.exists():
        meta = json.loads(marker.read_text())
        if meta['identity'] != config_digest(spec) or file_digest(target) != meta['sha256']:
            raise ValueError('CDF评分恢复身份错误')
        return
    start = time.perf_counter(); torch.set_num_threads(4)
    frame, raw, _ = read_raw(root, length)
    indices = np.flatnonzero(frame.role.eq('cdf').to_numpy())
    refs = frame.iloc[indices].reset_index(drop=True)
    if len(refs) != 2000 or not refs.subset.eq('real').all():
        raise ValueError('CDF不是既定2000独立real')
    model = torch.load(root / c['source_directory'] / f'models_{length}/models.pt', map_location=device, weights_only=True)
    batch = c['query_batch']; result = np.empty((len(refs), 4)); routes = np.empty(len(refs), dtype=np.int64)
    rows = list(refs.itertuples(index=False)); compute = 0.
    def load(row):
        return load_feature(root / base['cache_directory'] / (row.cache_key + '.npz'), spec['feature_identity'], row)
    # 只读约125MiB参考特征，不加载视频、不运行编码器。
    with ThreadPoolExecutor(max_workers=4) as pool:
        features = list(pool.map(load, rows))
    from statistical_experts.gaussian import context
    for begin in range(0, len(rows), batch):
        count = min(batch, len(rows) - begin); selected = features[begin:begin+count]
        while len(selected) < batch: selected.append(selected[-1])
        g = torch.from_numpy(np.stack(selected)); t, zero = transitions(g)
        routes[begin:begin+count] = (context(g).to(device) @ model['centers'].T).argmax(1).cpu().numpy()[:count]
        t = t.to(device); zero = zero.to(device); torch.cuda.synchronize(device); tick = time.perf_counter()
        for m in range(4):
            mean = model['offline_t']['mean'][m].expand(batch, -1).contiguous()
            chol = model['offline_t']['chol'][m].expand(batch, -1, -1).contiguous()
            values = energy(t, mean, chol).masked_fill(zero, float('inf')).min(-1).values
            result[begin:begin+count, m] = values.cpu().numpy()[:count]
        compute += time.perf_counter() - tick
        if (begin + count) % 200 == 0:
            print(f'[CDF {length}] {begin+count}/{len(rows)} real ×4专家，{time.perf_counter()-start:.1f}s', flush=True)
    expected_routes = np.asarray([raw[i]['cluster'] for i in indices])
    expected = np.asarray([float(raw[i]['scores']['offline'][1]) for i in indices])
    np.testing.assert_array_equal(routes, expected_routes)
    selected = result[np.arange(len(result)), routes]
    # 同一专家、同一batch与输入，必须逐位恢复首轮CDF raw，不用接近指标代替。
    np.testing.assert_array_equal(selected, expected)
    temporary = target.with_suffix('.tmp.npz')
    np.savez(temporary, scores=result, routes=routes, video_ids=refs.video_id.to_numpy(dtype=str))
    temporary.replace(target)
    paper_json(marker, dict(identity=config_digest(spec), sha256=file_digest(target),
        length=length, real_videos=len(refs), experts=4, source_raw_exact=True,
        elapsed_seconds=time.perf_counter()-start, scoring_seconds=compute,
        peak_gpu_mib=torch.cuda.max_memory_allocated(device)/2**20))


def evaluate(root):
    root = Path(root); spec = prepare(root); c = spec['config']; out = root / c['run_directory']
    if (out / 'evaluation_manifest.json').exists():
        check_files(out, 'evaluation_manifest.json'); return
    source = pd.read_csv(root / c['source_directory'] / 'evaluation/video_scores.csv.gz', float_precision='round_trip')
    metadata = source[source.variant.eq('official_final')].set_index('video_id', verify_integrity=True)
    def branch(name):
        return source[source.variant.eq(name)].set_index('video_id', verify_integrity=True).loc[metadata.index].final_score
    a = pd.Series(index=metadata.index, dtype=float); calibration_rows = []
    for length in (8, 16):
        stage = json.loads((out / f'cdf_scores_{length}.json').read_text())
        if stage['identity'] != config_digest(spec) or stage['sha256'] != file_digest(out / f'cdf_scores_{length}.npz'):
            raise ValueError('新CDF资产身份错误')
        frame, raw, _ = read_raw(root, length)
        with np.load(out / f'cdf_scores_{length}.npz') as z:
            matrix = z['scores'].copy()
            np.testing.assert_array_equal(z['video_ids'], frame.loc[frame.role.eq('cdf'), 'video_id'].to_numpy())
            routes_cdf = z['routes'].copy()
        routes = np.asarray([r['cluster'] for r in raw], dtype=np.int64)
        values = np.asarray([float(r['scores']['offline'][1]) for r in raw])
        mask = frame.role.eq('evaluation').to_numpy()
        a.loc[frame.loc[mask, 'video_id']] = expert_percentiles(values[mask], routes[mask], matrix)
        pooled_reference = matrix[np.arange(len(matrix)), routes_cdf]
        old = np.searchsorted(np.sort(pooled_reference), values[mask], side='right') / len(matrix)
        np.testing.assert_array_equal(old, branch('offline_temporal').loc[frame.loc[mask, 'video_id']].to_numpy())
        new = a.loc[frame.loc[mask, 'video_id']].to_numpy()
        for m in range(4):
            calibration_rows.append(dict(length=length, expert=m, reference_count=len(matrix),
                routed_reference_count=int((routes_cdf == m).sum()),
                reference_q10=float(np.quantile(matrix[:, m][np.isfinite(matrix[:, m])], .1)),
                reference_q50=float(np.median(matrix[:, m][np.isfinite(matrix[:, m])])),
                reference_q90=float(np.quantile(matrix[:, m][np.isfinite(matrix[:, m])], .9))))
    if a.isna().any(): raise ValueError('新百分位没有覆盖全评价集')
    variants = {'official_final': branch('official_final')}
    for method in ('pooled', 'offline', 'online', 'random'):
        suffix = '_b' if method in ('offline', 'online') else ''
        variants[f'official_s_{method}_t{suffix}'] = .5*branch('official_spatial') + .5*branch(method+'_temporal')
    variants.update(official_s_offline_t_a=.5*branch('official_spatial')+.5*a,
                    pooled_s_offline_t_b=.5*branch('pooled_spatial')+.5*branch('offline_temporal'),
                    pooled_s_offline_t_a=.5*branch('pooled_spatial')+.5*a,
                    offline_t_b=branch('offline_temporal'), offline_t_a=a)
    pairs = pd.read_csv(root / base_manifest(root) / 'pairs.csv'); videos = []; tables = {}
    for name, values in variants.items():
        q = metadata.copy(); q['final_score'] = values; q['variant'] = name
        q['method'] = name; q['branch'] = 'temporal' if name.startswith('offline_t_') else 'final'
        q = q.reset_index(); videos.append(q)
        for key, table in evaluate_fixed_pairs(q, pairs).items():
            if key == 'generator_metrics' and len(table) != 23: raise ValueError('生成器单元不完整')
            tables.setdefault(key, []).append(table.assign(variant=name))
    pd.concat(videos, ignore_index=True).to_csv(out / 'video_scores.csv.gz', index=False)
    for key, frames in tables.items():
        atomic_csv(out / f'{key}.csv', pd.concat(frames, ignore_index=True).replace({'scope': {'Macro-3': 'Average'}}))
    atomic_csv(out / 'cdf_diagnostics.csv', pd.DataFrame(calibration_rows))
    # A/B分数如何改变real/fake，按视频而非重叠窗口计算，不参与模型选择。
    changes = metadata[['dataset','subset','length']].copy()
    changes['delta'] = a-branch('offline_temporal')
    changes['changed'] = changes.delta.ne(0); changes['up'] = changes.delta.gt(0)
    diag = changes.groupby(['dataset','subset','length']).agg(n=('delta','size'),
        mean_delta=('delta','mean'), median_delta=('delta','median'), changed_fraction=('changed','mean'), up_fraction=('up','mean')).reset_index()
    atomic_csv(out / 'score_changes.csv', diag)
    files = ['video_scores.csv.gz', 'cdf_diagnostics.csv', 'score_changes.csv'] + [k+'.csv' for k in tables]
    paper_json(out / 'evaluation_manifest.json', dict(identity=config_digest(spec), status='point_estimates_complete',
        clip_ids=len(metadata), generator_cells=23, variants=list(variants),
        files={p: file_digest(out / p) for p in files}))
    print(pd.read_csv(out / 'macro_metrics.csv')[['variant','auc','real_positive_ap','fake_positive_ap','fake_tpr_at_1pct_real_fpr']].to_string(index=False), flush=True)


def base_manifest(root):
    return settings(root)['manifest_directory']


def check_files(out, name):
    meta = json.loads((Path(out) / name).read_text())
    for p, digest in meta['files'].items():
        if file_digest(Path(out) / p) != digest: raise ValueError(f'产物hash改变：{p}')
    return meta


def contrast(root, name):
    root = Path(root); spec = prepare(root); c = spec['config']; out = root / c['run_directory']
    check_files(out, 'evaluation_manifest.json'); scores_path = out / 'video_scores.csv.gz'
    scores = pd.read_csv(scores_path, float_precision='round_trip')
    pairs_path = root / base_manifest(root) / 'pairs.csv'; pairs = pd.read_csv(pairs_path)
    a, b = CONTRASTS[name]; directory = out / 'intervals' / name
    identity = dict(study=config_digest(spec), scores=file_digest(scores_path), pairs=file_digest(pairs_path),
        candidate=a, baseline=b, seed=c['bootstrap_seed'], iterations=c['bootstrap_iterations'])
    if (directory / 'manifest.json').exists():
        if check_files(directory, 'manifest.json')['inputs'] != identity: raise ValueError('区间恢复身份变化')
        return
    groups = scores[['video_id','source_group']].drop_duplicates().set_index('video_id').source_group
    begin = time.perf_counter()
    point, interval = paired_source_contrast(scores[scores.variant.eq(a)], scores[scores.variant.eq(b)], pairs, groups,
        seed=c['bootstrap_seed'], iterations=c['bootstrap_iterations'], progress=lambda d: print(name,d,flush=True))
    result = point.merge(interval, on=['dataset','metric'], validate='one_to_one').replace({'dataset': {'Macro-3':'Average'}})
    directory.mkdir(parents=True, exist_ok=True); atomic_csv(directory / 'difference.csv', result)
    paper_json(directory / 'manifest.json', dict(status='completed', inputs=identity,
        seconds=time.perf_counter()-begin, files={'difference.csv':file_digest(directory / 'difference.csv')}))


def analyze(root):
    root = Path(root); c = configuration(root); out = root / c['run_directory']; logs = out / 'logs'; logs.mkdir(exist_ok=True)
    def launch(name):
        env = dict(os.environ, OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='2', PYTHONPATH=str(root / 'src'))
        command = [sys.executable, '-m', 'statistical_experts.run', 'controls-contrast', '--contrast', name]
        with (logs / f'{name}.log').open('w') as stream:
            subprocess.run(command, cwd=root, env=env, stdout=stream, stderr=subprocess.STDOUT, check=True)
        print('completed', name, flush=True)
    with ThreadPoolExecutor(max_workers=c['analysis_workers']) as pool:
        list(pool.map(launch, CONTRASTS))
    frames = []
    for name in CONTRASTS:
        directory = out / 'intervals' / name; check_files(directory, 'manifest.json')
        frames.append(pd.read_csv(directory / 'difference.csv').assign(contrast=name))
    atomic_csv(out / 'confidence_intervals.csv', pd.concat(frames, ignore_index=True))
    paper_json(out / 'analysis_manifest.json', dict(status='completed', contrasts=list(CONTRASTS),
        scope='固定fit/CDF/路由seed；源组配对Poisson区间，1000次；未多重比较校正',
        files={'confidence_intervals.csv':file_digest(out / 'confidence_intervals.csv')}))


def verify(root):
    root = Path(root); spec = prepare(root); out = root / spec['config']['run_directory']
    e = check_files(out, 'evaluation_manifest.json'); check_files(out, 'analysis_manifest.json')
    scores = pd.read_csv(out / 'video_scores.csv.gz', float_precision='round_trip')
    pairs = pd.read_csv(root / base_manifest(root) / 'pairs.csv')
    reported = pd.read_csv(out / 'macro_metrics.csv', float_precision='round_trip').set_index('variant')
    if scores.groupby('variant').video_id.nunique().nunique() != 1 or e['clip_ids'] != 15569:
        raise ValueError('评价覆盖不一致')
    for variant, group in scores.groupby('variant'):
        actual = evaluate_fixed_pairs(group, pairs)['macro_metrics'].iloc[0]
        for metric in actual.index.drop('scope'):
            np.testing.assert_allclose(actual[metric], reported.loc[variant, metric], rtol=0, atol=1e-14)
    for length in (8,16):
        meta = json.loads((out / f'cdf_scores_{length}.json').read_text())
        if not meta['source_raw_exact'] or meta['sha256'] != file_digest(out / f'cdf_scores_{length}.npz'):
            raise ValueError('CDF来源核验失败')
        with np.load(out / f'cdf_scores_{length}.npz') as z:
            if z['scores'].shape != (2000,4) or len(set(z['video_ids'])) != 2000:
                raise ValueError('CDF形状/身份非法')
    intervals = pd.read_csv(out / 'confidence_intervals.csv')
    if set(intervals.contrast) != set(CONTRASTS) or len(intervals) != len(CONTRASTS)*8:
        raise ValueError('区间缺失')
    for name,(a,b) in CONTRASTS.items():
        for metric,column in [('auc','auc'),('ap_real','real_positive_ap')]:
            row = intervals[(intervals.contrast==name)&(intervals.dataset=='Average')&(intervals.metric==metric)].iloc[0]
            np.testing.assert_allclose(row.delta, reported.loc[a,column]-reported.loc[b,column], rtol=0, atol=1e-14)
    paper_json(out / 'verification.json', dict(status='verified', identity=config_digest(spec),
        evaluation_clip_ids=15569, generator_cells=23, variants=len(e['variants']), contrasts=len(CONTRASTS),
        cdf_videos_per_length=2000, experts=4, original_raw_and_B_percentiles_exact=True,
        metrics_independently_recomputed=True))
    print('controls verified', flush=True)
