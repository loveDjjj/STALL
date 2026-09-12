"""时序专家收益确认后，固定B校准拆分均值/协方差来源；不重新选簇。"""
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

from artifacts import atomic_csv, paper_json
from config import config_digest
from reference import file_digest
from data.prefetch import bounded_map
from evaluation.tables import evaluate_fixed_pairs
from evaluation.bootstrap import paired_source_contrast
from statistical_experts.controls import prepare, check_files
from statistical_experts.cache import load_feature
from statistical_experts.evaluation import read_raw
from statistical_experts.gaussian import context, transitions, energy
from statistical_experts.manifests import settings

MODELS=('pooled','expert_mean','expert_covariance','expert_both')
CONTRASTS={
    'mean_vs_pooled':('expert_mean_final','pooled_final'),
    'covariance_vs_pooled':('expert_covariance_final','pooled_final'),
    'both_vs_pooled':('expert_both_final','pooled_final'),
    'both_vs_covariance':('expert_both_final','expert_covariance_final'),
    'both_vs_mean':('expert_both_final','expert_mean_final'),
}


def raw_order_scores(values):
    """仅供raw排序指标：保留+inf为最高同分，不把它裁成任意数值。"""
    values=np.asarray(values,dtype=np.float64)
    if np.isnan(values).any() or np.isneginf(values).any():raise ValueError('raw含非法值')
    return pd.Series(values).rank(method='max',pct=True).to_numpy()


def setup(root):
    root=Path(root); parent=prepare(root); directory=root/parent['config']['run_directory']
    # 明确条件：官方空间固定、在线内容时序对同池时序的AUC/AP区间均为正。
    check_files(directory,'analysis_manifest.json')
    intervals=pd.read_csv(directory/'confidence_intervals.csv')
    gate=intervals[(intervals.contrast=='online_b_vs_pooled_t')&(intervals.dataset=='Average')]
    if len(gate)!=2 or not gate.ci95_low.gt(0).all():raise ValueError('尚未通过时序收益条件')
    spec=dict(parent=config_digest(parent),source=str(Path(__file__).relative_to(root)),
        code_sha256=file_digest(Path(__file__)),gate_sha256=file_digest(directory/'confidence_intervals.csv'),
        models=list(MODELS),covariance='既有0.5簇协方差＋0.5总体协方差＋1e-5I；不改变数值协议',
        calibration='每组独立重建B整流程8/16帧CDF；同一2000real',
        window='原版单窗，固定路由',query_batch=4)
    out=directory/'moments';out.mkdir(exist_ok=True)
    if (out/'identity.json').exists():
        if json.loads((out/'identity.json').read_text())!=spec:raise ValueError('四格身份改变')
    else:
        paper_json(out/'identity.json',spec);shutil.copy2(__file__,out/'source_snapshot.py')
    return out,parent,spec


def score(root,length,device):
    root=Path(root);out,parent,spec=setup(root);name=f'raw_{length}'
    if (out/(name+'.json')).exists():
        check_files(out,name+'.json');return
    start=time.perf_counter();torch.set_num_threads(4);c=settings(root)
    frame,raw,_=read_raw(root,length)
    model=torch.load(root/c['run_directory']/f'models_{length}/models.pt',map_location=device,weights_only=True)
    rows=list(frame.itertuples(index=False));n=len(rows);batch=4
    results=np.empty((n,4));routes=np.empty(n,dtype=np.int64)
    items=[rows[i:i+batch] for i in range(0,n,batch)]
    def load(group):
        feats=[load_feature(root/c['cache_directory']/(r.cache_key+'.npz'),parent['feature_identity'],r) for r in group]
        while len(feats)<batch:feats.append(feats[-1])
        return torch.from_numpy(np.stack(feats))
    stream=bounded_map(items,load,lambda group:batch*length*1024*4,workers=4,depth=8,budget=128*2**20)
    done=0;compute=0.
    try:
        for group,g in stream:
            t,zero=transitions(g); route=(context(g).to(device)@model['centers'].T).argmax(1)
            t=t.to(device);zero=zero.to(device)
            # 保留首轮总体参数的D/二维Cholesky广播，显式批复制会改变浮点末位。
            pm=model['pool_t']['mean']
            pc=model['pool_t']['chol']
            em=model['offline_t']['mean'][route];ec=model['offline_t']['chol'][route]
            torch.cuda.synchronize(device);tick=time.perf_counter();count=len(group)
            for j,(mean,chol) in enumerate(((pm,pc),(em,pc),(pm,ec),(em,ec))):
                value=energy(t,mean,chol).masked_fill(zero,float('inf')).min(-1).values
                results[done:done+count,j]=value.cpu().numpy()[:count]
            compute+=time.perf_counter()-tick
            routes[done:done+count]=route.cpu().numpy()[:count];done+=count
            if done%1000==0 or done==n:print(f'[moments {length}] {done}/{n} {time.perf_counter()-start:.1f}s',flush=True)
    finally:stream.close()
    np.testing.assert_array_equal(routes,np.array([r['cluster'] for r in raw]))
    np.testing.assert_array_equal(results[:,0],np.array([float(r['scores']['pooled'][1]) for r in raw]))
    np.testing.assert_array_equal(results[:,3],np.array([float(r['scores']['offline'][1]) for r in raw]))
    temporary=out/(name+'.tmp.npz');target=out/(name+'.npz')
    np.savez(temporary,video_ids=frame.video_id.to_numpy(dtype=str),scores=results,routes=routes)
    temporary.replace(target)
    paper_json(out/(name+'.json'),dict(identity=config_digest(spec),files={target.name:file_digest(target)},
        rows=n,elapsed_seconds=time.perf_counter()-start,scoring_seconds=compute,
        exact_endpoint_raw=True,peak_gpu_mib=torch.cuda.max_memory_allocated(device)/2**20))


def evaluate(root):
    root=Path(root);out,parent,spec=setup(root);base=root/parent['config']['source_directory']
    if (out/'evaluation_manifest.json').exists():check_files(out,'evaluation_manifest.json');return
    source=pd.read_csv(base/'evaluation/video_scores.csv.gz',float_precision='round_trip')
    official=source[source.variant=='official_spatial'].set_index('video_id',verify_integrity=True)
    outputs=[];cdf={}
    for length in (8,16):
        stage=check_files(out,f'raw_{length}.json')
        if stage['identity']!=config_digest(spec):raise ValueError('四格raw身份不符')
        frame,old,_=read_raw(root,length)
        with np.load(out/f'raw_{length}.npz') as z:
            np.testing.assert_array_equal(z['video_ids'],frame.video_id.to_numpy());values=z['scores'].copy()
        refs=frame.role.eq('cdf').to_numpy();mask=frame.role.eq('evaluation').to_numpy()
        if refs.sum()!=2000:raise ValueError('参考数不符')
        meta=official.loc[frame.loc[mask,'video_id']].copy();spatial=meta.final_score.to_numpy()
        for j,name in enumerate(MODELS):
            reference=np.sort(values[refs,j]);cdf[f'{name}_{length}']=reference
            pct=np.searchsorted(reference,values[mask,j],side='right')/len(reference)
            for branch,v in [('raw',values[mask,j]),('temporal',pct),('final',.5*spatial+.5*pct)]:
                q=meta.copy();q['raw_temporal_score']=values[mask,j];q['final_score']=v;q['variant']=name+'_'+branch;q['method']=name;q['branch']=branch
                outputs.append(q.reset_index())
    scores=pd.concat(outputs,ignore_index=True)
    # 当前有2条全静态评价片段。raw指标使用严格保序的有限秩，原始+inf另列保留。
    # 此变换只用于raw-only的AUC/AP/ROC计算，绝不用于检测融合或拟合CDF。
    for name in MODELS:
        selected=scores.variant.eq(name+'_raw')
        scores.loc[selected,'final_score']=raw_order_scores(scores.loc[selected,'raw_temporal_score'])
    if not np.isfinite(scores.final_score.to_numpy()).all():raise ValueError('评价分数非法')
    scores.to_csv(out/'video_scores.csv.gz',index=False);np.savez(out/'cdf_arrays.npz',**cdf)
    pairs=pd.read_csv(root/'data/manifests/global_experts/pairs.csv');tables={}
    for name,part in scores.groupby('variant',sort=False):
        for key,t in evaluate_fixed_pairs(part,pairs).items():tables.setdefault(key,[]).append(t.assign(variant=name))
    for key,t in tables.items():atomic_csv(out/(key+'.csv'),pd.concat(t,ignore_index=True).replace({'scope':{'Macro-3':'Average'}}))
    files=['video_scores.csv.gz','cdf_arrays.npz']+[k+'.csv' for k in tables]
    paper_json(out/'evaluation_manifest.json',dict(identity=config_digest(spec),status='point_estimates_complete',
        files={p:file_digest(out/p) for p in files},models=list(MODELS),branches=['raw','temporal','final'],
        raw_metrics='raw_temporal_score保留+inf；raw-only final_score为保序且保同分的有限秩，仅用于排序指标'))
    print(pd.read_csv(out/'macro_metrics.csv')[['variant','auc','real_positive_ap']].to_string(index=False),flush=True)


def contrast(root,name):
    root=Path(root);out,parent,spec=setup(root);check_files(out,'evaluation_manifest.json')
    directory=out/'intervals'/name;a,b=CONTRASTS[name]
    if (directory/'manifest.json').exists():check_files(directory,'manifest.json');return
    scores=pd.read_csv(out/'video_scores.csv.gz',float_precision='round_trip')
    pairs_path=root/'data/manifests/global_experts/pairs.csv';pairs=pd.read_csv(pairs_path)
    groups=scores[['video_id','source_group']].drop_duplicates().set_index('video_id').source_group
    start=time.perf_counter()
    point,interval=paired_source_contrast(scores[scores.variant==a],scores[scores.variant==b],pairs,groups,iterations=1000,seed=17)
    result=point.merge(interval,on=['dataset','metric'],validate='one_to_one').replace({'dataset':{'Macro-3':'Average'}})
    directory.mkdir(parents=True,exist_ok=True);atomic_csv(directory/'difference.csv',result)
    paper_json(directory/'manifest.json',dict(identity=config_digest(spec),candidate=a,baseline=b,iterations=1000,seed=17,
        input_scores_sha256=file_digest(out/'video_scores.csv.gz'),input_pairs_sha256=file_digest(pairs_path),
        elapsed_seconds=time.perf_counter()-start,files={'difference.csv':file_digest(directory/'difference.csv')}))


def analyze(root):
    root=Path(root);out,parent,spec=setup(root);logs=out/'logs';logs.mkdir(exist_ok=True)
    def job(name):
        env=dict(os.environ,OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='2',PYTHONPATH=str(root/'src'))
        with (logs/(name+'.log')).open('w') as log:
            subprocess.run([sys.executable,'-m','statistical_experts.run','moments-contrast','--contrast',name],
                           cwd=root,env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
        print('moments contrast completed',name,flush=True)
    with ThreadPoolExecutor(max_workers=5) as pool:list(pool.map(job,CONTRASTS))
    frames=[]
    for name in CONTRASTS:
        check_files(out/'intervals'/name,'manifest.json')
        frames.append(pd.read_csv(out/'intervals'/name/'difference.csv').assign(contrast=name))
    atomic_csv(out/'confidence_intervals.csv',pd.concat(frames,ignore_index=True))
    paper_json(out/'analysis_manifest.json',dict(status='completed',contrasts=list(CONTRASTS),
        files={'confidence_intervals.csv':file_digest(out/'confidence_intervals.csv')}))


def verify(root):
    root=Path(root);out,parent,spec=setup(root);check_files(out,'evaluation_manifest.json');check_files(out,'analysis_manifest.json')
    scores=pd.read_csv(out/'video_scores.csv.gz',float_precision='round_trip')
    original=pd.read_csv(root/parent['config']['run_directory']/'video_scores.csv.gz',float_precision='round_trip')
    for a,b in [('pooled_final','official_s_pooled_t'),('expert_both_final','official_s_offline_t_b')]:
        x=scores[scores.variant==a].set_index('video_id');y=original[original.variant==b].set_index('video_id').loc[x.index]
        np.testing.assert_array_equal(x.final_score.to_numpy(),y.final_score.to_numpy())
    for length in (8,16):
        frame,raw,_=read_raw(root,length);stage=check_files(out,f'raw_{length}.json')
        if stage['identity']!=config_digest(spec):raise ValueError('raw身份错误')
        with np.load(out/f'raw_{length}.npz') as z:
            vals=z['scores'];np.testing.assert_array_equal(z['video_ids'],frame.video_id.to_numpy())
            np.testing.assert_array_equal(vals[:,0],[float(r['scores']['pooled'][1]) for r in raw])
            np.testing.assert_array_equal(vals[:,3],[float(r['scores']['offline'][1]) for r in raw])
            reference=frame.role.eq('cdf').to_numpy();evaluation=frame.role.eq('evaluation').to_numpy()
            for j,name in enumerate(MODELS):
                expected=[];q=vals[evaluation,j];r=vals[reference,j]
                for start in range(0,len(q),128):expected.extend((r[:,None]<=q[None,start:start+128]).sum(0)/2000)
                actual=scores[(scores.variant==name+'_temporal')&(scores.length==length)].set_index('video_id').loc[frame.loc[evaluation,'video_id']]
                np.testing.assert_array_equal(expected,actual.final_score.to_numpy())
    pairs=pd.read_csv(root/'data/manifests/global_experts/pairs.csv')
    for name,group in scores.groupby('variant'):
        for table,keys in [('generator_metrics',['dataset','generator']),('dataset_metrics',['dataset']),('macro_metrics',['scope'])]:
            actual=evaluate_fixed_pairs(group,pairs)[table].replace({'scope':{'Macro-3':'Average'}}).set_index(keys).sort_index()
            saved=pd.read_csv(out/(table+'.csv'),float_precision='round_trip');saved=saved[saved.variant==name].drop(columns='variant').set_index(keys).sort_index()
            pd.testing.assert_frame_equal(actual,saved,check_dtype=False,check_exact=False,rtol=0,atol=1e-14)
    intervals=pd.read_csv(out/'confidence_intervals.csv')
    if len(intervals)!=40 or set(intervals.contrast)!=set(CONTRASTS):raise ValueError('四格区间缺失')
    paper_json(out/'verification.json',dict(status='verified',identity=config_digest(spec),source_sha256=file_digest(out/'source_snapshot.py'),
        models=4,variants=12,evaluation_clip_ids=15569,cells=23,contrasts=5,endpoint_raw_and_final_exact=True,
        all_cdf_direct_counts_exact=True,metrics_recomputed=True))
    print('moments verified',flush=True)
