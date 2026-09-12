"""同簇参考与连续精度组合：两个独立实验，复用固定Global特征。"""
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
from data.prefetch import bounded_map
from evaluation.tables import evaluate_fixed_pairs
from evaluation.bootstrap import paired_source_contrast
from statistical_experts.cache import feature_identity, load_feature
from statistical_experts.controls import check_files
from statistical_experts.engine import load_bank
from statistical_experts.evaluation import read_raw
from statistical_experts.gaussian import context, transitions, energy
from statistical_experts.manifests import settings

CONTRASTS={
 'expert_c_vs_b':('expert_c_final','expert_b_final'),
 'expert_c_vs_a':('expert_c_final','expert_a_final'),
 'pooled_c_vs_b':('pooled_c_final','pooled_b_final'),
 'expert_c_vs_pooled_c':('expert_c_final','pooled_c_final'),
 'expert_c_vs_official':('expert_c_final','official_final'),
 'uniform_vs_hard':('uniform_b_final','hard_b_final'),
 'soft_vs_hard':('soft_b_final','hard_b_final'),
 'soft_vs_uniform':('soft_b_final','uniform_b_final'),
 'soft_vs_official':('soft_b_final','official_final'),
 'uniform_vs_pooled':('uniform_b_final','pooled_b_final'),
}


def temperature(logits,floor=1e-6):
    x=np.asarray(logits,dtype=np.float64)
    if x.ndim!=2 or x.shape[1]<2 or not np.isfinite(x).all() or floor<=0:
        raise ValueError('真实拟合相似度非法')
    ordered=np.sort(x,axis=1);gaps=ordered[:,-1]-ordered[:,-2]
    median=float(np.median(gaps))
    return max(median,floor),dict(median_gap=median,min_gap=float(gaps.min()),max_gap=float(gaps.max()),
                                  sources=len(x),floor_used=median<floor)


def soft_weights(logits,tau):
    logits=np.asarray(logits,dtype=np.float64)
    if logits.ndim!=2 or not np.isfinite(logits).all() or np.isnan(tau) or tau<=0:
        raise ValueError('路由相似度或温度非法')
    x=(logits-logits.max(axis=1,keepdims=True))/tau
    w=np.exp(x);return w/w.sum(axis=1,keepdims=True)


def combine_positions(log_energy,zero,weights):
    """先在各转移上组合有限Gaussian能量，再统一屏蔽零T1并取min。"""
    e=np.asarray(log_energy,dtype=np.float64);zero=np.asarray(zero);w=np.asarray(weights,dtype=np.float64)
    if e.ndim!=3 or zero.shape!=e.shape[:2] or zero.dtype!=bool or w.shape!=(e.shape[0],e.shape[2]):
        raise ValueError('组合形状不符')
    if not np.isfinite(e).all() or not np.isfinite(w).all() or (w<0).any() or not np.allclose(w.sum(1),1.,rtol=0,atol=1e-14):
        raise ValueError('组合前能量必须有限且权重归一化')
    position=(e*w[:,None,:]).sum(axis=2)
    return np.where(zero,np.inf,position).min(axis=1)


def cluster_percentile(query,routes,reference,reference_routes,minimum=1):
    q=np.asarray(query,dtype=np.float64);r=np.asarray(routes);ref=np.asarray(reference,dtype=np.float64);rr=np.asarray(reference_routes)
    if q.ndim!=1 or r.shape!=q.shape or ref.ndim!=1 or rr.shape!=ref.shape or r.dtype.kind not in 'iu' or rr.dtype.kind not in 'iu':
        raise ValueError('条件参考形状错误')
    if np.isnan(q).any() or np.isnan(ref).any() or np.isneginf(q).any() or np.isneginf(ref).any():
        raise ValueError('条件参考分数非法')
    out=np.empty_like(q)
    for m in np.unique(r):
        selected=ref[rr==m]
        if len(selected)<minimum:raise ValueError('同簇CDF支持不足')
        out[r==m]=np.searchsorted(np.sort(selected),q[r==m],side='right')/len(selected)
    return out


def configuration(root):
    c=yaml.safe_load((Path(root)/'configs/global_expert_routing.yaml').read_text())
    keys={'protocol','source_directory','controls_directory','run_directory','query_batch','temperature_rule',
          'temperature_floor','min_cdf_sources','bootstrap_iterations','bootstrap_seed','analysis_workers'}
    if set(c)!=keys or c['temperature_rule']!='median_fit_top_two_gap' or c['query_batch']!=4 or c['temperature_floor']!=1e-6:
        raise ValueError('不符合冻结路由协议')
    if c['bootstrap_iterations']!=1000 or c['bootstrap_seed']!=17 or c['min_cdf_sources']!=128:
        raise ValueError('不符合预定参考/统计预算')
    if c['source_directory']!=settings(root)['run_directory'] or c['run_directory'] in (c['source_directory'],c['controls_directory']):
        raise ValueError('不能覆盖既有结果')
    return c


def prepare(root):
    root=Path(root);c=configuration(root);base=settings(root);source=root/c['source_directory'];controls=root/c['controls_directory']
    files=['configs/global_expert_routing.yaml','configs/global_experts.yaml','configs/paper.yaml',
           'src/statistical_experts/routing_study.py','src/statistical_experts/gaussian.py',
           'src/statistical_experts/cache.py','src/statistical_experts/engine.py',
           'src/evaluation/tables.py','src/evaluation/metrics.py','src/evaluation/bootstrap.py',
           'data/manifests/global_experts/windows.csv','data/manifests/global_experts/pairs.csv']
    files += [str((source/p).relative_to(root)) for p in ['manifest.json','evaluation/video_scores.csv.gz','models_8/models.pt','models_16/models.pt']]
    files += [str((controls/p).relative_to(root)) for p in ['manifest.json','video_scores.csv.gz','cdf_scores_8.npz','cdf_scores_16.npz',
                                                        'moments/raw_8.npz','moments/raw_16.npz','moments/video_scores.csv.gz']]
    spec=dict(config=c,files={p:file_digest(root/p) for p in files},feature_identity=config_digest(feature_identity(root,base)))
    out=root/c['run_directory'];marker=out/'identity.json'
    if marker.exists():
        if json.loads(marker.read_text())!=spec:raise ValueError('输入/源码改变，不能恢复同run')
    else:
        out.mkdir(parents=True,exist_ok=True)
        for p in files:
            if p.startswith(('src/','configs/')):
                dest=out/'source_snapshot'/p;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(root/p,dest)
        paper_json(marker,spec)
        paper_json(out/'status.json',dict(status='prepared',scope='同簇CDF及总体控制；共享均值hard/uniform/soft精度对照'))
    return out,spec


def score(root,length,device):
    root=Path(root);out,spec=prepare(root);c=spec['config'];base=settings(root);name=f'positions_{length}'
    if (out/(name+'.json')).exists():check_files(out,name+'.json');return
    start=time.perf_counter();torch.set_num_threads(4)
    model=torch.load(root/c['source_directory']/f'models_{length}/models.pt',map_location=device,weights_only=True)
    bank=load_bank(root,length,device)
    if bank['ids']!=model['fit_ids']:raise ValueError('拟合身份顺序变化')
    logits=(bank['context']@model['centers'].T).cpu().numpy();tau,diag=temperature(logits,c['temperature_floor'])
    del bank
    frame,raw,_=read_raw(root,length);rows=list(frame.itertuples(index=False));n=len(rows);batch=4
    positions=np.empty((n,length-1,4));zeros=np.empty((n,length-1),dtype=bool);similarity=np.empty((n,4));routes=np.empty(n,dtype=np.int64)
    items=[rows[i:i+batch] for i in range(0,n,batch)]
    def load(group):
        f=[load_feature(root/base['cache_directory']/(r.cache_key+'.npz'),spec['feature_identity'],r) for r in group]
        while len(f)<batch:f.append(f[-1])
        return torch.from_numpy(np.stack(f))
    stream=bounded_map(items,load,lambda group:batch*length*1024*4,workers=4,depth=8,budget=128*2**20)
    done=0
    try:
        for group,g in stream:
            count=len(group);t,z=transitions(g);sim=(context(g).to(device)@model['centers'].T).cpu().numpy()
            similarity[done:done+count]=sim[:count];routes[done:done+count]=sim[:count].argmax(1)
            zeros[done:done+count]=z.numpy()[:count];t=t.to(device)
            for m in range(4):
                chol=model['offline_t']['chol'][m].expand(batch,-1,-1).contiguous()
                # 与既有协方差-only相同的总体均值广播；不显式复制均值。
                value=energy(t,model['pool_t']['mean'],chol)
                positions[done:done+count,:,m]=value.cpu().numpy()[:count]
            done+=count
            if done%1000==0 or done==n:print(f'[routing {length}] {done}/{n} {time.perf_counter()-start:.1f}s',flush=True)
    finally:stream.close()
    np.testing.assert_array_equal(routes,[r['cluster'] for r in raw])
    hard=np.eye(4,dtype=np.float64)[routes]
    values=np.column_stack([combine_positions(positions,zeros,hard),
                            combine_positions(positions,zeros,np.full((n,4),.25)),
                            combine_positions(positions,zeros,soft_weights(similarity,tau))])
    with np.load(root/c['controls_directory']/f'moments/raw_{length}.npz') as old:
        np.testing.assert_array_equal(old['video_ids'],frame.video_id.to_numpy())
        np.testing.assert_array_equal(values[:,0],old['scores'][:,2])
    temp=out/(name+'.tmp.npz');target=out/(name+'.npz')
    np.savez(temp,video_ids=frame.video_id.to_numpy(dtype=str),positions=positions,zero=zeros,similarities=similarity,
             routes=routes,scores=values,temperature=np.asarray(tau),fit_logits=logits)
    temp.replace(target)
    paper_json(out/(name+'.json'),dict(identity=config_digest(spec),files={target.name:file_digest(target)},
        temperature=tau,temperature_diagnostic=diag,rows=n,elapsed_seconds=time.perf_counter()-start,
        hard_raw_exact=True,peak_gpu_mib=torch.cuda.max_memory_allocated(device)/2**20))


def evaluate(root):
    root=Path(root);out,spec=prepare(root);c=spec['config']
    if (out/'evaluation_manifest.json').exists():check_files(out,'evaluation_manifest.json');return
    old=pd.read_csv(root/c['source_directory']/'evaluation/video_scores.csv.gz',float_precision='round_trip')
    control=pd.read_csv(root/c['controls_directory']/'video_scores.csv.gz',float_precision='round_trip')
    moment=pd.read_csv(root/c['controls_directory']/'moments/video_scores.csv.gz',float_precision='round_trip')
    space=old[old.variant.eq('official_spatial')].set_index('video_id',verify_integrity=True)
    output=[];arrays={};support=[];routing=[]
    for length in (8,16):
        stage=check_files(out,f'positions_{length}.json')
        if stage['identity']!=config_digest(spec):raise ValueError('位置分数身份不符')
        frame,raw,_=read_raw(root,length);ref=frame.role.eq('cdf').to_numpy();mask=frame.role.eq('evaluation').to_numpy()
        routes=np.array([r['cluster'] for r in raw],dtype=np.int64)
        pooled=np.array([float(r['scores']['pooled'][1]) for r in raw]);expert=np.array([float(r['scores']['offline'][1]) for r in raw])
        if ref.sum()!=2000:raise ValueError('CDF身份不是2000')
        with np.load(root/c['controls_directory']/f'cdf_scores_{length}.npz') as z:
            matrix=z['scores'].copy();np.testing.assert_array_equal(z['routes'],routes[ref])
            np.testing.assert_array_equal(z['video_ids'],frame.loc[ref,'video_id'].to_numpy())
            np.testing.assert_array_equal(matrix[np.arange(2000),routes[ref]],expert[ref])
        meta=space.loc[frame.loc[mask,'video_id']];ids=meta.index
        def get(table,name):return table[table.variant.eq(name)].set_index('video_id',verify_integrity=True).loc[ids,'final_score'].to_numpy()
        pcts={'official':get(old,'official_temporal'),'pooled_b':get(old,'pooled_temporal'),
              'expert_b':get(old,'offline_temporal'),'expert_a':get(control,'offline_t_a')}
        for name,values in [('pooled_c',pooled),('expert_c',expert)]:
            pcts[name]=cluster_percentile(values[mask],routes[mask],values[ref],routes[ref],c['min_cdf_sources'])
            for m in range(4):arrays[f'{name}_{length}_{m}']=np.sort(values[ref][routes[ref]==m])
        with np.load(out/f'positions_{length}.npz') as z:
            np.testing.assert_array_equal(z['video_ids'],frame.video_id.to_numpy())
            v=z['scores'];sim=z['similarities'];weights=soft_weights(sim,float(z['temperature']))
            ordered=np.sort(sim,axis=1);gaps=ordered[:,-1]-ordered[:,-2]
            for j,name in enumerate(('hard_b','uniform_b','soft_b')):
                reference=np.sort(v[ref,j]);arrays[f'{name}_{length}']=reference
                pcts[name]=np.searchsorted(reference,v[mask,j],side='right')/len(reference)
            np.testing.assert_array_equal(pcts['hard_b'],get(moment,'expert_covariance_temporal'))
            entropy=-(weights*np.log(np.maximum(weights,np.finfo(float).tiny))).sum(axis=1)
            for i,r in enumerate(frame.itertuples(index=False)):
                routing.append(dict(video_id=r.video_id,role=r.role,dataset=r.dataset,subset=r.subset,length=length,
                    route=int(routes[i]),entropy=float(entropy[i]),effective_experts=float(np.exp(entropy[i])),
                    max_weight=float(weights[i].max()),top_gap=float(gaps[i])))
        for m in range(4):support.append(dict(length=length,expert=m,cdf_sources=int((routes[ref]==m).sum()),evaluation_clips=int((routes[mask]==m).sum())))
        for name,pct in pcts.items():
            for branch,values in [('temporal',pct),('final',.5*meta.final_score.to_numpy()+.5*pct)]:
                q=meta.copy();q['final_score']=values;q['variant']=name+'_'+branch;q['method']=name;q['branch']=branch
                output.append(q.reset_index())
    scores=pd.concat(output,ignore_index=True);scores.to_csv(out/'video_scores.csv.gz',index=False)
    pairs=pd.read_csv(root/'data/manifests/global_experts/pairs.csv');tables={}
    for name,q in scores.groupby('variant',sort=False):
        for key,t in evaluate_fixed_pairs(q,pairs).items():tables.setdefault(key,[]).append(t.assign(variant=name))
    for key,t in tables.items():atomic_csv(out/(key+'.csv'),pd.concat(t,ignore_index=True).replace({'scope':{'Macro-3':'Average'}}))
    atomic_csv(out/'cdf_support.csv',pd.DataFrame(support));rf=pd.DataFrame(routing);rf.to_csv(out/'routing.csv.gz',index=False)
    atomic_csv(out/'routing_summary.csv',rf.groupby(['length','role','dataset','subset'],as_index=False).agg(
        n=('video_id','size'),mean_entropy=('entropy','mean'),mean_effective_experts=('effective_experts','mean'),mean_max_weight=('max_weight','mean'),mean_gap=('top_gap','mean')))
    np.savez(out/'cdf_arrays.npz',**arrays)
    files=['video_scores.csv.gz','cdf_support.csv','routing.csv.gz','routing_summary.csv','cdf_arrays.npz']+[k+'.csv' for k in tables]
    paper_json(out/'evaluation_manifest.json',dict(status='point_estimates_complete',identity=config_digest(spec),
        files={p:file_digest(out/p) for p in files},methods=9,branches=2,evaluation_clip_ids=15569,cells=23))
    f=pd.read_csv(out/'macro_metrics.csv');print(f[f.variant.str.endswith('_final')][['variant','auc','real_positive_ap','fake_positive_ap','fake_tpr_at_1pct_real_fpr']].to_string(index=False),flush=True)


def contrast(root,name):
    root=Path(root);out,spec=prepare(root);check_files(out,'evaluation_manifest.json');directory=out/'intervals'/name
    a,b=CONTRASTS[name];p=root/'data/manifests/global_experts/pairs.csv'
    identity=dict(study=config_digest(spec),candidate=a,baseline=b,scores=file_digest(out/'video_scores.csv.gz'),pairs=file_digest(p),iterations=1000,seed=17)
    if (directory/'manifest.json').exists():
        if check_files(directory,'manifest.json')['inputs']!=identity:raise ValueError('区间身份改变')
        return
    s=pd.read_csv(out/'video_scores.csv.gz',float_precision='round_trip');pairs=pd.read_csv(p)
    groups=s[['video_id','source_group']].drop_duplicates().set_index('video_id').source_group
    start=time.perf_counter();point,interval=paired_source_contrast(s[s.variant.eq(a)],s[s.variant.eq(b)],pairs,groups,iterations=1000,seed=17)
    table=point.merge(interval,on=['dataset','metric'],validate='one_to_one').replace({'dataset':{'Macro-3':'Average'}})
    directory.mkdir(parents=True,exist_ok=True);atomic_csv(directory/'difference.csv',table)
    paper_json(directory/'manifest.json',dict(status='completed',inputs=identity,elapsed_seconds=time.perf_counter()-start,files={'difference.csv':file_digest(directory/'difference.csv')}))


def analyze(root):
    root=Path(root);out,spec=prepare(root);logs=out/'logs';logs.mkdir(exist_ok=True)
    def job(name):
        env=dict(os.environ,OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='2',PYTHONPATH=str(root/'src'))
        with (logs/(name+'.log')).open('w') as stream:
            subprocess.run([sys.executable,'-m','statistical_experts.run','routing-contrast','--contrast',name],cwd=root,env=env,stdout=stream,stderr=subprocess.STDOUT,check=True)
        print('completed',name,flush=True)
    with ThreadPoolExecutor(max_workers=spec['config']['analysis_workers']) as pool:list(pool.map(job,CONTRASTS))
    frames=[]
    for name in CONTRASTS:
        check_files(out/'intervals'/name,'manifest.json');frames.append(pd.read_csv(out/'intervals'/name/'difference.csv').assign(contrast=name))
    atomic_csv(out/'confidence_intervals.csv',pd.concat(frames,ignore_index=True))
    paper_json(out/'analysis_manifest.json',dict(status='completed',contrasts=list(CONTRASTS),files={'confidence_intervals.csv':file_digest(out/'confidence_intervals.csv')}))
