"""匹配独立源预算的T1来源诊断，含固定5折真实留出NLL。"""
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
from artifacts import atomic_csv,paper_json
from config import config_digest
from reference import file_digest
from data.prefetch import bounded_map
from evaluation.tables import evaluate_fixed_pairs
from evaluation.bootstrap import paired_source_contrast
from statistical_experts.source_data import prepare as prepare_data,target_identity
from statistical_experts.cache import load_feature
from statistical_experts.controls import check_files
from statistical_experts.evaluation import read_raw
from statistical_experts.gaussian import transitions,fitted,energy
from statistical_experts.manifests import settings
from statistical_experts.moment_controls import raw_order_scores

CONTRASTS={
 'target_vs_source':('target_matched_final','source_matched_final'),
 'target_raw_vs_source':('target_matched_raw','source_matched_raw'),
 'target_temporal_vs_source':('target_matched_temporal','source_matched_temporal'),
 'target_vs_vatex2200':('target_matched_final','vatex2200_final'),
 'target_raw_vs_vatex2200':('target_matched_raw','vatex2200_raw'),
 'target_temporal_vs_vatex2200':('target_matched_temporal','vatex2200_temporal'),
 'source_vs_vatex2200':('source_matched_final','vatex2200_final'),
 'target_vs_official':('target_matched_final','official_final'),
}


def prepare(root):
    root=Path(root);out,data=prepare_data(root);c=data['config']
    f=pd.read_csv(out/'target_windows.csv',keep_default_na=False)
    files={str(Path(c['cache_directory'])/(r.cache_key+'.npz')):file_digest(root/c['cache_directory']/(r.cache_key+'.npz')) for r in f.itertuples()}
    files.update({p:file_digest(root/p) for p in ['src/statistical_experts/source_study.py','src/statistical_experts/gaussian.py',
        'src/evaluation/tables.py','src/evaluation/metrics.py','src/evaluation/bootstrap.py',
        c['source_directory']+'/evaluation/video_scores.csv.gz',c['source_directory']+'/models_8/models.pt',c['source_directory']+'/models_16/models.pt']})
    spec=dict(data_identity=config_digest(data),config=c,files=files,target_feature_identity=target_identity(root,data))
    marker=out/'stat_identity.json'
    if marker.exists():
        if json.loads(marker.read_text())!=spec:raise ValueError('统计输入/源码改变')
    else:
        paper_json(marker,spec)
        for p in files:
            if p.startswith('src/'):
                dest=out/'source_snapshot'/p;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(root/p,dest)
    return out,data,spec


def banks(root,length,data,spec):
    root=Path(root);out=root/data['config']['run_directory'];c=data['config'];base=settings(root)
    target=pd.read_csv(out/'target_windows.csv',keep_default_na=False);target=target[target.length.eq(length)].reset_index(drop=True)
    old=pd.read_csv(root/'data/manifests/global_experts/windows.csv',keep_default_na=False)
    old=old[old.length.eq(length)&old.role.eq('fit')].copy()
    ids=pd.read_csv(out/'vatex_sources.csv').source_id.tolist();old['media_id']=old.video_path.map(lambda p:Path(p).stem)
    source=old.set_index('media_id',verify_integrity=True).loc[ids].reset_index()
    def read(frame,directory,identity):
        with ThreadPoolExecutor(max_workers=4) as pool:
            xs=list(pool.map(lambda r:load_feature(root/directory/(r.cache_key+'.npz'),identity,r),frame.itertuples(index=False)))
        return transitions(torch.from_numpy(np.stack(xs)))[0]
    t=read(target,c['cache_directory'],spec['target_feature_identity'])
    s=read(source,base['cache_directory'],data['old_feature_identity'])
    return target,t,source,s


def nll_per_video(x,model):
    # 与拟合一致，零T1仍参与真实NLL；检测时才在min前mask。含1/2 logdet。
    return (-energy(x,model['mean'],model['chol'])+model['chol'].diagonal().log().sum()).mean(dim=-1)


def fit(root,length,device):
    root=Path(root);out,data,spec=prepare(root);name=f'models_{length}'
    if (out/(name+'.json')).exists():check_files(out,name+'.json');return
    start=time.perf_counter();torch.set_num_threads(4)
    target,t,source,s=banks(root,length,data,spec);t=t.to(device);s=s.to(device)
    models={};nll=[];folds=[];support=[]
    for domain,n in data['config']['source_counts'].items():
        if length==8 and domain=='comgenvid':continue
        idx=np.flatnonzero(target.dataset.eq(domain).to_numpy());f=target.iloc[idx].reset_index(drop=True);x=t[idx]
        if len(f)!=n or f.source_group.nunique()!=n:raise ValueError('目标预算或源组重复')
        for label,z in [('source_matched',s[:n]),('target_matched',x)]:
            model=fitted(z.flatten(0,1),ridge=data['config']['ridge'])
            models[domain+'__'+label]={k:model[k].cpu() for k in ('mean','chol')}
            support.append(dict(dataset=domain,length=length,model=label,sources=n,vectors=n*(length-1),zero_vectors=int(z.norm(dim=-1).eq(0).sum())))
        for fold in range(5):
            train=np.flatnonzero(f.fold.ne(fold).to_numpy());val=np.flatnonzero(f.fold.eq(fold).to_numpy())
            if not len(train) or not len(val):raise ValueError('空留出折')
            tm=fitted(x[train].flatten(0,1),ridge=data['config']['ridge']);sm=fitted(s[:len(train)].flatten(0,1),ridge=data['config']['ridge'])
            tn=nll_per_video(x[val],tm).cpu().numpy();sn=nll_per_video(x[val],sm).cpu().numpy()
            folds.append(dict(dataset=domain,length=length,fold=fold,target_train=f.iloc[train].video_id.tolist(),
                target_validation=f.iloc[val].video_id.tolist(),source_train=source.iloc[:len(train)].video_id.tolist()))
            for j,k in enumerate(val):nll.append(dict(dataset=domain,length=length,fold=fold,video_id=f.iloc[k].video_id,
                source_group=f.iloc[k].source_group,training_sources=len(train),target_nll=float(tn[j]),source_nll=float(sn[j]),delta=float(tn[j]-sn[j])))
        print('fit/real holdout',domain,length,n,flush=True)
    temp=out/(name+'.tmp.pt');torch.save(models,temp);temp.replace(out/(name+'.pt'))
    atomic_csv(out/f'nll_{length}.csv',pd.DataFrame(nll));atomic_csv(out/f'support_{length}.csv',pd.DataFrame(support))
    paper_json(out/f'folds_{length}.json',dict(folds=folds,meaning='固定源级5折，仅诊断，不选择参数'))
    files=[name+'.pt',f'nll_{length}.csv',f'support_{length}.csv',f'folds_{length}.json']
    paper_json(out/(name+'.json'),dict(identity=config_digest(spec),status='completed',elapsed_seconds=time.perf_counter()-start,
        files={p:file_digest(out/p) for p in files}))


def score(root,length,device):
    root=Path(root);out,data,spec=prepare(root);c=data['config'];base=settings(root);name=f'raw_{length}'
    if (out/(name+'.json')).exists():check_files(out,name+'.json');return
    check_files(out,f'models_{length}.json');torch.set_num_threads(4);start=time.perf_counter()
    models=torch.load(out/f'models_{length}.pt',map_location=device,weights_only=True)
    original=torch.load(root/c['source_directory']/f'models_{length}/models.pt',map_location=device,weights_only=True)
    models['vatex2200']=original['pool_t'];keys=list(models)
    frame,old,_=read_raw(root,length);rows=list(frame.itertuples(index=False));batch=4
    values=np.empty((len(rows),len(keys)));items=[rows[i:i+batch] for i in range(0,len(rows),batch)]
    def load(group):
        xs=[load_feature(root/base['cache_directory']/(r.cache_key+'.npz'),data['old_feature_identity'],r) for r in group]
        while len(xs)<batch:xs.append(xs[-1])
        return torch.from_numpy(np.stack(xs))
    stream=bounded_map(items,load,lambda g:batch*length*1024*4,workers=4,depth=8,budget=128*2**20);done=0
    try:
        for group,g in stream:
            t,zero=transitions(g);t=t.to(device);zero=zero.to(device);n=len(group)
            for j,key in enumerate(keys):
                m=models[key];q=energy(t,m['mean'],m['chol']).masked_fill(zero,float('inf')).min(-1).values
                values[done:done+n,j]=q.cpu().numpy()[:n]
            done+=n
            if done%1000==0 or done==len(rows):print(f'[sources {length}] {done}/{len(rows)} {time.perf_counter()-start:.1f}s',flush=True)
    finally:stream.close()
    np.testing.assert_array_equal(values[:,keys.index('vatex2200')],[float(r['scores']['pooled'][1]) for r in old])
    temp=out/(name+'.tmp.npz');dest=out/(name+'.npz')
    np.savez(temp,scores=values,model_names=np.asarray(keys),video_ids=frame.video_id.to_numpy(dtype=str));temp.replace(dest)
    paper_json(out/(name+'.json'),dict(identity=config_digest(spec),status='completed',rows=len(rows),elapsed_seconds=time.perf_counter()-start,
        anchor_exact=True,files={dest.name:file_digest(dest)}))


def evaluate(root):
    root=Path(root);out,data,spec=prepare(root);c=data['config']
    if (out/'evaluation_manifest.json').exists():check_files(out,'evaluation_manifest.json');return
    original=pd.read_csv(root/c['source_directory']/'evaluation/video_scores.csv.gz',float_precision='round_trip')
    space=original[original.variant.eq('official_spatial')].set_index('video_id');records=[];cdf={}
    for length in (8,16):
        check_files(out,f'raw_{length}.json');frame,old,_=read_raw(root,length);ref=frame.role.eq('cdf').to_numpy()
        with np.load(out/f'raw_{length}.npz') as z:
            np.testing.assert_array_equal(z['video_ids'],frame.video_id.to_numpy());v=z['scores'];names=z['model_names'].tolist()
        for domain in c['source_counts']:
            mask=frame.role.eq('evaluation').to_numpy()&frame.dataset.eq(domain).to_numpy()
            if not mask.any():continue
            ids=frame.loc[mask,'video_id'];meta=space.loc[ids]
            for label in ('source_matched','target_matched','vatex2200','official'):
                if label=='official':
                    raw=frame.loc[mask,'expected_gt'].to_numpy(dtype=float)
                    pct=original[original.variant.eq('official_temporal')].set_index('video_id').loc[ids,'final_score'].to_numpy()
                else:
                    key=label if label=='vatex2200' else domain+'__'+label
                    raw=v[mask,names.index(key)];reference=np.sort(v[ref,names.index(key)])
                    if len(reference)!=2000:raise ValueError('CDF预算不符')
                    cdf[f'{domain}_{length}_{label}']=reference
                    pct=np.searchsorted(reference,raw,side='right')/len(reference)
                for branch,values in [('raw',raw),('temporal',pct),('final',.5*meta.final_score.to_numpy()+.5*pct)]:
                    q=meta.copy();q['raw_temporal_score']=raw;q['final_score']=values;q['variant']=label+'_'+branch;q['method']=label;q['branch']=branch
                    records.append(q.reset_index())
    scores=pd.concat(records,ignore_index=True)
    for label in ('source_matched','target_matched','vatex2200','official'):
        sel=scores.variant.eq(label+'_raw');scores.loc[sel,'final_score']=raw_order_scores(scores.loc[sel,'raw_temporal_score'])
    scores.to_csv(out/'video_scores.csv.gz',index=False);np.savez(out/'cdf_arrays.npz',**cdf)
    pairs=pd.read_csv(root/'data/manifests/global_experts/pairs.csv');tables={}
    for label,q in scores.groupby('variant',sort=False):
        for key,t in evaluate_fixed_pairs(q,pairs).items():tables.setdefault(key,[]).append(t.assign(variant=label))
    for key,t in tables.items():atomic_csv(out/(key+'.csv'),pd.concat(t,ignore_index=True).replace({'scope':{'Macro-3':'Average'}}))
    nll=pd.concat([pd.read_csv(out/f'nll_{l}.csv') for l in (8,16)],ignore_index=True);nll['improved']=nll.delta.lt(0)
    atomic_csv(out/'nll_all.csv',nll)
    atomic_csv(out/'nll_summary.csv',nll.groupby(['dataset','length'],as_index=False).agg(n=('video_id','size'),source_nll=('source_nll','mean'),target_nll=('target_nll','mean'),delta=('delta','mean'),median_delta=('delta','median'),fraction_improved=('improved','mean')))
    files=['video_scores.csv.gz','cdf_arrays.npz','nll_all.csv','nll_summary.csv']+[k+'.csv' for k in tables]
    paper_json(out/'evaluation_manifest.json',dict(identity=config_digest(spec),status='point_estimates_complete',files={p:file_digest(out/p) for p in files}))
    f=pd.read_csv(out/'macro_metrics.csv');print(f[['variant','auc','real_positive_ap']].to_string(index=False),flush=True)
    print(pd.read_csv(out/'nll_summary.csv').to_string(index=False),flush=True)


def contrast(root,name):
    root=Path(root);out,data,spec=prepare(root);check_files(out,'evaluation_manifest.json');directory=out/'intervals'/name
    a,b=CONTRASTS[name];pairs_path=root/'data/manifests/global_experts/pairs.csv'
    inputs=dict(study=config_digest(spec),candidate=a,baseline=b,scores=file_digest(out/'video_scores.csv.gz'),pairs=file_digest(pairs_path),iterations=1000,seed=17)
    if (directory/'manifest.json').exists():
        if check_files(directory,'manifest.json')['inputs']!=inputs:raise ValueError('区间身份不同')
        return
    scores=pd.read_csv(out/'video_scores.csv.gz',float_precision='round_trip');pairs=pd.read_csv(pairs_path)
    groups=scores[['video_id','source_group']].drop_duplicates().set_index('video_id').source_group
    start=time.perf_counter();point,interval=paired_source_contrast(scores[scores.variant.eq(a)],scores[scores.variant.eq(b)],pairs,groups,iterations=1000,seed=17)
    table=point.merge(interval,on=['dataset','metric'],validate='one_to_one').replace({'dataset':{'Macro-3':'Average'}})
    directory.mkdir(parents=True,exist_ok=True);atomic_csv(directory/'difference.csv',table)
    paper_json(directory/'manifest.json',dict(status='completed',inputs=inputs,elapsed_seconds=time.perf_counter()-start,files={'difference.csv':file_digest(directory/'difference.csv')}))


def analyze(root):
    root=Path(root);out,data,spec=prepare(root);logs=out/'logs';logs.mkdir(exist_ok=True)
    def job(name):
        env=dict(os.environ,OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='2',PYTHONPATH=str(root/'src'))
        with (logs/(name+'.log')).open('w') as stream:
            subprocess.run([sys.executable,'-m','statistical_experts.run','source-contrast','--contrast',name],cwd=root,env=env,stdout=stream,stderr=subprocess.STDOUT,check=True)
        print('completed',name,flush=True)
    with ThreadPoolExecutor(max_workers=data['config']['analysis_workers']) as pool:list(pool.map(job,CONTRASTS))
    frames=[]
    for name in CONTRASTS:
        check_files(out/'intervals'/name,'manifest.json');frames.append(pd.read_csv(out/'intervals'/name/'difference.csv').assign(contrast=name))
    atomic_csv(out/'confidence_intervals.csv',pd.concat(frames,ignore_index=True))
    paper_json(out/'analysis_manifest.json',dict(status='completed',contrasts=list(CONTRASTS),files={'confidence_intervals.csv':file_digest(out/'confidence_intervals.csv')}))
