"""固定参考下的配对差值，以及不重提特征的单分支专家归因。"""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import subprocess
import sys
import pandas as pd
from artifacts import paper_json
from reference import file_digest
from evaluation.bootstrap import paired_source_contrast
from evaluation.tables import evaluate_fixed_pairs
from statistical_experts.manifests import settings

CONTRASTS={
 'offline_vs_pooled':('offline_final','pooled_final'),
 'online_vs_pooled':('online_final','pooled_final'),
 'random_vs_pooled':('random_final','pooled_final'),
 'online_vs_random':('online_final','random_final'),
 'online_vs_offline':('online_final','offline_final'),
 'online_vs_official':('online_final','official_final'),
 'offline_vs_official':('offline_final','official_final'),
 'offline_spatial_vs_pooled':('offline_spatial','pooled_spatial'),
 'offline_temporal_vs_pooled':('offline_temporal','pooled_temporal'),
 'online_spatial_vs_pooled':('online_spatial','pooled_spatial'),
 'online_temporal_vs_pooled':('online_temporal','pooled_temporal'),
}
for method in ('offline','online'):
    for branch in ('spatial','temporal'):
        CONTRASTS[f'{method}_{branch}_hybrid_vs_pooled']=(f'{method}_{branch}_hybrid','pooled_final')


def hybrids(root):
    root=Path(root);c=settings(root);directory=root/c['run_directory']/'evaluation'
    source=directory/'video_scores.csv.gz';scores=pd.read_csv(source,float_precision='round_trip')
    pairs=pd.read_csv(root/c['manifest_directory']/'pairs.csv')
    pool=scores[scores.variant=='pooled_final'].set_index('video_id');extra=[];tables=[]
    for method in ('offline','online'):
        for branch in ('spatial','temporal'):
            opposite='temporal' if branch=='spatial' else 'spatial'
            a=scores[scores.variant==method+'_'+branch].set_index('video_id').loc[pool.index]
            b=scores[scores.variant=='pooled_'+opposite].set_index('video_id').loc[pool.index]
            p=pool.copy();p['final_score']=.5*a.final_score+.5*b.final_score
            p['variant']=method+'_'+branch+'_hybrid';p['method']=method;p['branch']=branch+'_hybrid';p=p.reset_index();extra.append(p)
            for name,table in evaluate_fixed_pairs(p,pairs).items():tables.append((name,table.assign(variant=p.variant.iloc[0])))
    pd.concat(extra).to_csv(directory/'hybrid_scores.csv.gz',index=False)
    for name in {name for name,_ in tables}:
        pd.concat([t for n,t in tables if n==name]).to_csv(directory/f'hybrid_{name}.csv',index=False)


def contrast(root,name):
    root=Path(root);c=settings(root);directory=root/c['run_directory']/'evaluation'
    scores=pd.concat([pd.read_csv(directory/'video_scores.csv.gz',float_precision='round_trip'),
                      pd.read_csv(directory/'hybrid_scores.csv.gz',float_precision='round_trip')],ignore_index=True)
    a,b=CONTRASTS[name];candidate=scores[scores.variant==a];baseline=scores[scores.variant==b]
    groups=scores[['video_id','source_group']].drop_duplicates().set_index('video_id').source_group
    pairs=pd.read_csv(root/c['manifest_directory']/'pairs.csv')
    out=root/c['run_directory']/'intervals'/name
    inputs=dict(scores=file_digest(directory/'video_scores.csv.gz'),hybrids=file_digest(directory/'hybrid_scores.csv.gz'),
        pairs=file_digest(root/c['manifest_directory']/'pairs.csv'),candidate=a,baseline=b,iterations=1000,seed=17)
    if (out/'manifest.json').exists():
        if json.loads((out/'manifest.json').read_text())['inputs']!=inputs:raise ValueError('区间输入改变')
        return
    point,interval=paired_source_contrast(candidate,baseline,pairs,groups,iterations=1000,seed=17)
    out.mkdir(parents=True,exist_ok=True)
    result=point.merge(interval,on=['dataset','metric'],validate='one_to_one').replace({'dataset':{'Macro-3':'Average'}})
    result.to_csv(out/'difference.csv',index=False)
    paper_json(out/'manifest.json',dict(status='completed',inputs=inputs,
        scope='固定fit/CDF及随机路由seed；源组配对Poisson区间，未进行多重比较修正',
        files={'difference.csv':file_digest(out/'difference.csv')}))
    print('interval completed',name,flush=True)


def all_contrasts(root,workers=6):
    root=Path(root);c=settings(root);directory=root/c['run_directory']/'evaluation'
    if not (directory/'hybrid_scores.csv.gz').exists():hybrids(root)
    env=dict(os.environ,PYTHONPATH=str(root/'src'),OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1')
    def run(name):
        subprocess.run([sys.executable,'-m','statistical_experts.analysis','--contrast',name],cwd=root,env=env,check=True)
    with ThreadPoolExecutor(max_workers=workers) as pool:list(pool.map(run,CONTRASTS))
    result=[]
    for name in CONTRASTS:
        p=root/c['run_directory']/'intervals'/name/'difference.csv'
        result.append(pd.read_csv(p).assign(contrast=name))
    pd.concat(result).to_csv(directory/'confidence_intervals.csv',index=False)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--contrast',choices=CONTRASTS);p.add_argument('--workers',type=int,default=6);a=p.parse_args()
    if a.contrast:contrast(Path.cwd(),a.contrast)
    else:all_contrasts(Path.cwd(),a.workers)
