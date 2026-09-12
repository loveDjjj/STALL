"""整流程CDF与23单元共享评价，不混用其他Gaussian旧参考。"""
import json
from pathlib import Path
import numpy as np
import pandas as pd
from artifacts import checkpoint_read,paper_json
from config import config_digest
from reference import file_digest
from evaluation.tables import evaluate_fixed_pairs
from statistical_experts.manifests import settings

METHODS=('pooled','offline','online','random')


def read_raw(root,length):
    root=Path(root);c=settings(root);directory=root/c['run_directory']/f'raw_{length}'
    frame=pd.read_csv(root/c['manifest_directory']/'windows.csv',keep_default_na=False)
    frame=frame[(frame.length==length)&(frame.role!='fit')].reset_index(drop=True)
    world=json.loads((directory/'completed_rank_0.json').read_text())['world_size'];identities={};first=None
    for rank in range(world):
        spec=json.loads((directory/f'identity_rank_{rank}.json').read_text());done=json.loads((directory/f'completed_rank_{rank}.json').read_text())
        identity=config_digest(spec)
        if done['identity']!=identity or done['world_size']!=world or spec['manifest_sha256']!=config_digest(frame.to_dict('records')):raise ValueError('统计分片身份错误')
        if first is not None and spec!=first:raise ValueError('分片不是同一算法')
        first=spec;identities[rank]=identity
    results=[]
    for i,row in enumerate(frame.itertuples(index=False)):
        payload=checkpoint_read(directory/'checkpoints'/f'{i:06d}.json',identities[i%world],row.video_id)
        if set(payload['scores'])!=set(METHODS) or payload['role']!=row.role or payload['length']!=length:raise ValueError('分数方法或数据职责不符')
        for key in ('online','random'):
            idx=payload['neighbors'][key]
            if len(idx)!=c['neighbors'] or len(set(idx))!=len(idx) or min(idx)<0 or max(idx)>=2200 or idx!=sorted(idx):raise ValueError('邻居身份不合法')
        results.append(payload)
    return frame,results,first


def evaluate(root):
    root=Path(root);c=settings(root);out=root/c['run_directory']/'evaluation'
    if (out/'manifest.json').exists():return
    inputs={};scores=[];calibration={};route_counts=[]
    with np.load(root/'precomputed/stall_params_vatex_dino_v3.npz') as z:
        official_cdf=[np.sort(z['calib_ll_spat'].max(1)),np.sort(z['calib_ll_temp'].min(1))]
    for length in (8,16):
        frame,raw,spec=read_raw(root,length);inputs[str(length)]=spec
        is_cdf=frame.role.eq('cdf').to_numpy()
        if is_cdf.sum()!=2000 or not frame.loc[is_cdf,'subset'].eq('real').all():raise ValueError('独立CDF身份不符')
        for method in METHODS:
            values=np.asarray([[float(v) for v in r['scores'][method]] for r in raw])
            if not np.isfinite(values[:,0]).all() or np.isnan(values).any() or np.isneginf(values).any():raise ValueError('非法原始分数')
            refs=np.sort(values[is_cdf],axis=0)
            calibration[f'{method}_{length}']=refs
            percentages=np.stack([np.searchsorted(refs[:,j],values[:,j],side='right')/len(refs) for j in range(2)],axis=1)
            for i,r in enumerate(frame.to_dict('records')):
                if r['role']!='evaluation':continue
                meta={key:r[key] for key in ('video_id','video_path','dataset','subset','source_model','source_group')}
                for branch,value in [('spatial',percentages[i,0]),('temporal',percentages[i,1]),('final',percentages[i].mean())]:
                    scores.append(dict(**meta,method=method,branch=branch,variant=method+'_'+branch,final_score=float(value),length=length))
        for i,r in enumerate(frame.to_dict('records')):
            route_counts.append(dict(video_id=r['video_id'],role=r['role'],dataset=r['dataset'],length=length,cluster=raw[i]['cluster']))
            if r['role']=='evaluation':
                values=[float(r['expected_gs']),float(r['expected_gt'])]
                pct=[np.searchsorted(official_cdf[j],values[j],side='right')/len(official_cdf[j]) for j in range(2)]
                if .5*(pct[0]+pct[1])!=float(r['expected_final']):raise ValueError('官方锚点融合不符')
                meta={key:r[key] for key in ('video_id','video_path','dataset','subset','source_model','source_group')}
                for branch,value in [('spatial',pct[0]),('temporal',pct[1]),('final',.5*(pct[0]+pct[1]))]:
                    scores.append(dict(**meta,method='official',branch=branch,variant='official_'+branch,final_score=float(value),length=length))
    scores=pd.DataFrame(scores);pairs=pd.read_csv(root/c['manifest_directory']/'pairs.csv',keep_default_na=False)
    expected=set(map(tuple,pairs[['dataset','generator']].drop_duplicates().to_numpy()))
    if len(expected)!=23:raise ValueError('不是23单元')
    generators=[];datasets=[];summaries=[]
    for variant,part in scores.groupby('variant',sort=False):
        tables=evaluate_fixed_pairs(part,pairs);g=tables['generator_metrics']
        if len(g)!=23 or set(map(tuple,g[['dataset','generator']].to_numpy()))!=expected:raise ValueError('方法覆盖不完整')
        generators.append(g.assign(variant=variant,method=part.method.iloc[0],branch=part.branch.iloc[0]))
        datasets.append(tables['dataset_metrics'].assign(variant=variant,method=part.method.iloc[0],branch=part.branch.iloc[0]))
        summaries.append(tables['macro_metrics'].assign(scope='Average',variant=variant,method=part.method.iloc[0],branch=part.branch.iloc[0]))
    out.mkdir(parents=True,exist_ok=True);scores.to_csv(out/'video_scores.csv.gz',index=False)
    pd.concat(generators).to_csv(out/'generator_metrics.csv',index=False)
    pd.concat(datasets).to_csv(out/'dataset_metrics.csv',index=False)
    summary=pd.concat(summaries);summary.to_csv(out/'summary.csv',index=False)
    pd.DataFrame(route_counts).to_csv(out/'routing.csv',index=False)
    np.savez(out/'cdf_arrays.npz',**calibration)
    paper_json(out/'manifest.json',dict(status='point_estimates_complete',inputs=inputs,config=c,cells=23,
        meaning='B式整体算法CDF；Average为三域等权；无独立部署阈值集',
        files={p.name:file_digest(p) for p in out.iterdir() if p.is_file()}))
    print(summary[['variant','auc','real_positive_ap']].to_string(index=False),flush=True)


if __name__=='__main__':evaluate(Path.cwd())
