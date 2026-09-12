"""原主线四格的匹配CDF、分支结果与源级配对区间。"""
from concurrent.futures import ProcessPoolExecutor
import json,multiprocessing,time
from pathlib import Path
import numpy as np,pandas as pd
from artifacts import atomic_csv,paper_json,checkpoint_read
from config import config_digest
from reference import file_digest,percentile,window_mean,local_video_cdfs
from evaluation.tables import evaluate_fixed_pairs
from evaluation.bootstrap import paired_source_contrast
from mainline_experts.run import configuration

CONTRASTS={'R1_vs_R0':('R1','R0'),'R2_vs_R0':('R2','R0'),'R3_vs_R0':('R3','R0'),
 'R3_vs_R1':('R3','R1'),'R3_vs_R2':('R3','R2'),
 'local_raw_expert_vs_base':('local1_raw','local0_raw'),
 'local_expert_vs_base':('local1','local0'),'GT_expert_vs_base':('GT1','GT0')}


def collect(root,allow_partial=False):
    root=Path(root);out=root/configuration(root)['run_directory'];jobs=json.loads((out/'plans.json').read_text())['jobs']
    spec=json.loads((out/'dense/identity.json').read_text());identity=config_digest(spec)
    records=[]
    for job in jobs:
        path=out/'dense'/(job['key']+'.json')
        if not path.exists():
            if allow_partial:continue
            raise ValueError('密集任务尚未完成：'+job['key'])
        payload=checkpoint_read(path,identity,job['key']);records.append((job,payload))
    return out,spec,records


def calibrations(records,recomputed=False):
    buckets={}
    for job,payload in records:
        if job['role']!='cdf':continue
        length=len(job['windows'][0])
        for d,meta in job['targets'].items():
            actual=payload['targets'][d];bucket=buckets.setdefault((d,length),dict(gs=[],gt0=[],gt1=[],local0=[],local1=[]))
            u=actual['uniform'] if recomputed else meta['uniform']
            bucket['gs'].append(float(u['gs']));bucket['gt0'].append(float(u['gt']))
            eu=actual['uniform'];bucket['gt1'].append(float(eu['gt_experts'][eu['route']]))
            windows=actual['windows'];base=windows if recomputed else meta['expected']
            bucket['local0'].append([float(w['lt']) for w in base]);bucket['local1'].append([float(w['lt_experts'][w['route']]) for w in windows])
    for (d,length),b in buckets.items():
        if len(b['gs'])!=2000:raise ValueError(f'{d}/{length} CDF数量不完整')
        for name in ('gs','gt0','gt1'):b[name]=np.sort(np.asarray(b[name],dtype=np.float64))
        for name in ('local0','local1'):
            b[name]=local_video_cdfs(b[name]) if length==16 else {1:np.sort(np.asarray([q[0] for q in b[name]]))}
    return buckets


def video_values(meta,actual,cdfs,recomputed=False):
    base=actual['windows'] if recomputed else meta['expected'];windows=actual['windows'];k=len(base)
    pgs=percentile([float(w['gs']) for w in base],cdfs['gs'])
    pgt0=percentile([float(w['gt']) for w in base],cdfs['gt0'],allow_positive_infinity=True)
    pgt1=percentile([float(w['gt_experts'][w['route']]) for w in windows],cdfs['gt1'],allow_positive_infinity=True)
    gs=window_mean(pgs);gt0=window_mean(pgt0);gt1=window_mean(pgt1)
    q0=window_mean([float(w['lt']) for w in base]);q1=window_mean([float(w['lt_experts'][w['route']]) for w in windows])
    l0=float(percentile([q0],cdfs['local0'][k])[0]);l1=float(percentile([q1],cdfs['local1'][k])[0])
    g0=window_mean(.5*pgs+.5*pgt0);g1=window_mean(.5*pgs+.5*pgt1)
    values=dict(R0=.5*g0+.5*l0,R1=.5*g1+.5*l0,R2=.5*g0+.5*l1,R3=.5*g1+.5*l1,
                GT0=gt0,GT1=gt1,global0=g0,global1=g1,local0=l0,local1=l1,local0_raw=q0,local1_raw=q1)
    return values


def evaluate(root,args=None):
    root=Path(root);out,spec,records=collect(root)
    if (out/'evaluation_manifest.json').exists():return
    refs=calibrations(records);actual_refs=calibrations(records,recomputed=True);rows=[];regression=[]
    baseline=pd.read_csv(root/'results/paper_complete/scores_fc3.csv.gz',float_precision='round_trip')
    baseline=baseline[baseline.variant.eq('full')].set_index('video_id',verify_integrity=True)
    for job,payload in records:
        if job['role']!='evaluation':continue
        length=len(job['windows'][0])
        for d,meta in job['targets'].items():
            actual=payload['targets'][d];v=video_values(meta,actual,refs[d,length]);rv=video_values(meta,actual,actual_refs[d,length],True)
            expected=float(baseline.loc[meta['video_id'],'final_score'])
            np.testing.assert_allclose(v['R0'],expected,rtol=0,atol=1e-10)
            np.testing.assert_allclose(rv['R0'],expected,rtol=0,atol=1e-10)
            np.testing.assert_allclose(v['R3']-v['R0'],v['R1']-v['R0']+v['R2']-v['R0'],rtol=0,atol=1e-12)
            regression.append(dict(video_id=meta['video_id'],expected=expected,R0=v['R0'],R0_recomputed=rv['R0'],raw_error=payload['max_baseline_error']))
            for variant,value in v.items():rows.append(dict(video_id=meta['video_id'],video_path=job['video_path'],dataset=d,
                subset=meta['subset'],source_model=meta['source_model'],source_group=meta['source_group'],length=length,effective_k=len(job['windows']),variant=variant,final_score=value))
    scores=pd.DataFrame(rows);pairs=pd.read_csv(out/'pairs.csv')
    if scores.video_id.nunique()!=15569 or scores.variant.nunique()!=12:raise ValueError('四格覆盖不完整')
    scores.to_csv(out/'video_scores.csv.gz',index=False);atomic_csv(out/'R0_regression.csv',pd.DataFrame(regression))
    arrays={}
    for (d,l),r in refs.items():
        for name,value in r.items():
            if isinstance(value,dict):
                for k,a in value.items():arrays[f'{d}_{l}_{name}_K{k}']=a
            else:arrays[f'{d}_{l}_{name}']=value
    np.savez(out/'cdf_arrays.npz',**arrays);tables={}
    for variant,q in scores.groupby('variant',sort=False):
        for key,t in evaluate_fixed_pairs(q,pairs).items():tables.setdefault(key,[]).append(t.assign(variant=variant))
    for key,t in tables.items():atomic_csv(out/(key+'.csv'),pd.concat(t,ignore_index=True).replace({'scope':{'Macro-3':'Average'}}))
    files=['video_scores.csv.gz','R0_regression.csv','cdf_arrays.npz']+[k+'.csv' for k in tables]
    paper_json(out/'evaluation_manifest.json',dict(status='point_estimates_complete',dense_identity=config_digest(spec),
        evaluator_sha256=file_digest(Path(__file__)),files={p:file_digest(out/p) for p in files}))
    print(pd.read_csv(out/'macro_metrics.csv')[['variant','auc','real_positive_ap']].to_string(index=False),flush=True)


def _contrast(task):
    root,name=task;root=Path(root);out=root/configuration(root)['run_directory'];a,b=CONTRASTS[name];dest=out/'intervals'/name
    if (dest/'manifest.json').exists():return name
    s=pd.read_csv(out/'video_scores.csv.gz',float_precision='round_trip');p=pd.read_csv(out/'pairs.csv');groups=s[['video_id','source_group']].drop_duplicates().set_index('video_id').source_group
    inputs=dict(scores=file_digest(out/'video_scores.csv.gz'),pairs=file_digest(out/'pairs.csv'),candidate=a,baseline=b,iterations=1000,seed=17)
    start=time.perf_counter();point,ci=paired_source_contrast(s[s.variant.eq(a)],s[s.variant.eq(b)],p,groups,iterations=1000,seed=17)
    dest.mkdir(parents=True,exist_ok=True);atomic_csv(dest/'difference.csv',point.merge(ci,on=['dataset','metric']).replace({'dataset':{'Macro-3':'Average'}}))
    paper_json(dest/'manifest.json',dict(status='completed',inputs=inputs,seconds=time.perf_counter()-start,files={'difference.csv':file_digest(dest/'difference.csv')}))
    return name


def analyze(root,args=None):
    root=Path(root);out=root/configuration(root)['run_directory']
    with ProcessPoolExecutor(max_workers=4,mp_context=multiprocessing.get_context('spawn')) as pool:
        for name in pool.map(_contrast,[(str(root),name) for name in CONTRASTS]):print('mainline CI',name,flush=True)
    atomic_csv(out/'confidence_intervals.csv',pd.concat([pd.read_csv(out/'intervals'/n/'difference.csv').assign(contrast=n) for n in CONTRASTS],ignore_index=True))
    paper_json(out/'analysis_manifest.json',dict(status='completed',files={'confidence_intervals.csv':file_digest(out/'confidence_intervals.csv')}))
