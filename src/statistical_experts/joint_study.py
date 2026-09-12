"""固定x边际，联合/独立/打乱评分的单次机制对照。"""
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import json,os,shutil,subprocess,sys,time
import numpy as np,pandas as pd,torch
from artifacts import atomic_csv,paper_json
from reference import file_digest
from config import config_digest
from data.prefetch import bounded_map
from evaluation.tables import evaluate_fixed_pairs
from evaluation.bootstrap import paired_source_contrast
from statistical_experts.gaussian import fitted,energy,transitions
from statistical_experts.predictive import position_scores
from statistical_experts.engine import load_bank
from statistical_experts.cache import load_feature
from statistical_experts.controls import check_files
from statistical_experts.moment_controls import raw_order_scores

RUN='results/runs/global_joint'
SOURCE='results/runs/global_predictive'
MODES={'independent':'marginal','joint':'predictive','shuffled_joint':'shuffled'}
CONTRASTS={'joint_vs_independent':('joint_final','independent_final'),
 'joint_vs_shuffled':('joint_final','shuffled_joint_final'),
 'joint_raw_vs_independent':('joint_raw','independent_raw'),
 'joint_raw_vs_shuffled':('joint_raw','shuffled_joint_raw'),
 'joint_vs_conditional':('joint_final','conditional_final'),
 'joint_vs_official':('joint_final','official_final')}


def aggregate_joint(x,y,valid):
    if x.shape!=y.shape or x.shape!=valid.shape:raise ValueError('联合评分形状不符')
    return (x+y).masked_fill(~valid,float('inf')).min(-1).values


def prepare(root):
    root=Path(root);out=root/RUN
    paths=['src/statistical_experts/joint_study.py','src/statistical_experts/predictive.py','src/statistical_experts/gaussian.py',
      'src/evaluation/tables.py','src/evaluation/metrics.py','src/evaluation/bootstrap.py',
      SOURCE+'/manifest.json',SOURCE+'/video_scores.csv.gz',SOURCE+'/pairs.csv',
      'results/runs/global_experts/evaluation/video_scores.csv.gz','results/runs/global_external/raw.csv','precomputed/stall_params_vatex_dino_v3.npz']
    for l in (8,16):paths += [SOURCE+f'/{n}_{l}.{ext}' for n,ext in [('models','pt'),('queries','csv'),('raw','npz')]]
    spec=dict(protocol='global_joint_information_v1',source=SOURCE,models=MODES,dimension=1024,ridge=1e-5,
              fit_sources=2200,cdf_sources=2000,query_batch=4,seed=17,bootstrap_iterations=1000,
              score='min_t(log energy x + conditional/marginal log energy y); identical mask; no new fusion weight',
              files={p:file_digest(root/p) for p in paths})
    if (out/'identity.json').exists():
        if json.loads((out/'identity.json').read_text())!=spec:raise ValueError('联合实验身份改变')
    else:
        out.mkdir(parents=True,exist_ok=True);paper_json(out/'identity.json',spec)
        for p in paths:
            if p.startswith('src/'):
                dest=out/'source_snapshot'/p;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(root/p,dest)
    return out,spec


def score(root,length,device):
    root=Path(root);out,spec=prepare(root);name=f'raw_{length}'
    if (out/(name+'.json')).exists():check_files(out,name+'.json');return
    torch.set_num_threads(4);start=time.perf_counter();bank=load_bank(root,length,device)
    xm=fitted(bank['temporal'][:,:-1].flatten(0,1),ridge=1e-5)
    torch.save({k:v.cpu() for k,v in xm.items()},out/f'x_model_{length}.pt')
    models=torch.load(root/SOURCE/f'models_{length}.pt',map_location=device,weights_only=True)
    f=pd.read_csv(root/SOURCE/f'queries_{length}.csv',keep_default_na=False);rows=list(f.itertuples(index=False));batch=4
    values=np.empty((len(f),3));old=np.empty((len(f),3));valid_count=np.empty(len(f),int)
    items=[rows[i:i+batch] for i in range(0,len(rows),batch)]
    def load(group):
        xs=[load_feature(root/r.cache_directory/(r.cache_key+'.npz'),r.feature_identity,r) for r in group]
        while len(xs)<batch:xs.append(xs[-1])
        return torch.from_numpy(np.stack(xs))
    stream=bounded_map(items,load,lambda g:batch*length*1024*4,workers=4,depth=8,budget=128*2**20);done=0
    try:
        for group,g in stream:
            t,z=transitions(g);valid=(~(z[:,:-1]|z[:,1:])).to(device);t=t.to(device);n=len(group)
            ex=energy(t[:,:-1],xm['mean'],xm['chol'])
            for j,m in enumerate(MODES.values()):
                ey=position_scores(t,models[m]);values[done:done+n,j]=aggregate_joint(ex,ey,valid).cpu().numpy()[:n]
                old[done:done+n,j]=ey.masked_fill(~valid,float('inf')).min(-1).values.cpu().numpy()[:n]
            valid_count[done:done+n]=valid.sum(-1).cpu().numpy()[:n];done+=n
            if done%1000==0 or done==len(rows):print('joint',length,done,len(rows),flush=True)
    finally:stream.close()
    with np.load(root/SOURCE/f'raw_{length}.npz') as z:
        np.testing.assert_array_equal(old,z['scores']);np.testing.assert_array_equal(valid_count,z['valid_pairs']);np.testing.assert_array_equal(f.video_id.to_numpy(),z['video_ids'])
    temp=out/(name+'.tmp.npz');np.savez(temp,scores=values,valid_pairs=valid_count,video_ids=f.video_id.to_numpy(dtype=str));temp.replace(out/(name+'.npz'))
    paper_json(out/(name+'.json'),dict(status='completed',identity=config_digest(spec),old_conditional_exact=True,seconds=time.perf_counter()-start,
        files={p:file_digest(out/p) for p in [name+'.npz',f'x_model_{length}.pt']}))


def evaluate(root):
    root=Path(root);out,spec=prepare(root)
    if (out/'evaluation_manifest.json').exists():check_files(out,'evaluation_manifest.json');return
    source=pd.read_csv(root/SOURCE/'video_scores.csv.gz',float_precision='round_trip');official=source[source.variant.eq('official_final')].copy();conditional=source[source.variant.eq('predictive_final')].copy();conditional['variant']='conditional_final'
    dev=pd.read_csv(root/'results/runs/global_experts/evaluation/video_scores.csv.gz',float_precision='round_trip');gs=dev[dev.variant.eq('official_spatial')].set_index('video_id').final_score
    er=pd.read_csv(root/'results/runs/global_external/raw.csv',float_precision='round_trip').set_index('video_id')
    with np.load(root/'precomputed/stall_params_vatex_dino_v3.npz') as z:sc=np.sort(z['calib_ll_spat'].max(1))
    gs=pd.concat([gs,pd.Series(np.searchsorted(sc,er['gs'],side='right')/len(sc),index=er.index)])
    output=[official,conditional];cdf={}
    for length in (8,16):
        check_files(out,f'raw_{length}.json');f=pd.read_csv(root/SOURCE/f'queries_{length}.csv',keep_default_na=False)
        with np.load(out/f'raw_{length}.npz') as z:v=z['scores'];cnt=z['valid_pairs'];np.testing.assert_array_equal(z['video_ids'],f.video_id.to_numpy())
        ref=f.role.eq('cdf').to_numpy();ev=f.role.eq('evaluation').to_numpy();meta=f[ev].copy()
        for j,m in enumerate(MODES):
            r=np.sort(v[ref,j]);assert len(r)==2000;cdf[m+'_'+str(length)]=r;pct=np.searchsorted(r,v[ev,j],side='right')/len(r)
            for branch,value in [('raw',v[ev,j]),('temporal',pct),('final',.5*gs.loc[meta.video_id].to_numpy()+.5*pct)]:
                q=meta.copy();q['raw_score']=v[ev,j];q['final_score']=value;q['variant']=m+'_'+branch;q['method']=m;q['branch']=branch;q['valid_pairs']=cnt[ev];output.append(q)
    scores=pd.concat(output,ignore_index=True)
    for m in MODES:
        mask=scores.variant.eq(m+'_raw');scores.loc[mask,'final_score']=raw_order_scores(scores.loc[mask,'raw_score'])
    scores.to_csv(out/'video_scores.csv.gz',index=False);np.savez(out/'cdf_arrays.npz',**cdf)
    pairs=pd.read_csv(root/SOURCE/'pairs.csv');tables={}
    for name,q in scores.groupby('variant'):
        for key,t in evaluate_fixed_pairs(q,pairs).items():tables.setdefault(key,[]).append(t.assign(variant=name))
    for key,t in tables.items():atomic_csv(out/(key+'.csv'),pd.concat(t,ignore_index=True).replace({'scope':{'Macro-3':'Average'}}))
    files=['video_scores.csv.gz','cdf_arrays.npz']+[k+'.csv' for k in tables];paper_json(out/'evaluation_manifest.json',dict(status='point_estimates_complete',files={p:file_digest(out/p) for p in files}))
    print(pd.read_csv(out/'macro_metrics.csv')[['variant','auc','real_positive_ap']].to_string(index=False),flush=True)
    d=pd.read_csv(out/'dataset_metrics.csv');print(d[d.dataset.isin(['genvidbench','vifbench'])&d.variant.str.endswith('_final')][['dataset','variant','auc','real_positive_ap']].to_string(index=False),flush=True)


def contrast(root,name):
    root=Path(root);out,spec=prepare(root);check_files(out,'evaluation_manifest.json');dest=out/'intervals'/name;a,b=CONTRASTS[name]
    inputs=dict(scores=file_digest(out/'video_scores.csv.gz'),pairs=file_digest(root/SOURCE/'pairs.csv'),candidate=a,baseline=b,iterations=1000,seed=17)
    if (dest/'manifest.json').exists():
        if check_files(dest,'manifest.json')['inputs']!=inputs:raise ValueError('配对输入不同')
        return
    scores=pd.read_csv(out/'video_scores.csv.gz',float_precision='round_trip');pairs=pd.read_csv(root/SOURCE/'pairs.csv');groups=scores[['video_id','source_group']].drop_duplicates().set_index('video_id').source_group
    point,ci=paired_source_contrast(scores[scores.variant.eq(a)],scores[scores.variant.eq(b)],pairs,groups,iterations=1000,seed=17)
    dest.mkdir(parents=True,exist_ok=True);atomic_csv(dest/'difference.csv',point.merge(ci,on=['dataset','metric']).replace({'dataset':{'Macro-3':'Average'}}))
    paper_json(dest/'manifest.json',dict(status='completed',inputs=inputs,files={'difference.csv':file_digest(dest/'difference.csv')}))


def analyze(root):
    root=Path(root);out,spec=prepare(root);logs=out/'logs';logs.mkdir(exist_ok=True)
    def job(name):
        env=dict(os.environ,PYTHONPATH=str(root/'src'),OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='2')
        with (logs/(name+'.log')).open('w') as stream:subprocess.run([sys.executable,'-m','statistical_experts.run','joint-contrast','--contrast',name],cwd=root,env=env,stdout=stream,stderr=subprocess.STDOUT,check=True)
        print('joint interval',name,flush=True)
    with ThreadPoolExecutor(max_workers=4) as pool:list(pool.map(job,CONTRASTS))
    atomic_csv(out/'confidence_intervals.csv',pd.concat([pd.read_csv(out/'intervals'/k/'difference.csv').assign(contrast=k) for k in CONTRASTS],ignore_index=True))
    paper_json(out/'analysis_manifest.json',dict(status='completed',files={'confidence_intervals.csv':file_digest(out/'confidence_intervals.csv')}))
