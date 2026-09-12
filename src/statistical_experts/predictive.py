"""相邻T1条件预测，边际/跨源打乱同复杂度控制；只拟合真实统计。"""
from concurrent.futures import ThreadPoolExecutor
import json,os,shutil,subprocess,sys,time
from pathlib import Path
import numpy as np
import pandas as pd
import torch,yaml
from artifacts import atomic_csv,paper_json
from config import config_digest
from reference import file_digest
from data.prefetch import bounded_map
from evaluation.tables import evaluate_fixed_pairs
from evaluation.bootstrap import paired_source_contrast
from statistical_experts.cache import load_feature,feature_identity
from statistical_experts.engine import load_bank
from statistical_experts.manifests import settings
from statistical_experts.gaussian import fitted,energy,transitions
from statistical_experts.moment_controls import raw_order_scores
from statistical_experts.controls import check_files

MODELS=('marginal','predictive','shuffled')
CONTRASTS={
 'predictive_final_vs_marginal':('predictive_final','marginal_final'),
 'predictive_final_vs_shuffled':('predictive_final','shuffled_final'),
 'predictive_raw_vs_marginal':('predictive_raw','marginal_raw'),
 'predictive_raw_vs_shuffled':('predictive_raw','shuffled_raw'),
 'predictive_temporal_vs_marginal':('predictive_temporal','marginal_temporal'),
 'predictive_final_vs_official':('predictive_final','official_final'),
}


def derangement(n,seed=17):
    if n<2:raise ValueError('跨源配对至少需要2个源')
    order=np.random.default_rng(seed).permutation(n);result=np.empty(n,dtype=np.int64)
    result[order]=np.roll(order,1)
    return result


def fit_model(t,mode,ridge=1e-5,seed=17):
    """行向量回归：x @ B。每视频固定相同转移对数，零向量保留拟合。"""
    if t.ndim!=3 or t.shape[1]<2 or mode not in MODELS:raise ValueError('条件拟合输入错误')
    t=t.double();x=t[:,:-1];y=t[:,1:];pairing=np.arange(len(t))
    if mode=='shuffled':
        pairing=derangement(len(t),seed);y=y[torch.as_tensor(pairing,device=t.device)]
    x=x.flatten(0,1);y=y.flatten(0,1);mx=x.mean(0);my=y.mean(0);d=x.shape[1]
    if mode=='marginal':b=torch.zeros((d,d),device=t.device,dtype=torch.float64)
    else:
        xc=x-mx;yc=y-my;den=len(x)-1
        cxx=xc.T@xc/den+ridge*torch.eye(d,device=t.device,dtype=torch.float64)
        cxy=xc.T@yc/den;b=torch.linalg.solve(cxx,cxy)
    residual=y-my-(x-mx)@b;model=fitted(residual,ridge=ridge)
    return dict(x_mean=mx,y_mean=my,B=b,residual_mean=model['mean'],chol=model['chol'],pairing=torch.as_tensor(pairing),
                training_pairs=torch.as_tensor(len(x)))


def position_scores(t,model):
    residual=t[...,1:,:].double()-model['y_mean']-(t[...,:-1,:].double()-model['x_mean'])@model['B']
    return energy(residual,model['residual_mean'],model['chol'])


def query_scores(g,models,device):
    t,z=transitions(g);valid=~(z[:,:-1]|z[:,1:]);t=t.to(device);valid=valid.to(device)
    scores={key:position_scores(t,m).masked_fill(~valid,float('inf')).min(-1).values.detach().cpu().numpy() for key,m in models.items()}
    return scores,valid.sum(-1).cpu().numpy()


def prepare(root):
    root=Path(root);c=yaml.safe_load((root/'configs/global_predictive.yaml').read_text())
    expected={'protocol','source_directory','external_directory','run_directory','dimension','ridge','seed','holdout_sources',
              'query_batch','bootstrap_iterations','bootstrap_seed','analysis_workers'}
    if set(c)!=expected or (c['dimension'],c['ridge'],c['seed'],c['holdout_sources'],c['query_batch'])!=(1024,1e-5,17,440,4):raise ValueError('预测协议配置变化')
    paths=['configs/global_predictive.yaml','configs/global_experts.yaml','configs/paper.yaml','src/statistical_experts/predictive.py',
           'src/statistical_experts/gaussian.py','src/statistical_experts/cache.py','src/statistical_experts/engine.py',
           'src/evaluation/tables.py','src/evaluation/metrics.py','src/evaluation/bootstrap.py',
           'data/manifests/global_experts/windows.csv','data/manifests/global_experts/pairs.csv',
           c['source_directory']+'/evaluation/video_scores.csv.gz',c['external_directory']+'/identity.json',
           c['external_directory']+'/windows.csv',c['external_directory']+'/pairs.csv',c['external_directory']+'/video_scores.csv.gz',
           c['external_directory']+'/raw.csv','precomputed/stall_params_vatex_dino_v3.npz']
    spec=dict(config=c,files={p:file_digest(root/p) for p in paths},feature_identity=config_digest(feature_identity(root,settings(root))),
        mask='query: both adjacent T1 nonzero; all three use identical predicted positions; fit/heldout NLL retain zero vectors',
        model='x@B + intercept; ridge on Cxx and residual covariance; no PCA/no MoE',
        evaluation='real high score; independent model CDF; GS official fixed; 0.5/0.5')
    out=root/c['run_directory'];marker=out/'identity.json'
    if marker.exists():
        if json.loads(marker.read_text())!=spec:raise ValueError('预测输入或源码变化，不能恢复同run')
    else:
        out.mkdir(parents=True,exist_ok=True)
        for p in paths:
            if p.startswith(('src/','configs/')):
                dest=out/'source_snapshot'/p;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(root/p,dest)
        paper_json(marker,spec);paper_json(out/'status.json',dict(status='prepared',new_predictive_experts=False))
    return out,spec


def fit(root,length,device):
    root=Path(root);out,spec=prepare(root);c=spec['config'];name=f'models_{length}'
    if (out/(name+'.json')).exists():check_files(out,name+'.json');return
    torch.set_num_threads(4);start=time.perf_counter();bank=load_bank(root,length,device);t=bank['temporal']
    if len(t)!=2200:raise ValueError('真实拟合池不是2200源')
    permutation=np.random.default_rng(17).permutation(2200);val=np.sort(permutation[:440]);train=np.sort(permutation[440:])
    models={};heldout=[];info=[]
    for name_model in MODELS:
        m=fit_model(t,name_model,c['ridge'],17);models[name_model]={k:v.cpu() for k,v in m.items()}
        hm=fit_model(t[train],name_model,c['ridge'],17)
        nll=(-position_scores(t[val],hm)+hm['chol'].diagonal().log().sum()).mean(1).cpu().numpy()
        for i,v in zip(val,nll):heldout.append(dict(video_id=bank['ids'][i],length=length,model=name_model,nll=float(v)))
        info.append(dict(model=name_model,B_frobenius=float(m['B'].norm()),residual_logdet=float(2*m['chol'].diagonal().log().sum()),pairs=int(m['training_pairs'])))
    temp=out/f'models_{length}.tmp.pt';torch.save(models,temp);temp.replace(out/f'models_{length}.pt')
    atomic_csv(out/f'heldout_{length}.csv',pd.DataFrame(heldout));atomic_csv(out/f'model_diagnostics_{length}.csv',pd.DataFrame(info))
    paper_json(out/f'fit_sources_{length}.json',dict(fit_ids=bank['ids'],heldout_train=[bank['ids'][i] for i in train],heldout_validation=[bank['ids'][i] for i in val],
         pairing_by_model={k:v['pairing'].tolist() for k,v in models.items()}))
    files=[f'models_{length}.pt',f'heldout_{length}.csv',f'model_diagnostics_{length}.csv',f'fit_sources_{length}.json']
    paper_json(out/f'models_{length}.json',dict(identity=config_digest(spec),status='completed',seconds=time.perf_counter()-start,files={p:file_digest(out/p) for p in files}))
    print('predictive fitted',length,pd.DataFrame(heldout).groupby('model').nll.mean().to_dict(),flush=True)


def score(root,length,device):
    root=Path(root);out,spec=prepare(root);c=spec['config'];check_files(out,f'models_{length}.json')
    if (out/f'raw_{length}.json').exists():check_files(out,f'raw_{length}.json');return
    torch.set_num_threads(4);start=time.perf_counter();models=torch.load(out/f'models_{length}.pt',map_location=device,weights_only=True)
    f=pd.read_csv(root/'data/manifests/global_experts/windows.csv',keep_default_na=False)
    f=f[f.length.eq(length)&f.role.ne('fit')].copy();f['feature_identity']=spec['feature_identity'];f['cache_directory']=settings(root)['cache_directory'];f['scope']='development'
    if length==16:
        ex=pd.read_csv(root/c['external_directory']/'windows.csv',keep_default_na=False);ex=ex[ex.role.eq('evaluation')].copy()
        ex['feature_identity']=config_digest(json.loads((root/c['external_directory']/'identity.json').read_text()))
        ex['cache_directory']='cache/global/external_single_windows';ex['scope']='external';f=pd.concat([f,ex],ignore_index=True)
    f=f.reset_index(drop=True);rows=list(f.itertuples(index=False));batch=4;items=[rows[i:i+batch] for i in range(0,len(rows),batch)]
    def load(group):
        xs=[load_feature(root/r.cache_directory/(r.cache_key+'.npz'),r.feature_identity,r) for r in group]
        while len(xs)<batch:xs.append(xs[-1])
        return torch.from_numpy(np.stack(xs))
    values=np.empty((len(rows),3));valid=np.empty(len(rows),int);done=0
    stream=bounded_map(items,load,lambda x:batch*length*1024*4,workers=4,depth=8,budget=128*2**20)
    try:
        for group,g in stream:
            result,count=query_scores(g,models,device);n=len(group)
            for j,m in enumerate(MODELS):values[done:done+n,j]=result[m][:n]
            valid[done:done+n]=count[:n];done+=n
            if done%1000==0 or done==len(rows):print('prediction',length,done,len(rows),round(time.perf_counter()-start,1),flush=True)
    finally:stream.close()
    atomic_csv(out/f'queries_{length}.csv',f)
    temp=out/f'raw_{length}.tmp.npz';np.savez(temp,video_ids=f.video_id.to_numpy(dtype=str),scores=values,valid_pairs=valid);temp.replace(out/f'raw_{length}.npz')
    paper_json(out/f'raw_{length}.json',dict(status='completed',identity=config_digest(spec),seconds=time.perf_counter()-start,rows=len(rows),files={p:file_digest(out/p) for p in [f'queries_{length}.csv',f'raw_{length}.npz']}))


def evaluate(root):
    root=Path(root);out,spec=prepare(root);c=spec['config']
    if (out/'evaluation_manifest.json').exists():check_files(out,'evaluation_manifest.json');return
    dev=pd.read_csv(root/c['source_directory']/'evaluation/video_scores.csv.gz',float_precision='round_trip')
    ex=pd.read_csv(root/c['external_directory']/'video_scores.csv.gz',float_precision='round_trip')
    # 外部单GS从官方空间raw与官方参考直接恢复，避免由Final反推的浮点末位误差。
    gs=dev[dev.variant.eq('official_spatial')].set_index('video_id').final_score
    er=pd.read_csv(root/c['external_directory']/'raw.csv',float_precision='round_trip').set_index('video_id')
    with np.load(root/'precomputed/stall_params_vatex_dino_v3.npz') as z:sc=np.sort(z['calib_ll_spat'].max(1))
    es=pd.Series(np.searchsorted(sc,er['gs'],side='right')/len(sc),index=er.index)
    gs=pd.concat([gs,es]);official=pd.concat([dev[dev.variant.eq('official_final')],ex[ex.variant.eq('official_final')]]).set_index('video_id')
    outputs=[];refs={}
    for length in (8,16):
        check_files(out,f'raw_{length}.json');f=pd.read_csv(out/f'queries_{length}.csv',keep_default_na=False)
        with np.load(out/f'raw_{length}.npz') as z:v=z['scores'];counts=z['valid_pairs'];np.testing.assert_array_equal(z['video_ids'],f.video_id.to_numpy())
        ref=f.role.eq('cdf').to_numpy();ev=f.role.eq('evaluation').to_numpy();ids=f.loc[ev,'video_id'];meta=f[ev].copy()
        if ref.sum()!=2000:raise ValueError('CDF必须独立2000源')
        for j,m in enumerate(MODELS):
            r=np.sort(v[ref,j]);refs[m+'_'+str(length)]=r;pct=np.searchsorted(r,v[ev,j],side='right')/len(r)
            for branch,s in [('raw',v[ev,j]),('temporal',pct),('final',.5*gs.loc[ids].to_numpy()+.5*pct)]:
                q=meta.copy();q['final_score']=s;q['raw_score']=v[ev,j];q['valid_pairs']=counts[ev];q['variant']=m+'_'+branch;q['method']=m;q['branch']=branch;outputs.append(q)
        q=meta.copy();q['final_score']=official.loc[ids,'final_score'].to_numpy();q['raw_score']=np.nan;q['valid_pairs']=counts[ev];q['variant']='official_final';q['method']='official';q['branch']='final';outputs.append(q)
    scores=pd.concat(outputs,ignore_index=True)
    for m in MODELS:
        mask=scores.variant.eq(m+'_raw');scores.loc[mask,'final_score']=raw_order_scores(scores.loc[mask,'raw_score'])
    scores.to_csv(out/'video_scores.csv.gz',index=False);np.savez(out/'cdf_arrays.npz',**refs)
    pairs=pd.concat([pd.read_csv(root/'data/manifests/global_experts/pairs.csv'),pd.read_csv(root/c['external_directory']/'pairs.csv')],ignore_index=True)
    atomic_csv(out/'pairs.csv',pairs);tables={}
    for name,q in scores.groupby('variant',sort=False):
        for key,t in evaluate_fixed_pairs(q,pairs).items():tables.setdefault(key,[]).append(t.assign(variant=name))
    for key,t in tables.items():atomic_csv(out/(key+'.csv'),pd.concat(t,ignore_index=True).replace({'scope':{'Macro-3':'Average'}}))
    held=pd.concat([pd.read_csv(out/f'heldout_{l}.csv') for l in (8,16)],ignore_index=True)
    atomic_csv(out/'heldout_summary.csv',held.groupby(['length','model'],as_index=False).nll.mean())
    files=['video_scores.csv.gz','pairs.csv','cdf_arrays.npz','heldout_summary.csv']+[k+'.csv' for k in tables]
    paper_json(out/'evaluation_manifest.json',dict(status='point_estimates_complete',files={p:file_digest(out/p) for p in files}))
    print(pd.read_csv(out/'macro_metrics.csv')[['variant','auc','real_positive_ap']].to_string(index=False),flush=True)
    d=pd.read_csv(out/'dataset_metrics.csv');print(d[d.dataset.isin(['genvidbench','vifbench'])&d.variant.str.endswith('_final')][['dataset','variant','auc','real_positive_ap']].to_string(index=False),flush=True)


def contrast(root,name):
    root=Path(root);out,spec=prepare(root);check_files(out,'evaluation_manifest.json');dest=out/'intervals'/name;a,b=CONTRASTS[name]
    inputs=dict(scores=file_digest(out/'video_scores.csv.gz'),pairs=file_digest(out/'pairs.csv'),candidate=a,baseline=b,iterations=1000,seed=17)
    if (dest/'manifest.json').exists():
        if check_files(dest,'manifest.json')['inputs']!=inputs:raise ValueError('区间身份不同')
        return
    s=pd.read_csv(out/'video_scores.csv.gz',float_precision='round_trip');pairs=pd.read_csv(out/'pairs.csv');groups=s[['video_id','source_group']].drop_duplicates().set_index('video_id').source_group
    point,ci=paired_source_contrast(s[s.variant.eq(a)],s[s.variant.eq(b)],pairs,groups,iterations=1000,seed=17)
    dest.mkdir(parents=True,exist_ok=True);atomic_csv(dest/'difference.csv',point.merge(ci,on=['dataset','metric']).replace({'dataset':{'Macro-3':'Average'}}))
    paper_json(dest/'manifest.json',dict(status='completed',inputs=inputs,files={'difference.csv':file_digest(dest/'difference.csv')}))


def analyze(root):
    root=Path(root);out,spec=prepare(root);logs=out/'logs';logs.mkdir(exist_ok=True)
    def job(name):
        env=dict(os.environ,PYTHONPATH=str(root/'src'),OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='2')
        with (logs/(name+'.log')).open('w') as stream:subprocess.run([sys.executable,'-m','statistical_experts.run','predictive-contrast','--contrast',name],cwd=root,env=env,stdout=stream,stderr=subprocess.STDOUT,check=True)
        print('predictive interval',name,flush=True)
    with ThreadPoolExecutor(max_workers=spec['config']['analysis_workers']) as pool:list(pool.map(job,CONTRASTS))
    atomic_csv(out/'confidence_intervals.csv',pd.concat([pd.read_csv(out/'intervals'/k/'difference.csv').assign(contrast=k) for k in CONTRASTS],ignore_index=True))
    paper_json(out/'analysis_manifest.json',dict(status='completed',files={'confidence_intervals.csv':file_digest(out/'confidence_intervals.csv')}))
