"""冻结六种native单窗Global方法的全量外部验证。"""
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import hashlib,json,os,shutil,subprocess,sys,time
import numpy as np
import pandas as pd
import torch,yaml
from artifacts import atomic_csv,paper_json
from config import config_digest
from reference import file_digest
from data.video import decode_bounded
from data.prefetch import bounded_map,frame_reservation
from features import AlphaStallFeatureExtractor
from evaluation.official_baseline import official_window,official_numpy_score
from evaluation.tables import evaluate_fixed_pairs
from evaluation.bootstrap import paired_source_contrast
from statistical_experts.source_data import select_sources
from statistical_experts.cache import load_feature
from statistical_experts.engine import Engine
from statistical_experts.gaussian import fitted,transitions,energy
from statistical_experts.controls import check_files

RUN='results/runs/global_external'
CACHE='cache/global/external_single_windows'
METHODS=('official','pooled','offline_a','online_b','source_matched','target_matched')
CONTRASTS={'offline_vs_pooled':('offline_a','pooled'),'online_vs_pooled':('online_b','pooled'),
 'offline_vs_official':('offline_a','official'),'online_vs_official':('online_b','official'),
 'target_vs_source':('target_matched','source_matched'),'target_vs_pooled':('target_matched','pooled'),
 'target_vs_official':('target_matched','official')}


def prepare(root):
    root=Path(root);out=root/RUN
    paths=['src/statistical_experts/external_study.py','src/statistical_experts/gaussian.py','src/statistical_experts/engine.py',
           'configs/global_experts.yaml','configs/paper.yaml','src/features.py','src/data/video.py',
           'src/evaluation/official_baseline.py','src/evaluation/tables.py','src/evaluation/bootstrap.py',
           'results/runs/global_expert_controls/cdf_scores_16.npz','results/runs/global_experts/evaluation/cdf_arrays.npz',
           'results/runs/global_experts/models_16/models.pt','results/runs/global_reference_sources/vatex_sources.csv',
           'precomputed/stall_params_vatex_dino_v3.npz']
    paths += [f'data/manifests/active/{d}/{name}.csv' for d in ('genvidbench','vifbench') for name in ('fit','evaluation','pairs')]
    spec=dict(protocol='frozen_native_global_external_v1',length=16,encoder_batch=32,pad_tail=False,query_batch=4,
        target_sources={'genvidbench':115,'vifbench':80},ridge=1e-5,seed=17,methods=list(METHODS),
        files={p:file_digest(root/p) for p in paths})
    marker=out/'identity.json'
    if marker.exists():
        if json.loads(marker.read_text())!=spec:raise ValueError('外部协议身份改变')
        check_files(out,'data_manifest.json');return out,spec
    rows=[];pairs=[]
    for d,n in spec['target_sources'].items():
        fit=select_sources(pd.read_csv(root/f'data/manifests/active/{d}/fit.csv',keep_default_na=False))
        ev=pd.read_csv(root/f'data/manifests/active/{d}/evaluation.csv',keep_default_na=False)
        if len(fit)!=n or set(fit.source_group)&set(ev.source_group):raise ValueError('外部源预算/隔离不符')
        pairs.append(pd.read_csv(root/f'data/manifests/active/{d}/pairs.csv',keep_default_na=False))
        for role,f in [('fit',fit),('evaluation',ev)]:
            for r in f.itertuples(index=False):
                idx=official_window(json.loads(r.downsample_idxs),16);state=(root/r.video_path).stat()
                rows.append(dict(video_id=r.video_id,video_path=r.video_path,dataset=d,subset=r.subset,source_model=r.source_model,
                    source_group=r.source_group,role=role,length=16,frame_indices=json.dumps(idx),
                    cache_key=hashlib.sha256((r.video_path+json.dumps(idx)).encode()).hexdigest(),
                    random_source_key=hashlib.sha256(Path(r.video_path).stem.encode()).hexdigest(),
                    source_bytes=state.st_size,source_mtime_ns=state.st_mtime_ns))
    f=pd.DataFrame(rows);p=pd.concat(pairs,ignore_index=True)
    if len(f)!=2411 or len(f[f.role=='evaluation'])!=2216:raise ValueError('外部覆盖不同')
    out.mkdir(parents=True,exist_ok=True);atomic_csv(out/'windows.csv',f);atomic_csv(out/'pairs.csv',p)
    for x in paths:
        if x.startswith(('src/','configs/')):
            dest=out/'source_snapshot'/x;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(root/x,dest)
    paper_json(marker,spec);paper_json(out/'data_manifest.json',dict(files={x:file_digest(out/x) for x in ['windows.csv','pairs.csv']},
        status='prepared',evaluation_clips=2216,target_fit=195,meaning='已被历史研究观察的外部域，不称untouched'))
    print('external prepared',len(f),flush=True);return out,spec


def extract(root,rank,world):
    root=Path(root);out,spec=prepare(root);identity=config_digest(spec);directory=root/CACHE;directory.mkdir(parents=True,exist_ok=True)
    rows=list(pd.read_csv(out/'windows.csv',keep_default_na=False).itertuples(index=False))[rank::world];pending=[]
    for r in rows:
        p=directory/(r.cache_key+'.npz')
        if p.exists():load_feature(p,identity,r)
        else:pending.append(r)
    if not pending:return
    cfg=yaml.safe_load((root/'configs/paper.yaml').read_text());enc=cfg['encoder']
    if file_digest(root/enc['weights'])!=enc['weights_sha256']:raise ValueError('权重改变')
    model=AlphaStallFeatureExtractor(f'cuda:{rank%2}',dino_repo=str(root/enc['repo']),dino_weights=str(root/enc['weights']),pad_tail_batch=False)
    stream=bounded_map(pending,lambda r:model.prepare_frames(decode_bounded(root/r.video_path,json.loads(r.frame_indices))),
        lambda r:frame_reservation(root/r.video_path,16),workers=4,depth=6,budget=2*2**30)
    start=time.perf_counter()
    try:
        for i,(r,frames) in enumerate(stream,1):
            g=model.frames_to_global_embeddings([frames],batch_size=32)[0];state=(root/r.video_path).stat()
            if (state.st_size,state.st_mtime_ns)!=(r.source_bytes,r.source_mtime_ns):raise ValueError('原视频改变')
            p=directory/(r.cache_key+'.npz');tmp=p.with_suffix('.tmp.npz')
            np.savez(tmp,global_features=g,video_id=np.asarray(r.video_id),identity=np.asarray(identity),
                frame_indices=np.asarray(json.loads(r.frame_indices)),tensor_sha256=np.asarray(hashlib.sha256(g.tobytes()).hexdigest()))
            tmp.replace(p)
            if i%64==0 or i==len(pending):print(f'[external cache {rank}] {i}/{len(pending)} {time.perf_counter()-start:.1f}s',flush=True)
    finally:stream.close()
    paper_json(out/f'cache_rank_{rank}.json',dict(identity=identity,status='completed',new_windows=len(pending),seconds=time.perf_counter()-start))


def score(root,device):
    root=Path(root);out,spec=prepare(root);identity=config_digest(spec)
    if (out/'raw_manifest.json').exists():check_files(out,'raw_manifest.json');return
    start=time.perf_counter();torch.set_num_threads(4);engine=Engine(root,16,device)
    frame=pd.read_csv(out/'windows.csv',keep_default_na=False)
    def read(r):return load_feature(root/CACHE/(r.cache_key+'.npz'),identity,r)
    with ThreadPoolExecutor(max_workers=4) as pool:g=np.stack(list(pool.map(read,frame.itertuples(index=False))))
    fit=frame.role.eq('fit').to_numpy();ev=frame.role.eq('evaluation').to_numpy();models={}
    source_ids=pd.read_csv(root/'results/runs/global_reference_sources/vatex_sources.csv').source_id.tolist()
    mapping={Path(p).stem:i for i,p in enumerate(engine.bank['frame'].video_path)}
    source_idx=[mapping[x] for x in source_ids]
    for d,n in spec['target_sources'].items():
        target=(frame.dataset.eq(d).to_numpy()&fit);t,_=transitions(torch.from_numpy(g[target]))
        models[d+'__target_matched']=fitted(t.flatten(0,1).to(device))
        models[d+'__source_matched']=fitted(engine.bank['temporal'][source_idx[:n]].flatten(0,1))
    model_file=out/'target_models.pt';torch.save({k:{a:b.cpu() for a,b in v.items() if a in ('mean','chol')} for k,v in models.items()},model_file)
    # 新来源模型各自重算同一独立VATEX2000参考，不改变冻结专家参考。
    old=pd.read_csv(root/'data/manifests/global_experts/windows.csv',keep_default_na=False)
    refs=old[(old.length==16)&old.role.eq('cdf')].reset_index(drop=True)
    from statistical_experts.cache import feature_identity
    from statistical_experts.manifests import settings
    bid=config_digest(feature_identity(root,settings(root)));directory=root/settings(root)['cache_directory']
    with ThreadPoolExecutor(max_workers=4) as pool:rg=np.stack(list(pool.map(lambda r:load_feature(directory/(r.cache_key+'.npz'),bid,r),refs.itertuples(index=False))))
    cdfraw=np.empty((len(refs),len(models)));keys=list(models)
    for i in range(0,len(refs),4):
        t,z=transitions(torch.from_numpy(rg[i:i+4]));t=t.to(device);z=z.to(device)
        for j,key in enumerate(keys):
            m=models[key];cdfraw[i:i+4,j]=energy(t,m['mean'],m['chol']).masked_fill(z,float('inf')).min(-1).values.cpu().numpy()
    raws=[]
    with np.load(root/'precomputed/stall_params_vatex_dino_v3.npz') as z:official={k:z[k].copy() for k in z.files}
    ef=frame[ev].reset_index(drop=True);eg=g[ev]
    for i in range(0,len(ef),4):
        actual=eg[i:i+4];x=list(actual);r=ef.iloc[i:i+4];random=r.random_source_key.tolist()
        while len(x)<4:x.append(x[-1]);random.append(random[-1])
        tensor=torch.from_numpy(np.stack(x));values,route,_=engine.score(tensor,random)
        t,z=transitions(tensor);t=t.to(device);z=z.to(device)
        extra={}
        for key,m in models.items():extra[key]=energy(t,m['mean'],m['chol']).masked_fill(z,float('inf')).min(-1).values.cpu().numpy()
        for j,row in enumerate(r.itertuples(index=False)):
            o=official_numpy_score(actual[j][None],official)
            raws.append(dict(video_id=row.video_id,cluster=int(route[j]),gs=float(o['global_spatial_raw'][0]),gt=float(o['global_temporal_raw'][0]),
                official=float(o['final_score'][0]),pooled=float(values['pooled'][j,1]),offline=float(values['offline'][j,1]),online=float(values['online'][j,1]),
                **{label:float(extra[row.dataset+'__'+label][j]) for label in ('source_matched','target_matched')}))
        if (i+len(r))%200==0 or i+len(r)==len(ef):print('external score',i+len(r),len(ef),flush=True)
    atomic_csv(out/'raw.csv',pd.DataFrame(raws));np.savez(out/'source_cdf_raw.npz',scores=cdfraw,model_names=np.asarray(keys),video_ids=refs.video_id.to_numpy(dtype=str))
    paper_json(out/'raw_manifest.json',dict(identity=identity,status='completed',seconds=time.perf_counter()-start,
        files={p:file_digest(out/p) for p in ['raw.csv','source_cdf_raw.npz','target_models.pt']}))


def evaluate(root):
    root=Path(root);out,spec=prepare(root);check_files(out,'raw_manifest.json')
    if (out/'evaluation_manifest.json').exists():check_files(out,'evaluation_manifest.json');return
    frame=pd.read_csv(out/'windows.csv',keep_default_na=False);frame=frame[frame.role.eq('evaluation')].set_index('video_id')
    raw=pd.read_csv(out/'raw.csv',float_precision='round_trip').set_index('video_id').loc[frame.index]
    with np.load(root/'precomputed/stall_params_vatex_dino_v3.npz') as z:
        sc=np.sort(z['calib_ll_spat'].max(1));tc=np.sort(z['calib_ll_temp'].min(1))
    spatial=np.searchsorted(sc,raw.gs,side='right')/len(sc)
    pct={'official':np.searchsorted(tc,raw.gt,side='right')/len(tc)}
    np.testing.assert_array_equal(.5*spatial+.5*pct['official'],raw.official.to_numpy())
    with np.load(root/'results/runs/global_experts/evaluation/cdf_arrays.npz') as z:
        for m in ('pooled','online_b'):
            label='online' if m=='online_b' else m;ref=z[label+'_16'][:,1]
            pct[m]=np.searchsorted(ref,raw[label],side='right')/len(ref)
    with np.load(root/'results/runs/global_expert_controls/cdf_scores_16.npz') as z:
        pct['offline_a']=np.empty(len(raw))
        for k in range(4):
            mask=raw.cluster.eq(k);ref=np.sort(z['scores'][:,k]);pct['offline_a'][mask]=np.searchsorted(ref,raw.loc[mask,'offline'],side='right')/len(ref)
    with np.load(out/'source_cdf_raw.npz') as z:
        for label in ('source_matched','target_matched'):
            pct[label]=np.empty(len(raw))
            for d in spec['target_sources']:
                mask=frame.dataset.eq(d);ref=np.sort(z['scores'][:,z['model_names'].tolist().index(d+'__'+label)])
                pct[label][mask]=np.searchsorted(ref,raw.loc[mask,label],side='right')/len(ref)
    scores=[];tables={};pairs=pd.read_csv(out/'pairs.csv')
    for method in METHODS:
        for branch,values in [('temporal',pct[method]),('final',.5*spatial+.5*pct[method])]:
            q=frame.copy();q['final_score']=values;q['variant']=method+'_'+branch;q['method']=method;q['branch']=branch;q=q.reset_index();scores.append(q)
            for key,t in evaluate_fixed_pairs(q,pairs).items():tables.setdefault(key,[]).append(t.assign(variant=method+'_'+branch))
    pd.concat(scores,ignore_index=True).to_csv(out/'video_scores.csv.gz',index=False)
    for key,t in tables.items():atomic_csv(out/(key+'.csv'),pd.concat(t,ignore_index=True))
    paper_json(out/'evaluation_manifest.json',dict(status='point_estimates_complete',files={p:file_digest(out/p) for p in ['video_scores.csv.gz']+[k+'.csv' for k in tables]}))
    f=pd.read_csv(out/'dataset_metrics.csv');print(f[f.variant.str.endswith('_final')][['dataset','variant','auc','real_positive_ap']].to_string(index=False),flush=True)


def contrast(root,name):
    root=Path(root);out,spec=prepare(root);check_files(out,'evaluation_manifest.json');dest=out/'intervals'/name
    if (dest/'manifest.json').exists():check_files(dest,'manifest.json');return
    scores=pd.read_csv(out/'video_scores.csv.gz',float_precision='round_trip');pairs=pd.read_csv(out/'pairs.csv')
    a,b=CONTRASTS[name];groups=scores[['video_id','source_group']].drop_duplicates().set_index('video_id').source_group
    point,ci=paired_source_contrast(scores[scores.variant.eq(a+'_final')],scores[scores.variant.eq(b+'_final')],pairs,groups,iterations=1000,seed=17)
    dest.mkdir(parents=True,exist_ok=True);atomic_csv(dest/'difference.csv',point.merge(ci,on=['dataset','metric']))
    paper_json(dest/'manifest.json',dict(status='completed',candidate=a,baseline=b,iterations=1000,seed=17,scores_sha256=file_digest(out/'video_scores.csv.gz'),files={'difference.csv':file_digest(dest/'difference.csv')}))


def analyze(root):
    root=Path(root);out,spec=prepare(root);logs=out/'logs';logs.mkdir(exist_ok=True)
    def job(name):
        env=dict(os.environ,PYTHONPATH=str(root/'src'),OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='2')
        with (logs/(name+'.log')).open('w') as stream:subprocess.run([sys.executable,'-m','statistical_experts.run','external-contrast','--contrast',name],cwd=root,env=env,stdout=stream,stderr=subprocess.STDOUT,check=True)
        print('external interval',name,flush=True)
    with ThreadPoolExecutor(max_workers=4) as pool:list(pool.map(job,CONTRASTS))
    atomic_csv(out/'confidence_intervals.csv',pd.concat([pd.read_csv(out/'intervals'/k/'difference.csv').assign(contrast=k) for k in CONTRASTS],ignore_index=True))
    paper_json(out/'analysis_manifest.json',dict(status='completed',files={'confidence_intervals.csv':file_digest(out/'confidence_intervals.csv')}))
