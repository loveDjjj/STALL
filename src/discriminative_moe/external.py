"""冻结pooled监督分类器的外部核对，不拟合目标real或改超参数。"""
import json,time
from pathlib import Path
import numpy as np
import pandas as pd
import torch,yaml
from artifacts import atomic_csv,paper_json
from config import config_digest
from reference import file_digest
from features import AlphaStallFeatureExtractor
from math_utils import StableGaussianParams
from branches.global_branch import score_global_raw
from data.video import decode_bounded
from data.prefetch import bounded_map,frame_reservation
from evaluation.tables import evaluate_fixed_pairs
from discriminative_moe.run import configuration
from discriminative_moe.features import projection,global_descriptors,local_descriptors
from discriminative_moe.evaluation import load_classifier,predict_diagnostics
from discriminative_moe.training import score_frame
from discriminative_moe.analysis import paired_seed_contrast


def prepare(root):
    root=Path(root);c=configuration(root);parent=root/c['run_directory'];out=parent/'external';out.mkdir(exist_ok=True)
    sources=['genvidbench','vifbench'];paths=['results/reference/baseline/window_scores.csv.gz']
    for d in sources:paths.extend([f'data/manifests/active/{d}/evaluation.csv',f'data/manifests/active/{d}/pairs.csv'])
    identity=dict(inputs={p:file_digest(root/p) for p in paths},selection=file_digest(parent/'selection.json'),code=file_digest(Path(__file__)))
    marker=out/'prepared.json'
    if marker.exists():
        if json.loads(marker.read_text())['identity']!=identity:raise ValueError('外部准备身份变化')
        return out
    f=pd.concat([pd.read_csv(root/f'data/manifests/active/{d}/evaluation.csv',keep_default_na=False) for d in sources],ignore_index=True)
    f['split_group']=f.source_group
    dev=pd.read_csv(parent/'videos.csv');assert not set(f.video_id)&set(dev.video_id)
    dev_files=set()
    for p in dev.video_path.unique():
        s=(root/p).stat();dev_files.add((s.st_dev,s.st_ino))
    raw=pd.read_csv(root/paths[0],float_precision='round_trip');jobs=[]
    for r in f.itertuples():
        s=(root/r.video_path).stat()
        if (s.st_dev,s.st_ino) in dev_files:raise ValueError('外部与开发共享物理文件')
        q=raw[raw.video_id.eq(r.video_id)].sort_values('window_id')
        if not len(q) or len(q)>3:raise ValueError('外部FC窗口缺失')
        windows=[json.loads(x) for x in q.frame_indices]
        if any(len(w)!=16 for w in windows):raise ValueError('当前外部应为16帧协议')
        jobs.append(dict(video_id=r.video_id,dataset=r.dataset,video_path=r.video_path,key=config_digest(dict(video_id=r.video_id,windows=windows)),
            windows=windows,source_bytes=s.st_size,source_mtime_ns=s.st_mtime_ns,
            gs=q.global_spatial_raw.tolist(),gt=q.global_temporal_raw.tolist()))
    if len(jobs)!=2216:raise ValueError('外部身份范围改变')
    atomic_csv(out/'videos.csv',f);atomic_csv(out/'pairs.csv',pd.concat([pd.read_csv(root/f'data/manifests/active/{d}/pairs.csv') for d in sources],ignore_index=True))
    paper_json(out/'jobs.json',dict(jobs=jobs));paper_json(marker,dict(status='prepared',identity=identity,
        files={p:file_digest(out/p) for p in ['videos.csv','pairs.csv','jobs.json']}))
    return out


def extract(root,args):
    root=Path(root);c=configuration(root);out=prepare(root);cfg=yaml.safe_load((root/'configs/paper.yaml').read_text())
    directory=out/'features';directory.mkdir(exist_ok=True)
    spec=dict(prepared=file_digest(out/'prepared.json'),encoder=cfg['encoder'],
        code={p:file_digest(root/p) for p in ['src/discriminative_moe/external.py','src/discriminative_moe/features.py','src/features.py']},
        reference_checks={d:file_digest(root/f'precomputed/target_reference/{d}.npz') for d in ['genvidbench','vifbench']})
    identity=config_digest(spec);marker=directory/'identity.json'
    if marker.exists():
        if json.loads(marker.read_text())!=spec:raise ValueError('外部提取合同改变')
    else:paper_json(marker,spec)
    jobs=json.loads((out/'jobs.json').read_text())['jobs']
    if args.pilot:
        selected=[]
        for domain in ['genvidbench','vifbench']:
            selected.extend([j for j in jobs if j['dataset']==domain][:2])
        jobs=selected
    else:jobs=[j for i,j in enumerate(jobs) if i%args.world_size==args.rank]
    pending=[]
    for j in jobs:
        p=directory/(j['key']+'.npz')
        if p.exists():
            with np.load(p) as z:
                if str(z['identity'])!=identity:raise ValueError('外部检查点身份变化')
        else:pending.append(j)
    if not pending:return
    device=f'cuda:{args.rank%2}';basis=projection().to(device)
    if file_digest(root/cfg['encoder']['weights'])!=cfg['encoder']['weights_sha256']:raise ValueError('骨干改变')
    extractor=AlphaStallFeatureExtractor(device,dino_repo=str(root/cfg['encoder']['repo']),dino_weights=str(root/cfg['encoder']['weights']),pad_tail_batch=True)
    refs={}
    for d in ['genvidbench','vifbench']:
        with np.load(root/f'precomputed/target_reference/{d}.npz') as z:
            refs[d]=[StableGaussianParams(z[b+'_mean'],z[b+'_whitening'],z[b+'_cdf']) for b in ['gs','gt']]
    def load(j):
        path=root/j['video_path'];s=path.stat()
        if (s.st_size,s.st_mtime_ns)!=(j['source_bytes'],j['source_mtime_ns']):raise ValueError('外部原视频改变')
        return extractor.prepare_frames(decode_bounded(path,sorted({i for w in j['windows'] for i in w})))
    start=time.perf_counter();error=0.
    stream=bounded_map(pending,load,lambda j:frame_reservation(root/j['video_path'],len({i for w in j['windows'] for i in w})),workers=4,depth=6,budget=4*2**30)
    try:
        for i,(j,frames) in enumerate(stream,1):
            f=extractor.frames_to_global_patch_embeddings([frames],batch_size=8)[0]
            indices=sorted({i for w in j['windows'] for i in w});where={x:i for i,x in enumerate(indices)};pick=[[where[x] for x in w] for w in j['windows']]
            g=np.stack([f['global'][ix] for ix in pick]);patch=torch.from_numpy(np.stack([f['patch'][ix] for ix in pick])).to(device)
            # 目标参考仅用于旧raw回归核验，不进入分类头输入、选择或预测。
            raw=score_global_raw(g,*refs[j['dataset']],device=device)
            for a,b in [(raw.spatial,j['gs']),(raw.temporal_t1,j['gt'])]:
                b=np.asarray(b,dtype=np.float64)
                np.testing.assert_allclose(a,b,rtol=0,atol=1e-8)
                finite=np.isfinite(a)&np.isfinite(b)
                if finite.any():error=max(error,float(np.abs(np.asarray(a)[finite]-np.asarray(b)[finite]).max()))
            a,t=global_descriptors(g,basis.cpu());l=local_descriptors(patch,basis).cpu().numpy()
            dest=directory/(j['key']+'.npz');temp=dest.with_suffix('.tmp.npz')
            np.savez(temp,G=a,T=t,L=l,identity=np.asarray(identity),video_id=np.asarray(j['video_id']));temp.replace(dest)
            if i%32==0 or i==len(pending):
                paper_json(directory/f'progress_{args.rank}.json',dict(completed=i,pending=len(pending),seconds=time.perf_counter()-start,max_global_raw_error=error))
                print('external features',args.rank,i,len(pending),flush=True)
    finally:stream.close()
    paper_json(directory/(f'pilot_{args.rank}.json' if args.pilot else f'completed_{args.rank}.json'),dict(status='completed',identity=identity,jobs=len(pending),seconds=time.perf_counter()-start,max_global_raw_error=error))


def evaluate(root,args):
    root=Path(root);c=configuration(root);parent=root/c['run_directory'];out=prepare(root)
    selection=json.loads((parent/'selection.json').read_text());chosen=pd.DataFrame(selection['selection']);chosen=chosen[chosen.fold=='pooled']
    meta=pd.read_csv(out/'videos.csv');pairs=pd.read_csv(out/'pairs.csv');jobs=json.loads((out/'jobs.json').read_text())['jobs']
    blocks={k:[] for k in ['G','T','L']};mask=[];hashes={};identity=config_digest(json.loads((out/'features/identity.json').read_text()))
    for j in jobs:
        p=out/'features'/(j['key']+'.npz');hashes[j['key']]=file_digest(p)
        with np.load(p) as z:
            if str(z['identity'])!=identity or str(z['video_id'])!=j['video_id']:raise ValueError('外部特征身份不符')
            k=len(z['G']);mask.append(np.arange(3)<k)
            for name in blocks:
                x=np.zeros((3,z[name].shape[-1]),np.float32);x[:k]=z[name];blocks[name].append(x)
    blocks={k:np.stack(v) for k,v in blocks.items()};mask=np.stack(mask);all_scores=[];tables={}
    for inp,names in [('G',['G']),('GT',['G','T']),('GTL',['G','T','L'])]:
        x=np.concatenate([blocks[n] for n in names],axis=-1);sel=chosen[chosen.input==inp].set_index('kind').experts.to_dict()
        for kind,n,label in [('linear',1,'linear'),('mlp',1,'mlp'),('uniform',sel['uniform'],'uniform'),('moe',sel['moe'],'moe'),('uniform',sel['moe'],'uniform_matched')]:
            for seed in c['model_seeds']:
                key=f'pooled__{inp}__{kind}{n}__s{seed}';dest=parent/'training'/key
                if file_digest(dest/'manifest.json')!=selection['training'][key]:raise ValueError('冻结模型改变')
                receipt=json.loads((dest/'manifest.json').read_text())
                if file_digest(dest/'model.pt')!=receipt['files']['model.pt']:raise ValueError('模型内容改变')
                model,mean,scale,state=load_classifier(dest/'model.pt',c,f'cuda:{args.rank%2}')
                p,g,e,h,odds=predict_diagnostics(model,x,mask,mean,scale,f'cuda:{args.rank%2}')
                frame=score_frame(meta,p).assign(input=inp,head=label,seed=seed,experts=n,real_log_odds=odds);all_scores.append(frame)
                for name,t in evaluate_fixed_pairs(frame,pairs).items():tables.setdefault(name,[]).append(t.assign(input=inp,head=label,seed=seed))
                del model
    scores=pd.concat(all_scores,ignore_index=True);scores.to_csv(out/'test_scores.csv.gz',index=False)
    for name,table in tables.items():atomic_csv(out/(name+'.csv'),pd.concat(table,ignore_index=True))
    ds=pd.concat(tables['dataset_metrics'],ignore_index=True);summary=ds.groupby(['dataset','input','head'])[['auc','real_positive_ap']].agg(['mean','std']);summary.columns=['_'.join(x) for x in summary.columns]
    atomic_csv(out/'seed_summary.csv',summary.reset_index())
    intervals=[]
    for inp in ['G','GT','GTL']:
        for control in ['mlp','uniform_matched']:
            q=paired_seed_contrast(scores,pairs,(inp,'moe'),(inp,control))
            intervals.append(q[q.dataset!='Average'].assign(contrast=f'{inp}_moe_vs_{control}'))
    atomic_csv(out/'confidence_intervals.csv',pd.concat(intervals,ignore_index=True))
    paper_json(out/'evaluation_manifest.json',dict(status='evaluated',selection=file_digest(parent/'selection.json'),feature_hashes=hashes,
        files={p:file_digest(out/p) for p in ['test_scores.csv.gz','seed_summary.csv','confidence_intervals.csv']+[n+'.csv' for n in tables]}))
    print(summary.to_string(),flush=True)
