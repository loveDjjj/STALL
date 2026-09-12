"""冻结主线窗口，共享前向完成四格；原始分数和CDF分别验收。"""
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import hashlib,json,os,subprocess,sys,time
import numpy as np,pandas as pd,torch,yaml
from artifacts import atomic_csv,paper_json,checkpoint_write,checkpoint_read
from config import config_digest
from reference import file_digest,percentile,window_mean,local_video_cdfs
from evaluation.study_tables import read_evidence
from evaluation.tables import evaluate_fixed_pairs
from evaluation.bootstrap import paired_source_contrast
from mainline_experts.models import load_models,load_anchors
from mainline_experts.run import configuration
from math_utils import StableGaussianParams,GaussianMeanCandidateScorerFloat64,l2_normalized_second_order,l2_normalized_first_order,score_gaussian_aggregate_float64
from branches.global_branch import score_global_raw
from evaluation.representations import mean_whitening_controls
from features import AlphaStallFeatureExtractor
from data.video import decode_bounded
from data.prefetch import bounded_map,frame_reservation
from statistical_experts.gaussian import context


def prepare(root,args=None):
    root=Path(root);c=configuration(root);out=root/c['run_directory'];marker=out/'data_manifest.json'
    if marker.exists():
        m=json.loads(marker.read_text())
        for p,h in m['files'].items():
            if file_digest(out/p)!=h:raise ValueError('窗口清单改变')
        return
    evaluation=pd.read_csv(root/'results/paper_complete/evaluation.csv',keep_default_na=False);wanted=set(evaluation.video_id);plans={};inputs={}
    def add(domain,role,row,record):
        selected=record['selected'] if role=='cdf' else record;windows=[w['frame_indices'] for w in selected['windows']]
        uniform=record['uniform']['frame_indices'] if role=='cdf' else []
        spec=dict(video_path=row['video_path'],windows=windows,uniform=uniform,role=role)
        key=config_digest(spec)
        if key not in plans:
            state=(root/row['video_path']).stat();plans[key]=dict(**spec,key=key,targets={},source_bytes=state.st_size,source_mtime_ns=state.st_mtime_ns)
        expected=[]
        for w in selected['windows']:
            expected.append(dict(gs=w['global_models']['target']['gs'],gt=w['global_models']['target']['gt'],lt=w['representations']['local_d2']['target']))
        plans[key]['targets'][domain]=dict(video_id=row['video_id'],dataset=domain,subset=row['subset'],source_model=row['source_model'],
            source_group=row.get('source_group',''),expected=expected,uniform=record['uniform']['global_models']['target'] if role=='cdf' else None)
    for domain in c['datasets']:
        for length in ([16] if domain=='comgenvid' else [16,8]):
            for role in ('evaluation','cdf'):
                if length==16:
                    mp=root/f'data/manifests/active/{domain if role=="evaluation" else "vatex"}/{"evaluation" if role=="evaluation" else "cdf"}.csv'
                    directory=root/'results/runs'/f'paper_{"evidence" if role=="evaluation" else "cdf"}_{domain}_fc3'
                else:
                    mp=root/f'data/manifests/short_video/{domain if role=="evaluation" else "vatex"}/{"evaluation" if role=="evaluation" else "cdf"}.csv'
                    directory=root/'results/runs'/f'complete23_{domain}_{role}_short8'
                f=pd.read_csv(mp,keep_default_na=False);records,spec=read_evidence(directory,f)
                inputs[str(mp)]=file_digest(mp);inputs[str(directory)]=config_digest(spec)
                for row,record in zip(f.to_dict('records'),records):
                    if role=='evaluation' and row['video_id'] not in wanted:continue
                    add(domain,role,row,record)
    jobs=list(plans.values())
    ids=[v['video_id'] for job in jobs if job['role']=='evaluation' for v in job['targets'].values()]
    if len(ids)!=15569 or set(ids)!=wanted:raise ValueError('主线评价身份不完整')
    out.mkdir(parents=True,exist_ok=True);paper_json(out/'plans.json',dict(jobs=jobs))
    atomic_csv(out/'pairs.csv',pd.read_csv(root/'results/paper_complete/pairs.csv'))
    paper_json(marker,dict(status='prepared',inputs=inputs,jobs=len(jobs),evaluation_clips=len(ids),
        files={p:file_digest(out/p) for p in ['plans.json','pairs.csv']}))
    print('prepared physical jobs',len(jobs),'evaluation',len(ids),flush=True)


class Scorer:
    def __init__(self,root,domains,device):
        self.root=Path(root);self.device=device;cfg=yaml.safe_load((self.root/'configs/paper.yaml').read_text());self.config=cfg
        self.extractor=AlphaStallFeatureExtractor(device,dino_repo=str(self.root/cfg['encoder']['repo']),dino_weights=str(self.root/cfg['encoder']['weights']),pad_tail_batch=True)
        if file_digest(self.root/cfg['encoder']['weights'])!=cfg['encoder']['weights_sha256']:raise ValueError('骨干权重改变')
        with np.load(self.root/'results/runs/paper_fit_source/gaussians.npz') as z:source=StableGaussianParams(z['lt_mean'],z['lt_whitening'],np.empty(0))
        self.models={}
        for d in domains:
            anchors=load_anchors(self.root,d);centers,models=load_models(self.root/configuration(self.root)['run_directory']/f'fit_{d}/models.npz')
            with np.load(self.root/f'results/runs/paper_representation_fit_{d}/gaussians.npz') as z:diag=StableGaussianParams(z['local_diagonal_d2_mean'],z['local_diagonal_d2_whitening'],np.empty(0))
            originals=mean_whitening_controls(source,anchors['lt'],diag)
            self.models[d]=dict(anchors=anchors,centers=centers,experts=models,
                original_local=GaussianMeanCandidateScorerFloat64(list(originals.values()),source.mean,device),
                expert_local=GaussianMeanCandidateScorerFloat64(models['lt'][1:],source.mean,device))

    def globals(self,g,d):
        spec=self.models[d];a=spec['anchors'];raw=score_global_raw(g,a['gs'],a['gt'],device=self.device)
        c=context(torch.from_numpy(g)).numpy();route=(c@spec['centers'].T).argmax(1)
        gt,zero=l2_normalized_first_order(torch.from_numpy(g));candidate=[]
        for m in spec['experts']['gt'][1:]:candidate.append(score_gaussian_aggregate_float64(gt,m,'min',self.device,invalid_mask=zero,compute_percentile=False)[0])
        e=np.stack(candidate,axis=1)
        return raw,route,e

    def score(self,job,prepared):
        uniform,frames=prepared;features=self.extractor.frames_to_global_patch_embeddings([frames],batch_size=8)[0]
        union=sorted({x for w in job['windows'] for x in w});where={x:i for i,x in enumerate(union)};pick=[[where[x] for x in w] for w in job['windows']]
        g=np.stack([features['global'][p] for p in pick]);patch=torch.from_numpy(np.stack([features['patch'][p] for p in pick]));unit=l2_normalized_second_order(patch)
        ug=self.extractor.frames_to_global_embeddings([uniform],batch_size=8)[0][None] if uniform is not None else None
        output={};max_error=0.
        for d,target in job['targets'].items():
            s=self.models[d];raw,route,eg=self.globals(g,d);local=s['original_local'].score(unit)[:,1];el=s['expert_local'].score(unit)
            records=[]
            for i,w in enumerate(job['windows']):
                actual=np.array([raw.spatial[i],raw.temporal_t1[i],local[i]]);expected=np.array([float(target['expected'][i][k]) for k in ('gs','gt','lt')])
                np.testing.assert_allclose(actual,expected,rtol=0,atol=1e-8)
                mask=np.isfinite(actual)&np.isfinite(expected)
                if mask.any():max_error=max(max_error,float(np.abs(actual[mask]-expected[mask]).max()))
                records.append(dict(gs=float(raw.spatial[i]),gt=float(raw.temporal_t1[i]),lt=float(local[i]),route=int(route[i]),gt_experts=eg[i].tolist(),lt_experts=el[i].tolist()))
            u=None
            if ug is not None:
                ur,uq,ue=self.globals(ug,d);u=dict(gs=float(ur.spatial[0]),gt=float(ur.temporal_t1[0]),route=int(uq[0]),gt_experts=ue[0].tolist())
                np.testing.assert_allclose([u['gs'],u['gt']],[float(target['uniform'][k]) for k in ('gs','gt')],rtol=0,atol=1e-8)
            output[d]=dict(windows=records,uniform=u)
        return dict(key=job['key'],video_id=job['key'],targets=output,max_baseline_error=max_error,global_features=g,uniform_global=ug)


def dense(root,args):
    root=Path(root);c=configuration(root);out=root/c['run_directory'];prepare(root);jobs=json.loads((out/'plans.json').read_text())['jobs'];spec=dict(
        config=c,plans=file_digest(out/'plans.json'),code={p:file_digest(root/p) for p in ['src/mainline_experts/experiment.py','src/mainline_experts/models.py','src/features.py','src/math_utils.py','src/branches/global_branch.py']},
        models={d:file_digest(out/f'fit_{d}/models.npz') for d in c['datasets']})
    identity=config_digest(spec);directory=out/('pilot' if args.pilot else 'dense');directory.mkdir(parents=True,exist_ok=True)
    marker=directory/'identity.json'
    if marker.exists():
        if json.loads(marker.read_text())!=spec:raise ValueError('密集评分合同改变')
    else:paper_json(marker,spec)
    if args.pilot:
        selected=[];counts={}
        for job in jobs:
            key=(job['role'],len(job['windows'][0]),tuple(job['targets']))
            if counts.get(key,0)<4:selected.append(job);counts[key]=counts.get(key,0)+1
        jobs=selected
    pending=[]
    for i,job in enumerate(jobs):
        if i%args.world_size!=args.rank:continue
        path=directory/(job['key']+'.json')
        if path.exists():checkpoint_read(path,identity,job['key'])
        elif not args.pilot and (out/'pilot'/(job['key']+'.json')).exists():
            payload=checkpoint_read(out/'pilot'/(job['key']+'.json'),identity,job['key']);checkpoint_write(path,identity,payload)
        else:pending.append(job)
    if not pending:print('dense already complete',args.rank,flush=True);return
    scorer=Scorer(root,c['datasets'],f'cuda:{args.rank%2}');start=time.perf_counter()
    def load(job):
        path=root/job['video_path'];state=path.stat()
        if (state.st_size,state.st_mtime_ns)!=(job['source_bytes'],job['source_mtime_ns']):raise ValueError('原视频改变')
        union=sorted({x for w in job['windows'] for x in w})
        frames=scorer.extractor.prepare_frames(decode_bounded(path,union))
        uniform=scorer.extractor.prepare_frames(decode_bounded(path,job['uniform'])) if job['uniform'] else None
        return uniform,frames
    stream=bounded_map(pending,load,lambda j:frame_reservation(root/j['video_path'],len({x for w in j['windows'] for x in w})+len(j['uniform'])),workers=4,depth=6,budget=4*2**30)
    error=0.
    try:
        for i,(job,prepared) in enumerate(stream,1):
            payload=scorer.score(job,prepared);error=max(error,payload['max_baseline_error'])
            gd=out/'global';gd.mkdir(exist_ok=True);gp=gd/(job['key']+'.npz');tmp=gp.with_suffix('.tmp.npz')
            g=payload.pop('global_features');ug=payload.pop('uniform_global')
            np.savez(tmp,global_features=g,uniform_global=np.empty((0,)) if ug is None else ug,identity=np.asarray(identity),key=np.asarray(job['key']))
            tmp.replace(gp);payload['global_sha256']=file_digest(gp)
            checkpoint_write(directory/(job['key']+'.json'),identity,payload)
            if i%16==0 or i==len(pending):
                status=dict(completed=i,pending=len(pending),seconds=time.perf_counter()-start,videos_per_second=i/(time.perf_counter()-start),max_baseline_error=error)
                paper_json(directory/f'progress_{args.rank}.json',status);print('mainline',args.rank,i,len(pending),round(status['videos_per_second'],2),flush=True)
    finally:stream.close()
    paper_json(directory/f'completed_{args.rank}.json',dict(status='completed',identity=identity,rank=args.rank,world=args.world_size,seconds=time.perf_counter()-start,max_baseline_error=error))


def evaluate(root,args=None):
    from mainline_experts.evaluation import evaluate as function
    return function(root,args)


def analyze(root,args=None):
    from mainline_experts.evaluation import analyze as function
    return function(root,args)


def verify(root,args=None):
    from mainline_experts.audit import verify as function
    return function(root,args)


def report(root,args=None):
    from mainline_experts.audit import report as function
    return function(root,args)


def all(root,args=None):
    from mainline_experts.driver import execute
    return execute(root,args)
