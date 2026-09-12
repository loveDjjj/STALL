"""独立新真实池的共享特征和成对模型；只存Global、抽样方向及pooled小向量。"""
import argparse,json,time
from pathlib import Path
import numpy as np,pandas as pd,torch,yaml
from artifacts import checkpoint_read,checkpoint_write,paper_json
from config import config_digest
from data.prefetch import bounded_map,frame_reservation
from data.video import decode_bounded
from evaluation.confirmation_data import context,DOMAINS
from evaluation.direction_features import direction_features,fit_source_balanced
from features import AlphaStallFeatureExtractor
from math_utils import l2_normalized_first_order
from reference import file_digest,fit_gaussian,publish_arrays
from reference_fit import sample_d2_positions
from selection import uniform_windows


def setup(root):
    c,out,data=context(root);prepared=json.loads((out/'prepared.json').read_text())
    for p,h in prepared['files'].items():
        if file_digest(out/p)!=h:raise ValueError('新参考清单改变')
    encoder=yaml.safe_load((root/'configs/paper.yaml').read_text())['encoder']
    code={p:file_digest(root/p) for p in ['src/evaluation/confirmation_fit.py','src/features.py','src/data/video.py',
        'src/math_utils.py','src/reference.py','src/reference_fit.py','src/evaluation/direction_features.py']}
    spec=dict(data=config_digest(data),prepared=file_digest(out/'prepared.json'),encoder=encoder,code=code,
        numeric='FP32 differences / batch8 pad-tail / FP64 Gaussian / equal clip weights / ridge 1e-5')
    identity=config_digest(spec);p=out/'fit_identity.json'
    if p.exists() and json.loads(p.read_text())!=spec:raise ValueError('拟合数值合同改变')
    if not p.exists():paper_json(p,spec)
    return c,out,encoder,identity


def feature_key(row):return config_digest(dict(video_id=row['video_id'],indices=row['downsample_idxs']))


def extract(root,rank=0,world=2):
    c,out,e,identity=setup(root);f=pd.read_csv(out/'fit_unique.csv',keep_default_na=False);pending=[]
    for i,r in enumerate(f.to_dict('records')):
        if i%world!=rank:continue
        key=feature_key(r);p=out/'fit_features'/(key+'.npz');marker=p.with_suffix('.json')
        if marker.exists():
            s=checkpoint_read(marker,identity,r['video_id'])
            if file_digest(p)!=s['sha256']:raise ValueError('新fit特征改变')
        else:pending.append((r,key))
    if not pending:return
    if file_digest(root/e['weights'])!=e['weights_sha256']:raise ValueError('编码器改变')
    extractor=AlphaStallFeatureExtractor(c['devices'][rank],dino_repo=str(root/e['repo']),dino_weights=str(root/e['weights']),pad_tail_batch=True)
    def windows(r):return [list(w.frame_indices) for w in uniform_windows(json.loads(r['downsample_idxs']),3)]
    def prepare(item):
        r,key=item;st=(root/r['video_path']).stat()
        if (st.st_size,st.st_mtime_ns)!=(r['source_bytes'],r['source_mtime_ns']):raise ValueError('原真实视频改变')
        union=sorted({x for w in windows(r) for x in w})
        return extractor.prepare_frames(decode_bounded(root/r['video_path'],union))
    stream=bounded_map(pending,prepare,lambda item:frame_reservation(root/item[0]['video_path'],len({x for w in windows(item[0]) for x in w})),
        workers=c['decode_workers'],depth=c['prefetch_depth'],budget=4*2**30)
    start=time.perf_counter()
    for n,((r,key),frames) in enumerate(stream,1):
        w=windows(r);union=sorted({x for win in w for x in win});where={x:i for i,x in enumerate(union)}
        feat=extractor.frames_to_global_patch_embeddings([frames],batch_size=8)[0];pick=[[where[x] for x in win] for win in w]
        g=np.stack([feat['global'][ix] for ix in pick]);patch=np.stack([feat['patch'][ix] for ix in pick]);p=torch.from_numpy(patch)
        d1=p[:,1:]-p[:,:-1];d2=p[:,2:]-2*p[:,1:-1]+p[:,:-2]
        u1=torch.nn.functional.normalize(d1,dim=-1,eps=1e-12);u2=torch.nn.functional.normalize(d2,dim=-1,eps=1e-12)
        x1=sample_d2_positions(u1.reshape(-1,1024),r['legacy_path']).numpy();x2=sample_d2_positions(u2.reshape(-1,1024),r['legacy_path']).numpy()
        pooled,scalar=direction_features(patch)
        path=out/'fit_features'/(key+'.npz')
        publish_arrays(path,dict(video_id=np.asarray(r['video_id']),global_windows=g,unit_d1=x1,unit_d2=x2,pooled=pooled,
            scalar=scalar,frame_indices=np.asarray(w)))
        checkpoint_write(path.with_suffix('.json'),identity,dict(video_id=r['video_id'],sha256=file_digest(path)))
        if n%16==0 or n==len(pending):
            elapsed=time.perf_counter()-start;paper_json(out/f'fit_progress_{rank}.json',dict(completed=n,total=len(pending),seconds=elapsed,eta_seconds=(len(pending)-n)*elapsed/n))
            print('new reference features',rank,n,len(pending),flush=True)
    paper_json(out/f'fit_done_{rank}.json',dict(status='completed',identity=identity,rank=rank,world=world))


def fit(root):
    c,out,e,identity=setup(root);selected=pd.read_csv(out/'fit_selections.csv',keep_default_na=False)
    for d in DOMAINS:
        for seed in c['seeds']:
            name=f'{d}_s{seed}';path=out/'models'/(name+'.npz');marker=path.with_suffix('.json')
            if marker.exists():
                m=checkpoint_read(marker,identity,name)
                if file_digest(path)!=m['sha256']:raise ValueError('新参考模型改变')
                continue
            f=selected[(selected.dataset==d)&(selected.fit_seed==seed)];groups=[]
            obs={k:[] for k in ('gs','gt','d1','d2','matched')}
            inputs=[]
            for r in f.to_dict('records'):
                p=out/'fit_features'/(feature_key(r)+'.npz');m=checkpoint_read(p.with_suffix('.json'),identity,r['video_id'])
                if file_digest(p)!=m['sha256']:raise ValueError('拟合特征发生改变')
                with np.load(p) as z:
                    g=z['global_windows'];gt,zero=l2_normalized_first_order(torch.from_numpy(g));match=int(np.prod(z['pooled'].shape[:-1]))
                    obs['gs'].append(g.reshape(-1,1024));obs['gt'].append(gt[~zero].reshape(-1,1024).numpy())
                    obs['d1'].append(z['unit_d1'].copy());obs['d2'].append(z['unit_d2'].copy());obs['matched'].append(z['unit_d2'][:match].copy())
                groups.append(r['source_group']);inputs.append(dict(video_id=r['video_id'],source_group=r['source_group'],sha256=m['sha256'],matched_positions=match))
            models={k:fit_gaussian(v,ridge=c['ridge']) for k,v in obs.items()}
            models['source_equal']=fit_source_balanced(obs['d2'],groups,ridge=c['ridge'])
            arrays={}
            for k,m in models.items():arrays[k+'_mean']=m.mean;arrays[k+'_whitening']=m.whitening
            publish_arrays(path,arrays);checkpoint_write(marker,identity,dict(video_id=name,sha256=file_digest(path),inputs=inputs,
                fit_clips=len(f),fit_sources=len(set(groups)),global_and_local_same_fit=True))
            print('new reference models',name,flush=True)
        # 原池位置预算控制，保持原Global以对照已完成pooled实验。
        path=out/'models'/f'{d}_original_matched.npz'
        f=pd.read_csv(root/f'data/manifests/active/{d}/fit.csv',keep_default_na=False);values=[]
        for r in f.itertuples():
            p=root/r.feature_asset
            if file_digest(p)!=r.feature_asset_sha256:raise ValueError('原fit资产改变')
            with np.load(p) as z:
                count=len(z['global_windows'])*14
                values.append(torch.nn.functional.normalize(torch.from_numpy(z['raw_d2'][:count].copy()),dim=-1,eps=1e-12).numpy())
        m=fit_gaussian(values,ridge=c['ridge']);publish_arrays(path,dict(matched_mean=m.mean,matched_whitening=m.whitening))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('action',choices=['features','fit']);p.add_argument('--rank',type=int,default=0);p.add_argument('--world',type=int,default=2);a=p.parse_args()
    torch.set_num_threads(2)
    if a.action=='features':extract(Path.cwd(),a.rank,a.world)
    else:fit(Path.cwd())
