"""一次冻结前向获得Local轻量矩摘要，逐视频核验原Global合同。"""
import json
import time
from pathlib import Path
import numpy as np
import torch
import yaml
from artifacts import paper_json
from config import config_digest
from reference import file_digest
from features import AlphaStallFeatureExtractor
from data.video import decode_bounded
from data.prefetch import bounded_map, frame_reservation
from discriminative_moe.run import configuration
from discriminative_moe.features import projection, local_descriptors


def extract(root,args):
    root=Path(root);c=configuration(root);out=root/c['run_directory'];old=root/c['evidence_directory']
    prepared=json.loads((out/'prepared.json').read_text())
    for name,digest in prepared['files'].items():
        if file_digest(out/name)!=digest:raise ValueError('准备产物hash改变：'+name)
    cfg=yaml.safe_load((root/'configs/paper.yaml').read_text())
    spec=dict(config=c,prepared=file_digest(out/'prepared.json'),encoder=cfg['encoder'],
        code={p:file_digest(root/p) for p in ['src/discriminative_moe/extract.py','src/discriminative_moe/features.py','src/features.py','src/data/video.py']})
    identity=config_digest(spec);directory=out/'local';directory.mkdir(exist_ok=True)
    marker=directory/'identity.json'
    if marker.exists():
        if json.loads(marker.read_text())!=spec:raise ValueError('Local恢复合同改变')
    else:paper_json(marker,spec)
    jobs=json.loads((out/'jobs.json').read_text())['jobs']
    if args.pilot:
        selected=[];counts={}
        for j in jobs:
            m=next(iter(j['targets'].values()));tag=(m['dataset'],m['subset'],len(j['windows'][0]))
            if counts.get(tag,0)<2:selected.append(j);counts[tag]=counts.get(tag,0)+1
        jobs=selected
    pending=[]
    for i,j in enumerate(jobs):
        if not args.pilot and i%args.world_size!=args.rank:continue
        p=directory/(j['key']+'.npz')
        if p.exists():
            with np.load(p) as z:
                if str(z['identity'])!=identity or str(z['key'])!=j['key'] or not np.isfinite(z['L']).all():raise ValueError('Local检查点错误')
        else:pending.append(j)
    if not pending:print('Local already complete',flush=True);return
    device=f'cuda:{args.rank%2}'
    if file_digest(root/cfg['encoder']['weights'])!=cfg['encoder']['weights_sha256']:raise ValueError('骨干改变')
    extractor=AlphaStallFeatureExtractor(device,dino_repo=str(root/cfg['encoder']['repo']),dino_weights=str(root/cfg['encoder']['weights']),pad_tail_batch=True)
    basis=projection(rank=c['projection_dimension'],seed=c['projection_seed']).to(device)
    def load(j):
        path=root/j['video_path'];st=path.stat()
        if (st.st_size,st.st_mtime_ns)!=(j['source_bytes'],j['source_mtime_ns']):raise ValueError('原视频改变')
        indices=sorted({x for w in j['windows'] for x in w})
        return extractor.prepare_frames(decode_bounded(path,indices))
    start=time.perf_counter();largest=0.;rows=[]
    stream=bounded_map(pending,load,lambda j:frame_reservation(root/j['video_path'],len({x for w in j['windows'] for x in w})),
        workers=c['decode_workers'],depth=c['prefetch_depth'],budget=c['prefetch_gib']*2**30)
    try:
        for i,(j,frames) in enumerate(stream,1):
            f=extractor.frames_to_global_patch_embeddings([frames],batch_size=c['feature_batch'])[0]
            indices=sorted({x for w in j['windows'] for x in w});where={x:i for i,x in enumerate(indices)}
            pick=[[where[x] for x in w] for w in j['windows']]
            g=np.stack([f['global'][idx] for idx in pick])
            with np.load(old/'global'/(j['key']+'.npz')) as z:np.testing.assert_array_equal(g,z['global_features'])
            patches=torch.from_numpy(np.stack([f['patch'][idx] for idx in pick])).to(device)
            with torch.no_grad():local=local_descriptors(patches,basis).cpu().numpy()
            if not np.isfinite(local).all():raise ValueError('Local摘要非有限')
            if args.pilot:
                p=patches[0].cpu().numpy();u=p[2:]-2*p[1:-1]+p[:-2];u/=np.maximum(np.linalg.norm(u,axis=-1,keepdims=True),1e-12)
                u=u.reshape(-1,1024).astype(np.float64);b=basis.cpu().numpy().astype(np.float64);z=u@b;cross=z.T@z/len(z)
                check=np.concatenate([u.mean(0),(u*u).mean(0),cross[np.triu_indices(b.shape[1],1)]])
                err=float(np.abs(check-local[0]).max());largest=max(largest,err)
                np.testing.assert_allclose(check,local[0],rtol=1e-5,atol=1e-7)
            path=directory/(j['key']+'.npz');temp=path.with_suffix('.tmp.npz')
            np.savez(temp,L=local,identity=np.asarray(identity),key=np.asarray(j['key']));temp.replace(path)
            if i%32==0 or i==len(pending):
                elapsed=time.perf_counter()-start
                paper_json(directory/f'progress_{args.rank}.json',dict(completed=i,pending=len(pending),seconds=elapsed,videos_per_second=i/elapsed,pilot=args.pilot,global_exact=True))
                print('Local',args.rank,i,len(pending),round(i/elapsed,2),flush=True)
    finally:stream.close()
    paper_json(directory/(f'pilot_{args.rank}.json' if args.pilot else f'completed_{args.rank}.json'),
        dict(status='completed',identity=identity,jobs=len(pending),seconds=time.perf_counter()-start,global_exact=True,independent_moment_error=largest if args.pilot else None))
