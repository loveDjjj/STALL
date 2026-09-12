"""单视频Global缓存：CPU有界解码预取，GPU保留官方前向批界。"""
import hashlib
import json
import os
import tempfile
import time
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import yaml
from artifacts import paper_json
from config import config_digest
from reference import file_digest
from features import AlphaStallFeatureExtractor
from data.prefetch import bounded_map,frame_reservation
from data.sequential_decode import decode_sequential
from data.video import decode_bounded
from evaluation.official_baseline import official_numpy_score
from statistical_experts.manifests import settings


def feature_identity(root,c):
    root=Path(root);base=yaml.safe_load((root/'configs/paper.yaml').read_text())
    return dict(encoder=base['encoder'],encoder_batch=32,pad_tail=False,dtype='float32',
        code={name:file_digest(root/name) for name in ['src/features.py','src/data/video.py','src/data/sequential_decode.py',
            'src/statistical_experts/cache.py','src/evaluation/official_baseline.py']},
        window_manifest_sha256=file_digest(root/c['manifest_directory']/'windows.csv'))


def load_feature(path,identity,row):
    with np.load(path,allow_pickle=False) as z:
        if str(z['identity'])!=identity or str(z['video_id'])!=row.video_id:raise ValueError('Global缓存身份不符')
        g=z['global_features'].copy()
        if g.shape!=(row.length,1024) or g.dtype!=np.float32 or not np.isfinite(g).all():raise ValueError('Global缓存形状/精度非法')
        if list(z['frame_indices'])!=json.loads(row.frame_indices):raise ValueError('Global缓存帧索引不同')
        if str(z['tensor_sha256'])!=hashlib.sha256(g.tobytes()).hexdigest():raise ValueError('Global缓存内容损坏')
    return g


def extract(root,rank,world_size,*,pilot=False):
    root=Path(root).resolve();c=settings(root);directory=root/c['cache_directory'];run=root/c['run_directory']/'cache'
    spec=feature_identity(root,c);identity=config_digest(spec);directory.mkdir(parents=True,exist_ok=True);run.mkdir(parents=True,exist_ok=True)
    spec_path=directory/'identity.json'
    if spec_path.exists():
        if json.loads(spec_path.read_text())!=spec:raise ValueError('Global缓存合同已改变')
    else:paper_json(spec_path,spec)
    f=pd.read_csv(root/c['manifest_directory']/'windows.csv',keep_default_na=False)
    if pilot:
        # 200条真实窗口：100参考、100带官方锚点的评价real，覆盖8/16帧与各域。
        refs=f[f.role=='fit'].groupby('length',sort=True).head(50)
        real=f[(f.role=='evaluation')&(f.subset=='real')].groupby(['dataset','length'],sort=True).head(20)
        f=pd.concat([refs,real],ignore_index=True)
    items=[]
    for i,row in enumerate(f.itertuples(index=False)):
        if i%world_size!=rank:continue
        path=directory/(row.cache_key+'.npz')
        if path.exists():load_feature(path,identity,row)
        else:items.append(row)
    if not items:
        paper_json(run/f'{"pilot_completed" if pilot else "completed"}_rank_{rank}.json',dict(identity=identity,rank=rank,world_size=world_size,pilot=pilot,elapsed=0,reused=True))
        print(f'cache rank {rank}: already done',flush=True);return
    base=yaml.safe_load((root/'configs/paper.yaml').read_text());device=f'cuda:{rank%2}'
    model=AlphaStallFeatureExtractor(device,dino_repo=str(root/base['encoder']['repo']),dino_weights=str(root/base['encoder']['weights']),pad_tail_batch=False)
    if file_digest(root/base['encoder']['weights'])!=base['encoder']['weights_sha256']:raise ValueError('编码器权重变化')
    with np.load(root/'precomputed/stall_params_vatex_dino_v3.npz') as z:official={k:z[k].copy() for k in z.files}
    def prepare(row):
        path=root/row.video_path;indices=json.loads(row.frame_indices);before=path.stat()
        frames=decode_sequential(path,indices)
        if pilot:
            exact=decode_bounded(path,indices)
            if not np.array_equal(frames,exact):raise ValueError('pilot顺序解码与官方帧像素不同')
        prepared=model.prepare_frames(frames)
        return prepared,(before.st_size,before.st_mtime_ns)
    stream=bounded_map(items,prepare,lambda r:frame_reservation(root/r.video_path,int(r.length)),
        workers=c['decode_workers'],depth=c['prefetch_depth'],budget=c['prefetch_memory_mb']*2**20)
    start=time.perf_counter();fallback=0
    try:
        for done,(row,(frames,state)) in enumerate(stream,1):
            g=model.frames_to_global_embeddings([frames],batch_size=32)[0]
            if row.expected_final!='':
                expected=np.asarray([float(row.expected_gs),float(row.expected_gt),float(row.expected_final)])
                def matches(g):
                    raw=official_numpy_score(g[None],official)
                    actual=np.asarray([raw['global_spatial_raw'][0],raw['global_temporal_raw'][0],raw['final_score'][0]])
                    return np.allclose(actual[:2],expected[:2],rtol=0,atol=1e-6) and actual[2]==expected[2]
                if not matches(g):
                    frames=model.prepare_frames(decode_bounded(root/row.video_path,json.loads(row.frame_indices)))
                    g=model.frames_to_global_embeddings([frames],batch_size=32)[0];fallback+=1
                    if not matches(g):raise ValueError(f'{row.video_id}不能复现官方Global锚点')
            after=(root/row.video_path).stat()
            if state!=(after.st_size,after.st_mtime_ns):raise ValueError('提取中视频变化')
            path=directory/(row.cache_key+'.npz')
            with tempfile.NamedTemporaryFile(dir=directory,suffix='.npz',delete=False) as tmp:
                temp=Path(tmp.name)
                np.savez(tmp,global_features=g,frame_indices=np.asarray(json.loads(row.frame_indices)),
                    identity=np.asarray(identity),video_id=np.asarray(row.video_id),tensor_sha256=np.asarray(hashlib.sha256(g.tobytes()).hexdigest()),
                    source_bytes=np.asarray(state[0]),source_mtime_ns=np.asarray(state[1]))
            temp.replace(path)
            if done%16==0 or done==len(items):
                status=dict(completed=done,pending_at_start=len(items),elapsed=time.perf_counter()-start,strict_fallbacks=fallback,pilot=pilot,rank=rank)
                paper_json(run/f'{"pilot" if pilot else "progress"}_rank_{rank}.json',status)
                print(f'[Global cache {rank}] {done}/{len(items)} {done/max(status["elapsed"],1e-9):.2f}/s',flush=True)
    finally:stream.close()
    paper_json(run/f'{"pilot_completed" if pilot else "completed"}_rank_{rank}.json',dict(identity=identity,rank=rank,world_size=world_size,pilot=pilot,elapsed=time.perf_counter()-start))
