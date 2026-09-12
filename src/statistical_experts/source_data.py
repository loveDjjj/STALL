"""按独立源冻结目标/源参考预算，并提取原版单窗目标Global。"""
import hashlib
import json
from pathlib import Path
import shutil
import time
import numpy as np
import pandas as pd
import yaml
from artifacts import atomic_csv,paper_json
from config import config_digest
from reference import file_digest
from evaluation.official_baseline import official_window
from data.video import decode_bounded
from data.prefetch import bounded_map,frame_reservation
from features import AlphaStallFeatureExtractor
from statistical_experts.cache import load_feature,feature_identity
from statistical_experts.manifests import settings


def configuration(root):
    c=yaml.safe_load((Path(root)/'configs/global_reference_sources.yaml').read_text())
    keys={'protocol','source_directory','run_directory','cache_directory','source_counts','seed','folds','ridge','dimension',
          'encoder_batch','query_batch','decode_workers','prefetch_depth','prefetch_memory_mb','bootstrap_iterations','bootstrap_seed','analysis_workers'}
    if set(c)!=keys or c['source_counts']!={'comgenvid':132,'videofeedback':200,'genvideo':200}:
        raise ValueError('源预算/配置不符')
    if c['folds']!=5 or c['ridge']!=1e-5 or c['dimension']!=1024 or c['encoder_batch']!=32 or c['query_batch']!=4 or c['seed']!=17:
        raise ValueError('冻结科学协议不同')
    if c['source_directory']!=settings(root)['run_directory'] or c['run_directory']==c['source_directory']:raise ValueError('输出不能覆盖原研究')
    return c


def select_sources(frame,seed=17):
    """源内按固定身份选一片段；源顺序仅由seed和源ID决定。"""
    if frame.source_group.isna().any() or frame.source_group.eq('').any() or not frame.subset.eq('real').all():
        raise ValueError('不是有源身份的真实拟合池')
    f=frame.sort_values(['source_group','legacy_video_id']).drop_duplicates('source_group').copy()
    f['order_key']=f.source_group.map(lambda x:hashlib.sha256(f'{seed}:{x}'.encode()).hexdigest())
    f=f.sort_values('order_key').reset_index(drop=True);n=len(f)
    permutation=np.random.default_rng(seed).permutation(n);fold=np.empty(n,dtype=int);fold[permutation]=np.arange(n)%5
    f['fold']=fold;f['source_order']=np.arange(n)
    return f


def prepare(root):
    root=Path(root);c=configuration(root);base=settings(root);out=root/c['run_directory']
    paths=['configs/global_reference_sources.yaml','configs/paper.yaml','src/statistical_experts/source_data.py',
           'src/features.py','src/data/video.py','src/evaluation/official_baseline.py',
           'data/catalog/vatex_source_fit200.csv','data/manifests/global_experts/windows.csv','data/manifests/global_experts/pairs.csv']
    paths += [f'data/manifests/active/{d}/fit.csv' for d in c['source_counts']]
    spec=dict(config=c,inputs={p:file_digest(root/p) for p in paths},old_feature_identity=config_digest(feature_identity(root,base)))
    marker=out/'data_identity.json'
    if marker.exists():
        if json.loads(marker.read_text())!=spec:raise ValueError('数据准备身份改变')
        meta=json.loads((out/'data_manifest.json').read_text())
        for p,h in meta['files'].items():
            if file_digest(out/p)!=h:raise ValueError('冻结清单改变')
        return out,spec
    old=pd.read_csv(root/'data/manifests/global_experts/windows.csv',keep_default_na=False)
    evaluation=old[old.role.eq('evaluation')];vatex=old[old.role.isin(['fit','cdf'])]
    source_small=pd.read_csv(root/'data/catalog/vatex_source_fit200.csv',keep_default_na=False).sort_values('source_id').reset_index(drop=True)
    source_small=source_small.iloc[np.random.default_rng(c['seed']).permutation(len(source_small))].reset_index(drop=True)
    if len(source_small)!=200 or source_small.source_id.nunique()!=200:raise ValueError('源参考不是200独立媒体')
    frames=[];selections=[];checks=[]
    for domain,n in c['source_counts'].items():
        original=pd.read_csv(root/f'data/manifests/active/{domain}/fit.csv',keep_default_na=False)
        selected=select_sources(original,c['seed'])
        if len(selected)!=n:raise ValueError('目标独立源数量与计划不符')
        if set(selected.source_group)&set(evaluation.source_group) or set(selected.video_path)&set(evaluation.video_path):
            raise ValueError('目标拟合与评价存在已知源交集')
        if domain=='comgenvid':
            ids=set(selected.video_path.map(lambda p:Path(p).stem[:11]))
            other=set(vatex.video_path.map(lambda p:Path(p).stem))
            other |= set(evaluation.loc[evaluation.dataset.eq(domain),'video_path'].map(lambda p:Path(p).stem[:11]))
            if ids&other:raise ValueError('已知MSVD媒体与参考/评价交叉')
        selections.append(selected)
        checks.append(dict(dataset=domain,original_clips=len(original),selected_clips=n,independent_sources=n,
                           fit_eval_source_intersection=0,source_count_match=True))
        for row in selected.itertuples(index=False):
            path=root/row.video_path
            if not path.is_file():raise FileNotFoundError(path)
            for length in ([16] if domain=='comgenvid' else [8,16]):
                indices=official_window(json.loads(row.downsample_idxs),length)
                saved=selected.loc[selected.video_id.eq(row.video_id),'1_sec_idxs' if length==8 else '2_sec_idxs'].iloc[0]
                if saved and json.loads(saved)!=indices:raise ValueError('既有官方窗口索引与重建不符')
                vid=f'target_native:{row.video_id}:window{length}'
                key=hashlib.sha256((row.video_path+json.dumps(indices)).encode()).hexdigest()
                state=path.stat()
                frames.append(dict(video_id=vid,legacy_video_id=row.video_id,video_path=row.video_path,source_group=row.source_group,
                    dataset=domain,subset='real',source_model=row.source_model,real_source=row.real_source,role='fit',length=length,
                    frame_indices=json.dumps(indices),cache_key=key,fold=int(row.fold),source_order=int(row.source_order),
                    source_bytes=state.st_size,source_mtime_ns=state.st_mtime_ns))
    f=pd.DataFrame(frames).sort_values(['length','dataset','source_order']).reset_index(drop=True)
    if len(f)!=932 or f.frame_indices.map(lambda x:len(json.loads(x))).sum()!=11712:raise ValueError('目标窗口预算不符')
    out.mkdir(parents=True,exist_ok=True)
    atomic_csv(out/'target_windows.csv',f);atomic_csv(out/'target_sources.csv',pd.concat(selections,ignore_index=True))
    atomic_csv(out/'vatex_sources.csv',source_small.assign(source_order=np.arange(200)))
    paper_json(out/'source_isolation.json',dict(checks=checks,meaning='已知源组及MSVD媒体ID隔离，不等于完整跨库语义去重'))
    for p in paths:
        if p.startswith(('src/','configs/')):
            dest=out/'source_snapshot'/p;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(root/p,dest)
    paper_json(marker,spec)
    paper_json(out/'data_manifest.json',dict(status='prepared',target_sources=532,target_windows=932,frames=11712,
        files={p:file_digest(out/p) for p in ['target_windows.csv','target_sources.csv','vatex_sources.csv','source_isolation.json']}))
    print('prepared 532 sources / 932 windows',flush=True)
    return out,spec


def target_identity(root,spec):
    out=Path(root)/spec['config']['run_directory']
    return config_digest(dict(data=config_digest(spec),manifest=file_digest(out/'target_windows.csv'),
                              encoder_batch=32,pad_tail=False,decoder='decode_bounded',dtype='float32'))


def extract(root,rank,world):
    root=Path(root);out,spec=prepare(root);c=spec['config'];identity=target_identity(root,spec);directory=root/c['cache_directory']
    directory.mkdir(parents=True,exist_ok=True)
    f=pd.read_csv(out/'target_windows.csv',keep_default_na=False);rows=list(f.itertuples(index=False))[rank::world]
    pending=[]
    for r in rows:
        p=directory/(r.cache_key+'.npz')
        if p.exists():load_feature(p,identity,r)
        else:pending.append(r)
    if not pending:
        print('all target windows verified',rank,flush=True);return
    config=yaml.safe_load((root/'configs/paper.yaml').read_text())
    if file_digest(root/config['encoder']['weights'])!=config['encoder']['weights_sha256']:raise ValueError('编码器权重改变')
    model=AlphaStallFeatureExtractor(f'cuda:{rank%2}',dino_repo=str(root/config['encoder']['repo']),dino_weights=str(root/config['encoder']['weights']),pad_tail_batch=False)
    def load(r):
        state=(root/r.video_path).stat()
        if state.st_size!=r.source_bytes or state.st_mtime_ns!=r.source_mtime_ns:raise ValueError('目标原视频改变')
        return model.prepare_frames(decode_bounded(root/r.video_path,json.loads(r.frame_indices)))
    stream=bounded_map(pending,load,lambda r:frame_reservation(root/r.video_path,r.length),workers=c['decode_workers'],depth=c['prefetch_depth'],budget=c['prefetch_memory_mb']*2**20)
    start=time.perf_counter();outputs=[]
    try:
        for i,(r,frames) in enumerate(stream,1):
            g=model.frames_to_global_embeddings([frames],batch_size=32)[0]
            state=(root/r.video_path).stat()
            if (state.st_size,state.st_mtime_ns)!=(r.source_bytes,r.source_mtime_ns):raise ValueError('提取中原视频改变')
            target=directory/(r.cache_key+'.npz');temp=directory/(r.cache_key+f'.{rank}.tmp.npz')
            np.savez(temp,identity=np.asarray(identity),video_id=np.asarray(r.video_id),global_features=g,
                frame_indices=np.asarray(json.loads(r.frame_indices)),tensor_sha256=np.asarray(hashlib.sha256(g.tobytes()).hexdigest()),
                source_bytes=np.asarray(state.st_size),source_mtime_ns=np.asarray(state.st_mtime_ns))
            temp.replace(target);load_feature(target,identity,r);outputs.append(target.name)
            if i%32==0 or i==len(pending):print(f'[target cache {rank}] {i}/{len(pending)} {time.perf_counter()-start:.1f}s',flush=True)
    finally:stream.close()
    paper_json(out/f'cache_rank_{rank}.json',dict(status='completed',identity=identity,rank=rank,world=world,new_windows=len(pending),elapsed_seconds=time.perf_counter()-start,files=outputs))
