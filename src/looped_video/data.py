"""复用已审计用途；所有变体共享视频、窗口和源级验证身份。"""
import json
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from artifacts import atomic_csv, paper_json
from reference import file_digest
from config import config_digest
from .run import configuration


def prepare(root):
    root=Path(root);c=configuration(root);out=root/c['run_directory'];src=root/c['source_directory']
    out.mkdir(parents=True,exist_ok=True)
    paths=['videos.csv','roles.csv','jobs.json','external/videos.csv','external/pairs.csv','external/jobs.json']
    spec=dict(protocol=c['protocol'],config=c,inputs={p:file_digest(src/p) for p in paths},
              paper_pairs=file_digest(root/'results/paper_complete/pairs.csv'),code=file_digest(Path(__file__)))
    identity=config_digest(spec)
    if (out/'prepared.json').exists():
        old=json.loads((out/'prepared.json').read_text())
        if old['identity']!=identity:raise ValueError('准备配置/输入改变')
        for p,h in old['files'].items():
            if file_digest(out/p)!=h:raise ValueError('准备产物改变')
        print('准备已核验',flush=True);return
    meta=pd.read_csv(src/'videos.csv',keep_default_na=False)
    roles=pd.read_csv(src/'roles.csv',keep_default_na=False)
    jobs=json.loads((src/'jobs.json').read_text())['jobs']
    for fold,q in roles.groupby('fold'):
        used=q[~q.role.eq('excluded_shared_source')]
        if used.groupby('split_group').role.nunique().max()!=1:raise ValueError('源组泄漏')
    if meta.video_id.duplicated().any() or set(roles.video_id)!=set(meta.video_id):
        raise ValueError('身份重复或覆盖缺失')
    external=pd.read_csv(src/'external/videos.csv',keep_default_na=False)
    exjobs=json.loads((src/'external/jobs.json').read_text())['jobs']
    keymap={j['video_id']:j['key'] for j in exjobs}
    external['key']=external.video_id.map(keymap)
    external['length']=16
    external['effective_k']=external.video_id.map({j['video_id']:len(j['windows']) for j in exjobs})
    if external.key.isna().any() or set(external.video_id)&set(meta.video_id):raise ValueError('外部身份不合法')
    dev_inodes={( (root/p).stat().st_dev,(root/p).stat().st_ino) for p in meta.video_path.unique()}
    for p in external.video_path.unique():
        s=(root/p).stat()
        if (s.st_dev,s.st_ino) in dev_inodes:raise ValueError('外部共享训练原文件')
    for j in jobs:j['scope']='development'
    for j in exjobs:j['scope']='external'
    alljobs=jobs+exjobs
    if len({j['key'] for j in alljobs})!=len(alljobs):raise ValueError('缓存键冲突')
    for j in alljobs:
        lengths={len(w) for w in j['windows']}
        if len(lengths)!=1 or next(iter(lengths)) not in (8,16):raise ValueError('窗口长度非法')
        if any(len(w)!=len(set(w)) or sorted(w)!=w for w in j['windows']):raise ValueError('窗口帧重复/乱序')
    atomic_csv(out/'videos.csv',meta);atomic_csv(out/'roles.csv',roles)
    atomic_csv(out/'external_videos.csv',external)
    atomic_csv(out/'pairs.csv',pd.read_csv(root/'results/paper_complete/pairs.csv'))
    atomic_csv(out/'external_pairs.csv',pd.read_csv(src/'external/pairs.csv'))
    paper_json(out/'jobs.json',dict(jobs=alljobs))
    tasks=[dict(key=f'{fold}__{variant}__s{seed}',fold=fold,variant=variant,seed=seed)
           for variant in c['variants'] for seed in c['seeds'] for fold in c['folds']]
    paper_json(out/'tasks.json',dict(tasks=tasks))
    counts=roles.groupby(['fold','role','dataset','subset','source_model']).agg(
        clips=('video_id','size'),sources=('split_group','nunique')).reset_index()
    atomic_csv(out/'split_counts.csv',counts)
    frames=sum(len({i for w in j['windows'] for i in w}) for j in alljobs)
    files=['videos.csv','roles.csv','external_videos.csv','pairs.csv','external_pairs.csv','jobs.json','tasks.json','split_counts.csv']
    paper_json(out/'prepared.json',dict(identity=identity,spec=spec,status='prepared',
        development_clips=len(meta),external_clips=len(external),tasks=len(tasks),unique_forward_frames=frames,
        patch_float32_gib=frames*196*1024*4/2**30,files={p:file_digest(out/p) for p in files}))
    print('准备完成',len(meta),len(external),'模型',len(tasks),'FP32 GiB',frames*196*1024*4/2**30,flush=True)


class PatchDataset(Dataset):
    def __init__(self,meta,cache,records):
        self.meta=meta.reset_index(drop=True);self.cache=Path(cache);self.records=records

    def __len__(self):return len(self.meta)

    def __getitem__(self,i):
        r=self.meta.iloc[i];receipt=self.records[r.key];p=self.cache/(r.key+'.npy')
        s=p.stat()
        if (s.st_size,s.st_mtime_ns)!=(receipt['bytes'],receipt['mtime_ns']):
            raise ValueError('训练期间缓存发生变化：'+r.key)
        raw=np.load(p,mmap_mode='r',allow_pickle=False)
        windows=np.asarray(receipt['window_positions'],dtype=np.int64)
        x=np.asarray(raw[windows],dtype=np.float32)
        return torch.from_numpy(x),int(r.subset=='annotated'),i


def collate_videos(batch):
    """仅补齐计算张量，不复制短片段帧；模型按有效帧/窗口计权。"""
    max_t=max(x.shape[1] for x,_,_ in batch)
    count=sum(x.shape[0] for x,_,_ in batch)
    n,d=batch[0][0].shape[-2:]
    patches=torch.zeros(count,max_t,n,d)
    valid=torch.zeros(count,max_t,dtype=torch.bool)
    owners=torch.empty(count,dtype=torch.long)
    cursor=0
    for i,(x,_,_) in enumerate(batch):
        k,t=x.shape[:2];patches[cursor:cursor+k,:t]=x
        valid[cursor:cursor+k,:t]=True;owners[cursor:cursor+k]=i;cursor+=k
    return dict(patches=patches,valid=valid,owners=owners,
                labels=torch.tensor([y for _,y,_ in batch],dtype=torch.float32),
                indices=torch.tensor([i for _,_,i in batch],dtype=torch.long))
