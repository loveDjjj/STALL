"""片段等权、窗口内容路由和可回归的目标协方差专家。"""
import hashlib,json
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from math_utils import StableGaussianParams,l2_normalized_first_order
from statistical_experts.gaussian import context,spherical_kmeans
from reference import file_digest


def sample_indices(identity,k):
    seed=int.from_bytes(hashlib.sha256(('17:'+identity).encode()).digest()[:8],'little')
    return np.random.default_rng(seed).choice(k*14*196,256,replace=False)


def load_assets(root,domain):
    root=Path(root);f=pd.read_csv(root/f'results/runs/paper_fit_{domain}/prepared_fit.csv',keep_default_na=False)
    assets=[]
    for r in f.itertuples():
        p=root/r.feature_asset
        if file_digest(p)!=r.feature_asset_sha256:raise ValueError('fit资产hash改变')
        with np.load(p) as z:
            g=z['global_windows'].copy();raw=z['raw_d2'].copy();identity=str(z['sampling_identity'])
            if str(z['video_id'])!=r.video_id:raise ValueError('拟合视频身份变化')
            if raw.shape!=(256,1024) or g.shape[1:]!=(16,1024):raise ValueError('拟合特征协议不同')
        gt,zero=l2_normalized_first_order(torch.from_numpy(g));gt=gt.numpy();zero=zero.numpy()
        gt_window=np.repeat(np.arange(len(g)),15)[~zero.reshape(-1)]
        lt=torch.nn.functional.normalize(torch.from_numpy(raw),dim=-1,eps=1e-12).numpy()
        index=sample_indices(identity,len(g));lt_window=index//(14*196)
        assets.append(dict(video_id=r.video_id,source_group=r.source_group,context=context(torch.from_numpy(g)).numpy(),
            gt=gt.reshape(-1,1024)[~zero.reshape(-1)],lt=lt,gt_window=gt_window,lt_window=lt_window,
            source_sha256=r.feature_asset_sha256,sampling_identity=identity,sample_indices=index))
    return assets


def weighted_clusters(assets,k=2,seed=17):
    c=torch.from_numpy(np.concatenate([a['context'] for a in assets]));w=torch.tensor(np.concatenate([np.full(len(a['context']),1/len(a['context'])) for a in assets]),dtype=torch.float64)
    if k==1:return c.mean(0,keepdim=True).numpy(),[np.zeros(len(a['context']),int) for a in assets]
    centers,labels,_=spherical_kmeans(c,k,seed)
    for step in range(100):
        new=[]
        for m in range(k):
            mask=labels.eq(m)
            if not mask.any():raise ValueError('拟合真实内容簇为空')
            v=(c[mask]*w[mask,None]).sum(0);new.append(v/v.norm().clamp_min(1e-15))
        centers=torch.stack(new);updated=(c@centers.T).argmax(1)
        if torch.equal(updated,labels):break
        labels=updated
    result=[];offset=0
    for a in assets:result.append(labels[offset:offset+len(a['context'])].numpy());offset+=len(a['context'])
    return centers.numpy(),result


def weighted_moments(values,weights):
    x=np.asarray(values,dtype=np.float64);w=np.asarray(weights,dtype=np.float64)
    if len(x)<2 or not np.isfinite(x).all() or not (w>0).all():raise ValueError('协方差支持非法')
    mean=np.average(x,axis=0,weights=w);cov=np.atleast_2d(np.cov(x.T,aweights=w,ddof=1))
    return mean,cov


def whitening(cov,ridge=1e-5):
    e,v=np.linalg.eigh(cov)
    if e.min()<-1e-8:raise ValueError('协方差非PSD')
    return v/np.sqrt(np.maximum(e,0)+ridge)


def fit_experts(assets,k=2,pool_weight=.5,ridge=1e-5,seed=17,anchors=None):
    centers,routes=weighted_clusters(assets,k,seed);models={};diagnostics=[]
    for branch in ('gt','lt'):
        values=[];weights=[];labels=[];owners=[]
        for i,(a,r) in enumerate(zip(assets,routes)):
            x=a[branch]
            if not len(x):continue
            values.append(x);weights.append(np.full(len(x),1/len(x)));labels.append(r[a[branch+'_window']]);owners.extend([i]*len(x))
        x=np.concatenate(values);w=np.concatenate(weights);label=np.concatenate(labels);owner=np.asarray(owners)
        mean,cov=weighted_moments(x,w)
        if anchors is None:base=StableGaussianParams(mean,whitening(cov,ridge),np.empty(0))
        else:
            base=anchors[branch];np.testing.assert_allclose(mean,base.mean,rtol=0,atol=1e-12)
            # 验证协方差重建与原白化度量一致，实际R0直接复用原参数。
            p=base.whitening@base.whitening.T
            np.testing.assert_allclose((cov+ridge*np.eye(len(mean)))@p,np.eye(len(mean)),rtol=0,atol=1e-8)
        branch_models=[base]
        for m in range(k):
            mask=label==m
            if mask.sum()<2:raise ValueError('专家观察数不足')
            _,cm=weighted_moments(x[mask],w[mask])
            if k==1 or pool_weight==1.:expert=base
            else:expert=StableGaussianParams(base.mean,whitening(pool_weight*cov+(1-pool_weight)*cm,ridge),np.empty(0))
            branch_models.append(expert)
            mass={}
            for i in np.unique(owner[mask]):
                group=assets[i]['source_group'];mass[group]=mass.get(group,0.)+float(w[mask&(owner==i)].sum())
            mv=np.asarray(list(mass.values()));diagnostics.append(dict(branch=branch,expert=m,observations=int(mask.sum()),
                clips=len(np.unique(owner[mask])),source_groups=len(mass),source_effective=float(mv.sum()**2/(mv@mv)),total_weight=float(w[mask].sum())))
        models[branch]=branch_models
    return centers,models,diagnostics


def save_models(path,centers,models):
    arrays={'centers':centers}
    for b,ms in models.items():
        for i,m in enumerate(ms):arrays[f'{b}_{i}_mean']=m.mean;arrays[f'{b}_{i}_whitening']=m.whitening
    np.savez(path,**arrays)


def load_models(path):
    with np.load(path) as z:
        centers=z['centers'].copy();models={b:[StableGaussianParams(z[f'{b}_{i}_mean'].copy(),z[f'{b}_{i}_whitening'].copy(),np.empty(0)) for i in range(len(centers)+1)] for b in ('gt','lt')}
    return centers,models


def load_anchors(root,domain):
    with np.load(Path(root)/f'results/runs/paper_fit_{domain}/gaussians.npz') as z:
        return {b:StableGaussianParams(z[key+'_mean'].copy(),z[key+'_whitening'].copy(),np.empty(0)) for b,key in [('gt','gt'),('lt','lt'),('gs','gs')]}
