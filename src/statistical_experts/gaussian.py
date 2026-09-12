"""共同完整维度上的高斯型能量；不把在线分数宣称为完整密度。"""
import math
import numpy as np
import torch


def context(g):
    x=g.to(torch.float64)
    norm=torch.linalg.vector_norm(x,dim=-1,keepdim=True)
    unit=x/torch.where(norm==0,1.,norm)
    c=unit.mean(dim=-2);norm=torch.linalg.vector_norm(c,dim=-1,keepdim=True)
    return c/torch.where(norm==0,1.,norm)


def transitions(g):
    # 保留官方float32差分/归一化，拟合保留零向量，查询时再屏蔽零项。
    a=g.detach().cpu().numpy().astype(np.float32,copy=False)
    difference=a[...,1:,:]-a[...,:-1,:]
    norm=np.linalg.norm(difference,axis=-1,keepdims=True)
    return torch.as_tensor(difference/np.where(norm==0,1.,norm),device=g.device,dtype=torch.float64),torch.as_tensor(norm.squeeze(-1)==0,device=g.device)


def moments(x):
    x=x.to(torch.float64)
    if x.shape[-2]<2:raise ValueError('至少两个拟合观察')
    mean=x.mean(dim=-2);centered=x-mean.unsqueeze(-2)
    covariance=centered.transpose(-2,-1)@centered/(x.shape[-2]-1)
    covariance=.5*(covariance+covariance.transpose(-2,-1))
    return mean,covariance


def fitted(x,pooled_cov=None,shrinkage=.5,ridge=1e-5):
    mean,covariance=moments(x)
    raw_covariance=covariance
    if pooled_cov is not None:covariance=(1-shrinkage)*covariance+shrinkage*pooled_cov
    covariance=covariance+ridge*torch.eye(x.shape[-1],device=x.device,dtype=torch.float64)
    chol=torch.linalg.cholesky(covariance)
    return dict(mean=mean,chol=chol,covariance=raw_covariance)


def energy(x,mean,chol):
    x=x.to(torch.float64)
    z=torch.linalg.solve_triangular(chol,(x-mean.unsqueeze(-2)).transpose(-2,-1),upper=False)
    return -.5*(x.shape[-1]*math.log(2*math.pi)+z.square().sum(dim=-2))


def score_pair(g,space,time,t=None,zero=None):
    if t is None:t,zero=transitions(g)
    spatial=energy(g,space['mean'],space['chol']).max(dim=-1).values
    temporal=energy(t,time['mean'],time['chol']).masked_fill(zero,float('inf')).min(dim=-1).values
    return torch.stack([spatial,temporal],dim=-1)


def spherical_kmeans(c,k=4,seed=17,iterations=100):
    if c.ndim!=2 or len(c)<k:raise ValueError('聚类输入不够')
    rng=np.random.default_rng(seed);chosen=[int(rng.integers(len(c)))]
    while len(chosen)<k:
        distance=(1-(c@c[chosen].T).max(dim=1).values).clamp_min(0)
        distance[chosen]=0;prob=distance.cpu().numpy()
        selected=int(rng.choice(len(c),p=prob/prob.sum())) if prob.sum()>1e-15 else next(i for i in range(len(c)) if i not in chosen)
        chosen.append(selected)
    centers=c[chosen].clone();previous=None
    for step in range(iterations):
        scores=c@centers.T;labels=scores.argmax(dim=1)
        if previous is not None and torch.equal(labels,previous):break
        updated=[]
        for m in range(k):
            members=c[labels==m]
            vector=members.mean(dim=0) if len(members) else c[scores.max(dim=1).values.argmin()]
            updated.append(vector/vector.norm().clamp_min(1e-15))
        centers=torch.stack(updated);previous=labels
    labels=(c@centers.T).argmax(dim=1)
    return centers,labels,step+1


def neighbor_indices(query_context,fit_context,k):
    if k>len(fit_context):raise ValueError('邻居数超过拟合库')
    scores=query_context.to(torch.float64)@fit_context.to(torch.float64).T
    # 确定集合后按拟合行号排序，避免内容排序改变协方差归约顺序。
    return torch.sort(torch.argsort(scores,dim=-1,descending=True,stable=True)[...,:k],dim=-1).values


def random_indices(keys,n,k,seed):
    output=[]
    for key in keys:
        s=int(str(key)[:16],16)^seed
        output.append(np.sort(np.random.default_rng(s).choice(n,k,replace=False)))
    return np.stack(output)
