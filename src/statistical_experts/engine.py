"""总体、离线和按查询统计共享同一Global输入与高斯能量定义。"""
import json
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from artifacts import paper_json
from config import config_digest
from reference import file_digest
from statistical_experts.cache import load_feature,feature_identity
from statistical_experts.manifests import settings
from statistical_experts.gaussian import context,transitions,fitted,score_pair,spherical_kmeans,neighbor_indices,random_indices


def load_bank(root,length,device):
    root=Path(root);c=settings(root)
    frame=pd.read_csv(root/c['manifest_directory']/'windows.csv',keep_default_na=False)
    fit=frame[(frame.role=='fit')&(frame.length==length)].reset_index(drop=True)
    identity=config_digest(feature_identity(root,c))
    g=torch.from_numpy(np.stack([load_feature(root/c['cache_directory']/(r.cache_key+'.npz'),identity,r) for r in fit.itertuples()]))
    # T1在CPU按官方float32规则计算一次，之后全部统计使用float64。
    t,_=transitions(g)
    s=g[torch.arange(len(g)),torch.as_tensor(fit.spatial_fit_index.to_numpy(dtype=np.int64))].double()
    return dict(spatial=s.to(device),temporal=t.to(device),context=context(g).to(device),ids=fit.video_id.tolist(),
        frame=fit,identity=identity)


def fit_models(root,length,device):
    root=Path(root);c=settings(root);out=root/c['run_directory']/f'models_{length}'
    if (out/'manifest.json').exists():
        m=json.loads((out/'manifest.json').read_text())
        if file_digest(out/'models.pt')!=m['models_sha256']:raise ValueError('专家模型hash不符')
        return
    bank=load_bank(root,length,device)
    pool_s=fitted(bank['spatial'],ridge=c['ridge']);pool_t=fitted(bank['temporal'].flatten(0,1),ridge=c['ridge'])
    centers,labels,steps=spherical_kmeans(bank['context'],c['experts'],c['seed'])
    models_s=[];models_t=[];counts=[];fallback=[]
    for k in range(c['experts']):
        idx=torch.where(labels==k)[0];n=len(idx);counts.append(n);fallback.append(n<c['min_fit_sources'])
        if fallback[-1]:ms,mt=pool_s,pool_t
        else:
            ms=fitted(bank['spatial'][idx],pool_s['covariance'],c['shrinkage'],c['ridge'])
            mt=fitted(bank['temporal'][idx].flatten(0,1),pool_t['covariance'],c['shrinkage'],c['ridge'])
        models_s.append(ms);models_t.append(mt)
    def compact(models):return {key:torch.stack([m[key] for m in models]).cpu() for key in ('mean','chol')}
    payload=dict(pool_s={k:v.cpu() for k,v in pool_s.items()},pool_t={k:v.cpu() for k,v in pool_t.items()},
        offline_s=compact(models_s),offline_t=compact(models_t),centers=centers.cpu(),labels=labels.cpu(),fit_ids=bank['ids'])
    out.mkdir(parents=True,exist_ok=True);temp=out/'models.tmp';torch.save(payload,temp);temp.replace(out/'models.pt')
    paper_json(out/'manifest.json',dict(status='completed',config=c,length=length,cache_identity=bank['identity'],
        fit_manifest_sha256=file_digest(root/c['manifest_directory']/'windows.csv'),cluster_counts=counts,
        fallback_clusters=fallback,kmeans_iterations=steps,fit_sources=len(bank['ids']),
        models_sha256=file_digest(out/'models.pt'),code_sha256=file_digest(Path(__file__)),
        gaussian_code_sha256=file_digest(root/'src/statistical_experts/gaussian.py')))
    print('fit',length,counts,'fallback',fallback,flush=True)


class Engine:
    def __init__(self,root,length,device):
        root=Path(root);self.config=settings(root);self.device=device;self.length=length
        out=root/self.config['run_directory']/f'models_{length}'
        manifest=json.loads((out/'manifest.json').read_text())
        if manifest['config']!=self.config or file_digest(out/'models.pt')!=manifest['models_sha256']:raise ValueError('统计配置/模型不同')
        self.model=torch.load(out/'models.pt',map_location=device,weights_only=True)
        self.bank=load_bank(root,length,device)
        if self.bank['ids']!=self.model['fit_ids']:raise ValueError('检索库身份顺序不同')
        self.identity=config_digest(dict(model=file_digest(out/'models.pt'),config=self.config,
            gaussian=file_digest(root/'src/statistical_experts/gaussian.py'),engine=file_digest(Path(__file__))))

    def score(self,g,keys,online=True):
        # g来自CPU缓存；差分和归一化在CPU精确对应官方定义。
        t,zero=transitions(g);cc=context(g).to(self.device)
        g=g.to(self.device);t=t.to(self.device);zero=zero.to(self.device)
        cluster=(cc@self.model['centers'].T).argmax(dim=1)
        scores={}
        scores['pooled']=score_pair(g,self.model['pool_s'],self.model['pool_t'],t,zero)
        os={k:v[cluster] for k,v in self.model['offline_s'].items()};ot={k:v[cluster] for k,v in self.model['offline_t'].items()}
        scores['offline']=score_pair(g,os,ot,t,zero)
        selected={}
        if online:
            selected['online']=neighbor_indices(cc,self.bank['context'],self.config['neighbors'])
            selected['random']=torch.as_tensor(random_indices(keys,len(self.bank['ids']),self.config['neighbors'],self.config['seed']),device=self.device)
            for name,idx in selected.items():
                ms=fitted(self.bank['spatial'][idx],self.model['pool_s']['covariance'],self.config['shrinkage'],self.config['ridge'])
                mt=fitted(self.bank['temporal'][idx].flatten(1,2),self.model['pool_t']['covariance'],self.config['shrinkage'],self.config['ridge'])
                scores[name]=score_pair(g,ms,mt,t,zero)
        return {k:v.detach().cpu().numpy() for k,v in scores.items()},cluster.cpu().numpy(),{k:v.cpu().numpy() for k,v in selected.items()}
