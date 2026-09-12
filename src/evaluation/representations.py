"""论文表示消融的统一定义；不改变默认D2检测器。"""
from dataclasses import dataclass

import numpy as np
import torch

from math_utils import GaussianMeanCandidateScorerFloat64, StableGaussianParams
from reference import fit_gaussian


REPRESENTATIONS = ('local_d1', 'local_d2', 'local_raw_d2', 'local_d2_norm', 'global_d2')


def representation_vectors(global_windows, patch_windows):
    """输出[窗口,位置,通道]，保留float32差分及零向量。"""
    g=torch.as_tensor(global_windows,dtype=torch.float32)
    p=torch.as_tensor(patch_windows,dtype=torch.float32)
    if g.ndim!=3 or p.ndim!=4 or p.shape[:2]!=g.shape[:2] or p.shape[-1]!=g.shape[-1]:
        raise ValueError('Global/Patch窗口维度不一致')
    if g.shape[1]<3 or not torch.isfinite(g).all() or not torch.isfinite(p).all():
        raise ValueError('特征非有限或时间长度不足')
    d1=p[:,1:]-p[:,:-1]
    d2=p[:,2:]-2.*p[:,1:-1]+p[:,:-2]
    gd2=g[:,2:]-2.*g[:,1:-1]+g[:,:-2]
    values=dict(local_d1=torch.nn.functional.normalize(d1,dim=-1,eps=1e-12),
                local_d2=torch.nn.functional.normalize(d2,dim=-1,eps=1e-12),
                local_raw_d2=d2,local_d2_norm=torch.linalg.vector_norm(d2,dim=-1,keepdim=True),
                global_d2=torch.nn.functional.normalize(gd2,dim=-1,eps=1e-12))
    return {key:value.reshape(len(value),-1,value.shape[-1]) for key,value in values.items()}


def fit_representation_models(video_vectors):
    """调用者先按视频固定抽样；所有候选仍采用片段等权、同ridge。"""
    if set(video_vectors)!=set(REPRESENTATIONS):raise ValueError('拟合表示集合不完整')
    return {name:fit_gaussian(video_vectors[name],ridge=1e-5) for name in REPRESENTATIONS}


def fit_target_bank(root, dataset, fit_run, output):
    """新提取D2/Global拟合；D1复用逐视频核对相同输入后的缓存。"""
    import json
    from pathlib import Path
    import pandas as pd
    from artifacts import paper_json
    from reference import file_digest,publish_arrays
    root=Path(root);fit_run=Path(fit_run);output=Path(output)
    output.mkdir(parents=True,exist_ok=False)
    fresh=pd.read_csv(fit_run/'prepared_fit.csv',keep_default_na=False)
    original=pd.read_csv(root/'data/manifests/active'/dataset/'fit.csv',keep_default_na=False).set_index('video_id')
    values={name:[] for name in REPRESENTATIONS};inputs=[]
    for row in fresh.itertuples():
        cached=original.loc[row.video_id]
        old_path=root/cached.feature_asset;new_path=Path(row.feature_asset)
        if file_digest(old_path)!=cached.feature_asset_sha256 or file_digest(new_path)!=row.feature_asset_sha256:
            raise ValueError('表示拟合输入hash错误')
        with np.load(new_path,allow_pickle=False) as new,np.load(old_path,allow_pickle=False) as old:
            # D1没有存于主线小资产；不能默默假定旧D1来自同一前向。
            for key in ('raw_d2','global_windows','frame_indices'):
                if not np.array_equal(new[key],old[key]):raise ValueError(f'{row.video_id}新旧{key}不同，须重提D1')
            raw=torch.from_numpy(new['raw_d2'].copy())
            g=torch.from_numpy(new['global_windows'].copy())
            gd2=g[:,2:]-2.*g[:,1:-1]+g[:,:-2]
            values['local_d1'].append(old['unit_d1'].copy())
            values['local_d2'].append(torch.nn.functional.normalize(raw,dim=-1,eps=1e-12).numpy())
            values['local_raw_d2'].append(raw.numpy())
            values['local_d2_norm'].append(torch.linalg.vector_norm(raw,dim=-1,keepdim=True).numpy())
            values['global_d2'].append(torch.nn.functional.normalize(gd2,dim=-1,eps=1e-12).reshape(-1,1024).numpy())
        inputs.append(dict(video_id=row.video_id,fresh_sha256=row.feature_asset_sha256,cached_sha256=cached.feature_asset_sha256))
    models=fit_representation_models(values)
    models['local_diagonal_d2']=fit_gaussian(values['local_d2'],diagonal=True)
    arrays={}
    for name,model in models.items():
        arrays[name+'_mean']=model.mean;arrays[name+'_whitening']=model.whitening
    publish_arrays(output/'gaussians.npz',arrays)
    paper_json(output/'manifest.json',dict(status='gaussian_complete',dataset=dataset,inputs=inputs,
        models_sha256=file_digest(output/'gaussians.npz'),
        provenance='D2/Global重新提取；D1缓存在raw_d2/global/frame_indices逐元素一致后复用；所有Gaussian重新拟合',
        cdf_status='not_started',evaluation_status='not_started'))
    return output


def mean_whitening_controls(source, target, diagonal_target):
    """四格只交换均值/白化，CDF必须由每格重新产生。"""
    if source.mean.shape!=target.mean.shape:raise ValueError('源目标维度不同')
    return dict(source=source,target=target,
        target_mean_source_whitening=StableGaussianParams(target.mean,source.whitening,np.empty(0)),
        source_mean_target_whitening=StableGaussianParams(source.mean,target.whitening,np.empty(0)),
        target_diagonal=diagonal_target)


@dataclass
class RepresentationScorers:
    """相同表示的多个Gaussian共用一次方向矩计算；不同表示不混用。"""
    models: dict
    device: str

    def __post_init__(self):
        self.scorers={};self.names={}
        for representation,models in self.models.items():
            if representation not in REPRESENTATIONS or not models:raise ValueError('未知或空表示模型')
            self.names[representation]=list(models)
            candidates=list(models.values())
            self.scorers[representation]=GaussianMeanCandidateScorerFloat64(candidates,candidates[0].mean,self.device)

    def score(self,global_windows,patch_windows):
        features=representation_vectors(global_windows,patch_windows)
        result={}
        for name,scorer in self.scorers.items():
            scores=scorer.score(features[name])
            result[name]={model:scores[:,i] for i,model in enumerate(self.names[name])}
        return result
