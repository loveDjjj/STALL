"""论文实验资产准备；复用主线拟合/评分，不恢复历史研究入口。"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

from artifacts import paper_json,checkpoint_read
from config import config_digest
from reference import file_digest,publish_arrays
from reference_fit import prepare_fit_assets,fit_from_assets
from workflow import video_file_states


def fit_source_bank(root,config,output):
    """同预算VATEX200原视频重新提取，角色明确且不与CDF/阈值交叉。"""
    root=Path(root);output=Path(output)
    source=root/'data/catalog/vatex_source_fit200.csv'
    frame=pd.read_csv(source,keep_default_na=False)
    frame['video_id']='vatex:'+frame.video_path
    frame['legacy_path']=frame.video_path
    frame['dataset']='vatex';frame['split']='fit';frame['source_group']='vatex:'+frame.source_id
    if len(frame)!=200 or frame.source_id.nunique()!=200:raise ValueError('源域拟合预算不是200独立ID')
    for role in ('cdf','threshold'):
        other=pd.read_csv(root/'data/manifests/active/vatex'/f'{role}.csv')
        if set(frame.video_path)&set(other.video_path):raise ValueError('源域fit与CDF/threshold交叉')
    states=video_file_states(root,frame)
    inputs=dict(source_sha256=file_digest(source),config=config,files=states)
    identity=config_digest(inputs)
    output.mkdir(parents=True,exist_ok=False)
    paper_json(output/'identity.json',inputs)
    frame.to_csv(output/'fit.csv',index=False)
    def progress(stage,n,total,vid):
        paper_json(output/'progress.json',dict(stage=stage,completed=n,total=total,video_id=vid))
        if n%16==0 or n==total:print(f'[source:{stage}] {n}/{total}',flush=True)
    prepared=prepare_fit_assets(root,config,frame,states,output,identity,progress)
    models,statistics=fit_from_assets(root,prepared,progress=progress)
    arrays={}
    for name,model in models.items():arrays[name+'_mean']=model.mean;arrays[name+'_whitening']=model.whitening
    publish_arrays(output/'gaussians.npz',arrays)
    paper_json(output/'manifest.json',dict(status='gaussian_complete',identity=identity,statistics=statistics,
        models_sha256=file_digest(output/'gaussians.npz'),cdf_status='not_started'))
    return output


def run_shared_evidence(root,config,dataset,manifest,output,*,include_uniform=False,rank=0,world_size=1,limit=None,sequential_decode=False,baseline_only=None):
    """一个视频前向服务主比较、表示和度量对照；selector由配置显式指定。"""
    from math_utils import StableGaussianParams
    from branches.global_branch import load_official_stall_parameters
    from evaluation.representations import mean_whitening_controls,REPRESENTATIONS
    from evaluation.evidence import SharedEvidenceScorer,score_evidence_manifest
    root=Path(root);output=Path(output)
    # 本轮FC1/Uniform3只回答观察预算问题；显式False仍可运行完整证据。
    if baseline_only is None:
        baseline_only=(config['selection']['name'],config['selection']['k']) in {('feature_change',1),('uniform',3)}
    paths=[root/'results/runs'/f'paper_fit_{dataset}'/'gaussians.npz',
           root/'results/runs/paper_fit_source/gaussians.npz',
           root/'results/runs'/f'paper_representation_fit_{dataset}'/'gaussians.npz',
           root/'precomputed/stall_params_vatex_dino_v3.npz']
    def load(path):
        with np.load(path,allow_pickle=False) as z:
            return {k[:-5]:StableGaussianParams(z[k].copy(),z[k[:-5]+'_whitening'].copy(),np.empty(0))
                    for k in z.files if k.endswith('_mean')}
    target,source,representations=[load(p) for p in paths[:3]]
    official=load_official_stall_parameters(paths[3])
    global_models=dict(target=(target['gs'],target['gt']),source=(source['gs'],source['gt']),
                       official=(official['global_spatial'],official['global_t1']))
    local={name:{'target':representations[name]} for name in REPRESENTATIONS}
    local['local_d2']=mean_whitening_controls(source['lt'],target['lt'],representations['local_diagonal_d2'])
    frame=pd.read_csv(manifest,keep_default_na=False)
    if frame.video_id.duplicated().any():raise ValueError('评价/CDF身份重复')
    roles=set(frame['split'])
    if roles=={'evaluation'}:
        if not frame.dataset.eq(dataset).all():raise ValueError('评价域与目标统计模型不一致')
    elif roles in ({'cdf'},{'threshold'}):
        if not frame.subset.eq('real').all() or not frame.dataset.eq('vatex').all():raise ValueError('外部参考必须为VATEX真实视频')
        if roles=={'cdf'} and not include_uniform:raise ValueError('CDF任务必须包含Global Uniform首窗')
    else:raise ValueError('共享评分清单角色不明确')
    if limit is not None:frame=frame.iloc[:limit].copy()
    if file_digest(root/config['encoder']['weights'])!=config['encoder']['weights_sha256']:raise ValueError('编码器权重错误')
    scorer_type=SharedEvidenceScorer
    if baseline_only:
        from evaluation.budget import BudgetEvidenceScorer
        scorer_type=BudgetEvidenceScorer
        config=dict(config,evidence_scope='baseline_budget',budget_scorer_sha256=file_digest(root/'src/evaluation/budget.py'))
    window_frames=config.get('evidence_window_frames',16)
    if window_frames not in (8,16):raise ValueError('证据窗口只支持已定义的8或16帧')
    if window_frames==8:
        from evaluation.short_video import ShortWindowMixin
        class ShortScorer(ShortWindowMixin,scorer_type):pass
        scorer_type=ShortScorer
        config=dict(config,short_scorer_sha256=file_digest(root/'src/evaluation/short_video.py'))
        if sequential_decode:raise ValueError('8帧使用独立单窗路径，不叠加16帧顺序解码规划器')
    if sequential_decode:
        from data.sequential_decode import SequentialDecodeMixin
        class SequentialScorer(SequentialDecodeMixin,scorer_type):pass
        scorer_type=SequentialScorer
        config=dict(config,evidence_decoder='sequential selected frames / original fallback',
                    evidence_decoder_sha256=file_digest(root/'src/data/sequential_decode.py'))
    done_path=output/f'completed_rank_{rank}.json'
    if done_path.exists():
        spec=json.loads((output/f'identity_rank_{rank}.json').read_text());done=json.loads(done_path.read_text())
        if spec['config']!=config or spec['manifest_sha256']!=config_digest(frame.to_dict('records')) or spec['include_uniform']!=include_uniform:
            raise ValueError('已完成分片配置/清单不匹配')
        if spec['models']!={str(p):file_digest(p) for p in paths} or spec['files']!=video_file_states(root,frame):raise ValueError('已完成分片输入改变')
        for name,digest in spec['code'].items():
            if file_digest(root/name)!=digest:raise ValueError('已完成分片实现改变')
        identity=config_digest(spec)
        if done['identity']!=identity or done['world_size']!=world_size:raise ValueError('完成分片身份不符')
        for i,row in enumerate(frame.itertuples()):
            if i%world_size==rank:checkpoint_read(output/'raw'/f'{i:06d}.json',identity,row.video_id)
        print(f'[复用完成分片] {output} rank={rank}',flush=True)
        return output
    if all((output/'raw'/f'{i:06d}.json').exists() for i in range(rank,len(frame),world_size)):
        # 重新分片后可能已有全部行而尚无该rank完成标记；原函数仍逐行验hash。
        score_evidence_manifest(root,config,frame,None,output,paths,include_uniform=include_uniform,rank=rank,world_size=world_size)
        print(f'[复用重分片] {output} rank={rank}',flush=True)
        return output
    args=(target['gs'],target['gt'],target['lt'],source['lt'].mean,config['runtime']['device']) if baseline_only else (global_models,local,config['runtime']['device'])
    scorer=scorer_type(*args,dino_repo=str(root/config['encoder']['repo']),dino_weights=str(root/config['encoder']['weights']))
    score_evidence_manifest(root,config,frame,scorer,output,paths,include_uniform=include_uniform,rank=rank,world_size=world_size)
    return output
