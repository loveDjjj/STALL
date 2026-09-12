"""固定三seed模型的源级配对区间；不把seed平均预测冒充平均AUC。"""
import hashlib
import json
from concurrent.futures import ProcessPoolExecutor
import multiprocessing
from pathlib import Path
import numpy as np
import pandas as pd
from artifacts import atomic_csv,paper_json
from reference import file_digest
from discriminative_moe.run import configuration


class RankedMetrics:
    """固定分数排序与并列组，变化仅为源组bootstrap权重。"""
    def __init__(self,labels,scores):
        labels=np.asarray(labels,dtype=bool);scores=np.asarray(scores)
        if labels.ndim!=1 or labels.shape!=scores.shape or not np.isfinite(scores).all():raise ValueError('评价输入非法')
        self.order=np.argsort(-scores,kind='stable');self.labels=labels[self.order]
        values=scores[self.order]
        self.starts=np.r_[0,np.flatnonzero(values[1:]!=values[:-1])+1]

    def __call__(self,weights):
        w=np.asarray(weights)[self.order]
        if not np.isfinite(w).all() or (w<0).any():raise ValueError('权重非法')
        positive=np.add.reduceat(w*self.labels,self.starts);negative=np.add.reduceat(w*(~self.labels),self.starts)
        p,n=positive.sum(),negative.sum()
        if p<=0 or n<=0:raise ValueError('重抽样缺少一类')
        tp=positive.cumsum();total=(positive+negative).cumsum()
        auc=(negative*(tp-.5*positive)).sum()/(p*n)
        precision=np.divide(tp,total,out=np.zeros_like(tp,dtype=float),where=total>0)
        ap=(positive*precision).sum()/p
        return np.array([auc,ap])


def contrasts():
    result={}
    for inp in ('G','GT','GTL'):
        result[f'{inp}_moe_vs_mlp']=((inp,'moe'),(inp,'mlp'))
        result[f'{inp}_moe_vs_uniform_matched']=((inp,'moe'),(inp,'uniform_matched'))
    for head in ('linear','mlp','moe'):
        result[f'{head}_GTL_vs_GT']=(('GTL',head),('GT',head))
        result[f'{head}_GT_vs_G']=(('GT',head),('G',head))
    return result


def paired_seed_contrast(scores,pairs,a,b,seeds=(17,29,43),iterations=1000,seed=17):
    """每次源组权重在同域所有生成器、两个模型和三个seed间共享。"""
    lookup={}
    for spec in (a,b):
        for s in seeds:
            q=scores[(scores.input==spec[0])&(scores['head']==spec[1])&(scores.seed==s)]
            lookup[spec,s]=q.set_index('video_id',verify_integrity=True)
    first=lookup[a,seeds[0]]
    for q in lookup.values():
        if set(q.index)!=set(first.index):raise ValueError('对比视频身份不一致')
        for column in ('dataset','subset','source_model','split_group'):
            if not q.loc[first.index,column].equals(first[column]):raise ValueError('来源或标签漂移')
    membership=first[['dataset','split_group']].drop_duplicates().groupby('split_group').dataset.nunique()
    if membership.gt(1).any():raise ValueError('存在跨域共享源，需要联合跨域重抽样')
    point_rows=[];draws={}
    for domain,dp in pairs.groupby('dataset'):
        domain_first=first[first.dataset.eq(domain)];groups=sorted(domain_first.split_group.unique());mapping={g:i for i,g in enumerate(groups)}
        cells=[]
        for generator,cell in dp.groupby('generator'):
            ids=cell.video_id.to_numpy();base=first.loc[ids];y=base.subset.eq('real').to_numpy()
            if not y.any() or y.all() or y.sum()!=(~y).sum():raise ValueError('配对表不平衡')
            if not base.loc[~y,'source_model'].eq(generator).all():raise ValueError('生成器错配')
            idx=np.array([mapping[g] for g in base.split_group])
            ranked=[(RankedMetrics(y,lookup[a,s].loc[ids,'final_score'].to_numpy()),
                     RankedMetrics(y,lookup[b,s].loc[ids,'final_score'].to_numpy())) for s in seeds]
            cells.append((idx,y,ranked))
        def compute(weights):
            values=[]
            for idx,y,ranked in cells:
                w=weights[idx]
                if not w[y].sum() or not w[~y].sum():return None
                values.append(np.mean([ra(w)-rb(w) for ra,rb in ranked],axis=0))
            return np.mean(values,axis=0)
        point=compute(np.ones(len(groups)));salt=int.from_bytes(hashlib.sha256(domain.encode()).digest()[:4],'little')
        rng=np.random.default_rng(np.random.SeedSequence([seed,salt]));samples=[];attempts=0
        while len(samples)<iterations:
            attempts+=1
            if attempts>100*iterations:raise RuntimeError('有效源组重抽样不足')
            value=compute(rng.poisson(1,len(groups)))
            if value is not None:samples.append(value)
        draws[domain]=np.asarray(samples)
        for i,metric in enumerate(('auc','ap_real')):
            point_rows.append(dict(dataset=domain,metric=metric,delta=point[i],ci95_low=np.quantile(draws[domain][:,i],.025),
                                   ci95_high=np.quantile(draws[domain][:,i],.975),source_groups=len(groups),generator_cells=len(cells)))
    all_draws=np.mean(list(draws.values()),axis=0)
    for i,metric in enumerate(('auc','ap_real')):
        point=np.mean([r['delta'] for r in point_rows if r['metric']==metric])
        point_rows.append(dict(dataset='Average',metric=metric,delta=point,ci95_low=np.quantile(all_draws[:,i],.025),ci95_high=np.quantile(all_draws[:,i],.975)))
    return pd.DataFrame(point_rows)


def _one(task):
    root,name=task;root=Path(root);out=root/configuration(root)['run_directory'];a,b=contrasts()[name];dest=out/'intervals'/name
    identity=dict(scores=file_digest(out/'test_scores.csv.gz'),pairs=file_digest(root/'results/paper_complete/pairs.csv'),
                  a=a,b=b,seeds=[17,29,43],iterations=1000,seed=17,code=file_digest(Path(__file__)))
    marker=dest/'manifest.json'
    # JSON将tuple归一为list，比较经序列化的同一身份。
    identity=json.loads(json.dumps(identity))
    if marker.exists():
        m=json.loads(marker.read_text())
        if m['inputs']!=identity or file_digest(dest/'difference.csv')!=m['files']['difference.csv']:raise ValueError('区间恢复合同改变')
        return name
    scores=pd.read_csv(out/'test_scores.csv.gz',float_precision='round_trip');pairs=pd.read_csv(root/'results/paper_complete/pairs.csv')
    value=paired_seed_contrast(scores,pairs,tuple(a),tuple(b));dest.mkdir(parents=True,exist_ok=True);atomic_csv(dest/'difference.csv',value)
    paper_json(marker,dict(status='completed',inputs=identity,files={'difference.csv':file_digest(dest/'difference.csv')}))
    return name


def route_summary(out):
    q=pd.read_csv(out/'test_routes.csv.gz');rows=[];interventions=[]
    for keys,b in q.groupby(['input','head','fold','seed']):
        inp,head,fold,seed=keys;n=int(b.experts.iloc[0]);gate_cols=[f'gate_{m}' for m in range(n)]
        blocks=[('all','all',b)]+[('subset',name,g) for name,g in b.groupby('subset')]+[('source_model',name,g) for name,g in b.groupby('source_model')]
        for by,name,g in blocks:
            mass=g[gate_cols].mean().to_numpy();entropy=float(-(mass*np.log(np.maximum(mass,1e-12))).sum())
            expert_stats={}
            for i in range(n):
                expert_stats[f'expert_{i}_mean_fake']=g[f'expert_fake_{i}'].mean()
                expert_stats[f'expert_{i}_std_fake']=g[f'expert_fake_{i}'].std(ddof=0)
            rows.append(dict(input=inp,head=head,fold=fold,seed=seed,by=by,group=name,clips=len(g),experts=n,
                average_window_effective_experts=g.effective_experts.mean(),aggregate_effective_experts=np.exp(entropy),
                dominant_expert_mass=float(mass.max()),**{f'mean_gate_{i}':v for i,v in enumerate(mass)},**expert_stats))
        if head=='moe':
            # 测试时移除路由只作干预诊断；不能替代独立训练的uniform对照。
            r=b[['video_id','dataset','subset','source_model','split_group']].copy()
            r['final_score']=1-b[[f'expert_fake_{i}' for i in range(n)]].mean(axis=1)
            interventions.append(r.assign(input=inp,head='uniform_intervention',seed=seed))
    atomic_csv(out/'route_summary.csv',pd.DataFrame(rows))
    pd.concat(interventions,ignore_index=True).to_csv(out/'routing_intervention_scores.csv.gz',index=False)


def analyze(root,args=None):
    root=Path(root);out=root/configuration(root)['run_directory']
    with ProcessPoolExecutor(max_workers=4,mp_context=multiprocessing.get_context('spawn')) as pool:
        for name in pool.map(_one,[(str(root),name) for name in contrasts()]):print('MoE CI',name,flush=True)
    atomic_csv(out/'confidence_intervals.csv',pd.concat([pd.read_csv(out/'intervals'/n/'difference.csv').assign(contrast=n) for n in contrasts()],ignore_index=True))
    route_summary(out)
    paper_json(out/'analysis_manifest.json',dict(status='completed',files={p:file_digest(out/p) for p in ['confidence_intervals.csv','route_summary.csv','routing_intervention_scores.csv.gz']}))
