"""逐资产/逐配置验收，完成状态不能仅由进程退出推断。"""
import json
from pathlib import Path
import numpy as np
import pandas as pd
from artifacts import paper_json
from config import config_digest
from reference import file_digest
from evaluation.tables import evaluate_fixed_pairs
from statistical_experts.manifests import settings
from statistical_experts.cache import load_feature,feature_identity
from statistical_experts.evaluation import read_raw,METHODS
from statistical_experts.gaussian import random_indices
from statistical_experts.analysis import CONTRASTS


def verify(root):
    root=Path(root);c=settings(root);run=root/c['run_directory'];directory=run/'evaluation'
    manifest=json.loads((root/c['manifest_directory']/'manifest.json').read_text())
    for p,h in manifest['inputs'].items():
        if file_digest(p)!=h:raise ValueError('研究输入清单发生变化')
    for p,h in manifest['files'].items():
        if file_digest(root/c['manifest_directory']/p)!=h:raise ValueError('研究清单损坏')
    frame=pd.read_csv(root/c['manifest_directory']/'windows.csv',keep_default_na=False)
    cached=config_digest(feature_identity(root,c));hash_roles={};cache_count=0
    for row in frame.itertuples(index=False):
        path=root/c['cache_directory']/(row.cache_key+'.npz');load_feature(path,cached,row)
        with np.load(path,allow_pickle=False) as z:
            stat=(root/row.video_path).stat()
            if int(z['source_bytes'])!=stat.st_size or int(z['source_mtime_ns'])!=stat.st_mtime_ns:raise ValueError('缓存原视频变化')
            digest=str(z['tensor_sha256']);hash_roles.setdefault((row.length,digest),set()).add(row.role)
        cache_count+=1
    overlap=[(length,digest,sorted(roles)) for (length,digest),roles in hash_roles.items() if len(roles)>1]
    if overlap:raise ValueError(f'参考/评价之间存在完全相同Global窗口：{overlap[:3]}')
    fit_groups=set(frame.loc[frame.role=='fit','source_group']);cdf_groups=set(frame.loc[frame.role=='cdf','source_group'])
    if len(fit_groups)!=2200 or len(cdf_groups)!=2000 or fit_groups&cdf_groups:raise ValueError('参考角色交叉或数量不同')
    scores=pd.read_csv(directory/'video_scores.csv.gz',float_precision='round_trip')
    index=scores.set_index(['video_id','variant'])
    hybrids=pd.read_csv(directory/'hybrid_scores.csv.gz',float_precision='round_trip')
    for method in ('offline','online'):
        for branch in ('spatial','temporal'):
            variant=f'{method}_{branch}_hybrid';part=hybrids[hybrids.variant==variant].set_index('video_id')
            opposite='temporal' if branch=='spatial' else 'spatial'
            a=index.xs(method+'_'+branch,level='variant').loc[part.index,'final_score']
            b=index.xs('pooled_'+opposite,level='variant').loc[part.index,'final_score']
            if not np.array_equal(part.final_score.to_numpy(),(.5*a+.5*b).to_numpy()) or not part.method.eq(method).all():raise ValueError('混合分支元数据/分数错误')
    refs=np.load(directory/'cdf_arrays.npz');raw_count=0;neighbor_checks=0
    for length in (8,16):
        f,records,spec=read_raw(root,length)
        if spec['input_features']!=cached:raise ValueError('统计输入不是验收缓存')
        model_path=run/f'models_{length}/models.pt'
        expected_engine=config_digest(dict(model=file_digest(model_path),config=c,
            gaussian=file_digest(root/'src/statistical_experts/gaussian.py'),engine=file_digest(root/'src/statistical_experts/engine.py')))
        if spec['engine_identity']!=expected_engine or spec['code_sha256']!=file_digest(root/'src/statistical_experts/scoring.py'):raise ValueError('模型或统计实现与检查点不一致')
        expected_random=random_indices(f.random_source_key.tolist(),2200,c['neighbors'],c['seed'])
        for i,row in enumerate(records):
            if row['neighbors']['random']!=expected_random[i].tolist():raise ValueError('随机邻居规则不一致')
            neighbor_checks+=1
        mask=f.role.eq('cdf').to_numpy()
        for method in METHODS:
            values=np.asarray([[float(v) for v in r['scores'][method]] for r in records])
            reference=np.sort(values[mask],axis=0)
            if not np.array_equal(reference,refs[f'{method}_{length}']):raise ValueError('CDF不对应整流程原始分数')
            pct=np.stack([np.searchsorted(reference[:,j],values[:,j],side='right')/2000 for j in (0,1)],axis=1)
            for i,row in enumerate(f.itertuples(index=False)):
                if row.role!='evaluation':continue
                for branch,value in [('spatial',pct[i,0]),('temporal',pct[i,1]),('final',pct[i].mean())]:
                    if float(index.loc[(row.video_id,method+'_'+branch),'final_score'])!=value:raise ValueError('导出分数与对应CDF不同')
        raw_count+=len(records)
    pairs=pd.read_csv(root/c['manifest_directory']/'pairs.csv');expected_cells=set(map(tuple,pairs[['dataset','generator']].drop_duplicates().to_numpy()))
    if len(expected_cells)!=23:raise ValueError('评价范围不是23单元')
    generators=pd.read_csv(directory/'generator_metrics.csv',float_precision='round_trip')
    summary=pd.read_csv(directory/'summary.csv',float_precision='round_trip')
    for variant,part in scores.groupby('variant'):
        if set(part.video_id)!=set(pairs.video_id):raise ValueError('方法评价身份不同')
        recalculated=evaluate_fixed_pairs(part,pairs)
        g=generators[generators.variant==variant]
        if len(g)!=23 or set(map(tuple,g[['dataset','generator']].to_numpy()))!=expected_cells:raise ValueError('单元缺失')
        for metric in ['auc','real_positive_ap','fake_positive_ap']:
            actual=recalculated['macro_metrics'][metric].iloc[0]
            if abs(actual-summary.loc[summary.variant==variant,metric].iloc[0])>1e-12:raise ValueError('三数据集等权Average错误')
    intervals=pd.read_csv(directory/'confidence_intervals.csv')
    if set(intervals.contrast)!=set(CONTRASTS) or len(intervals)!=len(CONTRASTS)*8:raise ValueError('配对区间不完整')
    for name in CONTRASTS:
        p=run/'intervals'/name;meta=json.loads((p/'manifest.json').read_text())
        if meta['inputs']['iterations']!=1000 or meta['status']!='completed':raise ValueError('区间次数/状态错误')
        for key,file in [('scores',directory/'video_scores.csv.gz'),('hybrids',directory/'hybrid_scores.csv.gz'),('pairs',root/c['manifest_directory']/'pairs.csv')]:
            if file_digest(file)!=meta['inputs'][key]:raise ValueError('区间引用其他分数')
        for file,h in meta['files'].items():
            if file_digest(p/file)!=h:raise ValueError('区间文件损坏')
    for p in [run/'models_8/manifest.json',run/'models_16/manifest.json',run/'runtime/manifest.json',run/'real_diagnostic/manifest.json']:
        m=json.loads(p.read_text())
        if m['status']!='completed':raise ValueError('拟合/诊断/计时未完成')
        if 'models_sha256' in m:
            if file_digest(p.parent/'models.pt')!=m['models_sha256'] or m['config']!=c:raise ValueError('拟合模型被替换')
        for file,h in m.get('files',{}).items():
            if file_digest(p.parent/file)!=h:raise ValueError('辅助资产损坏')
    timing=pd.read_csv(run/'runtime/timings.csv')
    if timing.queries.sum()!=480 or timing.max_score_error.max()>1e-10:raise ValueError('实际成本测试不完整或不对应评分')
    audit=dict(status='verified',feature_windows=cache_count,statistical_queries=raw_count,random_neighbor_checks=neighbor_checks,
        cross_role_identical_features=len(overlap),methods=5,branch_configurations=scores.variant.nunique(),generator_cells=23,
        evaluation_clip_ids=scores.video_id.nunique(),hybrid_configurations=hybrids.variant.nunique(),contrasts=len(CONTRASTS),bootstrap_iterations=1000,
        post_encoder_timing_queries=int(timing.queries.sum()),
        limits='固定参考/单随机seed；像素逐元素审计为200真实pilot，评价缓存另匹配官方GS/GT摘要及Final锚点；不声称完整语义去重')
    paper_json(run/'verification.json',audit);print(json.dumps(audit,ensure_ascii=False),flush=True)


if __name__=='__main__':verify(Path.cwd())
