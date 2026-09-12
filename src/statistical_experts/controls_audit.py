"""独立验收：直接计数重算全部A百分位与融合，不只相信完成标志。"""
import json
from pathlib import Path
import numpy as np
import pandas as pd
from artifacts import paper_json
from config import config_digest
from reference import file_digest
from evaluation.tables import evaluate_fixed_pairs
from statistical_experts.controls import prepare, check_files, CONTRASTS
from statistical_experts.evaluation import read_raw


def audit(root):
    root=Path(root); spec=prepare(root); out=root/spec['config']['run_directory']
    source=root/spec['config']['source_directory']
    for name,digest in spec['files'].items():
        if name.startswith(('src/','configs/')) and file_digest(out/'source_snapshot'/name)!=digest:
            raise ValueError('冻结源码快照不符')
    old=pd.read_csv(source/'evaluation/video_scores.csv.gz',float_precision='round_trip')
    new=pd.read_csv(out/'video_scores.csv.gz',float_precision='round_trip')
    indices=old.loc[old.variant.eq('official_final'),'video_id'].to_numpy()
    def vector(table,key):
        f=table[table.variant.eq(key)].set_index('video_id',verify_integrity=True)
        if set(f.index)!=set(indices):raise ValueError('配置视频覆盖不同')
        return f.loc[indices,'final_score'].to_numpy()
    direct=pd.Series(index=indices,dtype=float); checked=0
    for length in (8,16):
        frame,raw,_=read_raw(root,length)
        refs=frame.role.eq('cdf').to_numpy(); evaluation=frame.role.eq('evaluation').to_numpy()
        routes=np.array([r['cluster'] for r in raw]); q=np.array([float(r['scores']['offline'][1]) for r in raw])
        with np.load(out/f'cdf_scores_{length}.npz') as z:
            matrix=z['scores']
            np.testing.assert_array_equal(z['video_ids'],frame.loc[refs,'video_id'].to_numpy())
            np.testing.assert_array_equal(matrix[np.arange(2000),routes[refs]],q[refs])
            result=np.empty(evaluation.sum())
            qe=q[evaluation]; re=routes[evaluation]
            # 独立于searchsorted实现，分批直接计数，不分簇筛选参考身份。
            for start in range(0,len(qe),128):
                end=min(start+128,len(qe))
                result[start:end]=(matrix[:,re[start:end]]<=qe[None,start:end]).sum(axis=0)/2000
            direct.loc[frame.loc[evaluation,'video_id']]=result; checked+=len(result)
    if direct.isna().any():raise ValueError('A百分位覆盖缺失')
    a=direct.loc[indices].to_numpy(); s=vector(old,'official_spatial'); p=vector(old,'pooled_spatial')
    expected={'official_final':vector(old,'official_final'),
              'official_s_offline_t_a':.5*s+.5*a,'pooled_s_offline_t_a':.5*p+.5*a,
              'offline_t_a':a,'offline_t_b':vector(old,'offline_temporal'),
              'pooled_s_offline_t_b':.5*p+.5*vector(old,'offline_temporal')}
    for m in ('pooled','offline','online','random'):
        name=f'official_s_{m}_t'+('_b' if m in ('offline','online') else '')
        expected[name]=.5*s+.5*vector(old,m+'_temporal')
    if set(new.variant)!=set(expected):raise ValueError('方法集合错误')
    for name,values in expected.items():np.testing.assert_array_equal(vector(new,name),values)
    pairs=pd.read_csv(root/'data/manifests/global_experts/pairs.csv')
    keys={'generator_metrics':['dataset','generator'],'dataset_metrics':['dataset'],
          'macro_metrics':['scope'],'full_population_metrics':['dataset']}
    for name,group in new.groupby('variant'):
        tables=evaluate_fixed_pairs(group,pairs)
        for table,columns in keys.items():
            actual=tables[table].replace({'scope':{'Macro-3':'Average'}}).set_index(columns).sort_index()
            saved=pd.read_csv(out/(table+'.csv'),float_precision='round_trip')
            saved=saved[saved.variant.eq(name)].drop(columns='variant').set_index(columns).sort_index()
            pd.testing.assert_frame_equal(actual,saved,check_dtype=False,check_exact=False,rtol=0,atol=1e-14)
    for name,(a_name,b_name) in CONTRASTS.items():
        info=check_files(out/'intervals'/name,'manifest.json')
        if info['inputs']['candidate']!=a_name or info['inputs']['baseline']!=b_name or info['inputs']['iterations']!=1000:
            raise ValueError('区间方法/预算错误')
    summary=dict(status='verified',identity=config_digest(spec),
        independent_direct_percentile_queries=checked,exact_fusion_variants=len(expected),
        cdf_sources_per_length=2000,cdf_query_lengths=[8,16],unchanged_reference_route_raw=True,
        recomputed_tables=list(keys),paired_contrasts=len(CONTRASTS),
        source_snapshot_verified=True,auditor_sha256=file_digest(Path(__file__)))
    moment_dir=out/'moments'
    if (moment_dir/'verification.json').exists():
        models=('pooled','expert_mean','expert_covariance','expert_both')
        m=pd.read_csv(moment_dir/'video_scores.csv.gz',float_precision='round_trip')
        expected_raw=pd.DataFrame(index=indices,columns=models,dtype=float)
        for length in (8,16):
            f=pd.read_csv(root/'data/manifests/global_experts/windows.csv',keep_default_na=False)
            f=f[(f.length==length)&f.role.ne('fit')].reset_index(drop=True)
            mask=f.role.eq('evaluation').to_numpy()
            with np.load(moment_dir/f'raw_{length}.npz') as z:
                np.testing.assert_array_equal(z['video_ids'],f.video_id.to_numpy())
                expected_raw.loc[f.loc[mask,'video_id'],list(models)]=z['scores'][mask]
        for model in models:
            raw_frame=m[m.variant.eq(model+'_raw')].set_index('video_id').loc[indices]
            raw_values=expected_raw[model].to_numpy()
            np.testing.assert_array_equal(raw_frame.raw_temporal_score.to_numpy(),raw_values)
            # 独立以unique计数验证保序raw指标输入，而非复用pandas.rank。
            _,inverse,counts=np.unique(raw_values,return_inverse=True,return_counts=True)
            ranks=np.cumsum(counts)[inverse]/len(raw_values)
            np.testing.assert_array_equal(raw_frame.final_score.to_numpy(),ranks)
            temporal=vector(m,model+'_temporal')
            np.testing.assert_array_equal(vector(m,model+'_final'),.5*vector(old,'official_spatial')+.5*temporal)
        summary['moments']=dict(raw_and_fusion_models=4,raw_rank_order_independently_verified=True,
            static_clip_ids=int(np.isposinf(expected_raw.pooled.to_numpy()).sum()))
    paper_json(out/'independent_audit.json',summary)
    print(json.dumps(summary,ensure_ascii=False),flush=True)


if __name__=='__main__':audit(Path.cwd())
