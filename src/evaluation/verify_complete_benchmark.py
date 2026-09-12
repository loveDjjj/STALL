"""完整23单元验收：检查身份、旧分数保持、短窗共享、元数据和区间。"""
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
from artifacts import paper_json
from reference import file_digest
from evaluation.paper_export import STUDIES,CONTRASTS
from evaluation.complete_benchmark import checked_scores
from evaluation.study_tables import read_evidence


def verify(root,directory):
    root=Path(root).resolve();directory=Path(directory)
    manifest=json.loads((directory/'manifest.json').read_text())
    for name,digest in manifest['files'].items():
        if file_digest(directory/name)!=digest:raise ValueError('点表产物改变')
    pairs=pd.read_csv(directory/'pairs.csv',keep_default_na=False)
    frame=pd.read_csv(directory/'evaluation.csv',keep_default_na=False)
    cells=set(map(tuple,pairs[['dataset','generator']].drop_duplicates().to_numpy()))
    if len(cells)!=23 or frame.video_id.duplicated().any():raise ValueError('单元或身份错误')
    expected={'videofeedback':11,'genvideo':10,'comgenvid':2}
    if pairs.groupby('dataset').generator.nunique().to_dict()!=expected:raise ValueError('每域覆盖错误')
    for d in expected:
        fit=pd.read_csv(root/'data/manifests/active'/d/'fit.csv',keep_default_na=False)
        evaluation=frame[frame.dataset==d]
        if set(fit.video_path)&set(evaluation.video_path) or set(fit.source_group)&set(evaluation.source_group):
            raise ValueError(f'{d}拟合/评价源交叉')
    metadata={str(r['id']):r for s in ('train','test') for r in json.loads((root/'datasets/recovery/metadata'/f'videofeedback_{s}.json').read_text())}
    if manifest['scope'] in ('paper_filter','paper_aligned'):
        for r in frame[(frame.dataset=='videofeedback')&(frame.subset=='annotated')].itertuples():
            if metadata[Path(r.video_path).stem]['dynamic degree']<3:raise ValueError('主协议包含低动态fake')
    short=frame.video_id.str.endswith(':duration1');short_ids=set(frame.loc[short,'video_id'])
    for r in frame[short].itertuples():
        ids=json.loads(r.downsample_idxs)
        if len(ids)!=8 or any(a>=b for a,b in zip(ids,ids[1:])):raise ValueError('短窗重复或不足8帧')
    canonical=None;long_checks=0
    for mode in ('official','fc3','uniform1','fc1','uniform3'):
        new=pd.read_csv(directory/f'scores_{mode}.csv.gz',float_precision='round_trip')
        source=Path(manifest['official_native_source']) if mode=='official' and manifest.get('official_native_source') else root/'results/runs'/f'paper_tables_all_{mode}'
        old=checked_scores(source)
        for variant,p in new.groupby('variant'):
            if set(p.video_id)!=set(frame.video_id):raise ValueError('变体身份不全')
            long=p[~p.video_id.isin(short_ids)].set_index('video_id')
            original=old[old.variant==variant].set_index('video_id').loc[long.index]
            if not np.array_equal(long.final_score.to_numpy(),original.final_score.to_numpy()):raise ValueError('长窗源分数被改变')
            long_checks+=len(long)
        if mode!='official':
            values=new[(new.variant=='full')&new.video_id.isin(short_ids)].set_index('video_id').final_score.sort_index()
            if canonical is None:canonical=values
            elif not canonical.equals(values):raise ValueError('短窗在观察对照中不一致')
    configs=0
    for study,definitions in STUDIES.items():
        table=pd.read_csv(directory/f'{study}_generators.csv')
        summary=pd.read_csv(directory/f'{study}.csv')
        for mode,variant,_ in definitions:
            rows=table[(table.variant==variant)&(table.observation==mode)]
            if set(map(tuple,rows[['dataset','generator']].to_numpy()))!=cells or len(rows)!=23:
                raise ValueError('消融缺完整单元')
            values=summary[(summary.variant==variant)&(summary.observation==mode)]
            for metric in ('auc','real_positive_ap','fake_positive_ap'):
                a=rows[metric].mean();m=rows.groupby('dataset')[metric].mean().mean()
                if abs(values.loc[values.dataset=='Average',metric].iloc[0]-a)>1e-12 or abs(values.loc[values.dataset=='Macro-3',metric].iloc[0]-m)>1e-12:
                    raise ValueError('平均公式错误')
            configs+=1
    intervals=pd.read_csv(directory/'confidence_intervals.csv')
    if len(intervals)!=190 or intervals.contrast.nunique()!=19:raise ValueError('完整区间缺失')
    for contrast in CONTRASTS:
        p=directory/'intervals'/contrast;m=json.loads((p/'manifest.json').read_text())
        if m['status']!='completed' or m['iterations']!=1000:raise ValueError('区间未完成')
        for name,digest in m['inputs'].items():
            if file_digest(directory/name)!=digest:raise ValueError('区间来源不同')
        for name,digest in m['files'].items():
            if file_digest(p/name)!=digest:raise ValueError('区间文件改变')
    raw_count=0
    for d in ('genvideo','videofeedback'):
        for role in ('cdf','threshold','evaluation'):
            mf=root/'data/manifests/short_video'/('vatex' if role!='evaluation' else d)/f'{role}.csv'
            f=pd.read_csv(mf,keep_default_na=False)
            records,_=read_evidence(root/'results/runs'/f'complete23_{d}_{role}_short8',f);raw_count+=len(records)
    operating=pd.read_csv(directory/'operating_points.csv')
    fake_groups=set(map(tuple,operating.loc[operating.role=='fake',['dataset','group']].drop_duplicates().to_numpy()))
    if fake_groups!=cells or (operating.reference_real_fpr>operating.nominal_real_fpr+1e-12).any():raise ValueError('操作点覆盖/阈值错误')
    timing=pd.read_csv(root/'results/runs/complete23_short_runtime/timings.csv')
    if len(timing)!=96 or timing.score_error.max()>1e-10 or not timing.observed_unique_frames.eq(8).all():raise ValueError('短窗实际主方法成本核验失败')
    audit=dict(status='verified',scope=manifest['scope'],cells=23,table_configuration_rows=configs,
        evaluation_clip_ids=len(frame),physical_video_paths=frame.video_path.nunique(),source_groups=frame.source_group.nunique(),
        pair_rows=len(pairs),unchanged_long_score_rows=long_checks,verified_short_raw_records=raw_count,
        official_native_source=manifest.get('official_native_source'),
        contrasts=19,bootstrap_repetitions=1000,short_runtime_measurements=len(timing),
        limitations=['参考固定的区间，不包含重新拟合不确定性','同原文生成器覆盖与短窗规则，不等同逐视频池完全复现','原文AP文字/实现与整体Average仍存在未解歧义'])
    paper_json(directory/'verification.json',audit)
    print(json.dumps(audit,ensure_ascii=False),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('directory');a=p.parse_args();verify(Path.cwd(),a.directory)
