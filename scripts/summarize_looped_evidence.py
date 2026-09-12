"""当前方法讨论的固定seed17三域证据快照；不改变训练或按结果选模型。"""
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
import json
import numpy as np
import pandas as pd
from artifacts import atomic_csv,paper_json
from reference import file_digest
from evaluation.tables import evaluate_fixed_pairs
from discriminative_moe.analysis import paired_seed_contrast
from looped_video.run import configuration
from looped_video.training import model_arguments
from looped_video.model import LoopedDetector

c=configuration(ROOT);out=ROOT/c['run_directory'];dest=out/'method_review';dest.mkdir(exist_ok=True)
domains=['comgenvid','videofeedback','genvideo'];seed=17;frames=[];inputs={};metrics=[];structure=[]
for v in c['variants']:
    parts=[]
    for d in domains:
        p=out/'evaluation'/f'{d}__{v}__s{seed}'/'scores.csv'
        marker=json.loads((p.parent/'manifest.json').read_text())
        if file_digest(p)!=marker['scores_sha256']:raise ValueError('模型分数身份不符')
        if not (out/'groups'/f'{d}__s{seed}'/'postprocess.json').exists():raise ValueError('组尚未验收')
        parts.append(pd.read_csv(p,float_precision='round_trip'));inputs[str(p.relative_to(ROOT))]=file_digest(p)
    f=pd.concat(parts,ignore_index=True).assign(input='review',head=v);frames.append(f)
    args=model_arguments(c,v);model=LoopedDetector(**args)
    structure.append(dict(variant=v,loops=args['loops'],width=args['width'],heads=args['heads'],
        parameters=sum(p.numel() for p in model.parameters())));del model
p=ROOT/'results/runs/discriminative_moe/test_scores.csv.gz';old=[]
for f in pd.read_csv(p,chunksize=100000,float_precision='round_trip'):
    q=f[(f.input=='G')&(f['head']=='uniform')&(f.seed==seed)]
    if len(q):old.append(q)
old=pd.concat(old,ignore_index=True).assign(input='review',head='old_global_uniform')
frames.append(old);inputs[str(p.relative_to(ROOT))]=file_digest(p)
pairs=pd.read_csv(out/'pairs.csv');inputs[str((out/'pairs.csv').relative_to(ROOT))]=file_digest(out/'pairs.csv')
reference=old.set_index('video_id',verify_integrity=True)
for f in frames:
    if set(f.video_id)!=set(reference.index):raise ValueError('新旧测试范围不一致')
    for name in ['dataset','subset','source_model','split_group']:
        if not np.array_equal(f[name].to_numpy(),reference.loc[f.video_id,name].to_numpy()):raise ValueError('分组/标签漂移')
    result=evaluate_fixed_pairs(f,pairs)
    for name in ['dataset_metrics','macro_metrics']:
        metrics.append(result[name].assign(model=f['head'].iloc[0],table=name))
atomic_csv(dest/'metrics.csv',pd.concat(metrics,ignore_index=True));atomic_csv(dest/'models.csv',pd.DataFrame(structure))
scores=pd.concat(frames,ignore_index=True)
contrasts=[(v,'old_global_uniform') for v in ['looped','untied','loop2','wide512']]
contrasts += [('looped',v) for v in c['variants'] if v!='looped']
contrasts += [('loop2','loop1'),('loop2','untied')]
intervals=[]
for a,b in contrasts:
    table=paired_seed_contrast(scores,pairs,('review',a),('review',b),seeds=(seed,),iterations=1000)
    intervals.append(table.assign(contrast=a+'_vs_'+b))
    print(a,'vs',b,table[table.dataset=='Average'][['metric','delta','ci95_low','ci95_high']].to_dict('records'),flush=True)
atomic_csv(dest/'paired_intervals.csv',pd.concat(intervals,ignore_index=True))
paper_json(dest/'manifest.json',dict(scope='seed17_complete_three_development_domains',seed=seed,
    comparison='同源级划分/测试身份；旧Global训练30epoch，新模型20epoch，结构与优化器等亦不同，非纯循环因果对照',
    uncertainty='1000次源组Poisson配对bootstrap，固定seed和模型，未校正全部开发模型搜索',inputs=inputs,
    source_code_sha256=file_digest(Path(__file__)),
    outputs={p:file_digest(dest/p) for p in ['metrics.csv','models.csv','paired_intervals.csv']}))
