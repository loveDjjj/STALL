"""论文对照的源组配对区间；CPU任务与GPU队列独立。"""
from pathlib import Path
import json
import pandas as pd
from artifacts import paper_json
from reference import file_digest
from evaluation.bootstrap import paired_source_contrast
from data.manifest import load_active_groups


def contrast(root,score_file,variant,baseline_file,baseline_variant,output,iterations=1000):
    root=Path(root).resolve();output=Path(output)
    candidate=pd.read_csv(score_file,float_precision='round_trip');baseline=pd.read_csv(baseline_file,float_precision='round_trip')
    candidate=candidate[candidate.variant.eq(variant)].copy()
    baseline=baseline[baseline.variant.eq(baseline_variant)].copy()
    if candidate.empty or baseline.empty:raise ValueError('指定变体不存在')
    groups,group_inputs=load_active_groups(root,root/'data/manifests/active',candidate)
    pairs=pd.read_csv(root/'data/manifests/active/pairs.csv')
    pairs=pairs[pairs.dataset.isin(candidate.dataset.unique())]
    metadata=dict(candidate_sha256=file_digest(score_file),baseline_sha256=file_digest(baseline_file),
        candidate=variant,baseline=baseline_variant,source_manifests=group_inputs,iterations=iterations,seed=17,
        scope='fixed fitted references; paired source-group Poisson bootstrap; pointwise unadjusted intervals',
        csv_float_policy='round_trip')
    if output.exists():
        manifest=json.loads((output/'manifest.json').read_text())
        if manifest['inputs']!=metadata or manifest['status']!='completed':raise ValueError('已有区间输出不匹配')
        for name,digest in manifest['files'].items():
            if file_digest(output/name)!=digest:raise ValueError('已有区间产物hash改变')
        return
    point,interval=paired_source_contrast(candidate,baseline,pairs,groups,iterations=iterations,seed=17,
        progress=lambda domain:print(f'[interval:{variant}] {domain}',flush=True))
    output.mkdir(parents=True,exist_ok=False)
    point.to_csv(output/'deltas.csv',index=False);interval.to_csv(output/'intervals.csv',index=False)
    paper_json(output/'manifest.json',dict(status='completed',inputs=metadata,
        files={p.name:file_digest(p) for p in output.iterdir() if p.is_file()}))
