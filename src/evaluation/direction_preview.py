"""仅导出已完整评分的数据域；不发布未齐全三域Average、不改活动任务。"""
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import numpy as np
import pandas as pd
from artifacts import checkpoint_read,atomic_csv,paper_json
from reference import file_digest,local_video_cdfs,percentile,window_mean
from evaluation.tables import evaluate_fixed_pairs


def build(root):
    out=root/'results/runs/local_direction_evidence';meta=json.loads((out/'identity.json').read_text());c=meta['config']
    did=json.loads((out/'dense_identity.json').read_text())['identity']
    jobs=json.loads((root/c['plans']).read_text())['jobs'];existing={p.stem for p in (out/'raw').glob('*.json')}
    domains=[d for d in ('comgenvid','videofeedback','genvideo') if all(j['key'] in existing for j in jobs if d in j['targets'])]
    if not domains:raise RuntimeError('尚无完整域，不能输出不完整AUC')
    names=('d1_anchor','d2_anchor','pooled_d2','ttr','split');refs={};records={}
    wanted=[j for j in jobs if set(domains)&set(j['targets'])]
    def read(j):return j,checkpoint_read(out/'raw'/(j['key']+'.json'),did,j['key'])
    pool=ThreadPoolExecutor(max_workers=12)
    for number,(j,r) in enumerate(pool.map(read,wanted),1):
        records[j['key']]=r
        if number%2000==0:print('preview read',number,len(wanted),flush=True)
        if j['role']!='cdf':continue
        for d in set(domains)&set(j['targets']):
            length=len(j['windows'][0])
            for name in names:
                v=(-np.asarray(r['scalars'])[:,0 if name=='ttr' else 2]).tolist() if name in ('ttr','split') else r['targets'][d][name]
                refs.setdefault((d,length,name),[]).append(v)
    pool.shutdown(wait=True)
    cdfs={}
    for key,v in refs.items():
        if len(v)!=2000:raise ValueError('独立CDF不足2000')
        cdfs[key]=local_video_cdfs(v) if key[1]==16 else {1:np.sort(np.asarray(v)[:,0])}
    base=pd.read_csv(root/c['baseline']/'scores_fc3.csv.gz',float_precision='round_trip')
    g=base[base.variant.eq('global_only')].set_index('video_id').final_score;rows=[]
    for j in jobs:
        if j['role']!='evaluation' or j['key'] not in records:continue
        r=records[j['key']];length=len(j['windows'][0]);k=len(j['windows'])
        for d in set(domains)&set(j['targets']):
            m=j['targets'][d];fields={n:m[n] for n in ('video_id','dataset','subset','source_model')}
            rows.append(dict(**fields,variant='global_only',final_score=float(g[m['video_id']])))
            for name in names:
                values=(-np.asarray(r['scalars'])[:,0 if name=='ttr' else 2]).tolist() if name in ('ttr','split') else r['targets'][d][name]
                q=window_mean(values);l=float(percentile([q],cdfs[d,length,name][k])[0])
                rows.append(dict(**fields,variant=name,final_score=.5*float(g[m['video_id']])+.5*l))
                rows.append(dict(**fields,variant=name+'_raw',final_score=q))
    scores=pd.DataFrame(rows);pairs=pd.read_csv(root/c['baseline']/'pairs.csv');pairs=pairs[pairs.dataset.isin(domains)]
    tables=[]
    for name,part in scores.groupby('variant'):
        t=evaluate_fixed_pairs(part,pairs);tables.append(t['dataset_metrics'].assign(variant=name))
    table=pd.concat(tables);atomic_csv(out/'completed_domain_preview.csv',table)
    paper_json(out/'completed_domain_preview.json',dict(status='partial_study_complete_domains_only',domains=domains,
        identity=did,source_plan_sha256=file_digest(root/c['plans']),metrics_sha256=file_digest(out/'completed_domain_preview.csv'),
        caution='no incomplete-domain metrics or three-domain Average; paired intervals still pending'))
    print(table[['dataset','variant','auc','real_positive_ap']].to_string(index=False),flush=True)


if __name__=='__main__':build(Path.cwd())
