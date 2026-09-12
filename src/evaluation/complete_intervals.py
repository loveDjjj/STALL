"""完整23单元及共同过滤协议的源组区间，包含单元Average与Macro-3。"""
import argparse
from concurrent.futures import ProcessPoolExecutor
import json
from pathlib import Path
import pandas as pd
from artifacts import paper_json
from reference import file_digest
from evaluation.paper_export import CONTRASTS
from evaluation.bootstrap import paired_source_contrast


def worker(job):
    directory,name=job;directory=Path(directory)
    candidate=pd.read_csv(directory/'scores_fc3.csv.gz',float_precision='round_trip')
    candidate=candidate[candidate.variant=='full']
    mode=name if name in ('official','uniform1','fc1','uniform3') else 'fc3'
    variant='official_single_window' if mode=='official' else 'full' if mode!='fc3' else name
    baseline=pd.read_csv(directory/f'scores_{mode}.csv.gz',float_precision='round_trip');baseline=baseline[baseline.variant==variant]
    frame=pd.read_csv(directory/'evaluation.csv',keep_default_na=False)
    pairs=pd.read_csv(directory/'pairs.csv',keep_default_na=False)
    groups=frame.set_index('video_id').source_group
    output=directory/'intervals'/name
    inputs={p:file_digest(directory/p) for p in ['scores_fc3.csv.gz',f'scores_{mode}.csv.gz','evaluation.csv','pairs.csv']}
    if (output/'manifest.json').exists():
        m=json.loads((output/'manifest.json').read_text())
        if m['inputs']!=inputs:raise ValueError('区间来源改变')
        return name
    point,interval=paired_source_contrast(candidate,baseline,pairs,groups,iterations=1000,seed=17,
        include_generator_average=True)
    output.mkdir(parents=True,exist_ok=True)
    point.to_csv(output/'deltas.csv',index=False);interval.to_csv(output/'intervals.csv',index=False)
    paper_json(output/'manifest.json',dict(status='completed',inputs=inputs,iterations=1000,
        source_groups='原始源组保留；同一真实视频1秒/2秒共享Poisson权重',
        files={p.name:file_digest(p) for p in output.iterdir() if p.is_file()}))
    return name


def main(directory,workers=4):
    directory=Path(directory)
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for name in pool.map(worker,[(str(directory),n) for n in CONTRASTS]):print('interval completed',name,flush=True)
    parts=[]
    for name in CONTRASTS:
        p=directory/'intervals'/name
        d=pd.read_csv(p/'deltas.csv');c=pd.read_csv(p/'intervals.csv')
        parts.append(d.merge(c,on=['dataset','metric'],validate='one_to_one').assign(contrast='full minus '+name))
    pd.concat(parts).to_csv(directory/'confidence_intervals.csv',index=False)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('directory');p.add_argument('--workers',type=int,default=4)
    a=p.parse_args();main(a.directory,a.workers)
