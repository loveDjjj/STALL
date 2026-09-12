"""补测三个短视频生成器的主方法成本，不把全部消融评分成本当部署成本。"""
import json
import time
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import yaml
from artifacts import paper_json
from reference import file_digest,percentile
from math_utils import StableGaussianParams
from evaluation.budget import BudgetEvidenceScorer
from evaluation.short_video import ShortWindowMixin
from evaluation.study_tables import read_evidence


class ShortBudgetScorer(ShortWindowMixin,BudgetEvidenceScorer):pass


def measure(root,output,device='cuda:1'):
    root=Path(root);output=Path(output);config=yaml.safe_load((root/'configs/paper.yaml').read_text());rows=[]
    for domain in ('genvideo','videofeedback'):
        target=root/'results/runs'/f'paper_fit_{domain}/gaussians.npz'
        with np.load(target) as z:
            models=[StableGaussianParams(z[n+'_mean'],z[n+'_whitening'],np.empty(0)) for n in ('gs','gt','lt')]
        with np.load(root/'results/runs/paper_fit_source/gaussians.npz') as z:center=z['lt_mean']
        scorer=ShortBudgetScorer(*models,center,device,dino_repo=str(root/config['encoder']['repo']),dino_weights=str(root/config['encoder']['weights']))
        frame=pd.read_csv(root/'data/manifests/short_video'/domain/'evaluation.csv',keep_default_na=False)
        sample=frame.sort_values('video_id').groupby(['subset','source_model'],sort=True).head(2)
        cf=pd.read_csv(root/'data/manifests/short_video/vatex/cdf.csv',keep_default_na=False)
        cdf,_=read_evidence(root/'results/runs'/f'complete23_{domain}_cdf_short8',cf)
        gs=np.asarray([r['uniform']['global_models']['target']['gs'] for r in cdf]);gt=np.asarray([r['uniform']['global_models']['target']['gt'] for r in cdf])
        lt=np.asarray([r['selected']['windows'][0]['representations']['local_d2']['target'] for r in cdf])
        expected=pd.read_csv(root/'results/runs'/f'complete23_tables_{domain}_short8/video_scores.csv.gz',float_precision='round_trip')
        expected=expected[expected.variant=='full'].set_index('video_id').final_score
        for selector,k in [('uniform',1),('feature_change',1),('uniform',3),('feature_change',3)]:
            first=sample.iloc[0];scorer.score_raw(root/first.video_path,json.loads(first.downsample_idxs),selector=selector,k=k)
            for repeat in range(2):
                for r in sample.itertuples():
                    torch.cuda.synchronize(device);torch.cuda.reset_peak_memory_stats(device);start=time.perf_counter()
                    raw=scorer.score_raw(root/r.video_path,json.loads(r.downsample_idxs),selector=selector,k=k)
                    w=raw['windows'][0]
                    s=.25*float(percentile([w['global_models']['target']['gs']],gs)[0])+.25*float(percentile([w['global_models']['target']['gt']],gt,allow_positive_infinity=True)[0])+.5*float(percentile([w['representations']['local_d2']['target']],lt)[0])
                    torch.cuda.synchronize(device);elapsed=time.perf_counter()-start
                    error=abs(s-float(expected.loc[r.video_id]))
                    if error>1e-10:raise ValueError('短视频主方法计时评分与完整证据不一致')
                    rows.append(dict(dataset=domain,video_id=r.video_id,source_model=r.source_model,subset=r.subset,
                        selector=('fc' if selector=='feature_change' else 'uniform')+str(k),repeat=repeat,
                        total_seconds=elapsed,observed_unique_frames=8,coarse_frames=0,
                        peak_gpu_allocated_mib=torch.cuda.max_memory_allocated(device)/2**20,score_error=error))
    output.mkdir(parents=True,exist_ok=False);pd.DataFrame(rows).to_csv(output/'timings.csv',index=False)
    paper_json(output/'manifest.json',dict(status='completed',device=device,measurements=len(rows),
        description='每域每真实源/生成器固定前2身份；每策略一次未计时warmup；OS页缓存未控制；无特征缓存；实际主方法',
        files={'timings.csv':file_digest(output/'timings.csv')}))


if __name__=='__main__':measure(Path.cwd(),Path('results/runs/complete23_short_runtime'))
