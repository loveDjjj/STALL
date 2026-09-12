"""真实缓存上的逐方法统计耗时；不把共享四算法吞吐当单方法延迟。"""
import time
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from artifacts import paper_json
from reference import file_digest
from statistical_experts.manifests import settings
from statistical_experts.cache import load_feature
from statistical_experts.engine import Engine
from statistical_experts.gaussian import context,transitions,fitted,score_pair,neighbor_indices,random_indices


def compute(engine,g,keys,method):
    t,zero=transitions(g);gdev=g.to(engine.device);t=t.to(engine.device);zero=zero.to(engine.device)
    if method=='pooled':s,tm=engine.model['pool_s'],engine.model['pool_t']
    elif method=='offline':
        labels=(context(g).to(engine.device)@engine.model['centers'].T).argmax(1)
        s={k:v[labels] for k,v in engine.model['offline_s'].items()};tm={k:v[labels] for k,v in engine.model['offline_t'].items()}
    else:
        if method=='online':idx=neighbor_indices(context(g).to(engine.device),engine.bank['context'],engine.config['neighbors'])
        else:idx=torch.as_tensor(random_indices(keys,len(engine.bank['ids']),engine.config['neighbors'],engine.config['seed']),device=engine.device)
        s=fitted(engine.bank['spatial'][idx],engine.model['pool_s']['covariance'],engine.config['shrinkage'],engine.config['ridge'])
        tm=fitted(engine.bank['temporal'][idx].flatten(1,2),engine.model['pool_t']['covariance'],engine.config['shrinkage'],engine.config['ridge'])
    return score_pair(gdev,s,tm,t,zero).cpu().numpy()


def measure(root):
    root=Path(root);c=settings(root);out=root/c['run_directory']/'runtime'
    if out.exists():raise ValueError('不覆盖已有计时')
    frame=pd.read_csv(root/c['manifest_directory']/'windows.csv',keep_default_na=False)
    expected=pd.read_csv(root/c['run_directory']/'evaluation/video_scores.csv.gz',float_precision='round_trip')
    expected=expected[expected.branch=='final'].set_index(['video_id','method']).final_score
    references=np.load(root/c['run_directory']/'evaluation/cdf_arrays.npz')
    records=[];samples=[]
    for length in (8,16):
        engine=Engine(root,length,'cuda:0')
        selected=frame[(frame.role=='evaluation')&(frame.length==length)].sort_values('video_id').groupby(['dataset','subset','source_model'],sort=True).head(2)
        rows=list(selected.itertuples(index=False));samples.extend(selected.to_dict('records'))
        data=[load_feature(root/c['cache_directory']/(r.cache_key+'.npz'),engine.bank['identity'],r) for r in rows]
        if len(rows)%c['query_batch']:raise ValueError('计时子集应为完整批量')
        for method in ('pooled','offline','online','random'):
            ref=references[f'{method}_{length}']
            compute(engine,torch.from_numpy(np.stack(data[:4])),[r.random_source_key for r in rows[:4]],method)
            for repeat in range(2):
                for i in range(0,len(rows),c['query_batch']):
                    subset=rows[i:i+4];g=torch.from_numpy(np.stack(data[i:i+4]));keys=[r.random_source_key for r in subset]
                    torch.cuda.synchronize(engine.device);torch.cuda.reset_peak_memory_stats(engine.device)
                    resident=torch.cuda.memory_allocated(engine.device);start=time.perf_counter()
                    raw=compute(engine,g,keys,method)
                    pct=np.stack([np.searchsorted(ref[:,j],raw[:,j],side='right')/len(ref) for j in range(2)],1);final=pct.mean(1)
                    torch.cuda.synchronize(engine.device);elapsed=time.perf_counter()-start
                    errors=[abs(float(final[j])-float(expected.loc[(r.video_id,method)])) for j,r in enumerate(subset)]
                    if max(errors)>1e-10:raise ValueError('单方法计时与共享评分输出不同')
                    records.append(dict(method=method,length=length,repeat=repeat,clip_ids='|'.join(r.video_id for r in subset),
                        queries=len(subset),batch_seconds=elapsed,seconds_per_query=elapsed/len(subset),max_score_error=max(errors),
                        common_resident_mib=resident/2**20,peak_allocated_mib=torch.cuda.max_memory_allocated(engine.device)/2**20))
    out.mkdir(parents=True);pd.DataFrame(records).to_csv(out/'timings.csv',index=False);pd.DataFrame(samples).to_csv(out/'samples.csv',index=False)
    pd.DataFrame(records).groupby(['method','length']).agg(n_batches=('batch_seconds','size'),
        mean_seconds_per_query=('seconds_per_query','mean'),median_seconds_per_query=('seconds_per_query','median'),
        peak_allocated_mib=('peak_allocated_mib','max')).reset_index().to_csv(out/'summary.csv',index=False)
    paper_json(out/'manifest.json',dict(status='completed',scope='Global已在CPU缓存后的统计算法＋CDF；固定batch4吞吐口径，不含DINO/解码/磁盘加载；非单视频请求延迟',
        memory='共同驻留拟合库和所有模型，不能把共同底座内存当各方法独立部署需求',
        warmup='每方法/长度一个未计时批次，之后2遍；GPU0，无其他GPU任务',
        files={p.name:file_digest(p) for p in out.iterdir() if p.is_file()}))
    print(pd.read_csv(out/'summary.csv').to_string(index=False))


if __name__=='__main__':measure(Path.cwd())
