"""有界特征预取、固定查询批量与异步检查点写入。"""
from collections import deque
from concurrent.futures import ThreadPoolExecutor
import json
import time
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from artifacts import checkpoint_read,checkpoint_write,paper_json
from config import config_digest
from reference import file_digest
from data.prefetch import bounded_map
from statistical_experts.manifests import settings
from statistical_experts.cache import load_feature,feature_identity
from statistical_experts.engine import Engine


def score(root,length,rank,world_size,*,pilot=False):
    root=Path(root).resolve();c=settings(root);device=f'cuda:{rank%2}'
    engine=Engine(root,length,device)
    frame=pd.read_csv(root/c['manifest_directory']/'windows.csv',keep_default_na=False)
    frame=frame[(frame.length==length)&(frame.role!='fit')].reset_index(drop=True)
    if pilot:frame=frame[frame.role=='cdf'].head(100).reset_index(drop=True)
    directory=root/c['run_directory']/(f'pilot_{length}' if pilot else f'raw_{length}')
    directory.mkdir(parents=True,exist_ok=True)
    spec=dict(engine_identity=engine.identity,manifest_sha256=config_digest(frame.to_dict('records')),
        code_sha256=file_digest(Path(__file__)),query_batch=c['query_batch'],length=length,
        calibration='each real video runs complete algorithm; two end-to-end branch CDFs',
        input_features=engine.bank['identity'])
    identity=config_digest(spec);marker=directory/f'identity_rank_{rank}.json'
    if marker.exists():
        if json.loads(marker.read_text())!=spec:raise ValueError('统计评分恢复身份改变')
    else:paper_json(marker,spec)
    pending=[]
    for i,row in enumerate(frame.itertuples(index=False)):
        if i%world_size!=rank:continue
        path=directory/'checkpoints'/f'{i:06d}.json'
        if path.exists():checkpoint_read(path,identity,row.video_id)
        else:pending.append((i,row))
    batch=c['query_batch'];batches=[pending[i:i+batch] for i in range(0,len(pending),batch)]
    def prepare(items):
        features=[load_feature(root/c['cache_directory']/(r.cache_key+'.npz'),engine.bank['identity'],r) for _,r in items]
        keys=[r.random_source_key for _,r in items]
        while len(features)<batch:features.append(features[-1]);keys.append(keys[-1])
        return torch.from_numpy(np.stack(features)),keys
    stream=bounded_map(batches,prepare,lambda items:batch*length*1024*8,workers=4,depth=8,budget=128*2**20)
    start=time.perf_counter();compute=0.;done=0;writes=deque();pool=ThreadPoolExecutor(max_workers=2)
    check_error=0.
    try:
        for items,(g,keys) in stream:
            begin=time.perf_counter();scores,clusters,neighbors=engine.score(g,keys);compute+=time.perf_counter()-begin
            if pilot and done==0:
                # 固定batch尺寸下逐查询重复与不同查询并排的结果必须一致。
                for j in range(len(items)):
                    ss,cc,nn=engine.score(g[j:j+1].expand(batch,-1,-1).contiguous(),[keys[j]]*batch)
                    for name in scores:
                        np.testing.assert_allclose(scores[name][j],ss[name][0],rtol=0,atol=1e-6)
                        delta=np.abs(scores[name][j]-ss[name][0]);finite=delta[np.isfinite(delta)]
                        check_error=max(check_error,float(finite.max()) if len(finite) else 0.)
                    for name in neighbors:np.testing.assert_array_equal(neighbors[name][j],nn[name][0])
                    assert clusters[j]==cc[0]
            for j,(index,r) in enumerate(items):
                payload=dict(video_id=r.video_id,role=r.role,length=length,cluster=int(clusters[j]),
                    scores={name:value[j].tolist() for name,value in scores.items()},
                    neighbors={name:value[j].tolist() for name,value in neighbors.items()})
                writes.append(pool.submit(checkpoint_write,directory/'checkpoints'/f'{index:06d}.json',identity,payload))
            while len(writes)>16:writes.popleft().result()
            done+=len(items)
            if done%32==0 or done==len(pending):
                status=dict(completed=done,pending_at_start=len(pending),elapsed=time.perf_counter()-start,
                    scoring_seconds=compute,queries_per_second=done/max(time.perf_counter()-start,1e-9),
                    peak_gpu_mib=torch.cuda.max_memory_allocated(device)/2**20,pilot_batch_error=check_error)
                paper_json(directory/f'progress_rank_{rank}.json',status)
                print(f'[statistics {length}/{rank}] {done}/{len(pending)} {status["queries_per_second"]:.2f}/s',flush=True)
        for future in writes:future.result()
    finally:stream.close();pool.shutdown(wait=True,cancel_futures=True)
    paper_json(directory/f'completed_rank_{rank}.json',dict(identity=identity,rank=rank,world_size=world_size,
        elapsed=time.perf_counter()-start,compute_seconds=compute,queries=len(pending),pilot=pilot,batch_validation_error=check_error))
