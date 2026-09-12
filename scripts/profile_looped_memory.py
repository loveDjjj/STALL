"""内存供数探针：不访问视频、不修改训练；用环境变量控制本进程NumPy大页提示。"""
import gc
import json
import resource
import time
import numpy as np
import torch

torch.set_num_threads(8)
raw=np.ones((24,196,1024),dtype=np.float32)
pick=np.array([range(0,16),range(4,20),range(8,24)])
rows=[]
for _ in range(3):
    before=resource.getrusage(resource.RUSAGE_SELF);start=time.perf_counter()
    examples=[torch.from_numpy(raw[pick]) for _ in range(8)]
    batch=torch.zeros(24,16,196,1024)
    for i,x in enumerate(examples):batch[i*3:(i+1)*3]=x
    batch.share_memory_()
    del batch,examples;gc.collect()
    after=resource.getrusage(resource.RUSAGE_SELF)
    rows.append(dict(seconds=time.perf_counter()-start,user=after.ru_utime-before.ru_utime,
        system=after.ru_stime-before.ru_stime,minor_faults=after.ru_minflt-before.ru_minflt))
print(json.dumps(dict(numpy_hugepage=bool(np.core.multiarray._get_madvise_hugepage()),measurements=rows)))
