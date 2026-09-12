"""纯内存固定缓冲对照，不读取生产视频或更改正在训练的模型。"""
import json,time
import numpy as np
import pandas as pd
import torch
from looped_video.buffers import BatchBuffers

torch.set_num_threads(8)
raw=np.ones((24,196,1024),dtype=np.float32)
pick=[list(range(0,16)),list(range(4,20)),list(range(8,24))]
meta=pd.DataFrame([dict(key='sample',subset='real') for _ in range(8)])
ring=BatchBuffers(8,slots=1)
records={'sample':dict(window_positions=pick)};rows=[]
for i in range(4):
    start=time.perf_counter();ring.fill(0,meta,list(range(8)),{'sample':raw},records)
    rows.append(time.perf_counter()-start)
print(json.dumps(dict(fixed_shared_buffer_seconds=rows)))
