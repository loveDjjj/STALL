"""单一供数进程：惰性FP32热缓存、去重读取、双GPU队列共用原始数据。"""
import os
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import numpy as np
import torch
from .data import collate_videos


class SharedReader:
    def __init__(self,cache,records,ram_gib=48,workers=2):
        self.cache=Path(cache);self.records=records;self.budget=int(ram_gib*2**30)
        self.pool=ThreadPoolExecutor(max_workers=workers,thread_name_prefix='patch-reader')
        self.hot=set();self.arrays={};self.lock=threading.Lock()
        self.hits=self.misses=self.disk_bytes=0

    def configure(self,meta,weights):
        importance={}
        for key,w in zip(meta.key,weights):importance[key]=importance.get(key,0.)+float(w)
        chosen=set();size=0
        for key in sorted(importance,key=lambda k:(-importance[k]/self.records[k]['bytes'],k)):
            amount=self.records[key]['bytes']
            if size+amount<=self.budget:chosen.add(key);size+=amount
        self.hot=chosen
        self.arrays={k:v for k,v in self.arrays.items() if k in chosen}
        self.hits=self.misses=self.disk_bytes=0

    def read(self,key):
        r=self.records[key];p=self.cache/(key+'.npy');s=p.stat()
        if (s.st_size,s.st_mtime_ns)!=(r['bytes'],r['mtime_ns']):raise ValueError('训练输入改变：'+key)
        with self.lock:
            if key in self.arrays:self.hits+=1;return self.arrays[key]
            self.misses+=1;self.disk_bytes+=s.st_size
        # 整个唯一帧数组连续读取，避免memmap高级索引产生零碎页缺失。
        with p.open('rb') as f:
            raw=np.load(f,allow_pickle=False)
            if key in self.hot:
                with self.lock:self.arrays[key]=raw
                # 热数据已在用户态RAM中；尽量避免再占一份内核文件页缓存。
                if hasattr(os,'posix_fadvise'):os.posix_fadvise(f.fileno(),0,0,os.POSIX_FADV_DONTNEED)
        return raw

    def batches(self,meta,rank_indices):
        # 相同视频在全局batch内只读一次，重复抽样身份和次数仍保留。
        rows={i:meta.iloc[i] for indices in rank_indices for i in indices}
        unique={r.key for r in rows.values()}
        values=dict(zip(sorted(unique),self.pool.map(self.read,sorted(unique))))
        result=[]
        for indices in rank_indices:
            examples=[]
            for i in indices:
                r=rows[i];pick=self.records[r.key]['window_positions']
                x=values[r.key][np.asarray(pick,dtype=np.int64)]
                examples.append((torch.from_numpy(x),int(r.subset=='annotated'),i))
            result.append(collate_videos(examples))
        return result

    def stats(self):
        return dict(ram_gib=sum(x.nbytes for x in self.arrays.values())/2**30,
                    ram_hits=self.hits,reads=self.misses,read_gib=self.disk_bytes/2**30)

    def close(self):self.pool.shutdown(wait=True);self.arrays.clear()


def global_microbatches(draw,global_batch=32,local_batch=8,world=2,start_update=0):
    """保持全局抽样次序和每次更新样本数；空rank用零权重计算占位。"""
    for update,start in enumerate(range(0,len(draw),global_batch)):
        if update<start_update:continue
        group=list(draw[start:start+global_batch]);denom=len(group)
        chunks=list(range(0,denom,local_batch*world))
        for micro,offset in enumerate(chunks):
            real=[group[offset+r*local_batch:offset+(r+1)*local_batch] for r in range(world)]
            indices=[x if x else [group[0]] for x in real]
            yield dict(update=update,micro=micro,last_micro=micro+1==len(chunks),denom=denom,
                       indices=indices,counts=[len(x) for x in real])
