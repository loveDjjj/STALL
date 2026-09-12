"""固定共享环形缓冲：队列只传槽编号，不重复创建数百MiB共享张量。"""
import time
import numpy as np
import torch


class BatchBuffers:
    def __init__(self,local_batch,patches=196,dimension=1024,slots=2,max_frames=16,max_windows=3,shared=True,pinned=False):
        self.local_batch=local_batch;self.max_frames=max_frames;self.max_windows=max_windows
        self.slots=[]
        shapes=dict(patches=((local_batch*max_windows,max_frames,patches,dimension),torch.float32),
            valid=((local_batch*max_windows,max_frames),torch.bool),
            owners=((local_batch*max_windows,),torch.long),labels=((local_batch,),torch.float32))
        for _ in range(slots):
            slot={k:torch.empty(shape,dtype=dtype,pin_memory=pinned) for k,(shape,dtype) in shapes.items()}
            if shared:
                for x in slot.values():x.share_memory_()
            self.slots.append(slot)

    def view(self,meta):
        b=self.slots[meta['slot']];w=meta['windows'];t=meta['frames'];n=meta['videos']
        if not (0<w<=self.local_batch*self.max_windows and 0<t<=self.max_frames and 0<n<=self.local_batch):
            raise ValueError('缓冲区范围非法')
        return dict(patches=b['patches'][:w,:t],valid=b['valid'][:w,:t],owners=b['owners'][:w],labels=b['labels'][:n])

    def fill(self,slot_id,meta,indices,values,records):
        rows=[meta.iloc[i] for i in indices]
        lengths=[len(records[r.key]['window_positions'][0]) for r in rows]
        windows=sum(len(records[r.key]['window_positions']) for r in rows)
        info=dict(slot=slot_id,windows=windows,frames=max(lengths),videos=len(rows),indices=[int(i) for i in indices])
        b=self.view(info);b['valid'].zero_();target=b['patches'].numpy();cursor=0
        for owner,r in enumerate(rows):
            raw=values[r.key];positions=records[r.key]['window_positions'];t=lengths[owner]
            for pick in positions:
                ix=np.asarray(pick,dtype=np.int64)
                if len(ix)!=t or np.any(ix<0) or np.any(ix>=len(raw)):
                    raise ValueError('缓存帧位置非法，禁止静默截断')
                # 范围已验证；clip模式允许直接写入out，避免raise模式的额外整块暂存。
                np.take(raw,ix,axis=0,out=target[cursor,:t],mode='clip')
                b['valid'][cursor,:t]=True;b['owners'][cursor]=owner;cursor+=1
            b['labels'][owner]=int(r.subset=='annotated')
        return info


def fill_shared(reader,meta,rank_indices,rings,slot_ids):
    rows={i:meta.iloc[i] for indices in rank_indices for i in indices}
    keys=sorted({r.key for r in rows.values()});start=time.perf_counter()
    values=dict(zip(keys,reader.pool.map(reader.read,keys)));loaded=time.perf_counter()
    packets=[rings[r].fill(slot_ids[r],meta,indices,values,reader.records) for r,indices in enumerate(rank_indices)]
    return packets,dict(read_seconds=loaded-start,pack_seconds=time.perf_counter()-loaded)
