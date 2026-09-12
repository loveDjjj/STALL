"""双卡同模型、单供数进程和更新级恢复；真实抽样/有效batch不变。"""
import argparse
import contextlib
import datetime
import fcntl
import json
import math
import os
import queue
import socket
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from artifacts import atomic_csv,paper_json
from config import config_digest
from reference import file_digest
from discriminative_moe.training import sampling_weights,validation_pairs,validation_objective
from .model import LoopedDetector
from .stream import SharedReader,global_microbatches
from .buffers import BatchBuffers,fill_shared
from .training import model_arguments,save_torch,score_frame,EpochSampler
from .run import ROOT,configuration


def receive(q,device,transfer,stop,ring,pinned,free_slots,copy_events):
    while not stop.is_set():
        try:packet=q.get(timeout=1);break
        except queue.Empty:continue
    else:return dict(phase='stop')
    if 'buffer' in packet:
        meta=packet['buffer'];slot=meta['slot']
        if slot in copy_events:copy_events[slot].synchronize()
        source=ring.view(meta);b=pinned.view(meta)
        for key in b:b[key].copy_(source[key])
        # 锁页副本已完成，父进程可重用共享槽；GPU读取的是独立锁页副本。
        free_slots.put(slot)
        with torch.cuda.stream(transfer):
            gpu={k:b[k].to(device,non_blocking=True) for k in ('patches','valid','owners','labels')}
            event=torch.cuda.Event();event.record(transfer)
        copy_events[slot]=event
        packet['gpu']=gpu;packet['event']=event
    return packet


def worker(rank,world,q,port,root_string,out_string,c,task,spec,vm,validation,rt,ring,free_slots):
    root=Path(root_string);out=Path(out_string);dest=out/'training'/task['key'];identity=config_digest(spec)
    os.environ['NCCL_P2P_DISABLE']='1'
    torch.set_num_threads(2);torch.cuda.set_device(rank);device=f'cuda:{rank}'
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    # 单机文件握手避免当前环境TCPStore的主机名反查等待；通信仍为NCCL。
    dist.init_process_group('nccl',init_method=(dest/f'.rendezvous_{port}').as_uri(),
        rank=rank,world_size=world,timeout=datetime.timedelta(seconds=180))
    torch.manual_seed(task['seed']);torch.cuda.manual_seed_all(task['seed'])
    base=LoopedDetector(**model_arguments(c,task['variant'])).to(device)
    model=DDP(base,device_ids=[rank],broadcast_buffers=False)
    optimizer=torch.optim.AdamW(base.parameters(),lr=c['learning_rate'],weight_decay=c['weight_decay'])
    history=[];best=-math.inf;best_epoch=0;previous_seconds=0.;stats=torch.zeros(2,device=device,dtype=torch.float64)
    if (dest/'last.pt').exists():
        saved=torch.load(dest/'last.pt',map_location=device,weights_only=True)
        if saved['identity']!=identity:raise ValueError('DDP续跑身份改变')
        base.load_state_dict(saved['model']);optimizer.load_state_dict(saved['optimizer'])
        history=saved['history'];best=saved['best'];best_epoch=saved['best_epoch'];previous_seconds=saved['seconds']
        if saved['phase']=='train':stats.copy_(torch.tensor(saved['rank_stats'][rank],device=device))
    updates_per_epoch=math.ceil(len(spec['train_ids'])/rt['global_batch'])
    total_updates=updates_per_epoch*c['epochs'];warm=updates_per_epoch*c['warmup_epochs']
    transfer=torch.cuda.Stream(device=device);pool=ThreadPoolExecutor(max_workers=1);stop=threading.Event()
    n,d=ring.slots[0]['patches'].shape[-2:]
    pinned=BatchBuffers(rt['local_batch'],n,d,shared=False,pinned=True)
    copy_events={}
    future=pool.submit(receive,q,device,transfer,stop,ring,pinned,free_slots,copy_events)
    start=time.perf_counter();epoch_start=start;vals=[];data_wait=0.
    torch.cuda.reset_peak_memory_stats(device)

    def checkpoint(epoch,next_update,phase):
        gathered=[torch.empty_like(stats) for _ in range(world)]
        dist.all_gather(gathered,stats)
        if rank==0:
            save_torch(dest/'last.pt',dict(model=base.state_dict(),optimizer=optimizer.state_dict(),identity=identity,
                epoch=epoch,next_update=next_update,phase=phase,rank_stats=[x.cpu().tolist() for x in gathered],
                history=history,best=best,best_epoch=best_epoch,seconds=previous_seconds+time.perf_counter()-start))

    try:
        while True:
            waiting=time.perf_counter();packet=future.result();data_wait+=time.perf_counter()-waiting
            if packet['phase']=='stop':break
            future=pool.submit(receive,q,device,transfer,stop,ring,pinned,free_slots,copy_events)
            if 'gpu' in packet:
                torch.cuda.current_stream(device).wait_event(packet['event'])
                b=packet['gpu']
                for x in b.values():x.record_stream(torch.cuda.current_stream(device))
            phase=packet['phase'];epoch=packet['epoch']
            if phase=='epoch_begin':
                model.train();optimizer.zero_grad(set_to_none=True);epoch_start=time.perf_counter();data_wait=0.
                if not packet['resume']:stats.zero_()
            elif phase=='train':
                sync=contextlib.nullcontext() if packet['last_micro'] else model.no_sync()
                with sync:
                    with torch.autocast('cuda',dtype=torch.bfloat16,enabled=c['amp']):
                        logits=model(b['patches'],b['valid'],b['owners'],len(b['labels'])).float()
                    summed=torch.nn.functional.binary_cross_entropy_with_logits(logits,b['labels'],reduction='sum')
                    if packet['count']==0:summed=summed*0
                    (summed*(world/packet['denom'])).backward()
                stats[0]+=summed.detach().double();stats[1]+=packet['count']
                if packet['last_micro']:
                    update=packet['update'];step=(epoch-1)*updates_per_epoch+update+1
                    factor=step/max(warm,1) if step<=warm else .5*(1+math.cos(math.pi*(step-warm)/max(total_updates-warm,1)))
                    for group in optimizer.param_groups:group['lr']=c['learning_rate']*factor
                    torch.nn.utils.clip_grad_norm_(base.parameters(),5.,error_if_nonfinite=True)
                    optimizer.step();optimizer.zero_grad(set_to_none=True)
                    if (update+1)%rt['checkpoint_updates']==0 or update+1==updates_per_epoch:
                        checkpoint(epoch,update+1,'train')
                    if (update+1)%10==0 or update+1==updates_per_epoch:
                        total=stats.clone();dist.all_reduce(total)
                        if rank==0:
                            elapsed=time.perf_counter()-epoch_start
                            paper_json(dest/'progress.json',dict(status='training',runner='ddp_shared_io',epoch=epoch,
                                epochs=c['epochs'],update=update+1,updates=updates_per_epoch,train_loss=float(total[0]/total[1]),
                                epoch_seconds=elapsed,data_wait_seconds=data_wait,
                                peak_gpu_gib=torch.cuda.max_memory_allocated(device)/2**30))
                            print('DDP',task['key'],'epoch',epoch,'update',update+1,'/',updates_per_epoch,
                                  'seconds',round(elapsed,1),flush=True)
            elif phase=='validation_begin':
                model.eval();vals=[];train_seconds=time.perf_counter()-epoch_start
            elif phase=='validation':
                with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16,enabled=c['amp']):
                    logits=base(b['patches'],b['valid'],b['owners'],len(b['labels'])).float()
                if packet['count']:
                    vals.extend(zip(packet['buffer']['indices'],logits.cpu().double().tolist()))
            elif phase=='validation_end':
                gathered=[None]*world;dist.all_gather_object(gathered,vals)
                total=stats.clone();dist.all_reduce(total)
                if rank==0:
                    merged=[x for part in gathered for x in part]
                    if sorted(i for i,_ in merged)!=list(range(len(vm))):raise ValueError('验证遗漏/重复')
                    logits=np.array([v for _,v in sorted(merged)])
                    if not np.isfinite(logits).all():raise ValueError('验证预测非有限')
                    frame=score_frame(vm,logits);objective,_=validation_objective(frame,validation)
                    seconds=previous_seconds+time.perf_counter()-start
                    history.append(dict(epoch=epoch,train_loss=float(total[0]/total[1]),validation_objective=objective,
                        train_seconds=train_seconds,epoch_seconds=time.perf_counter()-epoch_start,seconds=seconds,
                        peak_gpu_gib=torch.cuda.max_memory_allocated(device)/2**30))
                    if objective>best:
                        best,best_epoch=objective,epoch
                        save_torch(dest/'model.pt',dict(model=base.state_dict(),identity=identity,epoch=epoch,
                            arguments=model_arguments(c,task['variant'])))
                        atomic_csv(dest/'validation_scores.csv',frame)
                    atomic_csv(dest/'history.csv',pd.DataFrame(history))
                    paper_json(dest/'progress.json',dict(status='epoch_complete',runner='ddp_shared_io',**history[-1]))
                    print('DDP epoch complete',task['key'],epoch,'validation',round(objective,5),flush=True)
                # rank0保存最终状态；rank1的best/history只用于本进程，不作为checkpoint来源。
                checkpoint(epoch,0,'epoch_complete');dist.barrier()
        if rank==0:
            files=['model.pt','history.csv','validation_scores.csv','validation_pairs.csv']
            paper_json(dest/'manifest.json',dict(status='trained',identity=identity,task=task,best_epoch=best_epoch,
                best_validation=best,parameters=sum(p.numel() for p in base.parameters()),
                seconds=previous_seconds+time.perf_counter()-start,test_used=False,runner='ddp_shared_io',
                files={p:file_digest(dest/p) for p in files}))
        dist.barrier()
    finally:
        stop.set();pool.shutdown(wait=True,cancel_futures=True);dist.destroy_process_group()


def put(q,packet,processes):
    while True:
        if any(p.exitcode is not None for p in processes):raise RuntimeError('DDP worker已提前结束')
        try:q.put(packet,timeout=2);return
        except queue.Full:continue


def acquire_slot(q,processes):
    while True:
        if any(p.exitcode is not None for p in processes):raise RuntimeError('DDP worker在供数前结束')
        try:return q.get(timeout=2)
        except queue.Empty:continue


def task_spec(root,out,c,task,tm,vm,rt):
    cache=root/c['cache_directory']
    return dict(task=task,config=c,runtime=rt,runner='ddp_shared_io_v1',transport='fixed_shared_ring_v1',prepared=file_digest(out/'prepared.json'),
        cache_manifest=file_digest(cache/'manifest.json'),train_ids=tm.video_id.tolist(),validation_ids=vm.video_id.tolist(),
        code={p:file_digest(root/p) for p in ['src/looped_video/ddp_train.py','src/looped_video/stream.py','src/looped_video/buffers.py','src/looped_video/model.py',
            'src/looped_video/data.py','src/looped_video/training.py','src/discriminative_moe/training.py','src/evaluation/tables.py']})


def run_task(root,out,c,task,meta,roles,reader,rt):
    cache=root/c['cache_directory'];dest=out/'training'/task['key'];dest.mkdir(parents=True,exist_ok=True)
    with (dest/'run.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        split=roles[roles.fold.eq(task['fold'])].set_index('video_id').loc[meta.video_id]
        tm=meta.loc[split.role.to_numpy()=='train'].reset_index(drop=True);vm=meta.loc[split.role.to_numpy()=='validation'].reset_index(drop=True)
        if set(tm.split_group)&set(vm.split_group):raise ValueError('源组泄漏')
        spec=task_spec(root,out,c,task,tm,vm,rt)
        identity=config_digest(spec)
        if (dest/'manifest.json').exists():
            r=json.loads((dest/'manifest.json').read_text())
            if r['identity']!=identity:raise ValueError('已完成训练身份变化')
            for p,h in r['files'].items():
                if file_digest(dest/p)!=h:raise ValueError('模型产物变化')
            return
        if (dest/'identity.json').exists() and json.loads((dest/'identity.json').read_text())!=spec:
            raise ValueError('已有DDP训练合同不同')
        paper_json(dest/'identity.json',spec);validation=validation_pairs(vm);atomic_csv(dest/'validation_pairs.csv',validation)
        weights=sampling_weights(tm);reader.configure(tm,weights)
        first_epoch=1;first_update=0
        if (dest/'last.pt').exists():
            saved=torch.load(dest/'last.pt',map_location='cpu',weights_only=True)
            if saved['identity']!=identity:raise ValueError('续跑身份不符')
            first_epoch=saved['epoch']+(saved['phase']=='epoch_complete')
            first_update=saved['next_update'] if saved['phase']=='train' else 0
            del saved
        with socket.socket() as sock:sock.bind(('127.0.0.1',0));port=sock.getsockname()[1]
        ctx=mp.get_context('spawn');queues=[ctx.Queue(maxsize=2) for _ in range(2)]
        free_slots=[ctx.Queue(maxsize=2) for _ in range(2)]
        first=reader.read(tm.iloc[0].key);n,d=first.shape[-2:]
        rings=[BatchBuffers(rt['local_batch'],n,d) for _ in range(2)]
        for q in free_slots:
            for slot in range(2):q.put(slot)
        processes=[ctx.Process(target=worker,args=(rank,2,queues[rank],port,str(root),str(out),c,task,spec,vm,validation,rt,
            rings[rank],free_slots[rank])) for rank in range(2)]
        for p in processes:p.start()
        paper_json(out/'pipeline.json',dict(status='ddp_training',pid=os.getpid(),task=task['key'],
            workers=[dict(pid=p.pid,rank=r) for r,p in enumerate(processes)]))
        try:
            io=dict(slot_wait_seconds=0.,read_seconds=0.,pack_seconds=0.)
            for epoch in range(first_epoch,c['epochs']+1):
                resume=epoch==first_epoch and first_update>0
                for q in queues:put(q,dict(phase='epoch_begin',epoch=epoch,resume=resume),processes)
                sampler=EpochSampler(weights,task['seed']);sampler.epoch=epoch;draw=list(sampler)
                for part in global_microbatches(draw,rt['global_batch'],rt['local_batch'],2,first_update if resume else 0):
                    waiting=time.perf_counter();slots=[acquire_slot(q,processes) for q in free_slots]
                    io['slot_wait_seconds']+=time.perf_counter()-waiting
                    packets,timing=fill_shared(reader,tm,part['indices'],rings,slots)
                    for k,v in timing.items():io[k]+=v
                    for rank,q in enumerate(queues):
                        put(q,dict(phase='train',epoch=epoch,update=part['update'],last_micro=part['last_micro'],
                            denom=part['denom'],count=part['counts'][rank],buffer=packets[rank]),processes)
                    if part['last_micro'] and (part['update']+1)%10==0:
                        paper_json(dest/'io_progress.json',dict(epoch=epoch,enqueued_update=part['update']+1,**reader.stats(),**io))
                for q in queues:put(q,dict(phase='validation_begin',epoch=epoch),processes)
                for start in range(0,len(vm),rt['local_batch']*2):
                    indices=[list(range(start+r*rt['local_batch'],min(start+(r+1)*rt['local_batch'],len(vm)))) for r in range(2)]
                    counts=[len(x) for x in indices];slots=[acquire_slot(q,processes) for q in free_slots]
                    packets,_=fill_shared(reader,vm,[x or [0] for x in indices],rings,slots)
                    for rank,q in enumerate(queues):put(q,dict(phase='validation',epoch=epoch,count=counts[rank],buffer=packets[rank]),processes)
                for q in queues:put(q,dict(phase='validation_end',epoch=epoch),processes)
            for q in queues:put(q,dict(phase='stop',epoch=c['epochs']),processes)
            while any(p.is_alive() for p in processes):
                if any(p.exitcode not in (None,0) for p in processes):raise RuntimeError('DDP worker失败')
                for p in processes:p.join(timeout=1)
            if any(p.exitcode for p in processes):raise RuntimeError('DDP退出失败')
        finally:
            for p in processes:
                if p.is_alive():p.terminate()
            for p in processes:p.join(timeout=10)
            for q in queues+free_slots:q.close()


def main():
    p=argparse.ArgumentParser();p.add_argument('--task',default='');p.add_argument('--ram-gib',type=int,default=48)
    p.add_argument('--reader-workers',type=int,default=2);p.add_argument('--checkpoint-updates',type=int,default=25);a=p.parse_args()
    if not 0<=a.ram_gib<=56 or not 1<=a.reader_workers<=8:raise ValueError('RAM/读取并行预算越界')
    import signal
    def interrupted(signum,frame):raise KeyboardInterrupt('用户停止，保留最近优化器更新检查点')
    signal.signal(signal.SIGTERM,interrupted)
    torch.set_num_threads(8)
    c=configuration(ROOT);out=ROOT/c['run_directory'];cache=ROOT/c['cache_directory']
    receipt=json.loads((cache/'manifest.json').read_text())
    if receipt['status'] not in ('verified','ready_for_training'):raise ValueError('缓存未准入')
    meta=pd.read_csv(out/'videos.csv',keep_default_na=False);roles=pd.read_csv(out/'roles.csv',keep_default_na=False)
    rt=dict(world_size=2,local_batch=c['batch_size'],global_batch=c['batch_size']*c['accumulation'],
        ram_gib=a.ram_gib,reader_workers=a.reader_workers,checkpoint_updates=a.checkpoint_updates)
    tasks=json.loads((out/'tasks.json').read_text())['tasks']
    tasks=sorted(tasks,key=lambda t:(c['variants'].index(t['variant']),c['folds'].index(t['fold']),c['seeds'].index(t['seed'])))
    if a.task:
        tasks=[t for t in tasks if t['key']==a.task]
        if len(tasks)!=1:raise ValueError('未知任务')
    global_lock=(out/'pipeline.lock').open('a')
    fcntl.flock(global_lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    reader=SharedReader(cache,receipt['records'],a.ram_gib,a.reader_workers)
    try:
        for task in tasks:
            run_task(ROOT,out,c,task,meta,roles,reader,rt)
            from .evaluation import evaluate,report
            evaluate(ROOT,argparse.Namespace(task=task['key'],rank=0,world_size=2))
            report(ROOT,argparse.Namespace())
        paper_json(out/'pipeline.json',dict(status='matrix_finished_pending_final_audit',pid=os.getpid(),tasks=len(tasks)))
    except BaseException as exc:
        paper_json(out/'pipeline.json',dict(status='stopped' if isinstance(exc,KeyboardInterrupt) else 'failed',pid=os.getpid(),error=str(exc)))
        raise
    finally:reader.close();global_lock.close()


if __name__=='__main__':main()
