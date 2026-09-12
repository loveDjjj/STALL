"""相同fold/seed的独立模型共用一次供数；不共享参数、梯度或优化器。"""
import argparse
import contextlib
import datetime
import fcntl
import json
import math
import os
import queue
import shutil
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
from evaluation.tables import evaluate_fixed_pairs
from discriminative_moe.training import sampling_weights,validation_pairs,validation_objective
from .ddp_train import receive,put,acquire_slot,task_spec
from .buffers import BatchBuffers,fill_shared
from .stream import SharedReader,global_microbatches
from .training import model_arguments,save_torch,score_frame,EpochSampler
from .model import LoopedDetector
from .migrate_transport import fingerprint
from .run import ROOT,configuration


def install_identity(dest,old,new):
    """仅升级执行调度身份；既有数值和采样进度逐张量核对。"""
    if old==new:return
    if (dest/'manifest.json').exists():raise ValueError('不能改写已完成模型')
    strip=lambda x:{k:v for k,v in x.items() if k not in ('code','runner','scheduling')}
    if strip(old)!=strip(new):raise ValueError('模型、输入或训练预算改变，不能迁移')
    for p,h in old['code'].items():
        if new['code'].get(p)!=h:raise ValueError('旧计算代码改变：'+p)
    receipt=dict(status='migrating',previous_spec=old,current_spec=new,checkpoints={})
    paper_json(dest/'group_migration.json',receipt)
    for name in ['last.pt','model.pt']:
        path=dest/name
        if not path.exists():continue
        backup=dest/(name+'.pre_group')
        if not backup.exists():shutil.copyfile(path,backup)
        state=torch.load(backup,map_location='cpu',weights_only=True)
        if state['identity']!=config_digest(old):raise ValueError('旧checkpoint身份错误')
        digest=fingerprint(state);state['identity']=config_digest(new);save_torch(path,state)
        if fingerprint(torch.load(path,map_location='cpu',weights_only=True))!=digest:raise ValueError('迁移改变了数值')
        receipt['checkpoints'][name]=dict(numerical_state_sha256=digest,previous_sha256=file_digest(backup),
            current_sha256=file_digest(path),epoch=state.get('epoch'),next_update=state.get('next_update'))
    paper_json(dest/'identity.json',new);receipt['status']='completed';paper_json(dest/'group_migration.json',receipt)


class Member:
    def __init__(self,rank,world,out,c,task,spec,vm,vpairs):
        self.rank=rank;self.world=world;self.c=c;self.task=task;self.spec=spec;self.identity=config_digest(spec)
        self.dest=out/'training'/task['key'];self.device=f'cuda:{rank}';self.vm=vm;self.vpairs=vpairs
        torch.manual_seed(task['seed']);torch.cuda.manual_seed_all(task['seed'])
        self.base=LoopedDetector(**model_arguments(c,task['variant'])).to(self.device)
        self.model=DDP(self.base,device_ids=[rank],broadcast_buffers=False)
        self.optimizer=torch.optim.AdamW(self.base.parameters(),lr=c['learning_rate'],weight_decay=c['weight_decay'])
        self.stats=torch.zeros(2,dtype=torch.float64,device=self.device)
        self.history=[];self.best=-math.inf;self.best_epoch=0;self.previous_seconds=0.
        self.cursor=(1,0);self.initial_phase='new';self.validated=0
        if (self.dest/'last.pt').exists():
            state=torch.load(self.dest/'last.pt',map_location=self.device,weights_only=True)
            if state['identity']!=self.identity:raise ValueError('成员恢复身份错误')
            self.base.load_state_dict(state['model']);self.optimizer.load_state_dict(state['optimizer'])
            self.history=state['history'];self.best=state['best'];self.best_epoch=state['best_epoch'];self.previous_seconds=state['seconds']
            self.initial_phase=state['phase'];self.validated=max((h['epoch'] for h in self.history),default=0)
            if state['phase']=='train':
                self.cursor=(state['epoch'],state['next_update']);self.stats.copy_(torch.tensor(state['rank_stats'][rank],device=self.device))
            else:self.cursor=(state['epoch']+1,0)
        self.initial_cursor=self.cursor;self.started=None;self.epoch_started=None
        self.updates=math.ceil(len(spec['train_ids'])/spec['runtime']['global_batch'])

    def seconds(self):
        return self.previous_seconds+(time.perf_counter()-self.started if self.started is not None else 0.)

    def begin_epoch(self,epoch):
        if epoch<self.initial_cursor[0]:return
        if self.started is None:self.started=time.perf_counter()
        self.epoch_started=time.perf_counter();self.model.train();self.optimizer.zero_grad(set_to_none=True)
        if not (epoch==self.initial_cursor[0] and self.initial_phase=='train'):self.stats.zero_()

    def checkpoint(self,epoch,update,phase):
        parts=[torch.empty_like(self.stats) for _ in range(self.world)];dist.all_gather(parts,self.stats)
        if self.rank==0:
            save_torch(self.dest/'last.pt',dict(model=self.base.state_dict(),optimizer=self.optimizer.state_dict(),
                identity=self.identity,epoch=epoch,next_update=update,phase=phase,rank_stats=[p.cpu().tolist() for p in parts],
                history=self.history,best=self.best,best_epoch=self.best_epoch,seconds=self.seconds()))

    def step(self,b,p):
        epoch,update=p['epoch'],p['update']
        if self.cursor!=(epoch,update):raise ValueError(f'模型进度不符：{self.task["key"]} {self.cursor} != {(epoch,update)}')
        with (contextlib.nullcontext() if p['last_micro'] else self.model.no_sync()):
            with torch.autocast('cuda',dtype=torch.bfloat16,enabled=self.c['amp']):
                logits=self.model(b['patches'],b['valid'],b['owners'],len(b['labels'])).float()
            loss=torch.nn.functional.binary_cross_entropy_with_logits(logits,b['labels'],reduction='sum')
            if not p['count']:loss=loss*0
            (loss*(self.world/p['denom'])).backward()
        self.stats[0]+=loss.detach().double();self.stats[1]+=p['count']
        if not p['last_micro']:return
        step=(epoch-1)*self.updates+update+1;warm=self.updates*self.c['warmup_epochs'];total=self.updates*self.c['epochs']
        factor=step/max(warm,1) if step<=warm else .5*(1+math.cos(math.pi*(step-warm)/max(total-warm,1)))
        for g in self.optimizer.param_groups:g['lr']=self.c['learning_rate']*factor
        torch.nn.utils.clip_grad_norm_(self.base.parameters(),5.,error_if_nonfinite=True)
        self.optimizer.step();self.optimizer.zero_grad(set_to_none=True);self.cursor=(epoch,update+1)
        if (update+1)%self.spec['runtime']['checkpoint_updates']==0 or update+1==self.updates:
            self.checkpoint(epoch,update+1,'train')
        if (update+1)%10==0 or update+1==self.updates:
            total=self.stats.clone();dist.all_reduce(total)
            if self.rank==0:
                paper_json(self.dest/'progress.json',dict(status='training',runner='shared_batch',epoch=epoch,epochs=self.c['epochs'],
                    update=update+1,updates=self.updates,train_loss=float(total[0]/total[1]),
                    epoch_seconds=time.perf_counter()-self.epoch_started,peak_gpu_gib=torch.cuda.max_memory_allocated(self.device)/2**30))

    def begin_scores(self,mode,epoch):
        if mode=='validation' and self.cursor!=(epoch,self.updates):raise ValueError('该epoch训练尚未完成')
        if mode=='test':
            state=torch.load(self.dest/'model.pt',map_location=self.device,weights_only=True)
            if state['identity']!=self.identity:raise ValueError('测试模型身份错误')
            self.base.load_state_dict(state['model'])
        self.base.eval();self.predictions=[]

    def infer(self,b,p):
        with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16,enabled=self.c['amp']):
            logits=self.base(b['patches'],b['valid'],b['owners'],len(b['labels'])).float()
        if p['count']:self.predictions.extend(zip(p['buffer']['indices'],logits.cpu().double().tolist()))

    def gather(self,n):
        parts=[None]*self.world;dist.all_gather_object(parts,self.predictions)
        if self.rank!=0:return None
        items=[x for p in parts for x in p]
        if sorted(i for i,_ in items)!=list(range(n)):raise ValueError('评分身份遗漏/重复')
        logits=np.array([v for _,v in sorted(items)])
        if not np.isfinite(logits).all():raise ValueError('非有限分数')
        return logits

    def validate(self,epoch):
        logits=self.gather(len(self.vm));total=self.stats.clone();dist.all_reduce(total)
        if self.rank==0:
            frame=score_frame(self.vm,logits);objective,_=validation_objective(frame,self.vpairs)
            self.history.append(dict(epoch=epoch,train_loss=float(total[0]/total[1]),validation_objective=objective,
                epoch_seconds=time.perf_counter()-self.epoch_started,seconds=self.seconds(),
                peak_gpu_gib=torch.cuda.max_memory_allocated(self.device)/2**30))
            if objective>self.best:
                self.best,self.best_epoch=objective,epoch
                save_torch(self.dest/'model.pt',dict(model=self.base.state_dict(),identity=self.identity,epoch=epoch,
                    arguments=model_arguments(self.c,self.task['variant'])))
                atomic_csv(self.dest/'validation_scores.csv',frame)
            atomic_csv(self.dest/'history.csv',pd.DataFrame(self.history))
            paper_json(self.dest/'progress.json',dict(status='epoch_complete',runner='shared_batch',**self.history[-1]))
            print('共享批次验证',self.task['key'],epoch,round(objective,6),flush=True)
        self.validated=epoch;self.checkpoint(epoch,0,'epoch_complete');self.cursor=(epoch+1,0);dist.barrier()

    def finish_training(self):
        if self.rank:return
        if len(self.history)!=self.c['epochs']:raise ValueError('训练epoch不足')
        if [h['epoch'] for h in self.history]!=list(range(1,self.c['epochs']+1)):
            raise ValueError('训练epoch重复或顺序错误')
        expected=max(self.history,key=lambda h:h['validation_objective'])
        if expected['epoch']!=self.best_epoch:raise ValueError('最佳checkpoint不是验证选择')
        frame=pd.read_csv(self.dest/'validation_scores.csv',float_precision='round_trip',dtype={'video_id':str})
        value,_=validation_objective(frame,self.vpairs)
        np.testing.assert_allclose(value,self.best,rtol=0,atol=1e-12)
        files=['model.pt','history.csv','validation_scores.csv','validation_pairs.csv']
        paper_json(self.dest/'manifest.json',dict(status='trained',identity=self.identity,task=self.task,best_epoch=self.best_epoch,
            best_validation=self.best,parameters=sum(p.numel() for p in self.base.parameters()),seconds=self.seconds(),
            time_scope='active wall time overlaps other members; group wall cost must be reported separately',
            test_used=False,runner='shared_batch',files={p:file_digest(self.dest/p) for p in files}))

    def collect_test(self,fm):
        logits=self.gather(len(fm))
        if self.rank:return
        self.test_logits=logits

    def finish_probe_and_test(self,out,fm,pairs,probe_indices):
        probe=self.gather(len(probe_indices))
        if self.rank:return
        expected=self.test_logits[probe_indices]
        np.testing.assert_allclose(probe,expected,rtol=2e-3,atol=2e-3)
        dest=out/'evaluation'/self.task['key'];dest.mkdir(parents=True,exist_ok=True)
        frame=score_frame(fm,self.test_logits).assign(variant=self.task['variant'],seed=self.task['seed'],fold=self.task['fold'])
        atomic_csv(dest/'scores.csv',frame)
        for name,t in evaluate_fixed_pairs(frame,pairs).items():atomic_csv(dest/(name+'.csv'),t)
        paper_json(dest/'manifest.json',dict(status='evaluated',runner='shared_batch',task=self.task,clips=len(frame),
            identity=dict(training=file_digest(self.dest/'manifest.json'),code=file_digest(Path(__file__))),
            seconds=None,time_scope='shared test pass; see group receipt',
            probe_max_error=float(np.max(np.abs(probe-expected))),probe_video_ids=fm.iloc[probe_indices].video_id.tolist(),
            scores_sha256=file_digest(dest/'scores.csv'),
            files={p.name:file_digest(p) for p in dest.glob('*.csv')}))


def group_worker(rank,world,q,port,root_string,out_string,c,tasks,specs,vm,vpairs,fm,tpairs,rt,ring,free_slots):
    out=Path(out_string);root=Path(root_string);group=out/'groups'/f'{tasks[0]["fold"]}__s{tasks[0]["seed"]}'
    os.environ['NCCL_P2P_DISABLE']='1';torch.set_num_threads(2);torch.cuda.set_device(rank);device=f'cuda:{rank}'
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    dist.init_process_group('nccl',init_method=(group/f'.rendezvous_{port}').as_uri(),rank=rank,world_size=world,timeout=datetime.timedelta(seconds=180))
    members=[Member(rank,world,out,c,t,s,vm,vpairs) for t,s in zip(tasks,specs)]
    transfer=torch.cuda.Stream(device=device);pool=ThreadPoolExecutor(max_workers=1);stop=threading.Event()
    n,d=ring.slots[0]['patches'].shape[-2:];pinned=BatchBuffers(rt['local_batch'],n,d,shared=False,pinned=True);events={}
    future=pool.submit(receive,q,device,transfer,stop,ring,pinned,free_slots,events);started=time.perf_counter();data_wait=0.
    torch.cuda.reset_peak_memory_stats(device)
    try:
        while True:
            waiting=time.perf_counter();p=future.result();data_wait+=time.perf_counter()-waiting
            if p['phase']=='stop':break
            future=pool.submit(receive,q,device,transfer,stop,ring,pinned,free_slots,events)
            if 'gpu' in p:
                torch.cuda.current_stream(device).wait_event(p['event']);b=p['gpu']
                for x in b.values():x.record_stream(torch.cuda.current_stream(device))
            phase=p['phase'];epoch=p['epoch'];selected=[members[i] for i in p.get('members',[])]
            if phase=='epoch_begin':
                for m in selected:m.begin_epoch(epoch)
            elif phase=='train':
                for m in selected:m.step(b,p)
                if p['last_micro'] and (p['update']+1)%10==0 and rank==0:
                    progress=dict(status='training',epoch=epoch,update=p['update']+1,updates=members[0].updates,
                        active_models=[m.task['variant'] for m in selected],wall_seconds=time.perf_counter()-started,
                        data_wait_seconds=data_wait,peak_gpu_gib=torch.cuda.max_memory_allocated(device)/2**30)
                    paper_json(group/'progress.json',progress);print('共享批次',tasks[0]['fold'],tasks[0]['seed'],epoch,p['update']+1,
                        '模型',len(selected),'秒',round(progress['wall_seconds'],1),flush=True)
            elif phase in ('validation_begin','test_begin','probe_begin'):
                if phase=='test_begin':
                    for m in members:m.finish_training()
                    dist.barrier()
                for m in selected:m.begin_scores(phase.split('_')[0],epoch)
                if rank==0:paper_json(group/'progress.json',dict(status=phase.split('_')[0],epoch=epoch,
                    active_models=[m.task['variant'] for m in selected],scored_videos=0,
                    wall_seconds=time.perf_counter()-started))
            elif phase in ('validation','test','probe'):
                for m in selected:m.infer(b,p)
                if rank==0 and p['buffer']['indices'] and p['buffer']['indices'][0] % 256==0:
                    paper_json(group/'progress.json',dict(status=phase,epoch=epoch,
                        active_models=[m.task['variant'] for m in selected],rank0_scored=len(selected[0].predictions),
                        wall_seconds=time.perf_counter()-started))
            elif phase=='validation_end':
                for m in selected:m.validate(epoch)
            elif phase=='test_end':
                for m in selected:m.collect_test(fm)
                dist.barrier()
            elif phase=='probe_end':
                ix=np.linspace(0,len(fm)-1,min(6,len(fm)),dtype=int)
                for m in selected:m.finish_probe_and_test(out,fm,tpairs,ix)
                dist.barrier()
        if rank==0:
            paper_json(group/'worker_complete.json',dict(status='completed',models=[t['key'] for t in tasks],
                wall_seconds=time.perf_counter()-started,data_wait_seconds=data_wait))
        dist.barrier()
    finally:stop.set();pool.shutdown(wait=True,cancel_futures=True);dist.destroy_process_group()


def run_group(root,out,c,tasks,meta,roles,reader,rt,test_meta,test_pairs):
    fold=tasks[0]['fold'];seed=tasks[0]['seed']
    if any(t['fold']!=fold or t['seed']!=seed for t in tasks):raise ValueError('只有同fold/seed才能共享抽样')
    group=out/'groups'/f'{fold}__s{seed}';group.mkdir(parents=True,exist_ok=True)
    with contextlib.ExitStack() as stack:
        for task in tasks:
            dest=out/'training'/task['key'];dest.mkdir(parents=True,exist_ok=True)
            lock=stack.enter_context((dest/'run.lock').open('a'));fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        s=roles[roles.fold.eq(fold)].set_index('video_id').loc[meta.video_id]
        tm=meta.loc[s.role.to_numpy()=='train'].reset_index(drop=True);vm=meta.loc[s.role.to_numpy()=='validation'].reset_index(drop=True)
        if set(tm.split_group)&set(vm.split_group):raise ValueError('源组交叉')
        if (set(tm.split_group)|set(vm.split_group))&set(test_meta.split_group):raise ValueError('测试源与训练/验证交叉')
        vpairs=validation_pairs(vm);specs=[];cursors=[];validated=[]
        for task in tasks:
            dest=out/'training'/task['key'];spec=task_spec(root,out,c,task,tm,vm,rt)
            spec['runner']='ddp_batch_reuse_v1';spec['scheduling']='same_fold_seed_shared_input_independent_updates'
            spec['code']['src/looped_video/group_train.py']=file_digest(Path(__file__))
            if (dest/'identity.json').exists():install_identity(dest,json.loads((dest/'identity.json').read_text()),spec)
            else:paper_json(dest/'identity.json',spec)
            atomic_csv(dest/'validation_pairs.csv',vpairs);specs.append(spec)
            cursor=(1,0);last_valid=0
            if (dest/'last.pt').exists():
                state=torch.load(dest/'last.pt',map_location='cpu',weights_only=True)
                if state['identity']!=config_digest(spec):raise ValueError('成员checkpoint身份不符')
                cursor=(state['epoch'],state['next_update']) if state['phase']=='train' else (state['epoch']+1,0)
                last_valid=max((h['epoch'] for h in state['history']),default=0)
            cursors.append(cursor);validated.append(last_valid)
        complete=True
        for task in tasks:
            td=out/'training'/task['key'];ed=out/'evaluation'/task['key']
            if not (ed/'manifest.json').exists():complete=False;continue
            tr=json.loads((td/'manifest.json').read_text());er=json.loads((ed/'manifest.json').read_text())
            for p,h in tr['files'].items():
                if file_digest(td/p)!=h:raise ValueError('已完成模型产物改变')
            if file_digest(ed/'scores.csv')!=er['scores_sha256']:raise ValueError('已完成测试改变')
        if complete:return
        paper_json(group/'plan.json',dict(tasks=tasks,initial_cursors=cursors,validated_epochs=validated,
            prepared=file_digest(out/'prepared.json'),runtime=rt,code=file_digest(Path(__file__))))
        weights=sampling_weights(tm);reader.configure(tm,weights)
        ctx=mp.get_context('spawn');queues=[ctx.Queue(maxsize=2) for _ in range(2)];free=[ctx.Queue(maxsize=2) for _ in range(2)]
        sample=reader.read(tm.iloc[0].key);n,d=sample.shape[-2:];rings=[BatchBuffers(rt['local_batch'],n,d) for _ in range(2)]
        for q in free:
            for slot in range(2):q.put(slot)
        with socket.socket() as sock:sock.bind(('127.0.0.1',0));port=sock.getsockname()[1]
        processes=[ctx.Process(target=group_worker,args=(rank,2,queues[rank],port,str(root),str(out),c,tasks,specs,
            vm,vpairs,test_meta,test_pairs,rt,rings[rank],free[rank])) for rank in range(2)]
        for p in processes:p.start()
        paper_json(out/'pipeline.json',dict(status='shared_batch_training',pid=os.getpid(),group=f'{fold}__s{seed}',
            models=[t['variant'] for t in tasks],workers=[dict(pid=p.pid,rank=r) for r,p in enumerate(processes)]))
        timing=dict(read_seconds=0.,pack_seconds=0.,slot_wait_seconds=0.);started=time.perf_counter()

        def control(phase,epoch,active):
            for q in queues:put(q,dict(phase=phase,epoch=epoch,members=active),processes)

        def send_data(frame,indices,counts,payload):
            waiting=time.perf_counter();slots=[acquire_slot(q,processes) for q in free]
            timing['slot_wait_seconds']+=time.perf_counter()-waiting
            packets,t=fill_shared(reader,frame,indices,rings,slots)
            for k,v in t.items():timing[k]+=v
            for rank,q in enumerate(queues):put(q,dict(**payload,count=counts[rank],buffer=packets[rank]),processes)

        def inference(phase,epoch,active,frame):
            control(phase+'_begin',epoch,active)
            for start in range(0,len(frame),2*rt['local_batch']):
                ids=[list(range(start+r*rt['local_batch'],min(start+(r+1)*rt['local_batch'],len(frame)))) for r in range(2)]
                send_data(frame,[x or [0] for x in ids],[len(x) for x in ids],dict(phase=phase,epoch=epoch,members=active))
            control(phase+'_end',epoch,active)

        try:
            first=min(x[0] for x in cursors)
            for epoch in range(first,c['epochs']+1):
                eligible=[i for i,x in enumerate(cursors) if epoch>=x[0]];control('epoch_begin',epoch,eligible)
                sampler=EpochSampler(weights,seed);sampler.epoch=epoch;draw=list(sampler)
                for part in global_microbatches(draw,rt['global_batch'],rt['local_batch']):
                    active=[i for i,x in enumerate(cursors) if (epoch,part['update'])>=x]
                    if not active:continue
                    send_data(tm,part['indices'],part['counts'],dict(phase='train',epoch=epoch,members=active,
                        update=part['update'],last_micro=part['last_micro'],denom=part['denom']))
                    if part['last_micro'] and (part['update']+1)%10==0:
                        paper_json(group/'io_progress.json',dict(epoch=epoch,enqueued_update=part['update']+1,
                            active_models=len(active),**reader.stats(),**timing))
                active=[i for i,x in enumerate(cursors) if epoch>=x[0] and epoch>validated[i]]
                if active:inference('validation',epoch,active,vm)
            inference('test',c['epochs'],list(range(len(tasks))),test_meta)
            probe_indices=np.linspace(0,len(test_meta)-1,min(6,len(test_meta)),dtype=int)
            inference('probe',c['epochs'],list(range(len(tasks))),test_meta.iloc[probe_indices].reset_index(drop=True))
            control('stop',c['epochs'],[])
            while any(p.is_alive() for p in processes):
                if any(p.exitcode not in (None,0) for p in processes):raise RuntimeError('共享批次worker失败')
                for p in processes:p.join(timeout=1)
            if any(p.exitcode for p in processes):raise RuntimeError('共享批次退出失败')
            paper_json(group/'manifest.json',dict(status='completed',models=[t['key'] for t in tasks],
                wall_seconds=time.perf_counter()-started,io=timing,reader=reader.stats(),
                results={t['key']:file_digest(out/'evaluation'/t['key']/'manifest.json') for t in tasks}))
        finally:
            for p in processes:
                if p.is_alive():p.terminate()
            for p in processes:p.join(timeout=10)
            for q in queues+free:q.close()


def main():
    import signal
    def interrupted(signum,frame):raise KeyboardInterrupt('停止共享批次，保留各模型更新检查点')
    signal.signal(signal.SIGTERM,interrupted)
    p=argparse.ArgumentParser();p.add_argument('--fold',default='');p.add_argument('--seed',type=int,default=0)
    p.add_argument('--ram-gib',type=int,default=48);a=p.parse_args()
    if not 0<=a.ram_gib<=56:raise ValueError('RAM预算越界')
    torch.set_num_threads(8);c=configuration(ROOT);out=ROOT/c['run_directory'];cache=ROOT/c['cache_directory']
    if (a.fold and a.fold not in c['folds']) or (a.seed and a.seed not in c['seeds']):raise ValueError('未知fold/seed')
    receipt=json.loads((cache/'manifest.json').read_text());meta=pd.read_csv(out/'videos.csv',keep_default_na=False)
    if receipt['status'] not in ('ready_for_training','verified'):raise ValueError('Patch缓存未准入')
    roles=pd.read_csv(out/'roles.csv',keep_default_na=False);external=pd.read_csv(out/'external_videos.csv',keep_default_na=False)
    tasks=json.loads((out/'tasks.json').read_text())['tasks']
    rt=dict(world_size=2,local_batch=c['batch_size'],global_batch=c['batch_size']*c['accumulation'],
        ram_gib=a.ram_gib,reader_workers=2,checkpoint_updates=25)
    reader=SharedReader(cache,receipt['records'],a.ram_gib,2)
    with (out/'pipeline.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        try:
            for seed in c['seeds']:
                for fold in c['folds']:
                    if (a.fold and fold!=a.fold) or (a.seed and seed!=a.seed):continue
                    group=[t for t in tasks if t['seed']==seed and t['fold']==fold]
                    if fold=='pooled':fm=external;pairs=pd.read_csv(out/'external_pairs.csv')
                    else:
                        ids=roles.loc[roles.fold.eq(fold)&roles.role.eq('test'),'video_id'];fm=meta[meta.video_id.isin(ids)].reset_index(drop=True)
                        pairs=pd.read_csv(out/'pairs.csv');pairs=pairs[pairs.dataset.eq(fold)]
                    run_group(ROOT,out,c,group,meta,roles,reader,rt,fm,pairs)
                    from .postprocess import process_group
                    paper_json(out/'pipeline.json',dict(status='group_evaluation',pid=os.getpid(),group=f'{fold}__s{seed}'))
                    process_group(ROOT,out,c,group,fm,pairs)
                    from .evaluation import report
                    report(ROOT,argparse.Namespace())
            all_tasks=[out/'evaluation'/t['key']/'manifest.json' for t in tasks]
            done=sum(p.exists() for p in all_tasks)
            paper_json(out/'pipeline.json',dict(status='group_matrix_finished_pending_final_audit' if done==len(tasks)
                else 'requested_groups_finished',pid=os.getpid(),evaluated_models=done,total_models=len(tasks)))
        except BaseException as exc:
            paper_json(out/'pipeline.json',dict(status='stopped' if isinstance(exc,KeyboardInterrupt) else 'failed',pid=os.getpid(),error=str(exc)))
            raise
        finally:reader.close()


if __name__=='__main__':main()
