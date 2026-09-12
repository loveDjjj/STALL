"""双GPU独立任务并行；按训练侧验证选epoch，源分组用途保持冻结。"""
import contextlib
import fcntl
import json
import math
import time
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Sampler
from artifacts import atomic_csv, paper_json
from config import config_digest
from reference import file_digest
from discriminative_moe.training import sampling_weights, validation_pairs, validation_objective
from .data import PatchDataset, collate_videos
from .model import LoopedDetector
from .run import configuration


def model_arguments(c,variant):
    arguments=dict(width=c['width'],heads=c['heads'],loops=c['loops'],tile=c['tile'],
                   mlp_ratio=c['mlp_ratio'],variant=variant)
    arguments.update(c.get('experiment_overrides',{}).get(variant,{}))
    return arguments


class EpochSampler(Sampler):
    def __init__(self,weights,seed):self.weights,self.seed,self.epoch=weights,seed,0
    def __iter__(self):
        rng=np.random.default_rng(np.random.SeedSequence([self.seed,self.epoch]))
        return iter(rng.choice(len(self.weights),len(self.weights),replace=True,p=self.weights).tolist())
    def __len__(self):return len(self.weights)


def loader(meta,cache,records,c,sampler=None):
    return DataLoader(PatchDataset(meta,cache,records),batch_size=c['batch_size'],sampler=sampler,
        shuffle=False,num_workers=c['loader_workers'],collate_fn=collate_videos,pin_memory=True,
        persistent_workers=c['loader_workers']>0,
        **({'prefetch_factor':c['loader_prefetch']} if c['loader_workers'] else {}))


def forward_batch(model,batch,device,amp):
    with torch.autocast('cuda',dtype=torch.bfloat16,enabled=amp):
        return model(batch['patches'].to(device,non_blocking=True),batch['valid'].to(device,non_blocking=True),
            batch['owners'].to(device,non_blocking=True),len(batch['labels'])).float()


def predict(model,stream,device,amp):
    model.eval();out=np.empty(len(stream.dataset),dtype=np.float64)
    with torch.inference_mode():
        for batch in stream:
            logits=forward_batch(model,batch,device,amp)
            out[batch['indices'].numpy()]=logits.cpu().numpy()
    if not np.isfinite(out).all():raise ValueError('预测非有限')
    return out


def score_frame(meta,logits):
    """高分为real；使用负logit避免sigmoid在极端值处的排序饱和。"""
    from scipy.special import expit
    f=meta[['video_id','dataset','subset','source_model','source_group','split_group']].copy()
    f['fake_logit']=logits;f['p_fake']=expit(logits);f['final_score']=-np.asarray(logits)
    return f


def save_torch(path,value):
    temp=path.with_suffix('.tmp.pt');torch.save(value,temp);temp.replace(path)


def train_job(root,out,cache,records,c,meta,roles,task,device):
    dest=out/'training'/task['key'];dest.mkdir(parents=True,exist_ok=True)
    with (dest/'run.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        split=roles[roles.fold.eq(task['fold'])].set_index('video_id').loc[meta.video_id]
        tm=meta.loc[split.role.to_numpy()=='train'].reset_index(drop=True)
        vm=meta.loc[split.role.to_numpy()=='validation'].reset_index(drop=True)
        if set(tm.split_group)&set(vm.split_group):raise ValueError('训练验证源重叠')
        spec=dict(task=task,config=c,prepared=file_digest(out/'prepared.json'),
            cache_manifest=file_digest(cache/'manifest.json'),train_ids=tm.video_id.tolist(),validation_ids=vm.video_id.tolist(),
            code={p:file_digest(root/p) for p in ['src/looped_video/model.py','src/looped_video/training.py',
                'src/looped_video/data.py','src/discriminative_moe/training.py','src/evaluation/tables.py']})
        identity=config_digest(spec)
        if (dest/'manifest.json').exists():
            old=json.loads((dest/'manifest.json').read_text())
            if old['identity']!=identity:raise ValueError('训练完成后合同改变')
            for p,h in old['files'].items():
                if file_digest(dest/p)!=h:raise ValueError('训练资产改变')
            print('训练缓存命中',task['key'],flush=True);return
        if (dest/'identity.json').exists() and json.loads((dest/'identity.json').read_text())!=spec:
            raise ValueError('中断任务合同改变，不能混合恢复')
        paper_json(dest/'identity.json',spec)
        pairs=validation_pairs(vm);atomic_csv(dest/'validation_pairs.csv',pairs)
        torch.manual_seed(task['seed']);torch.cuda.manual_seed_all(task['seed'])
        model=LoopedDetector(**model_arguments(c,task['variant'])).to(device)
        optimizer=torch.optim.AdamW(model.parameters(),lr=c['learning_rate'],weight_decay=c['weight_decay'])
        sampler=EpochSampler(sampling_weights(tm),task['seed'])
        tl=loader(tm,cache,records,c,sampler);vl=loader(vm,cache,records,c)
        start_epoch=1;history=[];best=-math.inf;best_epoch=0;previous_seconds=0.
        if (dest/'last.pt').exists():
            state=torch.load(dest/'last.pt',map_location=device,weights_only=True)
            if state['identity']!=identity:raise ValueError('续跑checkpoint身份不同')
            model.load_state_dict(state['model']);optimizer.load_state_dict(state['optimizer'])
            start_epoch=state['epoch']+1;history=state['history'];best=state['best'];best_epoch=state['best_epoch']
            previous_seconds=state['seconds']
        start=time.perf_counter();torch.cuda.reset_peak_memory_stats(device)
        updates_per_epoch=math.ceil(len(tl)/c['accumulation'])
        total_updates=updates_per_epoch*c['epochs'];warm=updates_per_epoch*c['warmup_epochs']
        for epoch in range(start_epoch,c['epochs']+1):
            model.train();sampler.epoch=epoch;optimizer.zero_grad(set_to_none=True)
            epoch_loss=0.;count=0;epoch_start=time.perf_counter();compute_start=epoch_start
            for i,batch in enumerate(tl):
                labels=batch['labels'].to(device,non_blocking=True)
                logits=forward_batch(model,batch,device,c['amp'])
                loss_sum=torch.nn.functional.binary_cross_entropy_with_logits(logits,labels,reduction='sum')
                group_start=(i//c['accumulation'])*c['accumulation']*c['batch_size']
                denominator=min(c['accumulation']*c['batch_size'],len(tm)-group_start)
                loss=loss_sum/denominator
                if not torch.isfinite(loss):raise ValueError('loss非有限')
                loss.backward();epoch_loss+=float(loss_sum.detach());count+=len(labels)
                if (i+1)%c['accumulation']==0 or i+1==len(tl):
                    update=(epoch-1)*updates_per_epoch+i//c['accumulation']+1
                    factor=update/max(warm,1) if update<=warm else .5*(1+math.cos(math.pi*(update-warm)/max(total_updates-warm,1)))
                    for group in optimizer.param_groups:group['lr']=c['learning_rate']*factor
                    norm=torch.nn.utils.clip_grad_norm_(model.parameters(),5.,error_if_nonfinite=True)
                    optimizer.step();optimizer.zero_grad(set_to_none=True)
                if (i+1)%50==0:
                    paper_json(dest/'progress.json',dict(status='training',epoch=epoch,epochs=c['epochs'],
                        batch=i+1,batches=len(tl),train_loss=epoch_loss/count,
                        epoch_seconds=time.perf_counter()-epoch_start,peak_gpu_gib=torch.cuda.max_memory_allocated(device)/2**30))
            train_seconds=time.perf_counter()-epoch_start
            logits=predict(model,vl,device,c['amp']);frame=score_frame(vm,logits)
            objective,_=validation_objective(frame,pairs)
            seconds=previous_seconds+time.perf_counter()-start
            row=dict(epoch=epoch,train_loss=epoch_loss/count,validation_objective=objective,
                train_seconds=train_seconds,epoch_seconds=time.perf_counter()-epoch_start,seconds=seconds,
                videos_per_second=count/train_seconds,peak_gpu_gib=torch.cuda.max_memory_allocated(device)/2**30)
            history.append(row)
            if objective>best:
                best,best_epoch=objective,epoch
                save_torch(dest/'model.pt',dict(model=model.state_dict(),identity=identity,epoch=epoch,
                    arguments=model_arguments(c,task['variant'])))
                atomic_csv(dest/'validation_scores.csv',frame)
            save_torch(dest/'last.pt',dict(model=model.state_dict(),optimizer=optimizer.state_dict(),identity=identity,
                epoch=epoch,history=history,best=best,best_epoch=best_epoch,seconds=seconds))
            atomic_csv(dest/'history.csv',pd.DataFrame(history))
            paper_json(dest/'progress.json',dict(status='epoch_complete',**row,epochs=c['epochs'],
                eta_seconds=(c['epochs']-epoch)*float(np.mean([h['epoch_seconds'] for h in history[-3:]]))))
            print('训练',task['key'],'epoch',epoch,'val',round(objective,5),'秒',round(row['epoch_seconds'],1),flush=True)
        files=['model.pt','history.csv','validation_scores.csv','validation_pairs.csv']
        paper_json(dest/'manifest.json',dict(status='trained',identity=identity,task=task,best_epoch=best_epoch,
            best_validation=best,parameters=sum(p.numel() for p in model.parameters()),
            seconds=previous_seconds+time.perf_counter()-start,test_used=False,
            files={p:file_digest(dest/p) for p in files}))
        del tl,vl,model,optimizer
        torch.cuda.empty_cache()


def train(root,args):
    root=Path(root);c=configuration(root);out=root/c['run_directory'];cache=root/c['cache_directory']
    receipt=json.loads((cache/'manifest.json').read_text())
    if receipt['status'] not in ('verified','ready_for_training'):raise ValueError('缓存未验收')
    if receipt['status']=='ready_for_training' and receipt.get('verification',{}).get('mode')!='producer_full_sha256_readback_plus_all_stat_headers_indices':
        raise ValueError('训练准入缺少明确检查模式')
    meta=pd.read_csv(out/'videos.csv',keep_default_na=False);roles=pd.read_csv(out/'roles.csv',keep_default_na=False)
    tasks=json.loads((out/'tasks.json').read_text())['tasks']
    tasks=sorted(tasks,key=lambda t:(c['variants'].index(t['variant']),c['folds'].index(t['fold']),c['seeds'].index(t['seed'])))
    if args.task:
        tasks=[t for t in tasks if t['key']==args.task]
        if len(tasks)!=1:raise ValueError('任务不在冻结矩阵')
    else:tasks=[t for i,t in enumerate(tasks) if i%args.world_size==args.rank]
    for task in tasks:
        train_job(root,out,cache,receipt['records'],c,meta,roles,task,f'cuda:{args.rank%2}')
