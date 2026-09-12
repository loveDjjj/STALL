"""只在训练侧验证选择epoch；此模块不计算任何留域测试指标。"""
import json
import math
import time
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from artifacts import atomic_csv, paper_json
from config import config_digest
from reference import file_digest
from evaluation.tables import evaluate_fixed_pairs
from discriminative_moe.models import Classifier, training_standardizer
from discriminative_moe.run import configuration


KINDS=[('linear',1),('mlp',1),('uniform',2),('uniform',4),('moe',2),('moe',4)]
TRAINING_PROTOCOL='length_matched_supervision_v2'


def sampling_weights(frame):
    """域/真假/fake生成器等权；real长度分布匹配fake，层内源等权。"""
    result=pd.Series(0.,index=frame.index)
    for domain,d in frame.groupby('dataset'):
        real=d[d.subset.eq('real')];fake=d[d.subset.eq('annotated')]
        if not len(real) or not len(fake) or set(real.length)!=set(fake.length):
            raise ValueError('域/长度层缺少一类，禁止隐式丢样本')
        for _,b in fake.groupby('source_model'):
            count=b.groupby('split_group').video_id.transform('size')
            result.loc[b.index]=1/(frame.dataset.nunique()*2*fake.source_model.nunique()*b.split_group.nunique()*count)
        for length,b in real.groupby('length'):
            mass=float(result.loc[fake[fake.length.eq(length)].index].sum())
            count=b.groupby('split_group').video_id.transform('size')
            result.loc[b.index]=mass/(b.split_group.nunique()*count)
    w=result.to_numpy()
    if not np.isfinite(w).all() or (w<=0).any():raise ValueError('训练采样权重非法')
    return w/w.sum()


def validation_pairs(frame,seed=17):
    """标签与身份确定的平衡验证表，在训练前固定，不按预测筛选。"""
    rows=[]
    import hashlib
    for domain,d in frame.groupby('dataset'):
        real=d[d.subset.eq('real')]
        for generator,fake in d[d.subset.eq('annotated')].groupby('source_model'):
            for length,f in fake.groupby('length'):
                r=real[real.length.eq(length)];n=min(len(r),len(f))
                if n<2:raise ValueError(f'长度匹配验证支持不足：{domain}/{generator}/{length}')
                def choose(q):
                    ids=sorted(q.video_id,key=lambda x:hashlib.sha256(f'{seed}:{domain}:{generator}:{length}:{x}'.encode()).hexdigest())[:n]
                    return q.set_index('video_id').loc[ids].reset_index()
                for q in [choose(r),choose(f)]:
                    rows.extend(q[['video_id','dataset','subset']].assign(generator=generator,length=length).to_dict('records'))
    return pd.DataFrame(rows)


def feature_arrays(out,input_name,require_local=True):
    with np.load(out/'global_features.npz') as z:
        parts=[z['G']]
        if input_name in ('GT','GTL'):parts.append(z['T'])
        k=z['K'];ids=z['video_ids']
    if input_name=='GTL':
        meta=pd.read_csv(out/'videos.csv');local=[];local_hashes={}
        spec=json.loads((out/'local/identity.json').read_text());identity=config_digest(spec)
        for r in meta.itertuples():
            p=out/'local'/(r.key+'.npz')
            if not p.exists():raise ValueError('Local尚未齐备：'+r.key)
            local_hashes[r.key]=file_digest(p)
            with np.load(p) as z:
                if str(z['identity'])!=identity or str(z['key'])!=r.key:raise ValueError('Local特征身份错误')
                a=z['L'];padded=np.zeros((3,a.shape[-1]),np.float32);padded[:len(a)]=a;local.append(padded)
        parts.append(np.stack(local))
        receipt=dict(identity=identity,files=local_hashes)
        receipt_path=out/'local_features_manifest.json'
        if receipt_path.exists():
            if json.loads(receipt_path.read_text())!=receipt:raise ValueError('完整Local摘要内容变化')
        else:paper_json(receipt_path,receipt)
    if input_name not in ('G','GT','GTL'):raise ValueError('输入组合非法')
    x=np.concatenate(parts,axis=-1)
    if not np.isfinite(x).all():raise ValueError('分类输入有非有限值')
    return x,np.arange(3)[None,:]<k[:,None],ids


def score_frame(meta,probability):
    frame=meta[['video_id','dataset','subset','source_model','source_group','split_group']].copy()
    frame['final_score']=1-np.asarray(probability)
    return frame


def validation_objective(frame,pairs):
    tables=evaluate_fixed_pairs(frame,pairs)
    d=tables['dataset_metrics']
    return float(d[['auc','real_positive_ap']].mean().mean()),tables


def predict(model,x,mask,batch=256):
    model.eval();p=[];gates=[]
    with torch.no_grad():
        for start in range(0,len(x),batch):
            r=model(x[start:start+batch],mask[start:start+batch]);p.append(r['p_fake'].cpu().numpy());gates.append(r['gate'].cpu().numpy())
    return np.concatenate(p),np.concatenate(gates)


def train_job(root,out,c,meta,roles,features,mask,task,device,pilot=False):
    fold,input_name,kind,experts,seed=task
    key=f'{fold}__{input_name}__{kind}{experts}__s{seed}'
    dest=out/('training_probe' if pilot else 'training')/key;dest.mkdir(parents=True,exist_ok=True)
    split=roles[roles.fold.eq(fold)].set_index('video_id').loc[meta.video_id]
    train_idx=np.flatnonzero(split.role.eq('train'));val_idx=np.flatnonzero(split.role.eq('validation'))
    if pilot:train_idx=np.sort(np.random.default_rng(seed).choice(train_idx,size=min(len(train_idx),512),replace=False))
    train_meta=meta.iloc[train_idx];val_meta=meta.iloc[val_idx]
    spec=dict(task=list(task),training_protocol=TRAINING_PROTOCOL,configuration=c,prepared=file_digest(out/'prepared.json'),
        code={p:file_digest(root/p) for p in ['src/discriminative_moe/training.py','src/discriminative_moe/models.py']},
        local_identity=file_digest(out/'local/identity.json') if input_name=='GTL' else None,
        local_content=file_digest(out/'local_features_manifest.json') if input_name=='GTL' else None,
        train_ids=train_meta.video_id.tolist(),validation_ids=val_meta.video_id.tolist(),pilot=pilot)
    identity=config_digest(spec)
    marker=dest/'manifest.json'
    if marker.exists():
        m=json.loads(marker.read_text())
        if m['identity']!=identity:raise ValueError('训练恢复合同改变')
        for p,h in m['files'].items():
            if file_digest(dest/p)!=h:raise ValueError('训练结果hash改变')
        print('training cached',key,flush=True);return
    if (dest/'identity.json').exists() and json.loads((dest/'identity.json').read_text())!=spec:
        raise ValueError('未完成训练的输入/源码已改变，不能原地恢复')
    paper_json(dest/'identity.json',spec)
    x=torch.from_numpy(features).to(device);valid=torch.from_numpy(mask).to(device)
    mean,scale=training_standardizer(x[train_idx],valid[train_idx])
    # 非训练行只应用冻结变换，不参与标准化估计。
    x=(x-mean)/scale
    x=x.masked_fill(~valid[...,None],0)
    xt=x[train_idx];mt=valid[train_idx];xv=x[val_idx];mv=valid[val_idx]
    del x,valid
    y=torch.tensor(train_meta.subset.eq('annotated').to_numpy(),dtype=torch.float32,device=device)
    torch.manual_seed(seed);torch.cuda.manual_seed_all(seed)
    model=Classifier(features.shape[-1],kind,experts,c['hidden_budget'],c['dropout']).to(device)
    optimizer=torch.optim.AdamW(model.parameters(),lr=c['learning_rate'],weight_decay=c['weight_decay'])
    weights=sampling_weights(train_meta.reset_index(drop=True));rng=np.random.default_rng(seed)
    pairs=validation_pairs(val_meta,c['split_seed']);atomic_csv(dest/'validation_pairs.csv',pairs)
    epochs=2 if pilot else c['epochs'];history=[];best=-math.inf;best_epoch=0;start=time.perf_counter()
    for epoch in range(1,epochs+1):
        model.train();draw=rng.choice(len(xt),size=len(xt),replace=True,p=weights);losses=[]
        for i in range(0,len(draw),c['batch_size']):
            idx=torch.as_tensor(draw[i:i+c['batch_size']],device=device)
            optimizer.zero_grad(set_to_none=True);r=model(xt[idx],mt[idx]);loss=model.loss(r,y[idx],c['balance_weight'])
            if not torch.isfinite(loss):raise ValueError('训练loss非有限')
            loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),5.);optimizer.step();losses.append(float(loss.detach()))
        prob,gate=predict(model,xv,mv);frame=score_frame(val_meta,prob);objective,_=validation_objective(frame,pairs)
        history.append(dict(epoch=epoch,train_loss=np.mean(losses),validation_objective=objective,seconds=time.perf_counter()-start))
        if objective>best:
            best=objective;best_epoch=epoch
            temporary=dest/'model.tmp.pt'
            torch.save(dict(model={k:v.cpu() for k,v in model.state_dict().items()},mean=mean.cpu(),scale=scale.cpu(),
                            kind=kind,experts=experts,dimension=features.shape[-1],identity=identity,epoch=epoch),temporary)
            temporary.replace(dest/'model.pt');atomic_csv(dest/'validation_scores.csv',frame)
            atomic_csv(dest/'validation_routes.csv',pd.DataFrame(gate,columns=[f'expert_{i}' for i in range(gate.shape[1])]).assign(video_id=val_meta.video_id.to_numpy()))
        if epoch%10==0 or epoch==epochs:print('fit',key,epoch,'validation',round(objective,5),flush=True)
    atomic_csv(dest/'history.csv',pd.DataFrame(history))
    paper_json(marker,dict(status='probe_complete' if pilot else 'trained',training_protocol=TRAINING_PROTOCOL,identity=identity,task=list(task),
        best_validation=best,best_epoch=best_epoch,parameters=sum(p.numel() for p in model.parameters()),seconds=time.perf_counter()-start,
        test_not_evaluated=True,files={p:file_digest(dest/p) for p in ['model.pt','history.csv','validation_pairs.csv','validation_scores.csv','validation_routes.csv']}))
    print('trained',key,round(best,5),round(time.perf_counter()-start,1),flush=True)
    del model,optimizer,xt,mt,xv,mv;torch.cuda.empty_cache()


def train(root,args):
    root=Path(root);c=configuration(root);out=root/c['run_directory'];meta=pd.read_csv(out/'videos.csv');roles=pd.read_csv(out/'roles.csv')
    prepared=json.loads((out/'prepared.json').read_text())
    for p,h in prepared['files'].items():
        if file_digest(out/p)!=h:raise ValueError('训练输入身份改变')
    inputs=['GT'] if args.pilot else ['G','GT','GTL']
    counter=0
    for input_name in inputs:
        x,mask,ids=feature_arrays(out,input_name)
        np.testing.assert_array_equal(ids,meta.video_id)
        tasks=[(fold,input_name,kind,n,seed) for fold in c['datasets']+['pooled'] for kind,n in KINDS for seed in c['model_seeds']]
        if args.pilot:tasks=[('comgenvid',input_name,'moe',2,17)]
        for task in tasks:
            use=args.pilot or counter%args.world_size==args.rank;counter+=1
            if use:train_job(root,out,c,meta,roles,x,mask,task,f'cuda:{args.rank%2}',args.pilot)
        del x,mask
