"""保留总updates的四组训练；记录专家梯度与独立验证表现。"""
import hashlib,json,time
from pathlib import Path
import numpy as np,pandas as pd,torch
from sklearn.cluster import KMeans
from torch.nn import functional as F
from artifacts import atomic_csv,paper_json
from reference import file_digest
from config import config_digest
from discriminative_moe.models import Classifier,training_standardizer
from discriminative_moe.training import feature_arrays,sampling_weights,validation_pairs,validation_objective,score_frame
from moe_training.run import configuration

MODES=('wide_mlp','uniform','joint','warm')


def set_epoch(model,mode,epoch,warmup):
    if model.router is not None:model.router.requires_grad_(mode!='warm' or epoch>warmup)


def gradient_norm(parameters):
    values=[p.grad.detach().square().sum() for p in parameters if p.grad is not None]
    return torch.stack(values).sum().sqrt() if values else torch.tensor(0.)


def prepare(root,args=None):
    root=Path(root);c=configuration(root);source=root/c['source_directory'];out=root/c['run_directory'];out.mkdir(parents=True,exist_ok=True)
    names=['videos.csv','roles.csv','global_features.npz','prepared.json']
    spec=dict(config=c,inputs={n:file_digest(source/n) for n in names},code=file_digest(Path(__file__)))
    marker=out/'prepared.json'
    if marker.exists():
        m=json.loads(marker.read_text())
        if m['spec']!=spec:raise ValueError('准备合同改变')
        for n,h in m['files'].items():
            if file_digest(out/n)!=h:raise ValueError('准备文件改变')
        return
    original=json.loads((source/'prepared.json').read_text())
    for n in ['videos.csv','roles.csv','global_features.npz']:
        if original['files'][n]!=spec['inputs'][n]:raise ValueError('源特征身份不符')
    meta=pd.read_csv(source/'videos.csv');roles=pd.read_csv(source/'roles.csv');x,mask,ids=feature_arrays(source,'G')
    np.testing.assert_array_equal(ids,meta.video_id);contexts=[];centers={}
    desc=(x[:,:,:1024]*mask[...,None]).sum(1)/mask.sum(1)[:,None];desc/=np.maximum(np.linalg.norm(desc,axis=1,keepdims=True),1e-12)
    for fold in c['datasets']+['pooled']:
        r=roles[roles.fold.eq(fold)].set_index('video_id').loc[ids];train=np.flatnonzero(r.role.eq('train'))
        km=KMeans(n_clusters=c['diagnostic_clusters'],random_state=c['diagnostic_seed'],n_init=5)
        km.fit(desc[train],sample_weight=sampling_weights(meta.iloc[train].reset_index(drop=True)))
        label=km.predict(desc);centers[fold]=km.cluster_centers_
        contexts.extend(dict(fold=fold,video_id=v,context_cluster=int(k)) for v,k in zip(ids,label))
    atomic_csv(out/'contexts.csv',pd.DataFrame(contexts));np.savez(out/'context_centers.npz',**centers)
    paper_json(marker,dict(status='prepared',spec=spec,files={p:file_digest(out/p) for p in ['contexts.csv','context_centers.npz']}))


def validation_outputs(model,x,mask,y):
    """评估专家NLL时使用log域，不将饱和概率裁剪后冒充精确损失。"""
    model.eval();prob=[];nll=[];gate=[]
    with torch.no_grad():
        for start in range(0,len(x),256):
            z=x[start:start+256];m=mask[start:start+256];label=y[start:start+256]
            r=model(z,m);prob.append(r['p_fake'].cpu().numpy());gate.append(r['gate'].cpu().numpy())
            logits=torch.cat([e(z) for e in model.experts],dim=-1);logk=m.sum(1).log()[:,None]
            lf=torch.logsumexp(F.logsigmoid(logits).masked_fill(~m[...,None],-torch.inf),1)-logk
            lr=torch.logsumexp(F.logsigmoid(-logits).masked_fill(~m[...,None],-torch.inf),1)-logk
            nll.append((-(label[:,None]*lf+(1-label[:,None])*lr)).cpu().numpy())
    return np.concatenate(prob),np.concatenate(nll),np.concatenate(gate)


def train_one(root,c,meta,roles,x,mask,task,device,pilot=False):
    fold,mode,seed=task;out=root/c['run_directory'];key=f'{fold}__{mode}__s{seed}';dest=out/('pilot' if pilot else 'training')/key;dest.mkdir(parents=True,exist_ok=True)
    split=roles[roles.fold.eq(fold)].set_index('video_id').loc[meta.video_id];ti=np.flatnonzero(split.role.eq('train'));vi=np.flatnonzero(split.role.eq('validation'))
    if pilot:ti=np.sort(np.random.default_rng(seed).choice(ti,512,replace=False))
    spec=dict(config=c,task=list(task),pilot=pilot,prepared=file_digest(out/'prepared.json'),
        code={p:file_digest(root/p) for p in ['src/moe_training/training.py','src/discriminative_moe/models.py','src/discriminative_moe/training.py']},
        train_ids=meta.iloc[ti].video_id.tolist(),validation_ids=meta.iloc[vi].video_id.tolist())
    identity=config_digest(spec);marker=dest/'manifest.json'
    if marker.exists():
        info=json.loads(marker.read_text())
        if info['identity']!=identity:raise ValueError('训练恢复身份改变')
        for p,h in info['files'].items():
            if file_digest(dest/p)!=h:raise ValueError('训练产物改变')
        return
    if (dest/'identity.json').exists() and json.loads((dest/'identity.json').read_text())!=spec:raise ValueError('未完成训练的源码/输入改变')
    paper_json(dest/'identity.json',spec)
    allx=torch.from_numpy(x).to(device);allmask=torch.from_numpy(mask).to(device);mean,scale=training_standardizer(allx[ti],allmask[ti])
    allx=((allx-mean)/scale).masked_fill(~allmask[...,None],0);xt=allx[ti];xv=allx[vi];mt=allmask[ti];mv=allmask[vi];del allx,allmask
    yt=torch.tensor(meta.iloc[ti].subset.eq('annotated').to_numpy(),dtype=torch.float32,device=device);yv=torch.tensor(meta.iloc[vi].subset.eq('annotated').to_numpy(),dtype=torch.float32,device=device)
    kind='mlp' if mode=='wide_mlp' else ('uniform' if mode=='uniform' else 'moe');n=1 if kind=='mlp' else 2
    torch.manual_seed(seed);torch.cuda.manual_seed_all(seed);model=Classifier(x.shape[-1],kind,n,c['hidden_budget'],c['dropout']).to(device)
    optimizer=torch.optim.AdamW(model.parameters(),lr=c['learning_rate'],weight_decay=c['weight_decay'])
    weights=sampling_weights(meta.iloc[ti].reset_index(drop=True));rng=np.random.default_rng(seed)
    pairs=validation_pairs(meta.iloc[vi]);atomic_csv(dest/'validation_pairs.csv',pairs)
    labels=pd.read_csv(out/'contexts.csv');labels=labels[labels.fold.eq(fold)].set_index('video_id').loc[meta.iloc[vi].video_id,'context_cluster'].to_numpy()
    grouping=meta.iloc[vi][['dataset','subset','source_model']].reset_index(drop=True).assign(context_cluster=labels)
    epochs=2 if pilot else c['epochs'];warmup=1 if pilot else c['warmup_epochs'];best=-np.inf;best_epoch=0;history=[];details=[];start=time.perf_counter();updates=0
    def save(path,epoch):
        temp=path.with_suffix('.tmp.pt');torch.save(dict(model={k:v.cpu() for k,v in model.state_dict().items()},mean=mean.cpu(),scale=scale.cpu(),
            kind=kind,experts=n,dimension=x.shape[-1],identity=identity,epoch=epoch),temp);temp.replace(path)
    for epoch in range(1,epochs+1):
        set_epoch(model,mode,epoch,warmup);model.train();draw=rng.choice(len(ti),len(ti),replace=True,p=weights)
        draw_hash=hashlib.sha256(draw.astype('<i8').tobytes()).hexdigest();loss_sum=0.;gate_sum=np.zeros(n);entropy_sum=0.;grad_sum=np.zeros(n);router_grad=0.;batches=0
        for off in range(0,len(draw),c['batch_size']):
            ix=torch.as_tensor(draw[off:off+c['batch_size']],device=device);optimizer.zero_grad(set_to_none=True)
            r=model(xt[ix],mt[ix]);loss=model.loss(r,yt[ix],c['balance_weight'])
            if not torch.isfinite(loss):raise ValueError('非有限loss')
            loss.backward()
            norms=[float(gradient_norm(e.parameters())) for e in model.experts]
            rg=float(gradient_norm(model.router.parameters())) if model.router is not None else 0.
            grad_sum+=norms;router_grad+=rg
            torch.nn.utils.clip_grad_norm_(model.parameters(),c['gradient_clip']);optimizer.step();updates+=1;batches+=1
            b=len(ix);loss_sum+=float(loss.detach())*b;gate_sum+=r['gate'].detach().sum(0).cpu().numpy()
            h=-(r['window_gate'].detach()*r['window_gate'].detach().clamp_min(1e-12).log()).sum(-1)
            entropy_sum+=float(((h*mt[ix]).sum(1)/mt[ix].sum(1)).sum())
        p,expert_nll,vg=validation_outputs(model,xv,mv,yv);frame=score_frame(meta.iloc[vi],p);objective,_=validation_objective(frame,pairs)
        row=dict(epoch=epoch,updates=updates,sampling_sha256=draw_hash,train_loss=loss_sum/len(draw),validation_objective=objective,
                 train_gate_entropy=entropy_sum/len(draw),router_grad_norm=router_grad/batches,gate_trainable=bool(model.router is not None and next(model.router.parameters()).requires_grad),seconds=time.perf_counter()-start)
        for e in range(n):
            row[f'train_gate_{e}']=gate_sum[e]/len(draw);row[f'expert_{e}_grad_norm']=grad_sum[e]/batches;row[f'expert_{e}_validation_nll']=float(expert_nll[:,e].mean())
        history.append(row)
        for by in ['dataset','subset','source_model','context_cluster']:
            for group,index in grouping.groupby(by).groups.items():
                ix=np.asarray(list(index))
                for e in range(n):details.append(dict(epoch=epoch,by=by,group=str(group),expert=e,clips=len(ix),nll=float(expert_nll[ix,e].mean()),gate=float(vg[ix,e].mean())))
        if epoch==warmup:save(dest/'warmup.pt',epoch)
        if objective>best:
            best=objective;best_epoch=epoch;save(dest/'model.pt',epoch);atomic_csv(dest/'validation_scores.csv',frame)
        atomic_csv(dest/'history.csv',pd.DataFrame(history))
        if epoch%10==0 or epoch==epochs:print('schedule',key,epoch,'best validation',round(best,6),flush=True)
    save(dest/'final.pt',epochs);atomic_csv(dest/'expert_validation.csv',pd.DataFrame(details))
    paper_json(marker,dict(status='pilot_complete' if pilot else 'trained',identity=identity,task=list(task),parameters=sum(p.numel() for p in model.parameters()),
        updates=updates,best_epoch=best_epoch,best_validation=best,selected_before_gate_enabled=mode=='warm' and best_epoch<=warmup,seconds=time.perf_counter()-start,
        files={p:file_digest(dest/p) for p in ['model.pt','warmup.pt','final.pt','validation_pairs.csv','validation_scores.csv','history.csv','expert_validation.csv']}))
    print('completed',key,round(time.perf_counter()-start,1),flush=True)


def train(root,args):
    root=Path(root);c=configuration(root);prepare(root);source=root/c['source_directory'];meta=pd.read_csv(source/'videos.csv');roles=pd.read_csv(source/'roles.csv')
    x,mask,ids=feature_arrays(source,'G');np.testing.assert_array_equal(ids,meta.video_id)
    tasks=[(fold,mode,seed) for fold in c['datasets']+['pooled'] for mode in MODES for seed in c['seeds']]
    if args.pilot:tasks=[('comgenvid',mode,17) for mode in ['uniform','warm']]
    for i,task in enumerate(tasks):
        if args.pilot or i%args.world_size==args.rank:train_one(root,c,meta,roles,x,mask,task,f'cuda:{args.rank%2}',args.pilot)
