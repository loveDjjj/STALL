"""训练侧冻结专家数后，才执行留域测试与分工诊断。"""
import json
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from artifacts import atomic_csv,paper_json
from config import config_digest
from reference import file_digest
from evaluation.tables import evaluate_fixed_pairs
from discriminative_moe.run import configuration
from discriminative_moe.models import Classifier
from discriminative_moe.training import KINDS,TRAINING_PROTOCOL,feature_arrays,score_frame,validation_objective


def choose_capacity(scores):
    """输入只能是2/4专家的验证seed结果；差距不超过最大SE则取2。"""
    values={n:np.asarray(scores[n],dtype=float) for n in (2,4)}
    if any(len(a)!=3 or not np.isfinite(a).all() for a in values.values()):
        raise ValueError('需要每容量三个验证seed，不能缺失或挑选')
    se={n:float(a.std(ddof=1)/np.sqrt(len(a))) for n,a in values.items()}
    delta=float(values[4].mean()-values[2].mean())
    return (4 if delta>max(se.values()) else 2),dict(delta_four_minus_two=delta,se_two=se[2],se_four=se[4])


def freeze_selection(root):
    root=Path(root);c=configuration(root);out=root/c['run_directory'];records=[];hashes={}
    for fold in c['datasets']+['pooled']:
        for inp in ('G','GT','GTL'):
            for kind,n in KINDS:
                for seed in c['model_seeds']:
                    key=f'{fold}__{inp}__{kind}{n}__s{seed}';dest=out/'training'/key
                    if not (dest/'manifest.json').exists():raise ValueError('模型尚未训练完成：'+key)
                    info=json.loads((dest/'manifest.json').read_text());spec=json.loads((dest/'identity.json').read_text())
                    if info['status']!='trained' or info['task']!=[fold,inp,kind,n,seed] or info['identity']!=config_digest(spec):
                        raise ValueError('训练身份错误：'+key)
                    if not info['test_not_evaluated'] or spec['pilot']:raise ValueError('不能用工程探针或测试结果选模型')
                    if spec['training_protocol']!=TRAINING_PROTOCOL or info['training_protocol']!=TRAINING_PROTOCOL:
                        raise ValueError('不能混入长度未匹配的旧训练')
                    if spec['configuration']!=c or spec['prepared']!=file_digest(out/'prepared.json'):
                        raise ValueError('训练配置或数据准备身份已改变')
                    for p,h in info['files'].items():
                        if file_digest(dest/p)!=h:raise ValueError('训练产物改变：'+key+'/'+p)
                    for p,h in spec['code'].items():
                        if file_digest(root/p)!=h:raise ValueError('训练源码改变')
                    scores=pd.read_csv(dest/'validation_scores.csv',float_precision='round_trip')
                    pairs=pd.read_csv(dest/'validation_pairs.csv');obj,_=validation_objective(scores,pairs)
                    np.testing.assert_allclose(obj,info['best_validation'],rtol=0,atol=1e-12)
                    history=pd.read_csv(dest/'history.csv')
                    assert len(history)==c['epochs'] and int(history.loc[history.validation_objective.idxmax(),'epoch'])==info['best_epoch']
                    records.append(dict(fold=fold,input=inp,kind=kind,experts=n,seed=seed,validation=obj,epoch=info['best_epoch'],parameters=info['parameters'],key=key))
                    hashes[key]=file_digest(dest/'manifest.json')
    table=pd.DataFrame(records);selections=[]
    for fold in c['datasets']+['pooled']:
        for inp in ('G','GT','GTL'):
            for kind in ('linear','mlp','uniform','moe'):
                q=table[(table.fold==fold)&(table.input==inp)&(table.kind==kind)]
                if kind in ('uniform','moe'):
                    n,diagnostic=choose_capacity({n:q[q.experts==n].sort_values('seed').validation.to_numpy() for n in (2,4)})
                else:n=1;diagnostic={}
                selections.append(dict(fold=fold,input=inp,kind=kind,experts=n,**diagnostic))
    receipt=dict(status='selection_frozen',training_protocol=TRAINING_PROTOCOL,training=hashes,configuration=c,
                 selection=selections,code=file_digest(Path(__file__)),test_results_used=False)
    marker=out/'selection.json'
    if marker.exists():
        if json.loads(marker.read_text())!=receipt:raise ValueError('已经冻结的选择发生改变')
    else:
        atomic_csv(out/'validation_model_grid.csv',table);atomic_csv(out/'selected_capacities.csv',pd.DataFrame(selections));paper_json(marker,receipt)
    return table,pd.DataFrame(selections)


def load_classifier(path,c,device):
    # 仅加载本run由训练器产生且manifest已校验的checkpoint。
    state=torch.load(path,map_location='cpu',weights_only=True)
    model=Classifier(state['dimension'],state['kind'],state['experts'],c['hidden_budget'],c['dropout']).to(device)
    model.load_state_dict(state['model']);model.eval()
    return model,state['mean'].to(device),state['scale'].to(device),state


def predict_diagnostics(model,features,mask,mean,scale,device,batch=256):
    probability=[];gates=[];experts=[];entropy=[];log_odds=[]
    with torch.no_grad():
        for start in range(0,len(features),batch):
            x=torch.from_numpy(features[start:start+batch]).to(device);m=torch.from_numpy(mask[start:start+batch]).to(device)
            x=((x-mean)/scale).masked_fill(~m[...,None],0);r=model(x,m)
            probability.append(r['p_fake'].cpu().numpy());gates.append(r['gate'].cpu().numpy())
            log_odds.append((r['log_real']-r['log_fake']).cpu().numpy())
            ex=(r['expert_probability']*m[...,None]).sum(1)/m.sum(1,keepdim=True);experts.append(ex.cpu().numpy())
            h=-(r['window_gate']*r['window_gate'].clamp_min(1e-12).log()).sum(-1)
            entropy.append(((h*m).sum(1)/m.sum(1)).cpu().numpy())
    return np.concatenate(probability),np.concatenate(gates),np.concatenate(experts),np.concatenate(entropy),np.concatenate(log_odds)


def evaluate(root,args):
    root=Path(root);c=configuration(root);out=root/c['run_directory'];_,chosen=freeze_selection(root)
    meta=pd.read_csv(out/'videos.csv');roles=pd.read_csv(out/'roles.csv');pairs=pd.read_csv(root/'results/paper_complete/pairs.csv')
    device=f'cuda:{args.rank%2}';all_scores=[];all_routes=[];all_tables={};correlations=[]
    for inp in ('G','GT','GTL'):
        features,mask,ids=feature_arrays(out,inp);np.testing.assert_array_equal(ids,meta.video_id)
        for fold in c['datasets']:
            r=roles[(roles.fold==fold)&(roles.role=='test')];idx=meta.index[meta.video_id.isin(r.video_id)].to_numpy();fm=meta.iloc[idx]
            assert fm.dataset.eq(fold).all()
            selection=chosen[(chosen.fold==fold)&(chosen.input==inp)].set_index('kind').experts.to_dict()
            requested=[('linear',1,'linear'),('mlp',1,'mlp'),('uniform',selection['uniform'],'uniform'),
                       ('moe',selection['moe'],'moe'),('uniform',selection['moe'],'uniform_matched')]
            for kind,n,label in requested:
                for seed in c['model_seeds']:
                    key=f'{fold}__{inp}__{kind}{n}__s{seed}';dest=out/'training'/key
                    model,mean,scale,state=load_classifier(dest/'model.pt',c,device)
                    info=json.loads((dest/'manifest.json').read_text())
                    if state['identity']!=info['identity']:raise ValueError('权重与训练身份不同')
                    p,gate,ex,h,odds=predict_diagnostics(model,features[idx],mask[idx],mean,scale,device)
                    frame=score_frame(fm,p).assign(input=inp,head=label,seed=seed,experts=n,fold=fold)
                    # 仅存数值诊断；主指标仍按冻结的1-p，不凭测试结果换评分方向。
                    frame['real_log_odds']=odds
                    all_scores.append(frame)
                    rt=fm[['video_id','dataset','subset','source_model','real_source','split_group']].reset_index(drop=True).copy()
                    for m in range(n):rt[f'gate_{m}']=gate[:,m];rt[f'expert_fake_{m}']=ex[:,m]
                    rt['window_entropy']=h;rt['effective_experts']=np.exp(h)
                    rt=rt.assign(input=inp,head=label,seed=seed,experts=n,fold=fold);all_routes.append(rt)
                    if n>1:
                        for a in range(n):
                            for b in range(a+1,n):
                                correlation=float(np.corrcoef(ex[:,a],ex[:,b])[0,1]) if min(ex[:,a].std(),ex[:,b].std())>0 else np.nan
                                correlations.append(dict(input=inp,head=label,fold=fold,seed=seed,a=a,b=b,pearson=correlation))
                    del model;torch.cuda.empty_cache()
            print('heldout scored',inp,fold,flush=True)
    scores=pd.concat(all_scores,ignore_index=True);routes=pd.concat(all_routes,ignore_index=True)
    for (inp,head,seed),q in scores.groupby(['input','head','seed']):
        for key,t in evaluate_fixed_pairs(q,pairs).items():
            all_tables.setdefault(key,[]).append(t.assign(input=inp,head=head,seed=seed))
    scores.to_csv(out/'test_scores.csv.gz',index=False);routes.to_csv(out/'test_routes.csv.gz',index=False)
    atomic_csv(out/'expert_correlations.csv',pd.DataFrame(correlations))
    for key,t in all_tables.items():atomic_csv(out/(key+'.csv'),pd.concat(t,ignore_index=True).replace({'scope':{'Macro-3':'Average'}}))
    metrics=pd.read_csv(out/'macro_metrics.csv')
    columns=['auc','real_positive_ap','fake_positive_ap','fake_tpr_at_1pct_real_fpr']
    summary=metrics.groupby(['input','head'])[columns].agg(['mean','std']);summary.columns=['_'.join(x) for x in summary.columns]
    atomic_csv(out/'seed_summary.csv',summary.reset_index())
    files=['test_scores.csv.gz','test_routes.csv.gz','expert_correlations.csv','seed_summary.csv']+[k+'.csv' for k in all_tables]
    paper_json(out/'evaluation_manifest.json',dict(status='point_estimates_complete',selection=file_digest(out/'selection.json'),
        code=file_digest(Path(__file__)),files={p:file_digest(out/p) for p in files}))
    print(summary[['auc_mean','real_positive_ap_mean']].to_string(),flush=True)
