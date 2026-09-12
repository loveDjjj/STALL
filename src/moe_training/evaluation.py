"""冻结epoch后统一评价开发和外部；训练日程与容量分别归因。"""
import json
from pathlib import Path
import numpy as np,pandas as pd,torch
from artifacts import atomic_csv,paper_json
from reference import file_digest
from config import config_digest
from evaluation.tables import evaluate_fixed_pairs
from discriminative_moe.training import feature_arrays,score_frame,validation_objective
from discriminative_moe.evaluation import load_classifier,predict_diagnostics
from discriminative_moe.analysis import paired_seed_contrast
from discriminative_moe.audit import independent_probability,independent_standardizer
from moe_training.run import configuration
from moe_training.training import MODES


CONTRASTS={'joint_vs_uniform':('joint','uniform'),'warm_vs_joint':('warm','joint'),'warm_vs_uniform':('warm','uniform'),
           'joint_vs_wide_mlp':('joint','wide_mlp'),'warm_vs_wide_mlp':('warm','wide_mlp'),
           'wide_mlp_vs_narrow':('wide_mlp','narrow_mlp'),'uniform_vs_narrow':('uniform','narrow_uniform'),
           'joint_vs_narrow':('joint','narrow_moe')}


def freeze(root):
    c=configuration(root);out=root/c['run_directory'];source=root/c['source_directory'];rows=[];hashes={}
    for fold in c['datasets']+['pooled']:
        for mode in MODES:
            for seed in c['seeds']:
                key=f'{fold}__{mode}__s{seed}';p=out/'training'/key;m=json.loads((p/'manifest.json').read_text());spec=json.loads((p/'identity.json').read_text())
                if m['status']!='trained' or m['task']!=[fold,mode,seed] or m['identity']!=config_digest(spec) or spec['pilot']:raise ValueError('模型身份不符')
                if spec['config']!=c or spec['prepared']!=file_digest(out/'prepared.json'):raise ValueError('配置/数据改变')
                for name,h in spec['code'].items():
                    if file_digest(root/name)!=h:raise ValueError('训练源码改变')
                for name,h in m['files'].items():
                    if file_digest(p/name)!=h:raise ValueError('模型产物改变')
                v=pd.read_csv(p/'validation_scores.csv',float_precision='round_trip');pairs=pd.read_csv(p/'validation_pairs.csv');objective,_=validation_objective(v,pairs)
                np.testing.assert_allclose(objective,m['best_validation'],rtol=0,atol=1e-12)
                history=pd.read_csv(p/'history.csv');assert len(history)==c['epochs']
                assert int(history.loc[history.validation_objective.idxmax(),'epoch'])==m['best_epoch']
                rows.append(dict(fold=fold,mode=mode,seed=seed,epoch=m['best_epoch'],validation=objective,parameters=m['parameters'],updates=m['updates'],selected_before_gate_enabled=m['selected_before_gate_enabled']))
                hashes[key]=file_digest(p/'manifest.json')
    selection=dict(status='frozen',models=hashes,config=c,code=file_digest(Path(__file__)))
    path=out/'selection.json'
    if path.exists():
        if json.loads(path.read_text())!=selection:raise ValueError('冻结选择改变')
    else:atomic_csv(out/'selected_epochs.csv',pd.DataFrame(rows));paper_json(path,selection)
    return pd.DataFrame(rows)


def external_arrays(source):
    folder=source/'external';m=json.loads((folder/'evaluation_manifest.json').read_text());jobs=json.loads((folder/'jobs.json').read_text())['jobs'];xs=[];mask=[]
    for j in jobs:
        p=folder/'features'/(j['key']+'.npz')
        if file_digest(p)!=m['feature_hashes'][j['key']]:raise ValueError('外部特征改变')
        with np.load(p) as z:g=z['G'];a=np.zeros((3,2048),np.float32);a[:len(g)]=g
        xs.append(a);mask.append(np.arange(3)<len(g))
    meta=pd.read_csv(folder/'videos.csv');np.testing.assert_array_equal(meta.video_id,[j['video_id'] for j in jobs])
    return np.stack(xs),np.stack(mask),meta,pd.read_csv(folder/'pairs.csv')


def evaluate(root,args):
    root=Path(root);c=configuration(root);out=root/c['run_directory'];source=root/c['source_directory'];freeze(root)
    x,mask,ids=feature_arrays(source,'G');meta=pd.read_csv(source/'videos.csv');roles=pd.read_csv(source/'roles.csv');pairs=pd.read_csv(root/'results/paper_complete/pairs.csv')
    ex,em,ext,ep=external_arrays(source);scores=[];routes=[]
    for fold in c['datasets']+['pooled']:
        if fold=='pooled':z,m,q,scope=ex,em,ext,'external'
        else:
            ti=roles[(roles.fold==fold)&(roles.role=='test')].video_id;index=meta.index[meta.video_id.isin(ti)].to_numpy()
            z,m,q,scope=x[index],mask[index],meta.iloc[index],'development'
        for mode in MODES:
            for seed in c['seeds']:
                p=out/'training'/f'{fold}__{mode}__s{seed}'/'model.pt';model,mean,scale,state=load_classifier(p,c,f'cuda:{args.rank%2}')
                pred,g,ev,h,odds=predict_diagnostics(model,z,m,mean,scale,f'cuda:{args.rank%2}')
                scores.append(score_frame(q,pred).assign(input='G',head=mode,seed=seed,scope=scope,real_log_odds=odds))
                r=q[['video_id','dataset','subset','source_model']].reset_index(drop=True)
                for i in range(g.shape[1]):r[f'gate_{i}']=g[:,i];r[f'expert_fake_{i}']=ev[:,i]
                routes.append(r.assign(head=mode,seed=seed,scope=scope,entropy=h))
                del model
    # 同一数据/预算合同下已有窄模型，单列为容量参照。
    for scope,folder in [('development',source),('external',source/'external')]:
        old=pd.read_csv(folder/'test_scores.csv.gz',float_precision='round_trip');old=old[(old.input=='G')&old['head'].isin(['mlp','uniform','moe'])].copy()
        old['head']='narrow_'+old['head'];old['scope']=scope;scores.append(old)
    scores=pd.concat(scores,ignore_index=True);scores.to_csv(out/'test_scores.csv.gz',index=False);pd.concat(routes,ignore_index=True).to_csv(out/'routes.csv.gz',index=False)
    tables={}
    for (scope,head,seed),q in scores.groupby(['scope','head','seed']):
        for name,t in evaluate_fixed_pairs(q,pairs if scope=='development' else ep).items():
            if len(t):tables.setdefault(name,[]).append(t.assign(scope_kind=scope,head=head,seed=seed))
    for name,items in tables.items():atomic_csv(out/(name+'.csv'),pd.concat(items,ignore_index=True).replace({'scope':{'Macro-3':'Average'}}))
    paper_json(out/'evaluation_manifest.json',dict(status='evaluated',selection=file_digest(out/'selection.json'),files={p:file_digest(out/p) for p in ['test_scores.csv.gz','routes.csv.gz']+[n+'.csv' for n in tables]}))
    print(pd.read_csv(out/'macro_metrics.csv').groupby('head')[['auc','real_positive_ap']].mean().to_string(),flush=True)


def analyze(root,args=None):
    root=Path(root);c=configuration(root);out=root/c['run_directory'];source=root/c['source_directory'];s=pd.read_csv(out/'test_scores.csv.gz',float_precision='round_trip');rows=[]
    for scope,pairs_path in [('development',root/'results/paper_complete/pairs.csv'),('external',source/'external/pairs.csv')]:
        q=s[s.scope==scope];pairs=pd.read_csv(pairs_path)
        for name,(a,b) in CONTRASTS.items():
            r=paired_seed_contrast(q,pairs,('G',a),('G',b),seeds=tuple(c['seeds']),iterations=c['bootstrap_iterations'],seed=c['bootstrap_seed'])
            if scope=='external':r=r[r.dataset!='Average']
            rows.append(r.assign(contrast=name,scope_kind=scope));print('schedule CI',scope,name,flush=True)
    atomic_csv(out/'confidence_intervals.csv',pd.concat(rows,ignore_index=True))
    paper_json(out/'analysis_manifest.json',dict(status='completed',scores=file_digest(out/'test_scores.csv.gz'),files={'confidence_intervals.csv':file_digest(out/'confidence_intervals.csv')}))


def verify(root,args=None):
    root=Path(root);c=configuration(root);out=root/c['run_directory'];source=root/c['source_directory'];freeze(root)
    x,mask,ids=feature_arrays(source,'G');meta=pd.read_csv(source/'videos.csv');roles=pd.read_csv(source/'roles.csv');prefix_error=0.;prob_error=0.;probes=0
    for fold in c['datasets']+['pooled']:
        r=roles[roles.fold==fold].set_index('video_id').loc[ids];ti=np.flatnonzero(r.role.eq('train'));vi=np.flatnonzero(r.role.eq('validation'));mu,sd=independent_standardizer(x[ti],mask[ti])
        for seed in c['seeds']:
            histories={}
            for mode in MODES:
                dest=out/'training'/f'{fold}__{mode}__s{seed}';h=pd.read_csv(dest/'history.csv');histories[mode]=h
                spec=json.loads((dest/'identity.json').read_text());assert spec['train_ids']==meta.iloc[ti].video_id.tolist() and spec['validation_ids']==meta.iloc[vi].video_id.tolist()
                model=torch.load(dest/'model.pt',weights_only=True,map_location='cpu');np.testing.assert_allclose(model['mean'],mu,rtol=1e-5,atol=1e-7);np.testing.assert_allclose(model['scale'],sd,rtol=1e-5,atol=1e-7)
                ix=vi[[0,-1]];p=independent_probability(model,x[ix],mask[ix]);v=pd.read_csv(dest/'validation_scores.csv',float_precision='round_trip').set_index('video_id');expected=1-v.loc[ids[ix],'final_score'].to_numpy()
                np.testing.assert_allclose(p,expected,rtol=0,atol=2e-5);prob_error=max(prob_error,float(np.abs(p-expected).max()));probes+=2
                if mode=='warm':
                    assert not h[h.epoch<=10].gate_trainable.any() and h[h.epoch>10].gate_trainable.all()
                    assert h[h.epoch<=10].router_grad_norm.eq(0).all()
            for mode in MODES:
                np.testing.assert_array_equal(histories[mode].sampling_sha256,histories['uniform'].sampling_sha256)
                np.testing.assert_array_equal(histories[mode].updates,histories['uniform'].updates)
            u=torch.load(out/'training'/f'{fold}__uniform__s{seed}'/'warmup.pt',weights_only=True,map_location='cpu')
            w=torch.load(out/'training'/f'{fold}__warm__s{seed}'/'warmup.pt',weights_only=True,map_location='cpu')
            for k,v in u['model'].items():
                error=float((v-w['model'][k]).abs().max());prefix_error=max(prefix_error,error);torch.testing.assert_close(v,w['model'][k],rtol=0,atol=0)
            assert all(torch.count_nonzero(v)==0 for k,v in w['model'].items() if k.startswith('router.'))
    for name in ['evaluation_manifest.json','analysis_manifest.json']:
        for p,h in json.loads((out/name).read_text())['files'].items():assert file_digest(out/p)==h
    scores=pd.read_csv(out/'test_scores.csv.gz',float_precision='round_trip');ex,em,ext,ep=external_arrays(source)
    for (scope,head,seed),q in scores.groupby(['scope','head','seed']):
        assert q.video_id.nunique()==(15569 if scope=='development' else 2216)
        pairs=pd.read_csv(root/'results/paper_complete/pairs.csv') if scope=='development' else ep
        actual=evaluate_fixed_pairs(q,pairs)
        for name,keys in [('generator_metrics',['dataset','generator']),('dataset_metrics',['dataset']),('full_population_metrics',['dataset'])]:
            a=actual[name].set_index(keys).sort_index();b=pd.read_csv(out/(name+'.csv'),float_precision='round_trip')
            b=b[(b.scope_kind==scope)&(b['head']==head)&(b.seed==seed)].drop(columns=['scope_kind','head','seed']).set_index(keys).sort_index()
            pd.testing.assert_frame_equal(a,b,check_dtype=False,check_exact=False,rtol=0,atol=1e-12)
        if head in MODES:
            for domain,d in q.groupby('dataset'):
                fold=domain if scope=='development' else 'pooled';state=torch.load(out/'training'/f'{fold}__{head}__s{seed}'/'model.pt',weights_only=True,map_location='cpu')
                base=meta if scope=='development' else ext;xx=x if scope=='development' else ex;mm=mask if scope=='development' else em
                for label in ['real','annotated']:
                    video=d[d.subset==label].video_id.iloc[0];ix=np.flatnonzero(base.video_id.eq(video))
                    prediction=independent_probability(state,xx[ix],mm[ix])[0];expected=1-float(d.set_index('video_id').loc[video,'final_score'])
                    np.testing.assert_allclose(prediction,expected,rtol=0,atol=2e-5);prob_error=max(prob_error,abs(prediction-expected));probes+=1
    ci=pd.read_csv(out/'confidence_intervals.csv');assert len(ci)==8*(8+4)
    macro=pd.read_csv(out/'macro_metrics.csv');ds=pd.read_csv(out/'dataset_metrics.csv')
    for r in ci.itertuples():
        a,b=CONTRASTS[r.contrast];table=macro if r.dataset=='Average' else ds[(ds.dataset==r.dataset)&(ds.scope_kind==r.scope_kind)]
        metric='auc' if r.metric=='auc' else 'real_positive_ap'
        expected=table[table['head']==a][metric].mean()-table[table['head']==b][metric].mean()
        np.testing.assert_allclose(r.delta,expected,rtol=0,atol=1e-12)
    paper_json(out/'verification.json',dict(status='verified',models=48,prefix_exact=prefix_error==0,identical_sampling_updates=True,
        independent_validation_probes=probes,probability_error=prob_error,code=file_digest(Path(__file__))))
    print('schedule verification passed',flush=True)


def report(root,args=None):
    from moe_training.reporting import report as implementation
    implementation(Path(root))
