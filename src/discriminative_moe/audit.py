"""对数据用途、训练标准化、保存分类器和最终指标独立核验。"""
import json
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from scipy.special import erf,expit,softmax
from artifacts import paper_json
from config import config_digest
from reference import file_digest
from evaluation.tables import evaluate_fixed_pairs
from discriminative_moe.run import configuration
from discriminative_moe.training import feature_arrays,TRAINING_PROTOCOL,sampling_weights
from discriminative_moe.analysis import contrasts


def independent_standardizer(x,mask):
    """分块双遍、每视频等权；不调用训练器函数。"""
    mean=np.zeros(x.shape[-1],np.float64)
    for start in range(0,len(x),256):
        b=x[start:start+256].astype(np.float64);m=mask[start:start+256]
        mean+=((b*m[...,None]).sum(1)/m.sum(1)[:,None]).sum(0)
    mean/=len(x);var=np.zeros_like(mean)
    for start in range(0,len(x),256):
        b=x[start:start+256].astype(np.float64);m=mask[start:start+256]
        var+=(((b-mean)**2*m[...,None]).sum(1)/m.sum(1)[:,None]).sum(0)
    return mean.astype(np.float32),np.maximum(np.sqrt(var/len(x)),1e-6).astype(np.float32)


def independent_probability(state,features,mask):
    """用NumPy/Scipy直接执行保存权重，不调用PyTorch分类器forward。"""
    weights={k:v.numpy().astype(np.float64) for k,v in state['model'].items()}
    z=(features.astype(np.float64)-state['mean'].numpy())/state['scale'].numpy();z=np.where(mask[...,None],z,0)
    n=state['experts'] if state['kind'] in ('uniform','moe') else 1;values=[]
    for i in range(n):
        if state['kind']=='linear':v=z@weights[f'experts.{i}.weight'].T+weights[f'experts.{i}.bias']
        else:
            h=z@weights[f'experts.{i}.0.weight'].T+weights[f'experts.{i}.0.bias']
            h=.5*h*(1+erf(h/np.sqrt(2)))
            v=h@weights[f'experts.{i}.3.weight'].T+weights[f'experts.{i}.3.bias']
        values.append(expit(v[...,0]))
    values=np.stack(values,axis=-1)
    gate=softmax(z@weights['router.weight'].T+weights['router.bias'],axis=-1) if state['kind']=='moe' else np.full_like(values,1/n)
    return ((values*gate).sum(-1)*mask).sum(1)/mask.sum(1)


def verify(root,args=None):
    root=Path(root);c=configuration(root);out=root/c['run_directory']
    checked=0
    for name in ('prepared.json','evaluation_manifest.json','analysis_manifest.json'):
        for p,h in json.loads((out/name).read_text())['files'].items():
            if file_digest(out/p)!=h:raise ValueError('产物hash改变：'+p)
            checked+=1
    prepared=json.loads((out/'prepared.json').read_text())
    assert prepared['spec']['config']==c
    for p,h in prepared['spec']['code'].items():assert file_digest(root/p)==h
    for p,h in prepared['spec']['inputs'].items():assert file_digest(root/p)==h
    meta=pd.read_csv(out/'videos.csv');roles=pd.read_csv(out/'roles.csv');selection=json.loads((out/'selection.json').read_text())
    assert selection['configuration']==c and not selection['test_results_used']
    for fold,q in roles.groupby('fold'):
        used=q[q.role!='excluded_shared_source']
        assert used.groupby('split_group').role.nunique().max()==1
        if fold!='pooled':assert q[q.role=='test'].dataset.eq(fold).all()
        assert not q[q.role.isin(['train','validation'])].dataset.eq(fold).any()
    local=json.loads((out/'local_features_manifest.json').read_text())
    for key,h in local['files'].items():
        assert file_digest(out/'local'/(key+'.npz'))==h
    assert len(local['files'])==len(meta)==15569
    probes=json.loads((out/'local/pilot_0.json').read_text());assert probes['global_exact'] and probes['independent_moment_error']<1e-7
    for rank in (0,1):assert json.loads((out/f'local/completed_{rank}.json').read_text())['global_exact']
    scores=pd.read_csv(out/'test_scores.csv.gz',float_precision='round_trip');pairs=pd.read_csv(root/'results/paper_complete/pairs.csv')
    grid=pd.read_csv(out/'validation_model_grid.csv');assert len(grid)==216 and len(selection['training'])==216
    mean_error=0.;prob_error=0.;count=0
    for inp in ('G','GT','GTL'):
        x,mask,ids=feature_arrays(out,inp);np.testing.assert_array_equal(ids,meta.video_id)
        for fold in c['datasets']+['pooled']:
            role=roles[roles.fold.eq(fold)].set_index('video_id').loc[ids]
            ti=np.flatnonzero(role.role.eq('train'));vi=np.flatnonzero(role.role.eq('validation'))
            training=meta.iloc[ti].reset_index(drop=True).copy();training['weight']=sampling_weights(training)
            masses=training.groupby(['dataset','length','subset']).weight.sum().unstack('subset')
            np.testing.assert_allclose(masses['real'],masses['annotated'],rtol=0,atol=1e-12)
            expected_mean,expected_scale=independent_standardizer(x[ti],mask[ti])
            for r in grid[(grid['input']==inp)&grid.fold.eq(fold)].itertuples():
                dest=out/'training'/r.key;spec=json.loads((dest/'identity.json').read_text());info=json.loads((dest/'manifest.json').read_text())
                assert spec['configuration']==c and spec['prepared']==file_digest(out/'prepared.json')
                assert spec['training_protocol']==TRAINING_PROTOCOL and info['training_protocol']==TRAINING_PROTOCOL
                assert spec['train_ids']==meta.iloc[ti].video_id.tolist() and spec['validation_ids']==meta.iloc[vi].video_id.tolist()
                assert info['identity']==config_digest(spec) and file_digest(dest/'manifest.json')==selection['training'][r.key]
                for p,h in info['files'].items():assert file_digest(dest/p)==h
                for p,h in spec['code'].items():assert file_digest(root/p)==h
                state=torch.load(dest/'model.pt',map_location='cpu',weights_only=True)
                assert state['identity']==info['identity'] and state['epoch']==r.epoch
                np.testing.assert_allclose(state['mean'].numpy(),expected_mean,rtol=1e-5,atol=1e-7)
                np.testing.assert_allclose(state['scale'].numpy(),expected_scale,rtol=1e-5,atol=1e-7)
                mean_error=max(mean_error,float(np.abs(state['mean'].numpy()-expected_mean).max()))
                # 预先按身份顺序抽首尾验证视频，不据预测挑样本。
                probe=vi[[0,-1]];p=independent_probability(state,x[probe],mask[probe])
                saved=pd.read_csv(dest/'validation_scores.csv',float_precision='round_trip').set_index('video_id')
                vp=pd.read_csv(dest/'validation_pairs.csv');observed=meta.set_index('video_id').loc[vp.video_id,'length'].to_numpy()
                np.testing.assert_array_equal(observed,vp.length)
                matched=vp.groupby(['dataset','generator','length','subset']).size().unstack('subset')
                np.testing.assert_array_equal(matched['real'],matched['annotated'])
                actual=1-saved.loc[meta.iloc[probe].video_id,'final_score'].to_numpy()
                np.testing.assert_allclose(p,actual,rtol=0,atol=2e-5);prob_error=max(prob_error,float(np.abs(p-actual).max()));count+=len(probe)
        del x,mask
    for (inp,head,seed),q in scores.groupby(['input','head','seed']):
        assert q.video_id.nunique()==15569
        actual=evaluate_fixed_pairs(q,pairs)
        for table,keys in [('generator_metrics',['dataset','generator']),('dataset_metrics',['dataset']),('macro_metrics',['scope'])]:
            a=actual[table].replace({'scope':{'Macro-3':'Average'}}).set_index(keys).sort_index()
            b=pd.read_csv(out/(table+'.csv'),float_precision='round_trip');b=b[(b.input==inp)&(b['head']==head)&(b.seed==seed)].drop(columns=['input','head','seed']).set_index(keys).sort_index()
            pd.testing.assert_frame_equal(a,b,check_dtype=False,check_exact=False,rtol=0,atol=1e-12)
    ci=pd.read_csv(out/'confidence_intervals.csv');assert len(ci)==len(contrasts())*8
    macro=pd.read_csv(out/'macro_metrics.csv');ds=pd.read_csv(out/'dataset_metrics.csv')
    for name,(a,b) in contrasts().items():
        receipt=json.loads((out/'intervals'/name/'manifest.json').read_text())
        assert receipt['inputs']['scores']==file_digest(out/'test_scores.csv.gz')
        assert receipt['files']['difference.csv']==file_digest(out/'intervals'/name/'difference.csv')
        for r in ci[ci.contrast.eq(name)].itertuples():
            table=macro if r.dataset=='Average' else ds[ds.dataset.eq(r.dataset)]
            metric='auc' if r.metric=='auc' else 'real_positive_ap'
            aa=table[(table.input==a[0])&(table['head']==a[1])][metric].mean()
            bb=table[(table.input==b[0])&(table['head']==b[1])][metric].mean()
            np.testing.assert_allclose(r.delta,aa-bb,rtol=0,atol=1e-12)
    paper_json(out/'verification.json',dict(status='verified',clips=15569,source_groups=meta.split_group.nunique(),models=216,
        seeds=3,inputs=3,heads_reported=5,independent_forward_probes=count,independent_probability_error=prob_error,
        training_mean_error=mean_error,checked_top_level_files=checked,code=file_digest(Path(__file__))))
    print('discriminative MoE verification passed',flush=True)
