"""主线专家回归、独立矩阵评分与四格验收。"""
from pathlib import Path
import json
import shutil
import xml.etree.ElementTree as ET
import numpy as np,pandas as pd,torch
from artifacts import paper_json
from reference import file_digest
from mainline_experts.run import configuration
from mainline_experts.evaluation import collect,calibrations,video_values,CONTRASTS
from mainline_experts.models import load_assets,load_models,load_anchors
from evaluation.tables import evaluate_fixed_pairs
from features import AlphaStallFeatureExtractor
from data.video import decode_bounded
from math_utils import l2_normalized_first_order,l2_normalized_second_order
from artifacts import atomic_csv
from config import config_digest


def covariance(x,w):
    x=np.asarray(x,dtype=np.float64);w=np.asarray(w,dtype=np.float64)
    mu=(x*w[:,None]).sum(0)/w.sum();centered=x-mu
    return mu,centered.T@(centered*w[:,None])/(w.sum()-(w*w).sum()/w.sum())


def model_audit(root):
    root=Path(root);c=configuration(root);out=root/c['run_directory'];maximum=0.;supports=[]
    for d in c['datasets']:
        assets=load_assets(root,d);centers,models=load_models(out/f'fit_{d}/models.npz');anchors=load_anchors(root,d)
        for b in ('gt','lt'):
            xs=[];ws=[];ls=[]
            for a in assets:
                if not len(a[b]):continue
                route=(a['context']@centers.T).argmax(1);xs.append(a[b]);ws.append(np.full(len(a[b]),1/len(a[b])));ls.append(route[a[b+'_window']])
            x=np.concatenate(xs);w=np.concatenate(ws);labels=np.concatenate(ls);mu,base=covariance(x,w)
            np.testing.assert_allclose(mu,anchors[b].mean,rtol=0,atol=1e-12)
            np.testing.assert_array_equal(models[b][0].whitening,anchors[b].whitening)
            for m in range(2):
                mask=labels==m;_,part=covariance(x[mask],w[mask]);expected=.5*base+.5*part+1e-5*np.eye(1024)
                p=models[b][m+1].whitening@models[b][m+1].whitening.T;check=expected@p
                error=float(np.abs(check-np.eye(1024)).max());maximum=max(maximum,error)
                np.testing.assert_allclose(check,np.eye(1024),rtol=0,atol=1e-8)
                np.testing.assert_array_equal(models[b][m+1].mean,anchors[b].mean)
            split=json.loads((out/f'fit_{d}/holdout_sources.json').read_text())
            groups={a['video_id']:a['source_group'] for a in assets}
            assert not {groups[x] for x in split['train']}&{groups[x] for x in split['validation']}
        supports.append(pd.read_csv(out/f'fit_{d}/support.csv').assign(dataset=d))
    atomic_csv(out/'all_fit_support.csv',pd.concat(supports,ignore_index=True))
    paper_json(out/'model_audit.json',dict(status='verified',covariance_precision_identity_max_error=maximum,means_fixed=True,source_holdout_disjoint=True))
    print('mainline models independently verified',maximum,flush=True)


def formula_audit(root):
    root=Path(root);out=root/configuration(root)['run_directory'];jobs=json.loads((out/'plans.json').read_text())['jobs']
    records=[]
    for job in jobs:
        targets={}
        for d,meta in job['targets'].items():
            windows=[]
            for b in meta['expected']:windows.append(dict(**b,route=0,gt_experts=[b['gt'],b['gt']],lt_experts=[b['lt'],b['lt']]))
            u=meta['uniform']
            if u is not None:u=dict(**u,route=0,gt_experts=[u['gt'],u['gt']])
            targets[d]=dict(windows=windows,uniform=u)
        records.append((job,dict(targets=targets)))
    refs=calibrations(records);baseline=pd.read_csv(root/'results/paper_complete/scores_fc3.csv.gz',float_precision='round_trip')
    baseline=baseline[baseline.variant.eq('full')].set_index('video_id');maximum=0.;n=0
    for job,p in records:
        if job['role']!='evaluation':continue
        for d,m in job['targets'].items():
            v=video_values(m,p['targets'][d],refs[d,len(job['windows'][0])]);expected=float(baseline.loc[m['video_id'],'final_score'])
            for key in ('R0','R1','R2','R3'):
                maximum=max(maximum,abs(v[key]-expected));np.testing.assert_allclose(v[key],expected,rtol=0,atol=1e-10)
            n+=1
    paper_json(out/'formula_audit.json',dict(status='verified',clips=n,maximum_error=maximum,
        meaning='使用全部冻结raw，验证M1/完全回退的CDF及融合；不替代真实前向回归'))
    print('mainline formula verified',n,maximum,flush=True)


def verify(root,args=None):
    if args is not None and args.pilot:return formula_audit(root)
    root=Path(root);out,spec,records=collect(root);c=configuration(root)
    if spec['config']!=c or spec['plans']!=file_digest(out/'plans.json'):
        raise ValueError('密集评分的配置或任务清单改变')
    for path,digest in spec['code'].items():
        if file_digest(root/path)!=digest:raise ValueError('密集评分源码改变：'+path)
    for domain,digest in spec['models'].items():
        if file_digest(out/f'fit_{domain}/models.npz')!=digest:raise ValueError('评分模型改变：'+domain)
    for manifest in ['data_manifest.json','evaluation_manifest.json','analysis_manifest.json']:
        for path,digest in json.loads((out/manifest).read_text())['files'].items():
            if file_digest(out/path)!=digest:raise ValueError('结果文件改变：'+path)
    model_audit(root)
    if json.loads((out/'fit_sample_audit.json').read_text())['status']!='verified':raise ValueError('拟合样本位置未经验证')
    scores=pd.read_csv(out/'video_scores.csv.gz',float_precision='round_trip');pairs=pd.read_csv(out/'pairs.csv')
    baseline=pd.read_csv(root/'results/paper_complete/scores_fc3.csv.gz',float_precision='round_trip');baseline=baseline[baseline.variant.eq('full')].set_index('video_id')
    piv=scores.pivot(index='video_id',columns='variant',values='final_score')
    if len(piv)!=15569 or len(piv.columns)!=12:raise ValueError('最终覆盖不完整')
    np.testing.assert_allclose(piv.R0,baseline.loc[piv.index,'final_score'],rtol=0,atol=1e-10)
    np.testing.assert_allclose(piv.R3-piv.R0,piv.R1-piv.R0+piv.R2-piv.R0,rtol=0,atol=1e-12)
    regression=pd.read_csv(out/'R0_regression.csv',float_precision='round_trip');np.testing.assert_allclose(regression.R0_recomputed,regression.expected,rtol=0,atol=1e-10)
    centers={d:load_models(out/f'fit_{d}/models.npz')[0] for d in c['datasets']};model_map={d:load_models(out/f'fit_{d}/models.npz')[1] for d in c['datasets']}
    route_checks=0
    for job,p in records:
        path=out/'global'/(job['key']+'.npz')
        if file_digest(path)!=p['global_sha256']:raise ValueError('Global缓存内容变化')
        with np.load(path) as z:
            if str(z['identity'])!=config_digest(spec):raise ValueError('Global缓存身份不同')
            for d,a in p['targets'].items():
                for key,expected in [('global_features',[w['route'] for w in a['windows']]),('uniform_global',[] if a['uniform'] is None else [a['uniform']['route']])]:
                    if not expected:continue
                    g=z[key].astype(np.float64);norm=np.linalg.norm(g,axis=-1,keepdims=True);unit=g/np.where(norm==0,1,norm)
                    v=unit.mean(1);v/=np.maximum(np.linalg.norm(v,axis=-1,keepdims=True),1e-15)
                    np.testing.assert_array_equal((v@centers[d].T).argmax(1),expected);route_checks+=len(expected)
    # 独立重建专家CDF：GT仅Uniform参考，Local严格使用旧effective-K索引。
    independent_refs={}
    for job,p in records:
        if job['role']!='cdf':continue
        length=len(job['windows'][0])
        for d,actual in p['targets'].items():
            bucket=independent_refs.setdefault((d,length),{'gt':[],'local':{}})
            u=actual['uniform'];bucket['gt'].append(float(u['gt_experts'][u['route']]))
            q=np.array([float(w['lt_experts'][w['route']]) for w in actual['windows']])
            for k in (1,2,3):
                if len(q)<k:continue
                idx=[0] if k==1 else ([0,len(q)-1] if k==2 else [0,1,2])
                bucket['local'].setdefault(k,[]).append(float(q[idx].mean()))
    with np.load(out/'cdf_arrays.npz') as archive:
        for (d,length),b in independent_refs.items():
            assert len(b['gt'])==2000
            np.testing.assert_array_equal(np.sort(b['gt']),archive[f'{d}_{length}_gt1'])
            for k,v in b['local'].items():np.testing.assert_array_equal(np.sort(v),archive[f'{d}_{length}_local1_K{k}'])
    percentile_checks=0
    for job,p in records:
        if job['role']!='evaluation':continue
        length=len(job['windows'][0]);k=len(job['windows'])
        for d,m in job['targets'].items():
            b=independent_refs[d,length];actual=p['targets'][d];ref=np.asarray(b['gt'])
            fractions=[np.mean(ref<=float(w['gt_experts'][w['route']])) for w in actual['windows']]
            gt=float(np.mean(fractions));lt_raw=float(pd.Series([float(w['lt_experts'][w['route']]) for w in actual['windows']]).groupby(np.zeros(k,dtype=np.int8)).mean().iloc[0])
            lt=float(np.mean(np.asarray(b['local'][k])<=lt_raw))
            np.testing.assert_allclose(gt,piv.loc[m['video_id'],'GT1'],rtol=0,atol=1e-14)
            np.testing.assert_array_equal(lt,piv.loc[m['video_id'],'local1']);percentile_checks+=2
    # 原图抽查每种长度/角色/目标组合，独立按逐位置白化计算Local专家均值。
    cfg=__import__('yaml').safe_load((root/'configs/paper.yaml').read_text())
    extractor=AlphaStallFeatureExtractor('cuda:0',dino_repo=str(root/cfg['encoder']['repo']),dino_weights=str(root/cfg['encoder']['weights']),pad_tail_batch=True)
    seen=set();probes=0;maximum=0.
    for job,p in records:
        key=(job['role'],len(job['windows'][0]),tuple(job['targets']))
        if key in seen:continue
        seen.add(key);union=sorted({x for w in job['windows'] for x in w});where={x:i for i,x in enumerate(union)}
        frames=extractor.prepare_frames(decode_bounded(root/job['video_path'],union));f=extractor.frames_to_global_patch_embeddings([frames],batch_size=8)[0]
        pick=[[where[x] for x in w] for w in job['windows']];g=np.stack([f['global'][idx] for idx in pick]);patch=torch.from_numpy(np.stack([f['patch'][idx] for idx in pick]))
        u=l2_normalized_second_order(patch).numpy();gt,zero=l2_normalized_first_order(torch.from_numpy(g));gt=gt.numpy();zero=zero.numpy()
        for d,a in p['targets'].items():
            for i,w in enumerate(a['windows']):
                for m in range(2):
                    model=model_map[d]['lt'][m+1];x=u[i].reshape(-1,1024).astype(np.float64);z=(x-model.mean)@model.whitening
                    actual=float((-.5*(1024*np.log(2*np.pi)+(z*z).sum(1))).mean());expected=float(w['lt_experts'][m])
                    maximum=max(maximum,abs(actual-expected));np.testing.assert_allclose(actual,expected,rtol=0,atol=1e-8)
                    model=model_map[d]['gt'][m+1];z=(gt[i].astype(np.float64)-model.mean)@model.whitening;v=-.5*(1024*np.log(2*np.pi)+(z*z).sum(1));actual=np.where(zero[i],np.inf,v).min()
                    np.testing.assert_allclose(actual,float(w['gt_experts'][m]),rtol=0,atol=1e-8);probes+=1
    for name,q in scores.groupby('variant'):
        tables=evaluate_fixed_pairs(q,pairs)
        for table,keys in [('generator_metrics',['dataset','generator']),('dataset_metrics',['dataset']),('macro_metrics',['scope'])]:
            actual=tables[table].replace({'scope':{'Macro-3':'Average'}}).set_index(keys).sort_index();saved=pd.read_csv(out/(table+'.csv'),float_precision='round_trip');saved=saved[saved.variant.eq(name)].drop(columns='variant').set_index(keys).sort_index()
            pd.testing.assert_frame_equal(actual,saved,check_dtype=False,check_exact=False,rtol=0,atol=1e-14)
    ci=pd.read_csv(out/'confidence_intervals.csv');assert len(ci)==64 and set(ci.contrast)==set(CONTRASTS)
    macro=pd.read_csv(out/'macro_metrics.csv').set_index('variant');domains=pd.read_csv(out/'dataset_metrics.csv').set_index(['variant','dataset'])
    for name,(a,b) in CONTRASTS.items():
        info=json.loads((out/'intervals'/name/'manifest.json').read_text())['inputs'];assert info['candidate']==a and info['baseline']==b and info['iterations']==1000
        assert info['scores']==file_digest(out/'video_scores.csv.gz')
        assert info['pairs']==file_digest(out/'pairs.csv') and info['seed']==17
        for path,digest in json.loads((out/'intervals'/name/'manifest.json').read_text())['files'].items():
            assert file_digest(out/'intervals'/name/path)==digest
        for r in ci[ci.contrast.eq(name)].itertuples():
            column='auc' if r.metric=='auc' else 'real_positive_ap';expected=macro.loc[a,column]-macro.loc[b,column] if r.dataset=='Average' else domains.loc[(a,r.dataset),column]-domains.loc[(b,r.dataset),column]
            np.testing.assert_allclose(r.delta,expected,rtol=0,atol=1e-14)
    paper_json(out/'verification.json',dict(status='verified',clips=15569,cells=23,variants=12,contrasts=8,
        real_forward_R0_max_error=float(regression.raw_error.max()),R0_final_recomputed_max_error=float(np.abs(regression.R0_recomputed-regression.expected).max()),
        route_checks=route_checks,expert_percentile_checks=percentile_checks,original_video_expert_probes=probes,local_direct_max_error=maximum,source_code=file_digest(Path(__file__))))
    print('mainline full verification passed',flush=True)


def report(root,args=None):
    from mainline_experts.reporting import report as function
    return function(root,args)
