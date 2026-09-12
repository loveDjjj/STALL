"""外部六行的独立验收与报告。"""
from pathlib import Path
import json,shutil,xml.etree.ElementTree as ET
import numpy as np,pandas as pd,torch
from artifacts import paper_json
from reference import file_digest
from config import config_digest
from evaluation.tables import evaluate_fixed_pairs
from evaluation.official_baseline import official_numpy_score,official_window
from statistical_experts.external_study import prepare,RUN,CACHE,METHODS,CONTRASTS
from statistical_experts.cache import load_feature
from statistical_experts.controls import check_files
from statistical_experts.engine import Engine
from statistical_experts.gaussian import transitions
from artifacts import atomic_csv


def evaluate(root):
    """独立评价入口；gt必须按列名读取，不能使用与pandas方法冲突的属性。"""
    root=Path(root);out,spec=prepare(root);check_files(out,'raw_manifest.json')
    if (out/'evaluation_manifest.json').exists():check_files(out,'evaluation_manifest.json');return
    frame=pd.read_csv(out/'windows.csv',keep_default_na=False);frame=frame[frame.role.eq('evaluation')].set_index('video_id')
    raw=pd.read_csv(out/'raw.csv',float_precision='round_trip').set_index('video_id').loc[frame.index]
    with np.load(root/'precomputed/stall_params_vatex_dino_v3.npz') as z:
        sc=np.sort(z['calib_ll_spat'].max(1));tc=np.sort(z['calib_ll_temp'].min(1))
    spatial=np.searchsorted(sc,raw['gs'],side='right')/len(sc)
    pct={'official':np.searchsorted(tc,raw['gt'],side='right')/len(tc)}
    np.testing.assert_array_equal(.5*spatial+.5*pct['official'],raw['official'].to_numpy())
    with np.load(root/'results/runs/global_experts/evaluation/cdf_arrays.npz') as z:
        for name,key in [('pooled','pooled'),('online_b','online')]:pct[name]=np.searchsorted(z[key+'_16'][:,1],raw[key],side='right')/2000
    with np.load(root/'results/runs/global_expert_controls/cdf_scores_16.npz') as z:
        pct['offline_a']=np.empty(len(raw))
        for k in range(4):
            mask=raw['cluster'].eq(k);pct['offline_a'][mask]=np.searchsorted(np.sort(z['scores'][:,k]),raw.loc[mask,'offline'],side='right')/2000
    with np.load(out/'source_cdf_raw.npz') as z:
        for name in ('source_matched','target_matched'):
            pct[name]=np.empty(len(raw))
            for d in spec['target_sources']:
                mask=frame.dataset.eq(d);ref=np.sort(z['scores'][:,z['model_names'].tolist().index(d+'__'+name)])
                pct[name][mask]=np.searchsorted(ref,raw.loc[mask,name],side='right')/len(ref)
    scores=[];tables={};pairs=pd.read_csv(out/'pairs.csv')
    for method in METHODS:
        for branch,values in [('temporal',pct[method]),('final',.5*spatial+.5*pct[method])]:
            q=frame.copy();q['final_score']=values;q['variant']=method+'_'+branch;q['method']=method;q['branch']=branch;q=q.reset_index();scores.append(q)
            for key,t in evaluate_fixed_pairs(q,pairs).items():tables.setdefault(key,[]).append(t.assign(variant=method+'_'+branch))
    pd.concat(scores,ignore_index=True).to_csv(out/'video_scores.csv.gz',index=False)
    for key,t in tables.items():atomic_csv(out/(key+'.csv'),pd.concat(t,ignore_index=True))
    paper_json(out/'evaluation_manifest.json',dict(status='point_estimates_complete',evaluator_sha256=file_digest(Path(__file__)),files={p:file_digest(out/p) for p in ['video_scores.csv.gz']+[k+'.csv' for k in tables]}))
    f=pd.read_csv(out/'dataset_metrics.csv');print(f[f.variant.str.endswith('_final')][['dataset','variant','auc','real_positive_ap']].to_string(index=False),flush=True)


def verify(root):
    root=Path(root);out,spec=prepare(root);identity=config_digest(spec)
    for p in ['data_manifest.json','raw_manifest.json','evaluation_manifest.json','analysis_manifest.json']:check_files(out,p)
    frame=pd.read_csv(out/'windows.csv',keep_default_na=False);ev=frame[frame.role.eq('evaluation')].reset_index(drop=True)
    pairs=pd.read_csv(out/'pairs.csv');scores=pd.read_csv(out/'video_scores.csv.gz',float_precision='round_trip')
    raw=pd.read_csv(out/'raw.csv',float_precision='round_trip').set_index('video_id').loc[ev.video_id]
    features=[]
    for r in frame.itertuples(index=False):features.append(load_feature(root/CACHE/(r.cache_key+'.npz'),identity,r))
    if len(features)!=2411 or len(ev)!=2216 or pairs[['dataset','generator']].drop_duplicates().shape[0]!=20:raise ValueError('外部覆盖不符')
    for d,n in spec['target_sources'].items():
        a=frame[frame.dataset.eq(d)&frame.role.eq('fit')];b=ev[ev.dataset.eq(d)]
        assert a.source_group.nunique()==len(a)==n and not set(a.source_group)&set(b.source_group)
        for role in ('fit','evaluation'):
            original=pd.read_csv(root/f'data/manifests/active/{d}/{"fit" if role=="fit" else "evaluation"}.csv',keep_default_na=False).set_index('video_id')
            for r in frame[frame.dataset.eq(d)&frame.role.eq(role)].itertuples():assert json.loads(r.frame_indices)==official_window(json.loads(original.loc[r.video_id,'downsample_idxs']),16)
    g=np.stack(features)[frame.role.eq('evaluation').to_numpy()]
    with np.load(root/'precomputed/stall_params_vatex_dino_v3.npz') as z:params={k:z[k].copy() for k in z.files}
    # 全部官方锚点重算，不从三位指标推测复现正确。
    for i in range(len(ev)):
        q=official_numpy_score(g[i][None],params)
        np.testing.assert_array_equal([q['global_spatial_raw'][0],q['global_temporal_raw'][0],q['final_score'][0]],raw.iloc[i][['gs','gt','official']].to_numpy(float))
    engine=Engine(root,16,'cuda:0');error=0.;torch.set_num_threads(4)
    # 独立重建目标/同预算源协方差，并核对各模型实际查询能量。
    target_models=torch.load(out/'target_models.pt',map_location='cpu',weights_only=True)
    source_ids=pd.read_csv(root/'results/runs/global_reference_sources/vatex_sources.csv').source_id.tolist()
    mapping={Path(p).stem:i for i,p in enumerate(engine.bank['frame'].video_path)}
    source_indices=[mapping[x] for x in source_ids]
    covariance_error=0.;source_query_error=0.
    all_features=np.stack(features)
    for d,n in spec['target_sources'].items():
        target_mask=frame.dataset.eq(d).to_numpy()&frame.role.eq('fit').to_numpy()
        tt,_=transitions(torch.from_numpy(all_features[target_mask]))
        for label,x in [('target_matched',tt.numpy()),('source_matched',engine.bank['temporal'][source_indices[:n]].cpu().numpy())]:
            a=x.reshape(-1,1024);mu=a.mean(0);cov=np.cov(a,rowvar=False,ddof=1)+1e-5*np.eye(1024)
            m=target_models[d+'__'+label]
            np.testing.assert_allclose(mu,m['mean'].numpy(),rtol=0,atol=1e-13)
            got=(m['chol']@m['chol'].T).numpy();np.testing.assert_allclose(got,cov,rtol=0,atol=1e-13)
            covariance_error=max(covariance_error,float(np.abs(got-cov).max()))
            chol=np.linalg.cholesky(cov)
            idx=np.flatnonzero(ev.dataset.eq(d).to_numpy())
            from scipy.linalg import solve_triangular
            for i in idx[np.unique(np.linspace(0,len(idx)-1,8,dtype=int))]:
                t,z=transitions(torch.from_numpy(g[i]));white=solve_triangular(chol,(t.numpy()-mu).T,lower=True)
                values=-.5*(1024*np.log(2*np.pi)+(white**2).sum(0));expected=np.where(z.numpy(),np.inf,values).min()
                actual=float(raw.iloc[i][label]);np.testing.assert_allclose(expected,actual,rtol=0,atol=1e-8)
                if np.isfinite(actual):source_query_error=max(source_query_error,float(abs(expected-actual)))
    for i in np.unique(np.linspace(0,len(ev)-1,12,dtype=int)):
        values,route,_=engine.score(torch.from_numpy(np.repeat(g[i:i+1],4,axis=0)),[ev.iloc[i].random_source_key]*4)
        assert int(route[0])==raw.iloc[i].cluster
        for name in ('pooled','offline','online'):
            expected=float(values[name][0,1]);got=float(raw.iloc[i][name]);np.testing.assert_allclose(expected,got,rtol=0,atol=1e-8)
            if np.isfinite(got):error=max(error,abs(expected-got))
    with np.load(root/'results/runs/global_experts/evaluation/cdf_arrays.npz') as z:pooled=z['pooled_16'][:,1].copy();online=z['online_16'][:,1].copy()
    with np.load(root/'results/runs/global_expert_controls/cdf_scores_16.npz') as z:offline=z['scores'].copy()
    with np.load(out/'source_cdf_raw.npz') as z:source_refs=z['scores'].copy();keys=z['model_names'].tolist();assert source_refs.shape==(2000,4)
    sc=np.sort(params['calib_ll_spat'].max(1));spatial=np.searchsorted(sc,raw.gs,side='right')/len(sc)
    for method in METHODS:
        q=scores[scores.variant.eq(method+'_temporal')].set_index('video_id').loc[ev.video_id]
        expected=[]
        for i,r in enumerate(ev.itertuples()):
            if method=='official':reference=params['calib_ll_temp'].min(1);value=raw.iloc[i]['gt']
            elif method=='pooled':reference=pooled;value=raw.iloc[i].pooled
            elif method=='online_b':reference=online;value=raw.iloc[i].online
            elif method=='offline_a':reference=offline[:,int(raw.iloc[i].cluster)];value=raw.iloc[i].offline
            else:reference=source_refs[:,keys.index(r.dataset+'__'+method)];value=raw.iloc[i][method]
            expected.append(float((reference<=value).sum()/len(reference)))
        np.testing.assert_array_equal(q.final_score.to_numpy(),expected)
        f=scores[scores.variant.eq(method+'_final')].set_index('video_id').loc[ev.video_id]
        np.testing.assert_array_equal(f.final_score.to_numpy(),.5*spatial+.5*np.asarray(expected))
    for name,q in scores.groupby('variant'):
        for key,cols in [('generator_metrics',['dataset','generator']),('dataset_metrics',['dataset']),('full_population_metrics',['dataset'])]:
            actual=evaluate_fixed_pairs(q,pairs)[key].set_index(cols).sort_index()
            stored=pd.read_csv(out/(key+'.csv'),float_precision='round_trip');stored=stored[stored.variant.eq(name)].drop(columns='variant').set_index(cols).sort_index()
            pd.testing.assert_frame_equal(actual,stored,check_dtype=False,check_exact=False,rtol=0,atol=1e-14)
    ci=pd.read_csv(out/'confidence_intervals.csv');assert len(ci)==28 and set(ci.contrast)==set(CONTRASTS)
    dm=pd.read_csv(out/'dataset_metrics.csv').set_index(['variant','dataset'])
    for name,(a,b) in CONTRASTS.items():
        info=check_files(out/'intervals'/name,'manifest.json')
        assert info['candidate']==a and info['baseline']==b and info['iterations']==1000 and info['seed']==17
        assert info['scores_sha256']==file_digest(out/'video_scores.csv.gz')
        for d in ('genvidbench','vifbench'):
            for metric,column in [('auc','auc'),('ap_real','real_positive_ap')]:
                row=ci[(ci.contrast==name)&(ci.dataset==d)&(ci.metric==metric)].iloc[0]
                np.testing.assert_allclose(row.delta,dm.loc[(a+'_final',d),column]-dm.loc[(b+'_final',d),column],rtol=0,atol=1e-14)
    paper_json(out/'verification.json',dict(status='verified',clips=2216,fit_sources=195,cache_windows=2411,cells=20,
        methods=6,contrasts=7,official_all_exact=True,expert_probes=12,expert_max_error=error,
        all_percentiles_and_fusions_exact=True,source_covariance_max_error=covariance_error,source_query_max_error=source_query_error,
        metrics_recomputed=True,auditor_sha256=file_digest(Path(__file__))))
    print('external verified',flush=True)


def report(root):
    root=Path(root);out,spec=prepare(root);v=json.loads((out/'verification.json').read_text());assert v['status']=='verified'
    suites=list(ET.parse(out/'tests.xml').getroot().iter('testsuite'))
    if not suites or any(int(s.attrib.get('failures',0))+int(s.attrib.get('errors',0)) for s in suites):raise ValueError('测试未通过')
    test_count=sum(int(s.attrib['tests']) for s in suites)
    f=pd.read_csv(out/'dataset_metrics.csv').set_index(['variant','dataset']);ci=pd.read_csv(out/'confidence_intervals.csv')
    names={'official':'官方STALL','pooled':'官方空间＋VATEX总体时序','offline_a':'官方空间＋离线专家A','online_b':'官方空间＋在线专家B',
           'source_matched':'官方空间＋同预算VATEX时序','target_matched':'官方空间＋目标时序'}
    lines=['# 冻结Global参考方法的全量外部验证','',
        '**决策：本轮未达到升级新主线的依据，停止当前纯Global专家的继续细化。保留开发集正结果和外部反例，原Alpha-STALLED论文主线不变。**','',
        '六行共享原版16帧单窗与官方空间。前四行不使用目标拟合real；后两行按GenVidBench115/ViF80独立源匹配预算。两个外部域曾被历史研究观察，不称untouched。','',
        '| 方法 | GenVidBench AUC/AP-real | ViF-Bench AUC/AP-real |','| --- | ---: | ---: |']
    for m in METHODS:
        vals=[]
        for d in ('genvidbench','vifbench'):
            r=f.loc[(m+'_final',d)];vals.append(f'{r.auc:.6f}/{r.real_positive_ap:.6f}')
        lines.append('| '+names[m]+' | '+' | '.join(vals)+' |')
    lines += ['', '## 单时序','', '| 方法 | GenVidBench AUC/AP | ViF AUC/AP |','| --- | ---: | ---: |']
    for m in METHODS:
        vals=[]
        for d in ('genvidbench','vifbench'):
            r=f.loc[(m+'_temporal',d)];vals.append(f'{r.auc:.6f}/{r.real_positive_ap:.6f}')
        lines.append('| '+names[m]+' | '+' | '.join(vals)+' |')
    lines += ['', '## 源组配对区间','', '| 对比 | 数据域 | 指标 | 差值 | 95%区间 |','| --- | --- | --- | ---: | ---: |']
    for r in ci.itertuples():lines.append(f'| {r.contrast} | {r.dataset} | {r.metric} | {r.delta:+.6f} | [{r.ci95_low:+.6f}, {r.ci95_high:+.6f}] |')
    lines += ['', '每对比1000次固定seed17源组Poisson bootstrap，ViF共享内容源共同加权。区间条件于固定参考、未校正全部研究探索；两个外部域单列，不与开发Average混成有利总分。',
        '', '## 为什么停止这一轮探索','',
        '- 离线专家相对同库总体，在两外部域的AUC/AP区间均跨零；在线同样未得到正区间。在线在GenVidBench相对官方AUC差值-0.005639，区间[-0.010852,-0.000160]。开发集的小幅专家收益没有被这次冻结外部实验确认。',
        '- 目标参考相对同预算VATEX在GenVidBench提高约0.030083/0.029582，区间为正；但相对VATEX2200或官方的区间均跨零。ViF的目标适配所有主要Final差值区间均跨零，点估计仍约0.62。',
        '- 在GenVidBench，目标时序单分支AUC0.810低于VATEX2200的0.832，Final反而更高。不能只凭融合结果说目标参考改善了所有域的原始时序取证；分数尺度及空间互补仍有作用。',
        '- 同预算源/目标比较支持参考来源影响统计；它不保证未知生成器泛化。既有源隔离不排除所有编码、拍摄来源或内容混杂。ViF是当前明确的适用边界。',
        '- 本轮已依次验证总体/离线/在线/随机、均值/协方差、A/B/C参考人口、硬/软/均匀度量、目标来源和两外部域。继续增加簇、温度、CDF或分支缺乏新的正面依据；按照用户要求停止，而不是在外部结果上回调参数。',
        '- 没有将效果不稳定的专家写成论文主创新，也没有把目标真实版本冒充零目标域版本。新主线的触发条件未满足，因此保留原论文工作稿，不替换其数据表。',
        '- 这不是证明所有Global表示或条件统计不可能有效；只是本次固定表示/参考预算/评分族的继续投入没有充分依据。下一项研究需要独立的新假设或新信息，而非将当前失败开关重新组合。',
        '', '## 协议与验收','',
        '- GenVidBench600评价片段、1生成器单元；ViF1616评价片段、19单元。原配对身份保留，未按结果筛视频。',
        '- 每源一目标拟合片段，与评价已知源组互斥；native Global新提取2411窗口约151MiB，原视频和旧缓存不变。',
        '- 离线A复用冻结4专家及每专家全VATEX参考；在线B复用冻结2200库、相似512、0.5总体协方差收缩与整体算法CDF。',
        '- 两种新来源Gaussian分别重算同一独立VATEX2000参考。目标参考使用目标信息，不称零目标域。',
        '- 全量官方锚点、全部百分位/融合及指标重新核验；新来源模型的均值/协方差用NumPy独立重建，在线/离线另作重复查询探针。ViF真实样本少，低FPR操作点不代表精确部署保证。',
        '- gt列名与Pandas方法同名导致初版表格入口异常，未产生任何指标；统一CLI现使用external_audit.evaluate的显式列读取。冻结提取/评分输入未改变，未重提或修改raw。',
        f'- 当前tests目录{test_count}项通过；未将结果快照测试重复计数。',
        '- [逐生成器](generator_metrics.csv)、[逐视频](video_scores.csv.gz)、[区间](confidence_intervals.csv)、[数据清单](windows.csv)。','']
    (out/'RESULTS_zh.md').write_text('\n'.join(lines))
    dest=out/'source_snapshot/src/statistical_experts/external_audit.py';dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(__file__,dest)
    shutil.copy2(root/'src/statistical_experts/run.py',out/'source_snapshot/src/statistical_experts/run.py')
    paper_json(out/'status.json',dict(status='completed',research_decision='stop_current_global_expert_expansion',new_mainline_adopted=False))
    paper_json(out/'manifest.json',dict(status='completed',identity=spec,tests_passed=test_count,files={str(p.relative_to(out)):file_digest(p) for p in out.rglob('*') if p.is_file() and p!=out/'manifest.json'}))
    print(out/'RESULTS_zh.md',flush=True)
