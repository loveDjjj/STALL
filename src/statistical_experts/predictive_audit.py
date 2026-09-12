"""条件T1独立拟合公式、同有效位置、CDF与外部结果验收。"""
from pathlib import Path
import json,shutil,xml.etree.ElementTree as ET
import numpy as np,pandas as pd,torch
from scipy.linalg import solve,solve_triangular
from artifacts import paper_json
from reference import file_digest
from config import config_digest
from evaluation.tables import evaluate_fixed_pairs
from statistical_experts.predictive import prepare,MODELS,CONTRASTS
from statistical_experts.cache import load_feature
from statistical_experts.engine import load_bank
from statistical_experts.gaussian import transitions
from statistical_experts.controls import check_files


def independent_fit(t,mode,permutation):
    x=t[:,:-1].reshape(-1,t.shape[-1]);y=t[permutation,1:].reshape(-1,t.shape[-1]);mx=x.mean(0);my=y.mean(0)
    xc=x-mx;yc=y-my;d=x.shape[1]
    b=np.zeros((d,d)) if mode=='marginal' else solve(xc.T@xc/(len(x)-1)+1e-5*np.eye(d),xc.T@yc/(len(x)-1),assume_a='pos')
    residual=yc-xc@b;mu=residual.mean(0);cov=np.cov(residual,rowvar=False,ddof=1)+1e-5*np.eye(d)
    return dict(x_mean=mx,y_mean=my,B=b,residual_mean=mu,chol=np.linalg.cholesky(cov))


def independent_positions(t,m):
    r=t[1:]-m['y_mean']-(t[:-1]-m['x_mean'])@m['B']-m['residual_mean']
    z=solve_triangular(m['chol'],r.T,lower=True)
    return -.5*(t.shape[-1]*np.log(2*np.pi)+(z*z).sum(0))


def verify(root):
    root=Path(root);out,spec=prepare(root);c=spec['config'];torch.set_num_threads(4)
    check_files(out,'evaluation_manifest.json');check_files(out,'analysis_manifest.json')
    scores=pd.read_csv(out/'video_scores.csv.gz',float_precision='round_trip');pairs=pd.read_csv(out/'pairs.csv')
    if scores.video_id.nunique()!=17785 or scores.variant.nunique()!=10 or pairs[['dataset','generator']].drop_duplicates().shape[0]!=43:raise ValueError('预测评价覆盖不完整')
    max_b_error=0.;max_q_error=0.;checked=0;held_checked=0
    for length in (8,16):
        for name in (f'models_{length}.json',f'raw_{length}.json'):
            if check_files(out,name)['identity']!=config_digest(spec):raise ValueError('预测阶段身份变化')
        bank=load_bank(root,length,'cpu');t=bank['temporal'].numpy();model=torch.load(out/f'models_{length}.pt',map_location='cpu',weights_only=True)
        ids=json.loads((out/f'fit_sources_{length}.json').read_text());assert ids['fit_ids']==bank['ids']
        assert set(ids['heldout_train']).isdisjoint(ids['heldout_validation']) and len(ids['heldout_validation'])==440
        f=pd.read_csv(out/f'queries_{length}.csv',keep_default_na=False)
        with np.load(out/f'raw_{length}.npz') as z:
            np.testing.assert_array_equal(z['video_ids'],f.video_id.to_numpy());values=z['scores'];valid=z['valid_pairs']
        for j,name in enumerate(MODELS):
            permutation=model[name]['pairing'].numpy();assert np.array_equal(np.sort(permutation),np.arange(2200))
            if name=='shuffled':assert (permutation!=np.arange(2200)).all()
            else:assert np.array_equal(permutation,np.arange(2200))
            m=independent_fit(t,name,permutation);stored=model[name]
            error=float(np.abs(m['B']-stored['B'].numpy()).max());max_b_error=max(max_b_error,error)
            np.testing.assert_allclose(m['B'],stored['B'].numpy(),rtol=0,atol=1e-10)
            np.testing.assert_allclose(m['chol']@m['chol'].T,(stored['chol']@stored['chol'].T).numpy(),rtol=0,atol=1e-12)
            for i in np.unique(np.linspace(0,len(f)-1,10,dtype=int)):
                r=f.iloc[i];g=load_feature(root/r.cache_directory/(r.cache_key+'.npz'),r.feature_identity,r)
                tt,zero=transitions(torch.from_numpy(g));mask=~(zero[:-1]|zero[1:]).numpy();assert mask.sum()==valid[i]
                q=np.where(mask,independent_positions(tt.numpy(),m),np.inf).min();actual=values[i,j]
                np.testing.assert_allclose(q,actual,rtol=0,atol=1e-8)
                if np.isfinite(q):max_q_error=max(max_q_error,abs(float(q-actual)))
            ref=f.role.eq('cdf').to_numpy();ev=f.role.eq('evaluation').to_numpy();query=values[ev,j];reference=values[ref,j]
            if ref.sum()!=2000:raise ValueError('参考集数量不同')
            actual=scores[scores.variant.eq(name+'_temporal')].set_index('video_id').loc[f.loc[ev,'video_id'],'final_score'].to_numpy();expected=np.empty(len(query))
            for start in range(0,len(query),128):expected[start:start+128]=(reference[:,None]<=query[None,start:start+128]).sum(0)/len(reference)
            np.testing.assert_array_equal(actual,expected);checked+=len(actual)
        # 每个留出模型仅用1760训练源；独立重算所有440条真实NLL。
        mapping={v:i for i,v in enumerate(bank['ids'])};tr=np.array([mapping[x] for x in ids['heldout_train']]);va=np.array([mapping[x] for x in ids['heldout_validation']])
        held=pd.read_csv(out/f'heldout_{length}.csv',float_precision='round_trip')
        rng=np.random.default_rng(17);order=rng.permutation(len(tr));shuffle=np.empty(len(tr),dtype=int);shuffle[order]=np.roll(order,1)
        for name in MODELS:
            m=independent_fit(t[tr],name,shuffle if name=='shuffled' else np.arange(len(tr)))
            nll=np.array([-independent_positions(t[i],m).mean()+np.log(np.diag(m['chol'])).sum() for i in va])
            actual=held[held.model.eq(name)].set_index('video_id').loc[ids['heldout_validation'],'nll'].to_numpy()
            np.testing.assert_allclose(nll,actual,rtol=0,atol=1e-8);held_checked+=len(nll)
    dev=pd.read_csv(root/c['source_directory']/'evaluation/video_scores.csv.gz',float_precision='round_trip')
    external=pd.read_csv(root/c['external_directory']/'video_scores.csv.gz',float_precision='round_trip')
    gs=dev[dev.variant.eq('official_spatial')].set_index('video_id').final_score
    er=pd.read_csv(root/c['external_directory']/'raw.csv',float_precision='round_trip').set_index('video_id')
    with np.load(root/'precomputed/stall_params_vatex_dino_v3.npz') as z:sc=np.sort(z['calib_ll_spat'].max(1))
    gs=pd.concat([gs,pd.Series(np.searchsorted(sc,er['gs'],side='right')/len(sc),index=er.index)])
    for name in MODELS:
        q=scores[scores.variant.eq(name+'_final')].set_index('video_id');tt=scores[scores.variant.eq(name+'_temporal')].set_index('video_id').loc[q.index]
        np.testing.assert_array_equal(q.final_score.to_numpy(),.5*gs.loc[q.index].to_numpy()+.5*tt.final_score.to_numpy())
        rr=scores[scores.variant.eq(name+'_raw')];_,inv,count=np.unique(rr.raw_score,return_inverse=True,return_counts=True)
        np.testing.assert_array_equal(rr.final_score.to_numpy(),np.cumsum(count)[inv]/len(rr))
    official=pd.concat([dev[dev.variant.eq('official_final')],external[external.variant.eq('official_final')]]).set_index('video_id')
    oo=scores[scores.variant.eq('official_final')].set_index('video_id');np.testing.assert_array_equal(oo.final_score.to_numpy(),official.loc[oo.index].final_score.to_numpy())
    for name,q in scores.groupby('variant'):
        tables=evaluate_fixed_pairs(q,pairs)
        for key,cols in [('generator_metrics',['dataset','generator']),('dataset_metrics',['dataset']),('macro_metrics',['scope'])]:
            actual=tables[key].replace({'scope':{'Macro-3':'Average'}}).set_index(cols).sort_index()
            saved=pd.read_csv(out/(key+'.csv'),float_precision='round_trip');saved=saved[saved.variant.eq(name)].drop(columns='variant').set_index(cols).sort_index()
            pd.testing.assert_frame_equal(actual,saved,check_dtype=False,check_exact=False,rtol=0,atol=1e-14)
    ci=pd.read_csv(out/'confidence_intervals.csv');assert len(ci)==72 and set(ci.contrast)==set(CONTRASTS)
    dm=pd.read_csv(out/'dataset_metrics.csv').set_index(['variant','dataset']);macro=pd.read_csv(out/'macro_metrics.csv').set_index('variant')
    for name,(a,b) in CONTRASTS.items():
        info=check_files(out/'intervals'/name,'manifest.json')['inputs'];assert info['candidate']==a and info['baseline']==b and info['iterations']==1000
        for r in ci[ci.contrast.eq(name)].itertuples():
            col='auc' if r.metric=='auc' else 'real_positive_ap'
            delta=macro.loc[a,col]-macro.loc[b,col] if r.dataset=='Average' else dm.loc[(a,r.dataset),col]-dm.loc[(b,r.dataset),col]
            np.testing.assert_allclose(delta,r.delta,rtol=0,atol=1e-14)
    paper_json(out/'verification.json',dict(status='verified',identity=config_digest(spec),evaluation_clip_ids=17785,cells=43,models=3,
        contrasts=6,independent_B_max_error=max_b_error,independent_query_max_error=max_q_error,heldout_nll_checked=held_checked,
        percentiles_checked=checked,all_fusions_exact=True,matched_valid_positions=True,auditor_sha256=file_digest(Path(__file__))))
    print('predictive verified',flush=True)


def report(root):
    root=Path(root);out,spec=prepare(root);v=json.loads((out/'verification.json').read_text());assert v['status']=='verified'
    suites=list(ET.parse(out/'tests.xml').getroot().iter('testsuite'));assert suites and not any(int(s.attrib.get('failures',0))+int(s.attrib.get('errors',0)) for s in suites)
    tests=sum(int(s.attrib['tests']) for s in suites);macro=pd.read_csv(out/'macro_metrics.csv').set_index('variant');dm=pd.read_csv(out/'dataset_metrics.csv').set_index(['variant','dataset']);ci=pd.read_csv(out/'confidence_intervals.csv')
    labels={'marginal':'匹配位置的边际T1','predictive':'真实相邻T1条件预测','shuffled':'跨源打乱配对预测','official':'官方STALL（完整T1观察）'}
    lines=['# 相邻T1条件动态：边际与打乱控制实验','',
        '**本轮不采纳条件预测，也不扩展成预测专家。真实时间关系改善了部分真实留出预测，却降低了当前检测排序；保留完整负结果，不改变主线或反转评分方向。**','',
        '固定官方空间、原版单窗、三个开发域23单元与两个外部域20单元。新评分器没有专家、PCA、Local或权重搜索。','',
        '## 1. 算法和数据','',
        '$$x=u_t,\quad y=u_{t+1},\quad B=(C_{xx}+10^{-5}I)^{-1}C_{xy},\quad r=y-\mu_y-(x-\mu_x)B.$$',
        '', '行向量约定；对拟合残差估计Gaussian，再对有效转移对取最小能量。边际模型B=0；打乱模型按固定seed17跨视频无自配对置换y，保留完整x/y边际与样本预算。三个模型使用完全相同的被预测位置与零差分mask。',
        '', '- 2200 VATEX真实源拟合，每长度2000独立CDF；8/16帧分别建立模型。零T1保留在拟合和NLL，检测时只评分前后T1均非零的对，全无有效对为+inf。',
        '- 16帧最多14个被预测位置，8帧最多6个；边际控制也使用同样位置。官方完整T1只是外部锚点，不是唯一因果控制。',
        '- 真实留出1760/440，重新拟合三个模型。NLL含logdet、按视频平均；只诊断，不选择正则。正式模型仍用全部2200源。',
        '- 每个模型独立重算CDF；raw-only保序秩仅用于兼容+inf的排序指标，不用于融合。','',
        '## 2. 开发Average（三域等权）','', '| 模型 | raw T1 AUC/AP | 校准T1 AUC/AP | Final AUC/AP |','| --- | ---: | ---: | ---: |']
    for m in MODELS:
        cells=[]
        for branch in ('raw','temporal','final'):
            r=macro.loc[m+'_'+branch];cells.append(f'{r.auc:.6f}/{r.real_positive_ap:.6f}')
        lines.append('| '+labels[m]+' | '+' | '.join(cells)+' |')
    r=macro.loc['official_final'];lines.append(f'| 官方STALL完整观察 | — | — | {r.auc:.6f}/{r.real_positive_ap:.6f} |')
    lines += ['', '## 3. 各域Final','', '| 模型 | ComGenVid | VideoFeedback | GenVideo | GenVidBench | ViF |','| --- | ---: | ---: | ---: | ---: | ---: |']
    for m in (*MODELS,'official'):
        cells=[]
        for d in ('comgenvid','videofeedback','genvideo','genvidbench','vifbench'):
            r=dm.loc[(m+'_final',d)];cells.append(f'{r.auc:.6f}/{r.real_positive_ap:.6f}')
        lines.append('| '+labels[m]+' | '+' | '.join(cells)+' |')
    lines += ['', '## 4. 独立真实留出NLL（低为好）','', '| 长度 | 模型 | 平均NLL |','| --- | --- | ---: |']
    for r in pd.read_csv(out/'heldout_summary.csv').itertuples():lines.append(f'| {r.length} | {labels[r.model]} | {r.nll:.6f} |')
    lines += ['', '## 5. 配对差值与区间','', '| 对比 | 域 | 指标 | 差值 | 95%区间 |','| --- | --- | --- | ---: | ---: |']
    for r in ci.itertuples():lines.append(f'| {r.contrast} | {r.dataset} | {r.metric} | {r.delta:+.6f} | [{r.ci95_low:+.6f}, {r.ci95_high:+.6f}] |')
    lines += ['', '每组1000次源组Poisson bootstrap；共享真实/ViF内容源同步加权。区间条件于固定真实库和当前设计，未校正历次模型选择；外部数据曾被观察，不称untouched。外部不并入开发Average。',
        '', '## 结果解释与停止决定','',
        '- 三个模型严格使用相同被预测位置、相同有效对mask、同一真实拟合/CDF身份；预测模型的下降不能解释为边际观察了更多极值位置。',
        '- 真实相邻模型在16帧留出NLL优于边际和打乱控制，8帧则只优于打乱控制。因此时间关系确实被学习，但其真实预测能力存在长度与样本支持差异。',
        '- 开发raw-only从边际约0.803324降至条件模型0.780365，损失已经发生在CDF之前；Final同样下降。不是最后校准才使模型失效。',
        '- 打乱模型的检测更接近边际，条件模型明显更弱，说明在本轮模型与评分定义下，真实时间依赖没有形成额外真假分离，反而改变了有用排序。不能据此宣称所有生成视频都更可预测，也不能推出任意条件动态模型必然无效。',
        '- GenVidBench的Final下降较大，ViF未形成稳定增强；没有依据把一个总体失败的模型继续分成多个预测专家。',
        '- 不根据fake指标反转残差方向、搜索PCA维数/正则或新增低误差异常分支。若未来研究双侧典型性，应另立假设和独立确认，不将当前负结果当作事后调分依据。',
        '- 当前Global专家研究继续保持收档；原论文Local/目标统计主线未改。这次实现属于用户重新提出的新假设，实验完成不等于方法有效。',
        '', '## 6. 验收','',f'- {tests}项tests通过；独立NumPy/SciPy重算回归及残差协方差、60个查询探针、{v["heldout_nll_checked"]}个真实留出NLL、{v["percentiles_checked"]}个CDF百分位。',
        f'- 回归系数最大误差{v["independent_B_max_error"]:.3g}，查询分数最大误差{v["independent_query_max_error"]:.3g}；全部融合和指标验收通过。',
        '- 不修改原论文主线；新结果单独保存。新模型是真实数据回归拟合，不称完全没有参数学习。',
        '- [43单元明细](generator_metrics.csv)、[逐视频](video_scores.csv.gz)、[配对区间](confidence_intervals.csv)、[真实留出](heldout_summary.csv)。','']
    (out/'RESULTS_zh.md').write_text('\n'.join(lines))
    for p in ['src/statistical_experts/predictive_audit.py','src/statistical_experts/run.py','tests/test_predictive.py']:
        target=out/'source_snapshot'/p;target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(root/p,target)
    paper_json(out/'status.json',dict(status='completed',models=3,cells=43))
    paper_json(out/'manifest.json',dict(status='completed',identity=spec,tests_passed=tests,files={str(p.relative_to(out)):file_digest(p) for p in out.rglob('*') if p.is_file() and p!=out/'manifest.json'}))
    print(out/'RESULTS_zh.md',flush=True)
