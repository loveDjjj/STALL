"""参考来源实验独立验收：源预算、特征、拟合、留出和全部指标。"""
from datetime import datetime,timezone
import json
from pathlib import Path
import shutil
import xml.etree.ElementTree as ET
import numpy as np
import pandas as pd
import torch
from artifacts import paper_json
from config import config_digest
from reference import file_digest
from evaluation.tables import evaluate_fixed_pairs
from evaluation.official_baseline import official_window
from statistical_experts.source_study import prepare,banks,CONTRASTS
from statistical_experts.source_data import select_sources
from statistical_experts.cache import load_feature
from statistical_experts.controls import check_files
from statistical_experts.evaluation import read_raw
from statistical_experts.gaussian import transitions
from statistical_experts.manifests import settings

LABELS={'official':'官方STALL','vatex2200':'VATEX2200总体时序','source_matched':'同预算VATEX时序','target_matched':'同预算目标时序'}


def verify(root):
    root=Path(root);out,data,spec=prepare(root);c=data['config'];torch.set_num_threads(4)
    check_files(out,'evaluation_manifest.json');check_files(out,'analysis_manifest.json')
    target=pd.read_csv(out/'target_windows.csv',keep_default_na=False)
    old_frame=pd.read_csv(root/'data/manifests/global_experts/windows.csv',keep_default_na=False)
    evaluation=old_frame[old_frame.role.eq('evaluation')];old_ref=old_frame[old_frame.role.ne('evaluation')]
    for domain,n in c['source_counts'].items():
        original=pd.read_csv(root/f'data/manifests/active/{domain}/fit.csv',keep_default_na=False)
        expected=select_sources(original,c['seed']);actual=target[target.dataset.eq(domain)]
        assert set(expected.video_id)==set(actual.legacy_video_id) and expected.source_group.nunique()==n
        assert not set(actual.source_group)&set(evaluation.source_group)
        for r in actual.itertuples(index=False):
            row=original[original.video_id.eq(r.legacy_video_id)].iloc[0]
            assert json.loads(r.frame_indices)==official_window(json.loads(row.downsample_idxs),r.length)
            load_feature(root/c['cache_directory']/(r.cache_key+'.npz'),spec['target_feature_identity'],r)
        # 对各域已知真实媒体身份作第二层核对，不只比较目录或带域前缀ID。
        def media(f):
            if domain=='comgenvid':return set(f.video_path.map(lambda p:Path(p).stem[:11]))
            if domain=='videofeedback':return set(f.video_path.map(lambda p:(Path(p).parent.name,Path(p).stem)))
            return set(f.video_path.map(lambda p:Path(p).stem))
        assert not media(actual)&media(evaluation[evaluation.dataset.eq(domain)&evaluation.subset.eq('real')])
        if domain=='comgenvid':assert not media(actual)&set(old_ref.video_path.map(lambda p:Path(p).stem))
    scores=pd.read_csv(out/'video_scores.csv.gz',float_precision='round_trip')
    official=pd.read_csv(root/c['source_directory']/'evaluation/video_scores.csv.gz',float_precision='round_trip')
    os=official[official.variant.eq('official_spatial')].set_index('video_id')
    if scores.variant.nunique()!=12 or scores.video_id.nunique()!=15569:raise ValueError('配置或覆盖不完整')
    nll_checked=0;model_error=0.;probe_error=0.;probe_n=0;percentiles=0
    for length in (8,16):
        for name in (f'models_{length}.json',f'raw_{length}.json'):
            if check_files(out,name)['identity']!=config_digest(spec):raise ValueError('阶段身份错误')
        frame,old,_=read_raw(root,length);ref=frame.role.eq('cdf').to_numpy()
        tf,t,sf,s=banks(root,length,data,spec)
        models=torch.load(out/f'models_{length}.pt',map_location='cpu',weights_only=True)
        folds=json.loads((out/f'folds_{length}.json').read_text())['folds']
        nll=pd.read_csv(out/f'nll_{length}.csv',float_precision='round_trip')
        with np.load(out/f'raw_{length}.npz') as z:
            np.testing.assert_array_equal(z['video_ids'],frame.video_id.to_numpy());v=z['scores'];keys=z['model_names'].tolist()
        np.testing.assert_array_equal(v[:,keys.index('vatex2200')],[float(r['scores']['pooled'][1]) for r in old])
        def independent_model(x):
            flat=x.numpy().reshape(-1,1024);mean=flat.mean(0);cov=np.cov(flat,rowvar=False,ddof=1)+1e-5*np.eye(1024)
            chol=np.linalg.cholesky(cov)
            return mean,cov,chol
        for domain,n in c['source_counts'].items():
            if length==8 and domain=='comgenvid':continue
            idx=np.flatnonzero(tf.dataset.eq(domain));f=tf.iloc[idx].reset_index(drop=True);x=t[idx]
            for label,features in [('source_matched',s[:n]),('target_matched',x)]:
                mean,cov,chol=independent_model(features);model=models[domain+'__'+label]
                actual=(model['chol']@model['chol'].T).numpy()
                np.testing.assert_allclose(mean,model['mean'].numpy(),rtol=0,atol=1e-13)
                np.testing.assert_allclose(cov,actual,rtol=0,atol=1e-13)
                model_error=max(model_error,float(np.max(np.abs(cov-actual))))
                # CPU独立求解，核验实际查询和各模型，不用指标代替raw正确性。
                indices=np.flatnonzero(ref|frame.dataset.eq(domain).to_numpy())
                for i in indices[np.unique(np.linspace(0,len(indices)-1,4,dtype=int))]:
                    r=frame.iloc[i];g=load_feature(root/settings(root)['cache_directory']/(r.cache_key+'.npz'),data['old_feature_identity'],r)
                    tt,zero=transitions(torch.from_numpy(g));d=tt.numpy()-mean
                    from scipy.linalg import solve_triangular
                    white=solve_triangular(chol,d.T,lower=True);energy=-.5*(1024*np.log(2*np.pi)+(white**2).sum(0))
                    raw=np.where(zero.numpy(),np.inf,energy).min();got=v[i,keys.index(domain+'__'+label)]
                    np.testing.assert_allclose(raw,got,rtol=0,atol=1e-8)
                    if np.isfinite(raw):probe_error=max(probe_error,abs(float(raw-got)))
                    probe_n+=1
            held=[]
            for fold in range(5):
                record=next(r for r in folds if r['dataset']==domain and r['fold']==fold)
                train=np.flatnonzero(f.fold.ne(fold));val=np.flatnonzero(f.fold.eq(fold))
                assert record['target_train']==f.iloc[train].video_id.tolist()
                assert record['target_validation']==f.iloc[val].video_id.tolist()
                assert record['source_train']==sf.iloc[:len(train)].video_id.tolist()
                assert not set(record['target_train'])&set(record['target_validation'])
                held.extend(record['target_validation'])
                # 使用独立torch分布接口检查所有留出NLL；协方差用NumPy独立估计。
                for label,train_features in [('source',s[:len(train)]),('target',x[train])]:
                    mu,cov,chol=independent_model(train_features)
                    distribution=torch.distributions.MultivariateNormal(torch.from_numpy(mu),scale_tril=torch.from_numpy(chol))
                    values=-distribution.log_prob(x[val]).mean(1).numpy()
                    rows=nll[(nll.dataset==domain)&(nll.fold==fold)].set_index('video_id').loc[f.iloc[val].video_id]
                    np.testing.assert_allclose(values,rows[label+'_nll'].to_numpy(),rtol=0,atol=1e-8)
                    nll_checked+=len(val)
            assert len(held)==len(set(held))==n
            mask=frame.role.eq('evaluation').to_numpy()&frame.dataset.eq(domain).to_numpy();ids=frame.loc[mask,'video_id']
            for label in ('source_matched','target_matched','vatex2200'):
                key=label if label=='vatex2200' else domain+'__'+label;raw=v[mask,keys.index(key)];refs=v[ref,keys.index(key)]
                actual=scores[scores.variant.eq(label+'_temporal')].set_index('video_id').loc[ids,'final_score'].to_numpy()
                expected=np.empty(len(raw))
                for start in range(0,len(raw),128):expected[start:start+128]=(refs[:,None]<=raw[None,start:start+128]).sum(0)/2000
                np.testing.assert_array_equal(actual,expected);percentiles+=len(raw)
                stored=scores[scores.variant.eq(label+'_raw')].set_index('video_id').loc[ids,'raw_temporal_score'].to_numpy()
                np.testing.assert_array_equal(stored,raw)
    for label in LABELS:
        t=scores[scores.variant.eq(label+'_temporal')].set_index('video_id');f=scores[scores.variant.eq(label+'_final')].set_index('video_id').loc[t.index]
        np.testing.assert_array_equal(f.final_score.to_numpy(),.5*os.loc[t.index].final_score.to_numpy()+.5*t.final_score.to_numpy())
        raw=scores[scores.variant.eq(label+'_raw')];u,inv,counts=np.unique(raw.raw_temporal_score,return_inverse=True,return_counts=True)
        np.testing.assert_array_equal(raw.final_score.to_numpy(),np.cumsum(counts)[inv]/len(raw))
    # 原生官方及2200总体的两分支/Final均保持已有逐视频锚点。
    for a,b in [('official_final','official_final'),('official_temporal','official_temporal'),('vatex2200_temporal','pooled_temporal')]:
        x=scores[scores.variant.eq(a)].set_index('video_id');y=official[official.variant.eq(b)].set_index('video_id').loc[x.index]
        np.testing.assert_array_equal(x.final_score.to_numpy(),y.final_score.to_numpy())
    pairs=pd.read_csv(root/'data/manifests/global_experts/pairs.csv')
    for name,q in scores.groupby('variant'):
        for table,keys in [('generator_metrics',['dataset','generator']),('dataset_metrics',['dataset']),('macro_metrics',['scope'])]:
            actual=evaluate_fixed_pairs(q,pairs)[table].replace({'scope':{'Macro-3':'Average'}}).set_index(keys).sort_index()
            saved=pd.read_csv(out/(table+'.csv'),float_precision='round_trip');saved=saved[saved.variant.eq(name)].drop(columns='variant').set_index(keys).sort_index()
            pd.testing.assert_frame_equal(actual,saved,check_dtype=False,check_exact=False,rtol=0,atol=1e-14)
    ci=pd.read_csv(out/'confidence_intervals.csv');macro=pd.read_csv(out/'macro_metrics.csv').set_index('variant')
    if len(ci)!=64 or set(ci.contrast)!=set(CONTRASTS):raise ValueError('区间缺失')
    for name,(a,b) in CONTRASTS.items():
        meta=check_files(out/'intervals'/name,'manifest.json')
        assert meta['inputs']['candidate']==a and meta['inputs']['baseline']==b and meta['inputs']['iterations']==1000
        for metric,column in [('auc','auc'),('ap_real','real_positive_ap')]:
            point=ci[(ci.contrast==name)&(ci.metric==metric)&(ci.dataset=='Average')].delta.iloc[0]
            np.testing.assert_allclose(point,macro.loc[a,column]-macro.loc[b,column],rtol=0,atol=1e-14)
    full_nll=pd.concat([pd.read_csv(out/f'nll_{l}.csv',float_precision='round_trip') for l in (8,16)],ignore_index=True)
    if full_nll.duplicated(['video_id']).any() or len(full_nll)!=932:raise ValueError('留出预测遗漏/重复')
    np.testing.assert_allclose(full_nll.target_nll-full_nll.source_nll,full_nll.delta,rtol=0,atol=1e-10)
    paper_json(out/'verification.json',dict(status='verified',identity=config_digest(spec),target_sources=532,target_windows=932,
        target_cache_all_checked=True,source_budgets=[132,200,200],known_source_isolation=True,folds=5,nll_values_verified=nll_checked,
        covariance_max_error=model_error,query_probes=probe_n,query_max_error=probe_error,percentiles_verified=percentiles,
        evaluation_clip_ids=15569,cells=23,contrasts=8,variants=12,metrics_recomputed=True,
        auditor_sha256=file_digest(Path(__file__))))
    print('source study verified',nll_checked,'NLL',percentiles,'CDF',flush=True)


def report(root):
    root=Path(root);out,data,spec=prepare(root);c=data['config'];v=json.loads((out/'verification.json').read_text())
    if v['status']!='verified':raise ValueError('未验收')
    suites=list(ET.parse(out/'tests.xml').getroot().iter('testsuite'))
    if not suites or any(int(s.attrib.get('failures',0))+int(s.attrib.get('errors',0)) for s in suites):raise ValueError('测试未通过')
    count=sum(int(s.attrib['tests']) for s in suites)
    macro=pd.read_csv(out/'macro_metrics.csv').set_index('variant');ds=pd.read_csv(out/'dataset_metrics.csv').set_index(['variant','dataset']);ci=pd.read_csv(out/'confidence_intervals.csv')
    lines=['# 原版单窗T1：真实参考来源的受控诊断','',
        '完成匹配独立源预算的VATEX/目标参考比较与源级五折真实NLL诊断。官方空间固定，无专家、Local、FC或新融合权重。原论文主线未改变。','',
        '**主要发现：目标参考改善真实留出NLL，同时改善原始T1排序及最终融合。当前参考来源确实构成限制；这不是证明Global T1已到上限，也不是新的零目标域部署结果。**','',
        '## 1. 数据和协议','',
        '- ComGenVid132、VideoFeedback200、GenVideo200个独立源，每源按固定身份取一个片段；原ComGenVid200片段中同源重复不重复计权。',
        '- 源参考使用已有VATEX source-fit200中固定seed17排序，132源为200源的嵌套前缀；不依据评价性能选择。',
        '- 每个Gaussian分别评分独立VATEX2000参考，8/16帧分别建CDF；目标Gaussian使用目标信息，不称zero-target方法或严格性能上界。',
        '- 532目标片段共932个原版窗口、11712帧，新Global约45.75MiB；已有评价和VATEX缓存直接复用。',
        '- 1024维、同ridge1e-5、T1 fp32差分归一化后fp64拟合评分；拟合保留零T1，检测min屏蔽零T1，全静态+inf。',
        '- 源级固定5折：每折目标训练源与同数量VATEX源，给同一目标留出real计算每视频平均NLL，包含logdet，零T1参与；只诊断、不调参。',
        '', '## 2. 全量检测','', '| 配置 | raw T1 AUC/AP | 校准T1 AUC/AP | Final AUC/AP | Recall@1% FPR |','| --- | ---: | ---: | ---: | ---: |']
    for name,label in LABELS.items():
        values=[]
        for branch in ('raw','temporal','final'):
            r=macro.loc[name+'_'+branch];values.append(f'{r.auc:.6f}/{r.real_positive_ap:.6f}')
        lines.append('| '+label+' | '+' | '.join(values)+f' | {100*macro.loc[name+"_final"].fake_tpr_at_1pct_real_fpr:.2f}% |')
    lines += ['', 'Average为域内生成器等权后、三域等权。AUC/AP-real高为真实；AP-fake及其他ROC指标见CSV。raw-only为兼容+inf以保序保同分有限秩计算排序指标，raw_temporal_score保留原值；该秩不用于融合。',
        '', '## 3. 各域Final','', '| 配置 | ComGenVid AUC/AP | VideoFeedback AUC/AP | GenVideo AUC/AP |','| --- | ---: | ---: | ---: |']
    for name,label in LABELS.items():
        vals=[]
        for d in ('comgenvid','videofeedback','genvideo'):
            r=ds.loc[(name+'_final',d)];vals.append(f'{r.auc:.6f}/{r.real_positive_ap:.6f}')
        lines.append('| '+label+' | '+' | '.join(vals)+' |')
    lines += ['', '## 4. 真实留出统计（NLL越低越好）','', '| 数据域 | 帧数 | 留出预测数 | VATEX NLL | 目标 NLL | 改善视频比例 |','| --- | ---: | ---: | ---: | ---: | ---: |']
    nf=pd.read_csv(out/'nll_summary.csv')
    for r in nf.itertuples():lines.append(f'| {r.dataset} | {r.length} | {r.n} | {r.source_nll:.6f} | {r.target_nll:.6f} | {100*r.fraction_improved:.2f}% |')
    lines += ['', '五折模型训练源互相重叠；上表是固定分折的描述性诊断，不把932条预测当成932个独立源，也不对重叠折声称独立重复实验保证。NLL是归一化方向上的Gaussian近似，不是真实性概率。',
        '', '## 5. 配对区间','', '| 对比 | ΔAUC [95% CI] | ΔAP-real [95% CI] |','| --- | ---: | ---: |']
    for name in CONTRASTS:
        vals=[]
        for metric in ('auc','ap_real'):
            r=ci[(ci.contrast==name)&(ci.dataset=='Average')&(ci.metric==metric)].iloc[0]
            vals.append(f'{r.delta:+.6f} [{r.ci95_low:+.6f}, {r.ci95_high:+.6f}]')
        lines.append('| '+name+' | '+' | '.join(vals)+' |')
    lines += ['', '8组各1000次源组配对Poisson bootstrap，固定seed17；区间条件于本轮拟合/CDF，未多重校正，不涵盖参考池重抽样。Recall为评价ROC操作点，没有新部署阈值保证。',
        '', '## 6. 原因与后续判断','',
        '- 同预算目标参考Final约0.846156/0.850704，高于同预算VATEX0.827195/0.832416，也高于VATEX2200的0.832329/0.837316。因此不能只解释为小VATEX模型不稳定；更多同源VATEX并未达到本轮目标参考水平。',
        '- 原始T1从同预算源参考0.797806/0.788593升至目标0.817249/0.811712；校准后的T1几乎相同。变化发生在原始统计测量层，不只是最终CDF重新排百分位。',
        '- 三域真实留出NLL均改善；16帧改善比例约80.3%/90.5%/86.0%（ComGenVid/VideoFeedback/GenVideo），说明更匹配目标真实统计与检测收益同时出现。NLL绝对值不可直接跨8/16长度和训练预算比较。',
        '- 目标参考相对同预算源参考的Final三个域点估计都提高；相对VATEX2200，GenVideo收益更明显，ComGenVid raw-T1变化较小、Final改善较多，存在与固定空间分支的尺度/互补作用。不能把全部Final收益称为独立时序判别力提升。',
        '- 低误报表现仍依赖域：目标Final的Recall@1%在VideoFeedback低于VATEX2200，不能写所有指标所有域都改善。独立部署阈值尚未验证。',
        '- 本轮只交换单Gaussian的来源，未再次交换均值/协方差，因此不能将本轮来源收益严格全部归因于目标协方差。先前专家四格是相关机制证据，不替代本轮直接归因。',
        '- 后续应围绕真实参考覆盖/目标适配及冻结外部确认，不再继续在同一VATEX池上扫专家、CDF或温度。若要恢复零目标域主张，需另作不含目标源的多真实域参考对照；不能把本轮目标参考结果改名为通用模型。',
        '- 单一源池/固定源选择仍可能受拍摄来源、编码条件、内容组成影响。源级隔离与真实留出不排除所有来源捷径；不承诺该收益能无损迁移到新域。',
        '', '## 7. 成本','']
    for rank in (0,1):
        marker=out/f'cache_rank_{rank}.json'
        if marker.exists():
            timing=json.loads(marker.read_text());lines.append(f'- GPU{rank}目标Global核心提取{timing["elapsed_seconds"]:.2f}s，{timing["new_windows"]}窗口；不含模型初始化。')
    for length in (8,16):
        fit_time=json.loads((out/f'models_{length}.json').read_text())['elapsed_seconds'];score_time=json.loads((out/f'raw_{length}.json').read_text())['elapsed_seconds']
        lines.append(f'- {length}帧：正式拟合＋五折NLL阶段{fit_time:.2f}s；全部模型参考/评价重评分{score_time:.2f}s。')
    lines += ['', '## 8. 验收和产物','',
        f'- {count}项tests通过；全部932个目标缓存、源选择/窗口/已知源级隔离、全部五折身份已验证。',
        f'- 独立NumPy协方差最大差{v["covariance_max_error"]:.3g}；独立Gaussian分布接口重算{v["nll_values_verified"]}条NLL；{v["percentiles_verified"]}次直接计数CDF验证。',
        f'- {v["query_probes"]}条模型/查询探针独立求解最大raw差{v["query_max_error"]:.3g}；VATEX2200全部raw逐位恢复首轮锚点。',
        '- 新目标缓存与原Uniform特征分开；没有修改原视频、旧参数、旧科学结果或main分支。',
        '- [23单元](generator_metrics.csv)、[三域](dataset_metrics.csv)、[Average](macro_metrics.csv)、[逐视频](video_scores.csv.gz)、[区间](confidence_intervals.csv)。',
        '- [源级选择](target_sources.csv)、[原版窗口](target_windows.csv)、[VATEX固定来源](vatex_sources.csv)、[真实NLL](nll_all.csv)。',
        '- data_identity/stat_identity与source_snapshot绑定输入/配置/源码；verification.json为独立验收。','']
    (out/'RESULTS_zh.md').write_text('\n'.join(lines))
    for p in ['src/statistical_experts/source_audit.py','src/statistical_experts/run.py','tests/test_reference_source_study.py','pytest.ini']:
        target=out/'source_snapshot'/p;target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(root/p,target)
    paper_json(out/'status.json',dict(status='completed',updated_utc=datetime.now(timezone.utc).isoformat(),target_sources=532,cells=23,contrasts=8))
    paper_json(out/'manifest.json',dict(status='completed',data_identity=data,stat_identity=spec,tests_passed=count,
        files={str(p.relative_to(out)):file_digest(p) for p in out.rglob('*') if p.is_file() and p!=out/'manifest.json'}))
    print(out/'RESULTS_zh.md',flush=True)
