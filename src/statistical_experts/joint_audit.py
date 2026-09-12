"""联合评分独立二次型/百分位验收与报告。"""
from pathlib import Path
import json,shutil,xml.etree.ElementTree as ET
import numpy as np,pandas as pd,torch
from scipy.linalg import solve_triangular
from artifacts import paper_json
from config import config_digest
from reference import file_digest
from evaluation.tables import evaluate_fixed_pairs
from statistical_experts.joint_study import prepare,MODES,CONTRASTS,SOURCE
from statistical_experts.engine import load_bank
from statistical_experts.cache import load_feature
from statistical_experts.gaussian import transitions
from statistical_experts.controls import check_files


def verify(root):
    root=Path(root);out,spec=prepare(root);torch.set_num_threads(4)
    check_files(out,'evaluation_manifest.json');check_files(out,'analysis_manifest.json')
    scores=pd.read_csv(out/'video_scores.csv.gz',float_precision='round_trip');pairs=pd.read_csv(root/SOURCE/'pairs.csv')
    assert scores.video_id.nunique()==17785 and scores.variant.nunique()==11 and pairs[['dataset','generator']].drop_duplicates().shape[0]==43
    checked=0;error=0.
    for length in (8,16):
        assert check_files(out,f'raw_{length}.json')['identity']==config_digest(spec)
        xm=torch.load(out/f'x_model_{length}.pt',map_location='cpu',weights_only=True);models=torch.load(root/SOURCE/f'models_{length}.pt',map_location='cpu',weights_only=True)
        t=load_bank(root,length,'cpu')['temporal'][:,:-1].numpy().reshape(-1,1024)
        np.testing.assert_allclose(xm['mean'],t.mean(0),rtol=0,atol=1e-13)
        np.testing.assert_allclose((xm['chol']@xm['chol'].T).numpy(),np.cov(t,rowvar=False)+1e-5*np.eye(1024),rtol=0,atol=1e-13)
        f=pd.read_csv(root/SOURCE/f'queries_{length}.csv',keep_default_na=False)
        with np.load(out/f'raw_{length}.npz') as z:v=z['scores'];valid_count=z['valid_pairs'];np.testing.assert_array_equal(z['video_ids'],f.video_id.to_numpy())
        with np.load(root/SOURCE/f'raw_{length}.npz') as z:np.testing.assert_array_equal(valid_count,z['valid_pairs'])
        for i in np.unique(np.linspace(0,len(f)-1,12,dtype=int)):
            r=f.iloc[i];g=load_feature(root/r.cache_directory/(r.cache_key+'.npz'),r.feature_identity,r);tt,zz=transitions(torch.from_numpy(g));tt=tt.numpy();valid=~(zz[:-1]|zz[1:]).numpy()
            x=solve_triangular(xm['chol'].numpy(),(tt[:-1]-xm['mean'].numpy()).T,lower=True);ex=-.5*(1024*np.log(2*np.pi)+(x*x).sum(0))
            for j,name in enumerate(MODES.values()):
                m={k:x.numpy() for k,x in models[name].items()};res=tt[1:]-m['y_mean']-(tt[:-1]-m['x_mean'])@m['B']-m['residual_mean']
                y=solve_triangular(m['chol'],res.T,lower=True);ey=-.5*(1024*np.log(2*np.pi)+(y*y).sum(0));expected=np.where(valid,ex+ey,np.inf).min()
                np.testing.assert_allclose(expected,v[i,j],rtol=0,atol=1e-8)
                if np.isfinite(expected):error=max(error,abs(float(expected-v[i,j])))
        ref=f.role.eq('cdf').to_numpy();ev=f.role.eq('evaluation').to_numpy();ids=f.loc[ev,'video_id']
        for j,name in enumerate(MODES):
            actual=scores[scores.variant.eq(name+'_temporal')].set_index('video_id').loc[ids].final_score.to_numpy();q=v[ev,j];r=v[ref,j];assert len(r)==2000
            expected=np.empty(len(q))
            for start in range(0,len(q),128):expected[start:start+128]=(r[:,None]<=q[None,start:start+128]).sum(0)/len(r)
            np.testing.assert_array_equal(actual,expected);checked+=len(q)
    dev=pd.read_csv(root/'results/runs/global_experts/evaluation/video_scores.csv.gz',float_precision='round_trip');gs=dev[dev.variant.eq('official_spatial')].set_index('video_id').final_score
    er=pd.read_csv(root/'results/runs/global_external/raw.csv',float_precision='round_trip').set_index('video_id')
    with np.load(root/'precomputed/stall_params_vatex_dino_v3.npz') as z:sc=np.sort(z['calib_ll_spat'].max(1))
    gs=pd.concat([gs,pd.Series(np.searchsorted(sc,er['gs'],side='right')/len(sc),index=er.index)])
    for name in MODES:
        a=scores[scores.variant.eq(name+'_final')].set_index('video_id');b=scores[scores.variant.eq(name+'_temporal')].set_index('video_id').loc[a.index]
        np.testing.assert_array_equal(a.final_score.to_numpy(),.5*gs.loc[a.index].to_numpy()+.5*b.final_score.to_numpy())
        rr=scores[scores.variant.eq(name+'_raw')];_,inv,cnt=np.unique(rr.raw_score,return_inverse=True,return_counts=True);np.testing.assert_array_equal(rr.final_score,np.cumsum(cnt)[inv]/len(rr))
    source=pd.read_csv(root/SOURCE/'video_scores.csv.gz',float_precision='round_trip')
    for name,original in [('official_final','official_final'),('conditional_final','predictive_final')]:
        a=scores[scores.variant.eq(name)].set_index('video_id');b=source[source.variant.eq(original)].set_index('video_id').loc[a.index]
        np.testing.assert_array_equal(a.final_score.to_numpy(),b.final_score.to_numpy())
    for name,q in scores.groupby('variant'):
        tables=evaluate_fixed_pairs(q,pairs)
        for key,cols in [('generator_metrics',['dataset','generator']),('dataset_metrics',['dataset']),('macro_metrics',['scope'])]:
            a=tables[key].replace({'scope':{'Macro-3':'Average'}}).set_index(cols).sort_index();b=pd.read_csv(out/(key+'.csv'),float_precision='round_trip');b=b[b.variant.eq(name)].drop(columns='variant').set_index(cols).sort_index()
            pd.testing.assert_frame_equal(a,b,check_dtype=False,check_exact=False,rtol=0,atol=1e-14)
    ci=pd.read_csv(out/'confidence_intervals.csv');assert len(ci)==72 and set(ci.contrast)==set(CONTRASTS)
    dm=pd.read_csv(out/'dataset_metrics.csv').set_index(['variant','dataset']);macro=pd.read_csv(out/'macro_metrics.csv').set_index('variant')
    for name,(a,b) in CONTRASTS.items():
        info=check_files(out/'intervals'/name,'manifest.json')['inputs'];assert info['candidate']==a and info['baseline']==b and info['iterations']==1000
        for r in ci[ci.contrast.eq(name)].itertuples():
            column='auc' if r.metric=='auc' else 'real_positive_ap';delta=macro.loc[a,column]-macro.loc[b,column] if r.dataset=='Average' else dm.loc[(a,r.dataset),column]-dm.loc[(b,r.dataset),column]
            np.testing.assert_allclose(delta,r.delta,rtol=0,atol=1e-14)
    paper_json(out/'verification.json',dict(status='verified',identity=config_digest(spec),evaluation_clips=17785,cells=43,contrasts=6,models=3,
        old_conditional_exact=True,query_probes=72,query_max_error=error,percentiles_checked=checked,all_fusions_exact=True,metrics_recomputed=True,auditor_sha256=file_digest(Path(__file__))))
    print('joint verified',flush=True)


def report(root):
    root=Path(root);out,spec=prepare(root);v=json.loads((out/'verification.json').read_text());assert v['status']=='verified'
    suites=list(ET.parse(out/'tests.xml').getroot().iter('testsuite'));assert suites and not any(int(x.attrib.get('failures',0))+int(x.attrib.get('errors',0)) for x in suites)
    count=sum(int(x.attrib['tests']) for x in suites);macro=pd.read_csv(out/'macro_metrics.csv').set_index('variant');dm=pd.read_csv(out/'dataset_metrics.csv').set_index(['variant','dataset']);ci=pd.read_csv(out/'confidence_intervals.csv')
    labels={'independent':'独立边际对','joint':'真实相邻联合对','shuffled_joint':'跨源打乱联合对','conditional':'已有仅条件残差','official':'官方STALL'}
    lines=['# 联合动态评分：保留输入边际信息的机制对照','',
      '**本轮不采纳联合预测。补回x边际仅部分恢复仅条件残差的损失，未超过独立边际对/打乱控制，外部也没有稳定增强。结束这次限定机制检查，不继续扩展预测专家。**','',
      '固定官方空间、原版单窗、2200 VATEX fit和每长度2000独立CDF。复用预测器，只拟合同一x边际；三组逐对相加后再min，不搜索权重，不建立专家。','',
      '$$q_{ind}=\min_t[\ell_x(x_t)+\ell_y(y_t)],\qquad q_{joint}=\min_t[\ell_x(x_t)+\ell_{cond}(y_t|x_t)].$$','',
      '所有新组共享相同x边际、有效位置和mask，各自重建CDF；整体固定常数不影响同模型CDF排名。这是相邻T1对的Gaussian型联合能量，窗口min不称作整个视频的概率密度。raw-only保序秩仅用于兼容+inf的排序指标，不用于融合。','',
      '## 开发Average（仅三域等权）','', '| 配置 | raw AUC/AP | 时序CDF AUC/AP | Final AUC/AP |','| --- | ---: | ---: | ---: |']
    for name in MODES:
        vals=[]
        for b in ('raw','temporal','final'):
            r=macro.loc[name+'_'+b];vals.append(f'{r.auc:.6f}/{r.real_positive_ap:.6f}')
        lines.append('| '+labels[name]+' | '+' | '.join(vals)+' |')
    for name in ('conditional','official'):
        r=macro.loc[name+'_final'];lines.append(f'| {labels[name]} | — | — | {r.auc:.6f}/{r.real_positive_ap:.6f} |')
    lines += ['', '## 各域Final','', '| 配置 | ComGenVid | VideoFeedback | GenVideo | GenVidBench | ViF |','| --- | ---: | ---: | ---: | ---: | ---: |']
    for name in labels:
        vals=[]
        for d in ('comgenvid','videofeedback','genvideo','genvidbench','vifbench'):
            r=dm.loc[(name+'_final',d)];vals.append(f'{r.auc:.6f}/{r.real_positive_ap:.6f}')
        lines.append('| '+labels[name]+' | '+' | '.join(vals)+' |')
    lines += ['', '## 配对区间','', '| 对比 | 域 | 指标 | 差值 | 95%区间 |','| --- | --- | --- | ---: | ---: |']
    for r in ci.itertuples():lines.append(f'| {r.contrast} | {r.dataset} | {r.metric} | {r.delta:+.6f} | [{r.ci95_low:+.6f}, {r.ci95_high:+.6f}] |')
    lines += ['', '每组1000次seed17源组Poisson bootstrap。参考固定，未多重校正；两外部域已被观察，不称untouched。不能通过恢复到边际控制便宣称条件预测带来新增能力。','',
      '## 解释与停止决定','',
      '- 联合Final约0.821093/0.828160，虽高于已有条件残差0.819762/0.825288，但低于匹配独立边际对0.825020/0.830906及打乱联合对0.824351/0.830637。补边际不是新增时间判别能力的证明。',
      '- 联合raw-only也低于独立边际对，退化不是仅由CDF造成；GenVidBench联合Final约0.755522，独立对约0.801756，损失明显。',
      '- 独立边际对本身还弱于上一轮单边际控制0.831399/0.836461，说明逐对相加再min也改变了有用的极值排序。因此不能把所有损失归因给预测参数或某一个边际项。',
      '- 此结果不支持“只要保留p(x)，真实预测就能增强检伪”的假设。本轮结束，不反转评分、不追加新分支/权重、不扩成多个预测专家。',
      '- 当前停止结论只针对已测Global T1/Gaussian/聚合协议，不声称所有联合时序模型无效。原论文Local D2及目标统计主线不变。','',
      '## 验收与产物','',f'- {count}项tests通过；已有条件raw及有效位置逐位恢复。独立求解72个查询探针最大误差{v["query_max_error"]:.3g}，53355次直接计数CDF和全部融合/指标已核验。',
      '- 无新DINO提取，无新视频；输入边际只用原fit真实。结果不覆盖原论文工作稿。',
      '- [43单元](generator_metrics.csv)、[逐视频](video_scores.csv.gz)、[区间](confidence_intervals.csv)。','']
    (out/'RESULTS_zh.md').write_text('\n'.join(lines))
    for p in ['src/statistical_experts/joint_audit.py','src/statistical_experts/run.py','tests/test_joint.py']:
        dest=out/'source_snapshot'/p;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(root/p,dest)
    paper_json(out/'manifest.json',dict(status='completed',identity=spec,tests_passed=count,files={str(p.relative_to(out)):file_digest(p) for p in out.rglob('*') if p.is_file() and p!=out/'manifest.json'}))
    print(out/'RESULTS_zh.md',flush=True)
