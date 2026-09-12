"""新候选的独立公式核验和结果导出，不修改既有研究结果。"""
from datetime import datetime,timezone
import json
from pathlib import Path
import shutil
import subprocess
import xml.etree.ElementTree as ET
import numpy as np
import pandas as pd
import torch
from artifacts import paper_json
from config import config_digest
from reference import file_digest
from evaluation.tables import evaluate_fixed_pairs
from statistical_experts.routing_study import prepare,CONTRASTS
from statistical_experts.controls import check_files
from statistical_experts.evaluation import read_raw
from statistical_experts.cache import load_feature
from statistical_experts.gaussian import transitions
from statistical_experts.manifests import settings

LABELS={'official':'官方STALL','pooled_b':'总体时序＋全库B','expert_b':'离线时序＋整体B',
        'expert_a':'离线时序＋全库A','pooled_c':'总体时序＋同簇C','expert_c':'离线时序＋同簇C',
        'hard_b':'共享均值硬路由＋B','uniform_b':'固定平均精度＋B','soft_b':'连续内容精度＋B'}


def verify(root):
    root=Path(root);out,spec=prepare(root);c=spec['config'];base=settings(root)
    check_files(out,'evaluation_manifest.json');check_files(out,'analysis_manifest.json')
    for name,digest in spec['files'].items():
        if name.startswith(('src/','configs/')) and file_digest(out/'source_snapshot'/name)!=digest:
            raise ValueError('源码快照不符')
    scores=pd.read_csv(out/'video_scores.csv.gz',float_precision='round_trip')
    official=pd.read_csv(root/c['source_directory']/'evaluation/video_scores.csv.gz',float_precision='round_trip')
    os=official[official.variant.eq('official_spatial')].set_index('video_id')
    expected_ids=set(os.index)
    if set(scores.method)!=set(LABELS) or scores.variant.nunique()!=18:raise ValueError('方法集合不完整')
    if expected_ids!=set(scores.video_id) or len(expected_ids)!=15569:raise ValueError('视频集合不同')
    for name in LABELS:
        t=scores[scores.variant.eq(name+'_temporal')].set_index('video_id',verify_integrity=True)
        f=scores[scores.variant.eq(name+'_final')].set_index('video_id',verify_integrity=True).loc[t.index]
        if set(t.index)!=expected_ids:raise ValueError('某配置缺视频')
        np.testing.assert_array_equal(f.final_score.to_numpy(),.5*os.loc[t.index].final_score.to_numpy()+.5*t.final_score.to_numpy())
    direct_error=0.;checked=0;probe_count=0;torch.set_num_threads(4)
    for length in (8,16):
        frame,old,_=read_raw(root,length);ref=frame.role.eq('cdf').to_numpy();ev=frame.role.eq('evaluation').to_numpy()
        route=np.array([r['cluster'] for r in old]);stage=check_files(out,f'positions_{length}.json')
        if stage['identity']!=config_digest(spec):raise ValueError('位置分数身份不同')
        with np.load(out/f'positions_{length}.npz') as z:
            np.testing.assert_array_equal(z['video_ids'],frame.video_id.to_numpy())
            np.testing.assert_array_equal(z['routes'],route)
            e=z['positions'];zero=z['zero'];v=z['scores'];sim=z['similarities'];fit=z['fit_logits'];tau=float(z['temperature'])
            ordered=np.sort(fit,axis=1);expected_tau=max(float(np.median(ordered[:,-1]-ordered[:,-2])),1e-6)
            assert tau==expected_tau and len(fit)==2200
            # 独立用torch softmax与einsum验证numpy位置组合。
            w=torch.softmax(torch.from_numpy(sim)/tau,dim=1).numpy()
            raw_soft=np.einsum('btm,bm->bt',e,w)
            raw_uniform=np.einsum('btm,m->bt',e,np.full(4,.25))
            raw_hard=e[np.arange(len(e)),:,route]
            for j,raw in enumerate([raw_hard,raw_uniform,raw_soft]):
                value=np.where(zero,np.inf,raw).min(1)
                np.testing.assert_allclose(value,v[:,j],rtol=0,atol=1e-10)
            # 验证32个固定片段的实际特征与完整精度矩阵二次型，覆盖两长度。
            model=torch.load(root/c['source_directory']/f'models_{length}/models.pt',map_location='cpu',weights_only=True)
            precision=torch.cholesky_inverse(model['offline_t']['chol'])
            for i in np.unique(np.linspace(0,len(frame)-1,16,dtype=int)):
                row=frame.iloc[i]
                g=torch.from_numpy(load_feature(root/base['cache_directory']/(row.cache_key+'.npz'),spec['feature_identity'],row))
                t,zmask=transitions(g);delta=t-model['pool_t']['mean']
                direct=-.5*(1024*np.log(2*np.pi)+torch.einsum('td,mde,te->tm',delta,precision,delta).numpy())
                direct_error=max(direct_error,float(np.max(np.abs(direct-e[i]))))
                np.testing.assert_allclose(direct,e[i],rtol=0,atol=1e-8)
                np.testing.assert_array_equal(zmask.numpy(),zero[i]);probe_count+=1
        queries={
            'pooled_c':np.array([float(r['scores']['pooled'][1]) for r in old]),
            'expert_c':np.array([float(r['scores']['offline'][1]) for r in old]),
            'hard_b':v[:,0],'uniform_b':v[:,1],'soft_b':v[:,2]}
        ids=frame.loc[ev,'video_id'];r_eval=route[ev]
        for name,q in queries.items():
            actual=scores[scores.variant.eq(name+'_temporal')].set_index('video_id').loc[ids].final_score.to_numpy()
            expected=np.empty(len(ids));q_eval=q[ev]
            if name.endswith('_c'):
                for m in range(4):
                    r=q[ref & (route==m)];selected=np.flatnonzero(r_eval==m)
                    if len(r)<128:raise ValueError('C参考支持不足')
                    for begin in range(0,len(selected),128):
                        idx=selected[begin:begin+128];expected[idx]=(r[:,None]<=q_eval[None,idx]).sum(0)/len(r)
            else:
                r=q[ref]
                for begin in range(0,len(ids),128):expected[begin:begin+128]=(r[:,None]<=q_eval[None,begin:begin+128]).sum(0)/2000
            np.testing.assert_array_equal(actual,expected);checked+=len(ids)
    pairs=pd.read_csv(root/'data/manifests/global_experts/pairs.csv')
    for name,q in scores.groupby('variant'):
        tables=evaluate_fixed_pairs(q,pairs)
        for key,cols in [('generator_metrics',['dataset','generator']),('dataset_metrics',['dataset']),('macro_metrics',['scope']),('full_population_metrics',['dataset'])]:
            actual=tables[key].replace({'scope':{'Macro-3':'Average'}}).set_index(cols).sort_index()
            stored=pd.read_csv(out/(key+'.csv'),float_precision='round_trip');stored=stored[stored.variant.eq(name)].drop(columns='variant').set_index(cols).sort_index()
            pd.testing.assert_frame_equal(actual,stored,check_dtype=False,check_exact=False,rtol=0,atol=1e-14)
    ci=pd.read_csv(out/'confidence_intervals.csv');macro=pd.read_csv(out/'macro_metrics.csv').set_index('variant')
    if len(ci)!=80 or set(ci.contrast)!=set(CONTRASTS):raise ValueError('区间不完整')
    for name,(a,b) in CONTRASTS.items():
        meta=check_files(out/'intervals'/name,'manifest.json')
        if meta['inputs']['candidate']!=a or meta['inputs']['baseline']!=b or meta['inputs']['iterations']!=1000:raise ValueError('区间对比身份不同')
        for metric,col in [('auc','auc'),('ap_real','real_positive_ap')]:
            point=ci[(ci.contrast==name)&(ci.dataset=='Average')&(ci.metric==metric)].delta.iloc[0]
            np.testing.assert_allclose(point,macro.loc[a,col]-macro.loc[b,col],rtol=0,atol=1e-14)
    paper_json(out/'verification.json',dict(status='verified',identity=config_digest(spec),evaluation_clip_ids=15569,
        cells=23,methods=9,branches=2,contrasts=10,direct_percentile_checks=checked,precision_probe_clips=probe_count,
        precision_formula_max_error=direct_error,temperature_real_fit_only=True,all_fusions_exact=True,
        metrics_recomputed=True,auditor_sha256=file_digest(Path(__file__))))
    print('routing verified',checked,'percentiles; direct error',direct_error,flush=True)


def report(root):
    root=Path(root);out,spec=prepare(root);c=spec['config']
    proof=json.loads((out/'verification.json').read_text())
    if proof['status']!='verified':raise ValueError('未验收')
    tests=list(ET.parse(out/'tests.xml').getroot().iter('testsuite'))
    if not tests or any(int(t.attrib.get('failures',0))+int(t.attrib.get('errors',0)) for t in tests):raise ValueError('测试未通过')
    count=sum(int(t.attrib['tests']) for t in tests)
    macro=pd.read_csv(out/'macro_metrics.csv').set_index('variant');domains=pd.read_csv(out/'dataset_metrics.csv').set_index(['variant','dataset'])
    ci=pd.read_csv(out/'confidence_intervals.csv')
    lines=['# 同簇参考与连续内容精度：实验结果','',
      '本轮仅新增四个配置：总体/专家的同簇C校准，固定平均/连续内容精度。其余为逐条复用或恢复的控制。原论文主线不变。','',
      '**结论：同簇C未稳定超过已有A/B方案，连续内容度量未稳定超过硬路由；四个新增配置均不进入主方法。内容依赖仍有作用，但本轮不支持继续扫CDF、温度或专家数。**','',
      '## 1. 固定协议','',
      '官方空间固定；时序沿原版单窗T1、min与等权融合。2200 VATEX fit、独立2000 VATEX CDF，8/16帧分别处理；15569评价片段身份、23生成器单元。Average为三个域等权。',
      '', 'C校准保留原Gaussian/路由，仅使用同路由簇参考；总体Gaussian＋C控制用于分离人口作用。连续精度使用共享总体均值与既有四套时序精度矩阵，温度只由fit real相似度差中位数确定。三种度量分别重建B式整体算法CDF，不与C叠加。',
      '', '## 2. Final结果','',
      '| 配置（均含官方空间） | AUC | AP-real | AP-fake | Recall@1% FPR |',
      '| --- | ---: | ---: | ---: | ---: |']
    for name,label in LABELS.items():
        r=macro.loc[name+'_final'];lines.append(f'| {label} | {r.auc:.6f} | {r.real_positive_ap:.6f} | {r.fake_positive_ap:.6f} | {100*r.fake_tpr_at_1pct_real_fpr:.2f}% |')
    lines += ['', '## 3. 三域结果','', '| 配置 | ComGenVid AUC/AP | GenVideo AUC/AP | VideoFeedback AUC/AP |','| --- | ---: | ---: | ---: |']
    for name,label in LABELS.items():
        values=[]
        for domain in ('comgenvid','genvideo','videofeedback'):
            r=domains.loc[(name+'_final',domain)];values.append(f'{r.auc:.6f}/{r.real_positive_ap:.6f}')
        lines.append('| '+label+' | '+' | '.join(values)+' |')
    lines += ['', '## 4. 单时序结果','', '| 配置 | AUC | AP-real |','| --- | ---: | ---: |']
    for name,label in LABELS.items():
        r=macro.loc[name+'_temporal'];lines.append(f'| {label} | {r.auc:.6f} | {r.real_positive_ap:.6f} |')
    lines += ['', '## 5. 固定参考下的配对区间','', '| 对比 | ΔAUC [95% CI] | ΔAP-real [95% CI] |','| --- | ---: | ---: |']
    for name in CONTRASTS:
        vals=[]
        for metric in ('auc','ap_real'):
            r=ci[(ci.dataset=='Average')&(ci.contrast==name)&(ci.metric==metric)].iloc[0]
            vals.append(f'{r.delta:+.6f} [{r.ci95_low:+.6f}, {r.ci95_high:+.6f}]')
        lines.append('| '+name+' | '+' | '.join(vals)+' |')
    lines += ['', '每个对比1000次源组配对Poisson bootstrap，固定seed17。区间未多重比较校正，条件于固定参考，不覆盖温度规则/模型开发选择或真实库重抽样。Recall为评价ROC操作点，不是部署阈值迁移保证。',
      '', '### 结果解释','',
      '- 专家C相对原B的AUC差值约+0.000231、区间跨零；AP-real约-0.001354、区间为负。相对A，两项差值区间都跨零。因此“同内容参考”没有自然变成更好的检测。',
      '- 总体Gaussian直接换同簇CDF，AUC/AP均下降且区间为负；专家C对这个较弱控制的增益，不能被当成超过已有专家基线的新增能力。末端分组不能替代合适的时序统计。',
      '- 连续度量相对硬路由的AUC/AP差值区间都跨零。它相对固定平均有正区间，支持内容依赖的价值，但没有证明连续化优于简单硬选择。',
      '- 固定平均的AUC接近总体模型，AP-real略低；简单集成不能代替内容选择。三个度量使用同一总体均值、相同4个协方差、相同B-CDF规则，未混入额外数据或不同采样。',
      '- 软路由确实运行了：评价fake平均有效专家数约3.73–3.79，real约3.17–3.43，均非hard。当前真实拟合集确定的温度在评价fake上接近均匀混合，可能削弱部分专门度量的作用；这是机制线索，不证明调低温度一定更好，也不将路由熵增加为真假分支。',
      '- 同簇参考更少，且会消除部分跨簇分数差异。这些都是可能的退化因素，当前实验没有把样本数量和人群效应进一步拆开；不声称已证明唯一原因。',
      '- 按原方案停止本轮CDF/路由扩展，保留先前官方空间＋在线B/离线A的研究候选。下一项若启动，应是冻结后的外部确认或明确的参考覆盖对照，而非继续在这批开发数据上寻找更好的温度。',
      '', '## 6. 路由及成本','']
    for length in (8,16):
        m=json.loads((out/f'positions_{length}.json').read_text())
        lines.append(f'- {length}帧：温度{m["temperature"]:.8f}，仅2200 fit描述决定；{m["rows"]}条参考/评价查询重评分{m["elapsed_seconds"]:.2f}s。未重提DINO。')
    rf=pd.read_csv(out/'routing_summary.csv')
    lines += ['', '| 长度 | 角色/数据域/类别 | 平均有效专家数 exp(H) | 平均最大权重 |','| --- | --- | ---: | ---: |']
    for r in rf.itertuples():lines.append(f'| {r.length} | {r.role}/{r.dataset}/{r.subset} | {r.mean_effective_experts:.4f} | {r.mean_max_weight:.4f} |')
    lines += ['', '有效专家数范围1–4；该描述用于判断软路由的实际行为，不将real/fake权重差异当作新判别特征。CDF簇大小与分母见cdf_support.csv，不保证小簇精确低FPR。',
      '', '## 7. 实现与验收','',
      '- 按精度矩阵组合，先混合逐转移能量再做时间min；不是平均各专家窗口min，也不是混合Gaussian密度。',
      '- 温度趋零（唯一最大时）/无穷分别恢复hard/uniform数学边界；零T1在混合后统一mask，全静态继续+inf。',
      '- 所有查询的hard raw及百分位逐位恢复已有协方差-only控制。独立直接计数复核全部新增映射，18种输出全部复核融合和指标。',
      f'- 完整测试{count}项通过；32个实际缓存片段（含参考与评价）用显式精度矩阵独立验算，最大能量误差{proof["precision_formula_max_error"]:.3g}。',
      '', '## 8. 文件入口','',
      '- [23单元](generator_metrics.csv)、[三域](dataset_metrics.csv)、[Average](macro_metrics.csv)、[逐视频](video_scores.csv.gz)。',
      '- [配对区间](confidence_intervals.csv)、[参考支持](cdf_support.csv)、[路由诊断](routing_summary.csv)。',
      '- positions_8/16.npz为轻量逐转移能量、零值标志、相似度和fit温度依据；未保存新Patch或每查询完整矩阵。',
      '- identity.json与source_snapshot绑定本轮输入/配置/源码，verification.json与tests.xml为验收证据。','']
    (out/'RESULTS_zh.md').write_text('\n'.join(lines))
    for p in ['src/statistical_experts/routing_audit.py','src/statistical_experts/run.py','tests/test_expert_routing.py','pytest.ini']:
        target=out/'source_snapshot'/p;target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(root/p,target)
    paper_json(out/'status.json',dict(status='completed',updated_utc=datetime.now(timezone.utc).isoformat(),methods=9,new_methods=4,cells=23,contrasts=10))
    paper_json(out/'manifest.json',dict(status='completed',identity=spec,tests_passed=count,
        git_head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=root,text=True).strip(),
        files={str(p.relative_to(out)):file_digest(p) for p in out.rglob('*') if p.is_file() and p!=out/'manifest.json'}))
    print(out/'RESULTS_zh.md',flush=True)
