"""冻结checkpoint测试；所有负结果、种子和数据域均保留。"""
import json
import time
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from artifacts import atomic_csv,paper_json
from reference import file_digest
from config import config_digest
from evaluation.tables import evaluate_fixed_pairs
from discriminative_moe.training import validation_objective
from discriminative_moe.analysis import paired_seed_contrast
from .model import LoopedDetector
from .training import loader,predict,score_frame
from .run import configuration


def evaluate(root,args):
    root=Path(root);c=configuration(root);out=root/c['run_directory'];cache=root/c['cache_directory']
    records=json.loads((cache/'manifest.json').read_text())['records']
    tasks=json.loads((out/'tasks.json').read_text())['tasks']
    if args.task:
        tasks=[t for t in tasks if t['key']==args.task]
        if len(tasks)!=1:raise ValueError('未知任务')
    else:tasks=[t for i,t in enumerate(tasks) if i%args.world_size==args.rank]
    meta=pd.read_csv(out/'videos.csv',keep_default_na=False);roles=pd.read_csv(out/'roles.csv',keep_default_na=False)
    external=pd.read_csv(out/'external_videos.csv',keep_default_na=False)
    device=f'cuda:{args.rank%2}'
    for task in tasks:
        trained=out/'training'/task['key'];dest=out/'evaluation'/task['key']
        if not (trained/'manifest.json').exists():
            print('尚未训练完成',task['key'],flush=True);continue
        receipt=json.loads((trained/'manifest.json').read_text());spec=json.loads((trained/'identity.json').read_text())
        if receipt['identity']!=config_digest(spec) or receipt['test_used'] or spec['config']!=c:
            raise ValueError('训练身份/测试隔离改变')
        if spec['prepared']!=file_digest(out/'prepared.json') or spec['cache_manifest']!=file_digest(cache/'manifest.json'):
            raise ValueError('数据身份改变')
        for p,h in spec['code'].items():
            if file_digest(root/p)!=h:raise ValueError('训练后源码改变')
        for p,h in receipt['files'].items():
            if file_digest(trained/p)!=h:raise ValueError('训练结果改变')
        history=pd.read_csv(trained/'history.csv')
        if len(history)!=c['epochs'] or int(history.loc[history.validation_objective.idxmax(),'epoch'])!=receipt['best_epoch']:
            raise ValueError('epoch选择不合法')
        obj,_=validation_objective(pd.read_csv(trained/'validation_scores.csv',float_precision='round_trip'),
                                   pd.read_csv(trained/'validation_pairs.csv'))
        np.testing.assert_allclose(obj,receipt['best_validation'],rtol=0,atol=1e-12)
        identity=dict(training=file_digest(trained/'manifest.json'),code=file_digest(Path(__file__)))
        if (dest/'manifest.json').exists():
            r=json.loads((dest/'manifest.json').read_text())
            if r['identity']!=identity or file_digest(dest/'scores.csv')!=r['scores_sha256']:raise ValueError('测试产物改变')
            continue
        if task['fold']=='pooled':fm=external
        else:
            test_ids=roles.loc[(roles.fold==task['fold'])&(roles.role=='test'),'video_id']
            fm=meta[meta.video_id.isin(test_ids)].reset_index(drop=True)
        train_groups=set(meta.loc[meta.video_id.isin(spec['train_ids']+spec['validation_ids']),'split_group'])
        if train_groups&set(fm.split_group):raise ValueError('测试源泄漏')
        state=torch.load(trained/'model.pt',map_location=device,weights_only=True)
        if state['identity']!=receipt['identity']:raise ValueError('模型身份错误')
        model=LoopedDetector(**state['arguments']).to(device);model.load_state_dict(state['model'])
        stream=loader(fm,cache,records,c)
        start=time.perf_counter();logits=predict(model,stream,device,c['amp'])
        frame=score_frame(fm,logits).assign(variant=task['variant'],seed=task['seed'],fold=task['fold'])
        # 同一冻结模型对独立重排小批再次前向，检查数据归属和存储结果。
        probe=fm.iloc[np.linspace(0,len(fm)-1,min(6,len(fm)),dtype=int)].reset_index(drop=True)
        check=predict(model,loader(probe,cache,records,c),device,c['amp'])
        expected=frame.set_index('video_id').loc[probe.video_id,'fake_logit'].to_numpy()
        np.testing.assert_allclose(check,expected,rtol=2e-3,atol=2e-3)
        dest.mkdir(parents=True,exist_ok=True);atomic_csv(dest/'scores.csv',frame)
        pairs=pd.read_csv(out/('external_pairs.csv' if task['fold']=='pooled' else 'pairs.csv'))
        pairs=pairs[pairs.dataset.isin(fm.dataset.unique())]
        for name,table in evaluate_fixed_pairs(frame,pairs).items():atomic_csv(dest/(name+'.csv'),table)
        paper_json(dest/'manifest.json',dict(status='evaluated',identity=identity,task=task,clips=len(frame),
            seconds=time.perf_counter()-start,probe_max_error=float(np.max(np.abs(check-expected))),scores_sha256=file_digest(dest/'scores.csv')))
        print('测试完成',task['key'],len(frame),flush=True)
        del model,stream;torch.cuda.empty_cache()


def report(root,args):
    root=Path(root);c=configuration(root);out=root/c['run_directory']
    tasks=json.loads((out/'tasks.json').read_text())['tasks'];scores=[];missing=[];cost=[]
    for task in tasks:
        d=out/'evaluation'/task['key']
        if not (d/'manifest.json').exists():missing.append(task['key']);continue
        r=json.loads((d/'manifest.json').read_text())
        if file_digest(d/'scores.csv')!=r['scores_sha256']:raise ValueError('分数hash改变')
        scores.append(pd.read_csv(d/'scores.csv',float_precision='round_trip'))
        tr=json.loads((out/'training'/task['key']/'manifest.json').read_text())
        cost.append(dict(**task,train_seconds=tr['seconds'],head_parameters=tr['parameters'],test_seconds=r['seconds']))
    if not scores:
        paper_json(out/'status.json',dict(status='running',models_evaluated=0,missing=missing));return
    scores=pd.concat(scores,ignore_index=True);tables={};probability=[]
    for (variant,seed,scope),q in scores.assign(scope=np.where(scores.fold.eq('pooled'),'external','development')).groupby(['variant','seed','scope']):
        pairs=pd.read_csv(out/('external_pairs.csv' if scope=='external' else 'pairs.csv'))
        pairs=pairs[pairs.dataset.isin(q.dataset.unique())]
        for name,t in evaluate_fixed_pairs(q,pairs).items():
            tables.setdefault(name,[]).append(t.assign(variant=variant,seed=seed,evaluation_scope=scope))
        for domain,g in q.groupby('dataset'):
            y=g.subset.eq('annotated').to_numpy();z=g.fake_logit.to_numpy();p=g.p_fake.to_numpy()
            probability.append(dict(variant=variant,seed=seed,dataset=domain,n=len(g),
                brier=float(np.mean((p-y)**2)),nll=float(np.mean(np.logaddexp(0,z)-y*z))))
    for name,parts in tables.items():atomic_csv(out/(name+'.csv'),pd.concat(parts,ignore_index=True))
    atomic_csv(out/'probability_metrics.csv',pd.DataFrame(probability));atomic_csv(out/'costs.csv',pd.DataFrame(cost))
    groups=[]
    for p in (out/'groups').glob('*/manifest.json'):
        g=json.loads(p.read_text())
        groups.append(dict(group=p.parent.name,models=len(g['models']),shared_wall_seconds=g['wall_seconds'],
            read_seconds=g['io']['read_seconds'],pack_seconds=g['io']['pack_seconds']))
    atomic_csv(out/'group_costs.csv',pd.DataFrame(groups))
    ds=pd.concat(tables['dataset_metrics'],ignore_index=True)
    summary=ds.groupby(['variant','dataset'])[['auc','real_positive_ap','fake_positive_ap']].agg(['mean','std'])
    summary.columns=['_'.join(x) for x in summary.columns];atomic_csv(out/'seed_summary.csv',summary.reset_index())
    metric=pd.concat(tables['macro_metrics'],ignore_index=True)
    if not metric.empty:
        macro=metric.groupby('variant')[['auc','real_positive_ap','fake_positive_ap']].agg(['mean','std','count'])
        macro.columns=['_'.join(x) for x in macro.columns];atomic_csv(out/'average_summary.csv',macro.reset_index())
    lines=['# Looped十配置：自动结果汇总','',f'已完成测试 {len(tasks)-len(missing)} / {len(tasks)} 模型。',
        '未完成域或seed不以零补齐；每个seed须三个开发域齐备后才计算该seed的Average。',
        '各模型独立选取最佳验证epoch。外部结果单列，已经观察过的数据不称未接触确认集。','',
        '| 配置 | 数据域 | AUC（seed均值） | AP-real | AP-fake | 已有seed数 |',
        '| --- | --- | ---: | ---: | ---: | ---: |']
    for (variant,domain),r in summary.iterrows():
        count=ds[(ds.variant==variant)&(ds.dataset==domain)].seed.nunique()
        lines.append(f'| {variant} | {domain} | {r.auc_mean:.3f} | {r.real_positive_ap_mean:.3f} | {r.fake_positive_ap_mean:.3f} | {count} |')
    lines+=['','逐生成器结果：generator_metrics.csv；概率诊断：probability_metrics.csv；',
        '组级真实耗时：group_costs.csv；每组配对差值：groups/<fold>__s<seed>/confidence_intervals.csv。',
        '独立测试ROC上的低FPR召回不等于可迁移部署阈值；ViF真实样本量不足以精确验证极低误报。','']
    tmp=out/'RESULTS_zh.tmp.md';tmp.write_text('\n'.join(lines));tmp.replace(out/'RESULTS_zh.md')
    if missing:
        paper_json(out/'status.json',dict(status='partial_results',models_evaluated=len(tasks)-len(missing),missing=missing))
        print('部分结果，未完成',len(missing),'模型',flush=True);return
    intervals=[]
    for scope in ('development','external'):
        q=scores[scores.fold.eq('pooled') if scope=='external' else ~scores.fold.eq('pooled')].copy()
        q['input']='patch';q['head']=q.variant
        pairs=pd.read_csv(out/('external_pairs.csv' if scope=='external' else 'pairs.csv'))
        for control in c['variants'][1:]:
            table=paired_seed_contrast(q,pairs,('patch','looped'),('patch',control),
                                      seeds=tuple(c['seeds']),iterations=c['bootstrap_replicates'])
            intervals.append(table.assign(contrast='looped_vs_'+control,evaluation_scope=scope))
    atomic_csv(out/'confidence_intervals.csv',pd.concat(intervals,ignore_index=True))
    scores.to_csv(out/'all_scores.csv.gz',index=False)
    paper_json(out/'status.json',dict(status='evaluated',models_evaluated=len(tasks),missing=[],
        files={p:file_digest(out/p) for p in ['seed_summary.csv','confidence_intervals.csv','costs.csv','all_scores.csv.gz']}))
    print(summary.to_string(),flush=True)
