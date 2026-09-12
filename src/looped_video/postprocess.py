"""每组自动验收、逐生成器指标和配对区间；不据测试选择后续任务。"""
import json
from pathlib import Path
import numpy as np
import pandas as pd
from artifacts import atomic_csv,paper_json
from reference import file_digest
from config import config_digest
from discriminative_moe.analysis import paired_seed_contrast
from discriminative_moe.training import validation_objective
from evaluation.tables import evaluate_fixed_pairs
from .model import LoopedDetector
from .training import model_arguments


def process_group(root,out,c,tasks,meta,pairs):
    root=Path(root);out=Path(out);fold=tasks[0]['fold'];seed=tasks[0]['seed']
    dest=out/'groups'/f'{fold}__s{seed}';rows=[];frames=[];tables={};checks={}
    expected=set(meta.video_id)
    for task in tasks:
        train=out/'training'/task['key'];ev=out/'evaluation'/task['key']
        receipt=json.loads((train/'manifest.json').read_text());spec=json.loads((train/'identity.json').read_text())
        if receipt['identity']!=config_digest(spec) or receipt['test_used'] or spec['config']!=c:
            raise ValueError('训练身份或数据角色错误')
        if spec['prepared']!=file_digest(out/'prepared.json'):raise ValueError('准备身份改变')
        for p,h in spec['code'].items():
            if file_digest(root/p)!=h:raise ValueError('执行源码改变：'+p)
        for p,h in receipt['files'].items():
            if file_digest(train/p)!=h:raise ValueError('训练产物改变')
        er=json.loads((ev/'manifest.json').read_text())
        if er['identity']['training']!=file_digest(train/'manifest.json'):raise ValueError('测试模型身份改变')
        for p,h in er['files'].items():
            if file_digest(ev/p)!=h:raise ValueError('测试产物改变')
        f=pd.read_csv(ev/'scores.csv',float_precision='round_trip',dtype={'video_id':str,'source_group':str,'split_group':str})
        if set(f.video_id)!=expected or f.video_id.duplicated().any():raise ValueError('测试覆盖错误')
        if not f.variant.eq(task['variant']).all() or not f.seed.eq(seed).all() or not f.fold.eq(fold).all():
            raise ValueError('测试任务字段错误')
        history=pd.read_csv(train/'history.csv')
        if history.epoch.tolist()!=list(range(1,c['epochs']+1)):raise ValueError('未完成训练预算')
        best=history.loc[history.validation_objective.idxmax()]
        if int(best.epoch)!=receipt['best_epoch']:raise ValueError('选模不是首个最佳验证epoch')
        val=pd.read_csv(train/'validation_scores.csv',float_precision='round_trip',dtype={'video_id':str})
        vp=pd.read_csv(train/'validation_pairs.csv',dtype={'video_id':str})
        objective,_=validation_objective(val,vp)
        np.testing.assert_allclose(objective,receipt['best_validation'],rtol=0,atol=1e-12)
        args=model_arguments(c,task['variant']);model=LoopedDetector(**args)
        n=sum(p.numel() for p in model.parameters());del model
        if n!=receipt['parameters']:raise ValueError('模型结构与报告参数量不符')
        rows.append(dict(variant=task['variant'],fold=fold,seed=seed,loops=args['loops'],width=args['width'],heads=args['heads'],
            head_parameters=n,best_epoch=receipt['best_epoch'],best_validation=objective))
        frames.append(f)
        for name,t in evaluate_fixed_pairs(f,pairs).items():tables.setdefault(name,[]).append(t.assign(variant=task['variant'],seed=seed))
        checks[task['key']]=dict(training=file_digest(train/'manifest.json'),evaluation=file_digest(ev/'manifest.json'))
    atomic_csv(dest/'model_summary.csv',pd.DataFrame(rows))
    for name,values in tables.items():atomic_csv(dest/(name+'.csv'),pd.concat(values,ignore_index=True))
    scores=pd.concat(frames,ignore_index=True).assign(input='patch');scores['head']=scores.variant
    intervals=[]
    for control in c['variants'][1:]:
        intervals.append(paired_seed_contrast(scores,pairs,('patch','looped'),('patch',control),seeds=(seed,),
            iterations=c['bootstrap_replicates']).assign(contrast='looped_vs_'+control,seed=seed))
    atomic_csv(dest/'confidence_intervals.csv',pd.concat(intervals,ignore_index=True))
    paper_json(dest/'postprocess.json',dict(status='passed',models=checks,
        uncertainty='源级配对bootstrap，条件于当前单个seed；不代表跨seed或模型搜索不确定性',
        tests_used_for_selection=False,files={p.name:file_digest(p) for p in dest.glob('*metrics.csv')}|
            {p:file_digest(dest/p) for p in ['model_summary.csv','confidence_intervals.csv']}))
    metric=pd.concat(tables['dataset_metrics'],ignore_index=True)
    lines=[f'# {fold} / seed {seed}：十配置结果','',
        '各模型独立训练、独立验证选epoch；共享输入读取。AUC/AP按生成器单元等权汇总，分数高为real。',
        '当前为单seed结果；未据测试结果改变模型或后续矩阵。','',
        '| 配置 | 数据域 | AUC | AP-real | AP-fake |','| --- | --- | ---: | ---: | ---: |']
    for r in metric.itertuples():lines.append(f'| {r.variant} | {r.dataset} | {r.auc:.3f} | {r.real_positive_ap:.3f} | {r.fake_positive_ap:.3f} |')
    lines+=['','完整精度及逐生成器、低误报ROC指标见generator_metrics.csv；配对差值见confidence_intervals.csv。',
        '区间按源组重抽样，条件于该seed及已拟合模型，不代表模型搜索已校正，也不保证独立部署阈值的误报率。',
        '组的实际总耗时见manifest.json；多个模型的重叠wall time不能相加。','']
    p=dest/'RESULTS_zh.md';temp=p.with_suffix('.tmp.md');temp.write_text('\n'.join(lines));temp.replace(p)
    print('组级评价/验收通过',fold,seed,len(tasks),'模型',flush=True)
