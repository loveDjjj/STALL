"""新真实池与重编码结果的独立覆盖、指标、源隔离与编码合同核验。"""
import argparse,json
from pathlib import Path
import numpy as np,pandas as pd
from sklearn.metrics import roc_auc_score,average_precision_score
from artifacts import paper_json
from reference import file_digest
from config import config_digest
from evaluation.confirmation_engine import Chunks
from evaluation.confirmation_run import wait_pid


def metrics(scores,pairs,printed,conditions=False):
    keys=['condition','variant'] if conditions else ['variant'];err=0.;count=0
    indexed=printed.set_index(keys+['dataset','generator'])
    for tag,part in scores.groupby(keys):
        tag=tag if isinstance(tag,tuple) else (tag,);f=part.set_index('video_id',verify_integrity=True)
        if set(f.index)!=set(pairs.video_id):raise ValueError('条件或变体身份不完整')
        for (d,g),pair in pairs.groupby(['dataset','generator']):
            p=f.loc[pair.video_id];y=pair.subset.eq('real').to_numpy()
            if not (p.subset.to_numpy()==pair.subset.to_numpy()).all():raise ValueError('源配对标签改变')
            actual=[roc_auc_score(y,p.final_score),average_precision_score(y,p.final_score)]
            expected=indexed.loc[(*tag,d,g),['auc','real_positive_ap']].to_numpy(dtype=float)
            err=max(err,float(np.abs(actual-expected).max()));count+=1
    if err>1e-12:raise ValueError('独立指标不符')
    return dict(metric_rows=count,max_error=err)


def reference(root):
    out=root/'results/runs/reference_confirmation';dest=out/'results';m=json.loads((dest/'manifest.json').read_text())
    if m['status']!='complete':raise ValueError('新池评分未完成')
    for p,h in m['files'].items():
        if file_digest(dest/p)!=h:raise ValueError('结果产物改变')
    selections=pd.read_csv(out/'fit_selections.csv',keep_default_na=False)
    if not selections.groupby(['dataset','fit_seed']).size().eq(200).all():raise ValueError('新fit预算错误')
    blocked=[pd.read_csv(root/'results/paper_complete/evaluation.csv',keep_default_na=False)]
    blocked += [pd.read_csv(root/f'data/manifests/active/{d}/fit.csv',keep_default_na=False) for d in ('comgenvid','videofeedback','genvideo')]
    blocked += [pd.read_csv(root/f'data/manifests/active/vatex/{r}.csv',keep_default_na=False) for r in ('cdf','threshold')]
    forbidden=set(pd.concat(blocked).source_group)
    if forbidden&set(selections.source_group):raise ValueError('新fit与旧角色源交叉')
    scores=pd.read_csv(dest/'video_scores.csv.gz',float_precision='round_trip');pairs=pd.read_csv(root/'results/paper_complete/pairs.csv')
    result=metrics(scores,pairs,pd.read_csv(dest/'generator_metrics.csv'))
    original=pd.read_csv(root/'results/paper_complete/scores_fc3.csv.gz',float_precision='round_trip');reg=[]
    for a,b in [('original_global','global_only'),('original_full','full')]:
        x=scores[scores.variant==a].set_index('video_id').final_score;y=original[original.variant==b].set_index('video_id').final_score.loc[x.index]
        error=float(np.abs(x.to_numpy()-y.to_numpy()).max())
        if error>1e-12:raise ValueError('旧锚点改变')
        reg.append(dict(variant=a,max_error=error))
    paper_json(dest/'independent_audit.json',dict(status='passed',**result,regression=reg,new_fit_unique=selections.video_id.nunique(),
        fit_pools=15,clips_per_pool=200,old_fit_eval_reference_source_intersection=0,mutually_disjoint_new_pools=False,
        code=file_digest(Path(__file__)),scores=file_digest(dest/'video_scores.csv.gz'),pairs=file_digest(root/'results/paper_complete/pairs.csv')))
    print('new reference audit passed',result,flush=True)


def encoding(root):
    out=root/'results/runs/reference_confirmation/encoding';m=json.loads((out/'results.json').read_text());spec=json.loads((out/'identity.json').read_text());identity=config_digest(spec)
    if m['status']!='completed' or identity!=m['identity']:raise ValueError('编码结果未完成或身份不符')
    for p,h in m['files'].items():
        if file_digest(out/p)!=h:raise ValueError('编码结果改变')
    pairs=pd.read_csv(out.parent/'robust_pairs.csv');scores=pd.read_csv(out/'video_scores.csv.gz',float_precision='round_trip')
    result=metrics(scores,pairs,pd.read_csv(out/'generator_metrics.csv'),True)
    receipts=[]
    for rank in (0,1):
        for r in Chunks(out/'raw'/f'rank{rank}',identity).records.values():
            receipt=r['receipt']
            if receipt['max_timestamp_error']>1e-3 or receipt['frames']<8 or len(receipt['shape'])!=2:raise ValueError('编码帧合同失败')
            cmd=receipt['command']
            if '-r' in cmd or '-vf' in cmd or cmd[cmd.index('-vsync')+1]!='0':raise ValueError('编码改变采样/尺寸')
            receipts.append(receipt)
    paper_json(out/'independent_audit.json',dict(status='passed',**result,unique_evaluation=scores.video_id.nunique(),encoding_receipts=len(receipts),
        max_timestamp_error=max(r['max_timestamp_error'] for r in receipts),fixed_windows=True,clean_references=True,
        code=file_digest(Path(__file__)),scores=file_digest(out/'video_scores.csv.gz')))
    print('encoding independent audit passed',result,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('action',choices=['reference','encoding']);p.add_argument('--wait-pid',type=int);a=p.parse_args()
    if a.wait_pid is not None:wait_pid(a.wait_pid)
    globals()[a.action](Path.cwd())
