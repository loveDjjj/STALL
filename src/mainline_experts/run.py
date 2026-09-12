"""主线专家实验入口；固定配置、独立产物，不覆盖原论文。"""
import os,sys
from pathlib import Path
os.environ.setdefault('OPENBLAS_NUM_THREADS','4')
os.environ.setdefault('OMP_NUM_THREADS','4')
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'src'))
import argparse,json,time
import numpy as np,pandas as pd,torch,yaml
from artifacts import atomic_csv,paper_json
from config import config_digest
from reference import file_digest
from mainline_experts.models import load_assets,fit_experts,save_models,load_anchors


def configuration(root=ROOT):
    c=yaml.safe_load((root/'configs/mainline_experts.yaml').read_text())
    if c['experts']!=2 or c['pool_weight']!=.5 or c['ridge']!=1e-5 or c['seed']!=17:raise ValueError('冻结实验参数不同')
    return c


def fit(domain):
    c=configuration();out=ROOT/c['run_directory']/f'fit_{domain}';out.mkdir(parents=True,exist_ok=True)
    a=load_assets(ROOT,domain);anchors=load_anchors(ROOT,domain)
    spec=dict(config=c,code={p:file_digest(ROOT/p) for p in ['src/mainline_experts/models.py','src/mainline_experts/run.py']},
        inputs={x['video_id']:x['source_sha256'] for x in a},anchor=file_digest(ROOT/f'results/runs/paper_fit_{domain}/gaussians.npz'))
    if (out/'identity.json').exists():
        if json.loads((out/'identity.json').read_text())!=spec:raise ValueError('专家拟合恢复身份改变')
        if (out/'manifest.json').exists():return
    else:paper_json(out/'identity.json',spec)
    start=time.perf_counter();centers,models,support=fit_experts(a,anchors=anchors)
    save_models(out/'models.npz',centers,models);atomic_csv(out/'support.csv',pd.DataFrame(support))
    # M=1不重估R0；所有参数对象完全回退原模型。
    _,one,_=fit_experts(a,k=1,anchors=anchors)
    for b in one:
        np.testing.assert_array_equal(one[b][0].whitening,one[b][1].whitening)
    # 源级一次留出：总体/聚类/专家都在训练折重拟合，不使用全池先验。
    groups=sorted({x['source_group'] for x in a});order=np.random.default_rng(17).permutation(len(groups));hold={groups[i] for i in order[:max(1,int(len(groups)*.2))]}
    train=[x for x in a if x['source_group'] not in hold];val=[x for x in a if x['source_group'] in hold]
    hc,hm,hs=fit_experts(train);rows=[]
    logdets={b:[np.linalg.slogdet(m.whitening)[1] for m in ms] for b,ms in hm.items()}
    for x in val:
        route=(x['context']@hc.T).argmax(1)
        for b in ('gt','lt'):
            v=x[b].astype(np.float64)
            if not len(v):continue
            nll=[]
            for mi,m in enumerate(hm[b]):
                white=(v-m.mean)@m.whitening
                # log|W|=-.5log|C|；与Gaussian近似NLL一致。
                value=.5*(1024*np.log(2*np.pi)+(white*white).sum(1))-logdets[b][mi]
                nll.append(value)
            labels=route[x[b+'_window']]
            selected=np.stack(nll[1:])[labels,np.arange(len(v))]
            rows.append(dict(video_id=x['video_id'],source_group=x['source_group'],branch=b,pooled_nll=float(nll[0].mean()),expert_nll=float(selected.mean())))
    atomic_csv(out/'holdout.csv',pd.DataFrame(rows));atomic_csv(out/'holdout_support.csv',pd.DataFrame(hs))
    paper_json(out/'holdout_sources.json',dict(train=[x['video_id'] for x in train],validation=[x['video_id'] for x in val],validation_groups=sorted(hold)))
    paper_json(out/'manifest.json',dict(status='fit_complete',identity=config_digest(spec),seconds=time.perf_counter()-start,M1_exact=True,
        files={p:file_digest(out/p) for p in ['models.npz','support.csv','holdout.csv','holdout_support.csv','holdout_sources.json']}))
    print(domain,'fit complete',pd.DataFrame(rows).groupby('branch')[['pooled_nll','expert_nll']].mean().to_dict(),flush=True)


def main():
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['fit','prepare','dense','evaluate','analyze','verify','report','all']);p.add_argument('--dataset');p.add_argument('--rank',type=int,default=0);p.add_argument('--world-size',type=int,default=2);p.add_argument('--pilot',action='store_true');a=p.parse_args()
    torch.set_num_threads(4)
    if a.stage=='fit':fit(a.dataset)
    else:
        from mainline_experts import experiment
        getattr(experiment,a.stage)(ROOT,a)

if __name__=='__main__':main()
