"""完整单元的8/16帧操作点分开校准，禁止把窗口长度与真假标签绑定。"""
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from evaluation.complete_benchmark import checked_scores
from evaluation.study_tables import read_evidence,calibrated_scores
from evaluation.operating_points import real_threshold,wilson


def build(root,directory,*,nominal_levels=(.001,.01),output=None):
    root=Path(root);directory=Path(directory)
    if not nominal_levels or any(not 0<float(x)<1 for x in nominal_levels):
        raise ValueError('名义FPR必须在(0,1)且非空')
    destination=Path(output) if output is not None else directory/'operating_points.csv'
    if output is not None and destination.exists():raise ValueError('新增操作点不可覆盖已有文件')
    evaluation=pd.read_csv(directory/'scores_fc3.csv.gz',float_precision='round_trip')
    metadata=pd.read_csv(directory/'evaluation.csv',keep_default_na=False).set_index('video_id')
    evaluation['window_frames']=[8 if str(v).endswith(':duration1') else 16 for v in evaluation.video_id]
    rows=[]
    for domain in ('comgenvid','genvideo','videofeedback'):
        long=pd.read_csv(root/'results/runs'/f'paper_operating_points_{domain}/threshold_scores.csv.gz',float_precision='round_trip')
        references={16:long}
        if domain!='comgenvid':
            tf=pd.read_csv(root/'data/manifests/short_video/vatex/threshold.csv',keep_default_na=False)
            cf=pd.read_csv(root/'data/manifests/short_video/vatex/cdf.csv',keep_default_na=False)
            if set(tf.video_id)&set(cf.video_id):raise ValueError('阈值与CDF身份交叉')
            tr,ti=read_evidence(root/'results/runs'/f'complete23_{domain}_threshold_short8',tf)
            cr,ci=read_evidence(root/'results/runs'/f'complete23_{domain}_cdf_short8',cf)
            if ti['models']!=ci['models'] or ti['config']!=ci['config']:raise ValueError('短窗阈值和参考配置不匹配')
            references[8]=calibrated_scores(root,domain,tf,tr,cr,1)
        for (length,variant),part in evaluation[evaluation.dataset==domain].groupby(['window_frames','variant']):
            reference=references[length]
            values=reference[reference.variant==variant].final_score.to_numpy()
            if len(values)!=2000:raise ValueError('每个评分器必须有2000独立阈值real')
            for nominal in nominal_levels:
                tau=real_threshold(values,nominal)
                real=part[part.subset=='real'].copy();fake=part[part.subset=='annotated']
                real['real_source']=metadata.loc[real.video_id,'real_source'].to_numpy()
                groups=[('real','all_real',real),* [('real',name,g) for name,g in real.groupby('real_source')],
                        * [('fake',name,g) for name,g in fake.groupby('source_model')]]
                for role,name,g in groups:
                    n=len(g);count=int((g.final_score<tau).sum());lo,hi=wilson(count,n)
                    rows.append(dict(dataset=domain,variant=variant,window_frames=length,role=role,group=name,
                        nominal_real_fpr=nominal,threshold=tau,threshold_n=len(values),n=n,count=count,rate=count/n,
                        ci95_low=lo,ci95_high=hi,reference_real_fpr=float(np.mean(values<tau))))
    result=pd.DataFrame(rows)
    if (result.reference_real_fpr>result.nominal_real_fpr+1e-12).any():raise ValueError('阈值参考FPR越界')
    destination.parent.mkdir(parents=True,exist_ok=True)
    result.to_csv(destination,index=False)
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('directory')
    p.add_argument('--levels',nargs='+',type=float,default=[.001,.01]);p.add_argument('--output',type=Path)
    a=p.parse_args();build(Path.cwd(),a.directory,nominal_levels=a.levels,output=a.output)
