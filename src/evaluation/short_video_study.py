"""8帧参考及完整消融共享评分入口；标准16帧资产只读。"""
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
import yaml
from artifacts import paper_json
from reference import file_digest
from evaluation.study_runner import run_shared_evidence


def prepare_reference(root,role):
    root=Path(root);source=root/'data/manifests/active/vatex'/f'{role}.csv'
    output=root/'data/manifests/short_video/vatex'/f'{role}.csv'
    frame=pd.read_csv(source,keep_default_na=False)
    indices=[]
    for row in frame.itertuples():
        full=json.loads(row.downsample_idxs)
        if len(full)<8:raise ValueError('参考不足8帧')
        start=np.random.RandomState(42).randint(0,len(full)-8+1)
        indices.append(json.dumps(full[start:start+8]))
    frame['full_downsample_idxs']=frame.downsample_idxs
    frame['downsample_idxs']=indices
    frame['one_second_rule']='official seed42 first draw; one 8-frame window'
    text=frame.to_csv(index=False)
    output.parent.mkdir(parents=True,exist_ok=True)
    if output.exists():
        if output.read_text()!=text:raise ValueError('短视频参考身份改变')
    else:
        output.write_text(text)
        paper_json(output.with_suffix('.json'),dict(source_sha256=file_digest(source),
            rule='same identity, 8fps 1sec seed42 first draw',rows=len(frame)))
    return output


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--domain',required=True,choices=['videofeedback','genvideo'])
    parser.add_argument('--role',required=True,choices=['cdf','threshold','evaluation','official'])
    parser.add_argument('--device',default='cuda:0')
    parser.add_argument('--rank',type=int,default=0)
    parser.add_argument('--world-size',type=int,default=1)
    args=parser.parse_args();root=Path.cwd()
    config=yaml.safe_load((root/'configs/paper.yaml').read_text())
    config['runtime']['device']=args.device
    config['selection'].update(name='uniform',k=1,frames=8)
    config['evidence_window_frames']=8
    config['protocol']['id']='complete23_short8_same_real_fit'
    manifest=prepare_reference(root,args.role) if args.role in ('cdf','threshold') else root/'data/manifests/short_video'/args.domain/'evaluation.csv'
    output=root/'results/runs'/f'complete23_{args.domain}_{args.role}_short8'
    if args.role=='official':
        from evaluation.official_baseline import run_official
        return run_official(root,config,args.domain,output,rank=args.rank,world_size=args.world_size,
                            manifest=manifest,window_frames=8)
    run_shared_evidence(root,config,args.domain,manifest,output,include_uniform=args.role=='cdf',
        rank=args.rank,world_size=args.world_size,baseline_only=False)


if __name__=='__main__':main()
