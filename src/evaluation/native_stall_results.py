"""官方单视频默认前向的完整检查点与batch8对照，不覆盖既有分数。"""
import json
from pathlib import Path
import numpy as np
import pandas as pd
from artifacts import checkpoint_read,paper_json
from config import config_digest
from reference import file_digest
from evaluation.tables import evaluate_fixed_pairs


def aggregate(root):
    root=Path(root);rows=[];inputs={}
    for length,label,world_size in [(8,'8',2),(16,'16',2),(16,'supplement',1)]:
        mf=root/'data/manifests/native_stall'/f'evaluation_{label}.csv'
        frame=pd.read_csv(mf,keep_default_na=False)
        directory=root/'results/runs'/f'complete23_native_stall_{label}'
        world_size=json.loads((directory/'completed_rank_0.json').read_text())['world_size']
        identities={}
        for rank in range(world_size):
            spec=json.loads((directory/f'identity_rank_{rank}.json').read_text())
            done=json.loads((directory/f'completed_rank_{rank}.json').read_text());identity=config_digest(spec)
            if done['identity']!=identity or done['world_size']!=world_size or spec['manifest_sha256']!=config_digest(frame.to_dict('records')):
                raise ValueError('原生官方检查点身份不符')
            if spec['frame_batch']!=32 or spec['pad_tail'] is not False:raise ValueError('不是默认无补齐前向')
            identities[rank]=identity;inputs[str(directory/f'identity_rank_{rank}.json')]=file_digest(directory/f'identity_rank_{rank}.json')
        for i,r in enumerate(frame.to_dict('records')):
            payload=checkpoint_read(directory/'raw'/f'{i:06d}.json',identities[i%world_size],r['video_id'])
            expected=json.loads(r['downsample_idxs'] if length==8 else r['2_sec_idxs'])
            if payload['frame_indices']!=expected:raise ValueError('原生单窗帧身份不符')
            rows.append({k:r[k] for k in ('video_id','dataset','subset','source_model','video_path')}|
                        dict(variant='official_single_window',final_score=payload['final_score'],window_frames=length))
    scores=pd.DataFrame(rows)
    old=pd.concat([pd.read_csv(root/'results/runs/complete23_coverage_only/scores_official.csv.gz',float_precision='round_trip'),
                   pd.read_csv(root/'results/runs/paper_tables_all_official/video_scores.csv.gz',float_precision='round_trip')]).drop_duplicates('video_id').set_index('video_id')
    indexed=scores.set_index('video_id')
    if indexed.index.duplicated().any() or not set(indexed.index).issubset(old.index):raise ValueError('默认前向缺少逐身份batch8对照')
    indexed['batch8_score']=old.loc[indexed.index,'final_score'];indexed['difference']=indexed.final_score-indexed.batch8_score
    output=root/'results/runs/complete23_native_stall_tables';output.mkdir(parents=True,exist_ok=False)
    scores.to_csv(output/'video_scores.csv.gz',index=False)
    indexed.reset_index().to_csv(output/'batch_comparison.csv.gz',index=False)
    comparison=indexed.groupby(['dataset','window_frames']).difference.agg(n='size',changed=lambda x:int(np.count_nonzero(x)),max_abs=lambda x:float(abs(x).max()),mean_abs=lambda x:float(abs(x).mean()))
    comparison.to_csv(output/'batch_difference_summary.csv')
    for scope,path in [('coverage_only','results/runs/complete23_coverage_only'),('paper_filter','results/paper_complete')]:
        pairs=pd.read_csv(root/path/'pairs.csv');selected=scores[scores.video_id.isin(set(pairs.video_id))]
        tables=evaluate_fixed_pairs(selected,pairs)
        for key,table in tables.items():table.to_csv(output/f'{scope}_{key}.csv',index=False)
    paper_json(output/'manifest.json',dict(status='completed',inputs=inputs,rows=len(scores),
        protocol='official single-video inference default frame batch cap32; no tail padding; original NPZ',
        files={p.name:file_digest(p) for p in output.iterdir() if p.is_file()}))
    print(comparison.to_string(),flush=True)


if __name__=='__main__':aggregate(Path.cwd())
