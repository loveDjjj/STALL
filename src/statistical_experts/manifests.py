"""固定纯真实参考职责、原版单窗索引与23单元评价身份。"""
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd
import yaml
from artifacts import paper_json,checkpoint_read
from config import config_digest
from reference import file_digest
from evaluation.official_baseline import official_window


def settings(root):
    root=Path(root);p=root/'configs/global_experts.yaml'
    c=yaml.safe_load(p.read_text())
    expected={'protocol','dimension','experts','neighbors','shrinkage','ridge','min_fit_sources','seed',
        'spatial_frame_seed','encoder_batch','query_batch','decode_workers','prefetch_depth','prefetch_memory_mb',
        'cache_directory','manifest_directory','run_directory'}
    if set(c)!=expected:raise ValueError('专家配置字段不符')
    if c['dimension']!=1024 or c['experts']!=4 or c['neighbors']!=512 or c['encoder_batch']!=32:
        raise ValueError('改变预定科学配置需要另立协议')
    return c


def official_expectations(root):
    root=Path(root);result={}
    for label,length in [('8',8),('16',16),('supplement',16)]:
        mf=root/'data/manifests/native_stall'/f'evaluation_{label}.csv'
        f=pd.read_csv(mf,keep_default_na=False);directory=root/'results/runs'/f'complete23_native_stall_{label}'
        done=json.loads((directory/'completed_rank_0.json').read_text());world=done['world_size'];identities={}
        for rank in range(world):
            spec=json.loads((directory/f'identity_rank_{rank}.json').read_text())
            complete=json.loads((directory/f'completed_rank_{rank}.json').read_text())
            identity=config_digest(spec)
            if complete['identity']!=identity or spec['manifest_sha256']!=config_digest(f.to_dict('records')):
                raise ValueError('官方锚点身份错误')
            identities[rank]=identity
        for i,r in enumerate(f.itertuples()):
            raw=checkpoint_read(directory/'raw'/f'{i:06d}.json',identities[i%world],r.video_id)
            result[r.video_id]=raw
    return result


def prepare(root):
    root=Path(root).resolve();c=settings(root);out=root/c['manifest_directory']
    if (out/'manifest.json').exists():
        m=json.loads((out/'manifest.json').read_text())
        for name,digest in m['files'].items():
            if file_digest(out/name)!=digest:raise ValueError('已冻结专家清单改变')
        return
    source_paths=[root/'data/catalog/vatex_source_fit200.csv',root/'data/manifests/active/vatex/threshold.csv',
                  root/'data/manifests/active/vatex/cdf.csv',root/'results/paper_complete/evaluation.csv',root/'results/paper_complete/pairs.csv']
    small,former_threshold,reference,evaluation,pairs=[pd.read_csv(p,keep_default_na=False) for p in source_paths]
    fit=pd.concat([small,former_threshold],ignore_index=True).sort_values('source_id').fillna('')
    reference=reference.sort_values('source_id')
    if not fit.subset.eq('real').all() or not reference.subset.eq('real').all():raise ValueError('专家参考必须全部为预先确认的real')
    if len(fit)!=2200 or fit.source_id.nunique()!=2200 or len(reference)!=2000 or set(fit.source_id)&set(reference.source_id):
        raise ValueError('参考不是2200/2000独立身份')
    known_msvd=set(evaluation.loc[(evaluation.dataset=='comgenvid')&(evaluation.subset=='real'),'video_path'].map(lambda p:Path(p).stem[:11]))
    if (set(fit.source_id)|set(reference.source_id))&known_msvd:raise ValueError('已知原视频来源与评价交叉')
    expectations=official_expectations(root);rows=[]
    for length in (8,16):
        spatial=np.random.RandomState(c['spatial_frame_seed']).randint(0,length,len(fit))
        for role,frame in [('fit',fit),('cdf',reference)]:
            for index,r in enumerate(frame.to_dict('records')):
                indices=official_window(json.loads(r['downsample_idxs']),length)
                vid=f"vatex:{r['source_id']}:window{length}"
                rows.append(dict(video_id=vid,video_path=r['video_path'],source_group='vatex:'+r['source_id'],
                    dataset='vatex',subset='real',source_model='VATEX',role=role,length=length,
                    frame_indices=json.dumps(indices),spatial_fit_index=int(spatial[index]) if role=='fit' else -1,
                    random_source_key=hashlib.sha256(str(r['source_id']).encode()).hexdigest(),
                    expected_gs='',expected_gt='',expected_final=''))
    for r in evaluation.to_dict('records'):
        length=8 if r['video_id'].endswith(':duration1') else 16
        indices=official_window(json.loads(r['downsample_idxs']),length);expected=expectations[r['video_id']]
        if indices!=expected['frame_indices']:raise ValueError('与官方锚点窗口不一致')
        neutral=Path(str(r['source_group']).rsplit(':',1)[-1]).stem
        rows.append(dict(video_id=r['video_id'],video_path=r['video_path'],source_group=r['source_group'],
            dataset=r['dataset'],subset=r['subset'],source_model=r['source_model'],role='evaluation',length=length,
            frame_indices=json.dumps(indices),spatial_fit_index=-1,random_source_key=hashlib.sha256(neutral.encode()).hexdigest(),
            expected_gs=expected['global_spatial_raw'],expected_gt=expected['global_temporal_raw'],expected_final=expected['final_score']))
    frame=pd.DataFrame(rows)
    if frame.video_id.duplicated().any():raise ValueError('窗口身份重复')
    if set(frame.loc[frame.role!='evaluation','video_path'])&set(evaluation.video_path):raise ValueError('参考/评价视频交叉')
    frame['cache_key']=[hashlib.sha256((r.video_path+'|'+r.frame_indices).encode()).hexdigest() for r in frame.itertuples()]
    if frame.cache_key.duplicated().any():raise ValueError('不同实验身份实际重复了同一窗口，需显式处理')
    for r in frame.itertuples():
        if not (root/r.video_path).is_file():raise FileNotFoundError(root/r.video_path)
    out.mkdir(parents=True,exist_ok=True);frame.to_csv(out/'windows.csv',index=False);pairs.to_csv(out/'pairs.csv',index=False)
    paper_json(out/'manifest.json',dict(status='prepared',config=c,inputs={str(p):file_digest(p) for p in source_paths},
        windows=len(frame),fit_videos=2200,cdf_videos=2000,evaluation_clip_ids=len(evaluation),
        role_change='旧threshold2000用于新fit；本实验不使用它作独立阈值',
        source_overlap_scope='路径与VATEX/MSVD已知媒体ID；不声称完整跨库语义去重',
        files={p.name:file_digest(p) for p in out.glob('*.csv')}))
    print(f'prepared {len(frame)} windows, evaluation {len(evaluation)}',flush=True)
