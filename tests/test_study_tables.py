"""完整性门禁：不能用少一个视频或错身份的检查点生成论文表。"""
import json,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
import pandas as pd
import pytest
from artifacts import paper_json,checkpoint_write
from config import config_digest
from evaluation.study_tables import read_evidence
from evaluation.study_tables import combine_domains
from evaluation.study_tables import evidence_ready
from reference import file_digest


def test_evidence_requires_all_rows_and_matching_manifest(tmp_path):
    frame=pd.DataFrame([dict(video_id='a'),dict(video_id='b')])
    spec=dict(manifest_sha256=config_digest(frame.to_dict('records')))
    identity=config_digest(spec)
    for rank in (0,1):
        paper_json(tmp_path/f'identity_rank_{rank}.json',spec)
        paper_json(tmp_path/f'completed_rank_{rank}.json',dict(identity=identity,world_size=2))
    checkpoint_write(tmp_path/'raw/000000.json',identity,dict(video_id='a',evidence={'windows':[]}))
    with pytest.raises(FileNotFoundError):read_evidence(tmp_path,frame)
    checkpoint_write(tmp_path/'raw/000001.json',identity,dict(video_id='wrong',evidence={'windows':[]}))
    with pytest.raises(ValueError):read_evidence(tmp_path,frame)
    checkpoint_write(tmp_path/'raw/000001.json',identity,dict(video_id='b',evidence={'windows':[]}))
    assert len(read_evidence(tmp_path,frame)[0])==2
    with pytest.raises(ValueError):read_evidence(tmp_path,frame.iloc[::-1])
    for rank in (0,1):
        mismatched={**spec,'models':{'reference':str(rank)}}
        paper_json(tmp_path/f'identity_rank_{rank}.json',mismatched)
        paper_json(tmp_path/f'completed_rank_{rank}.json',dict(identity=config_digest(mismatched),world_size=2))
    with pytest.raises(ValueError,match='分片'):read_evidence(tmp_path,frame)


def test_csv_round_trip_preserves_ranking_and_roc(tmp_path):
    domains=('comgenvid','videofeedback','genvideo');inputs=[];pairs=[]
    for d in domains:
        folder=tmp_path/d;folder.mkdir();inputs.append(folder)
        rows=[]
        for subset,score in [('real',0.011993760301389215),('annotated',0.0119937603013892)]:
            vid=d+':'+subset
            rows.append(dict(video_id=vid,dataset=d,subset=subset,source_model='M',video_path=vid,variant='full',final_score=score))
            pairs.append(dict(video_id=vid,dataset=d,subset=subset,generator='M'))
        path=folder/'video_scores.csv.gz';pd.DataFrame(rows).to_csv(path,index=False)
        paper_json(folder/'manifest.json',dict(status='completed',files={path.name:file_digest(path)}))
    path=tmp_path/'data/manifests/active/pairs.csv';path.parent.mkdir(parents=True);pd.DataFrame(pairs).to_csv(path,index=False)
    scores=combine_domains(tmp_path,inputs,tmp_path/'combined')
    assert scores[scores.subset.eq('real')].final_score.iloc[0]>scores[scores.subset.eq('annotated')].final_score.iloc[0]
    metrics=pd.read_csv(tmp_path/'combined/macro_metrics.csv')
    assert metrics.auc.iloc[0]==1. and metrics.fake_tpr_at_1pct_real_fpr.iloc[0]==1.


def test_six_shards_reuse_same_gpu_checkpoint_identities(tmp_path):
    frame=pd.DataFrame([dict(video_id=str(i)) for i in range(12)])
    specs={rank:dict(manifest_sha256=config_digest(frame.to_dict('records')),
        config={'runtime':{'device':f'cuda:{rank%2}'}}) for rank in range(6)}
    # 原二分片已完成的奇数编号仍属于GPU1，不改其载荷或身份。
    for i in (1,3,5,7,9):checkpoint_write(tmp_path/'raw'/f'{i:06d}.json',config_digest(specs[1]),dict(video_id=str(i),evidence={}))
    for rank in range(6):
        paper_json(tmp_path/f'identity_rank_{rank}.json',specs[rank])
        paper_json(tmp_path/f'completed_rank_{rank}.json',dict(identity=config_digest(specs[rank]),world_size=6))
        if rank==1:assert not evidence_ready(tmp_path)
    for i in (0,2,4,6,8,10,11):checkpoint_write(tmp_path/'raw'/f'{i:06d}.json',config_digest(specs[i%6]),dict(video_id=str(i),evidence={}))
    assert evidence_ready(tmp_path)
    assert len(read_evidence(tmp_path,frame)[0])==12
