"""多个论文对照共享相同DINO前向，仅保存轻量证据。"""
import json
import time
from pathlib import Path

import numpy as np
import torch

from artifacts import checkpoint_read,checkpoint_write,paper_json
from branches.global_branch import score_global_raw
from config import config_digest
from evaluation.representations import RepresentationScorers
from reference import file_digest
from workflow import RawVideoScorer,prefetched_raw_scores,video_file_states


class SharedEvidenceScorer(RawVideoScorer):
    """保留原batch8与每视频选窗/前向顺序，不跨选择器拼接帧。"""
    def __init__(self,global_models,representation_models,device,**kwargs):
        target=global_models['target']
        super().__init__(target[0],target[1],representation_models['local_d2']['target'],device,**kwargs)
        self.global_models=global_models
        self.representations=RepresentationScorers(representation_models,device)

    def global_from_prepared(self,window,frames):
        features=self.extractor.frames_to_global_embeddings([frames],batch_size=8)[0][None]
        scores={}
        for name,(gs,gt) in self.global_models.items():
            raw=score_global_raw(features,gs,gt,device=self.device)
            scores[name]=dict(gs=float(raw.spatial[0]),gt=float(raw.temporal_t1[0]))
        return dict(frame_indices=list(window.frame_indices),global_models=scores)

    def score_dense(self,plan,frames):
        start=time.perf_counter()
        windows=plan['windows'];union=plan['union']
        features=self.extractor.frames_to_global_patch_embeddings([frames],batch_size=8)[0]
        if self.device.startswith('cuda'):torch.cuda.synchronize(self.device)
        feature_seconds=time.perf_counter()-start
        where={f:i for i,f in enumerate(union)}
        picks=[[where[f] for f in w.frame_indices] for w in windows]
        g=np.stack([features['global'][p] for p in picks])
        p=np.stack([features['patch'][p] for p in picks])
        start=time.perf_counter()
        local=self.representations.score(g,p)
        glob={name:score_global_raw(g,gs,gt,device=self.device) for name,(gs,gt) in self.global_models.items()}
        if self.device.startswith('cuda'):torch.cuda.synchronize(self.device)
        scoring_seconds=time.perf_counter()-start
        records=[]
        for i,w in enumerate(windows):
            records.append(dict(rank=i,start_position=w.start_position,frame_indices=list(w.frame_indices),change=w.change,
                global_models={name:dict(gs=float(x.spatial[i]),gt=float(x.temporal_t1[i])) for name,x in glob.items()},
                representations={name:{model:float(x[i]) for model,x in scores.items()} for name,scores in local.items()}))
        return dict(selector=plan['selector'],coarse_frames=plan['coarse_count'],dense_unique_frames=len(union),
            feature_seconds=feature_seconds,scoring_seconds=scoring_seconds,windows=records)


def score_evidence_manifest(root,config,frame,scorer,output,model_paths,*,include_uniform=False,rank=0,world_size=1):
    """逐视频原子检查点，模型/清单/代码hash绑定；不同进程按行号互斥分片。"""
    root=Path(root);output=Path(output);output.mkdir(parents=True,exist_ok=True)
    if not 0<=rank<world_size:raise ValueError('非法分片')
    states=video_file_states(root,frame)
    payload=dict(config=config,manifest_sha256=config_digest(frame.to_dict('records')),
        files=states,models={str(p):file_digest(p) for p in model_paths},include_uniform=include_uniform,
        code={name:file_digest(root/name) for name in ('src/evaluation/evidence.py','src/evaluation/representations.py',
             'src/workflow.py','src/features.py','src/selection.py','src/math_utils.py','src/branches/global_branch.py')})
    identity=config_digest(payload)
    marker=output/f'identity_rank_{rank}.json'
    if marker.exists():
        if json.loads(marker.read_text())!=payload:raise ValueError('证据任务身份已改变')
    else:paper_json(marker,payload)
    pending=[]
    for index,row in enumerate(frame.itertuples()):
        if index%world_size!=rank:continue
        path=output/'raw'/f'{index:06d}.json'
        if path.exists():checkpoint_read(path,identity,row.video_id)
        else:pending.append((index,row))
    start=time.perf_counter()
    for completed,((index,row),raw) in enumerate(prefetched_raw_scores(root,config,scorer,pending,include_uniform=include_uniform),1):
        checkpoint_write(output/'raw'/f'{index:06d}.json',identity,dict(video_id=row.video_id,evidence=raw))
        if completed%16==0 or completed==len(pending):
            paper_json(output/f'progress_rank_{rank}.json',dict(completed=completed,pending_at_start=len(pending),
                elapsed_seconds=time.perf_counter()-start,video_id=row.video_id))
            print(f'[evidence:{rank}] {completed}/{len(pending)}',flush=True)
    if states!=video_file_states(root,frame):raise ValueError('评分中原视频发生变化')
    paper_json(output/f'completed_rank_{rank}.json',dict(identity=identity,rank=rank,world_size=world_size,
        total_rows=sum(i%world_size==rank for i in range(len(frame))),elapsed_seconds=time.perf_counter()-start))
