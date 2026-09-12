"""官方STALL原生CSV单窗控制；数学与固定随机窗按官方实现核对。"""
import numpy as np
from selection import validate_indices


def official_window(indices, window_frames=16):
    """官方video_index.compute_windows(seed=42)：先抽1秒，再抽2秒窗。"""
    indices=validate_indices(indices)
    if window_frames not in (8,16) or len(indices)<window_frames:raise ValueError('官方单窗需要8或16个有效帧')
    rng=np.random.RandomState(42)
    first=rng.randint(0,len(indices)-8+1)
    if window_frames==8:return indices[first:first+8]
    start=rng.randint(0,len(indices)-16+1)
    return indices[start:start+16]


def official_numpy_score(global_windows,parameters):
    """保持官方NumPy归一化和矩阵运算dtype，不将Torch路径冒称逐元素相同。"""
    g=np.asarray(global_windows)
    d=np.diff(g,axis=1).astype(np.float32)
    norm=np.linalg.norm(d,axis=-1,keepdims=True)
    v=d/np.where(norm==0,1.,norm)
    def likelihood(x,mu,w):
        z=np.matmul(x-mu,w)
        return -.5*(z.shape[-1]*np.log(2.*np.pi)+(z**2).sum(-1))
    spatial=likelihood(g,parameters['mu_spat'],parameters['W_spat']).max(1)
    temporal=likelihood(v,parameters['mu_temp'],parameters['W_temp'])
    temporal[np.linalg.norm(np.diff(g,axis=1),axis=-1)==0]=np.inf
    temporal=temporal.min(1)
    rs=np.sort(parameters['calib_ll_spat'].max(1));rt=np.sort(parameters['calib_ll_temp'].min(1))
    score=.5*(np.searchsorted(rs,spatial,side='right')/len(rs)+np.searchsorted(rt,temporal,side='right')/len(rt))
    return dict(global_spatial_raw=spatial,global_temporal_raw=temporal,final_score=score)


def run_official(root,config,dataset,output,rank=0,world_size=2,*,manifest=None,window_frames=16,frame_batch_size=8,pad_tail=True):
    """官方固定单窗、原始NumPy参数评分；工程前向仍固定batch8。"""
    import json,time
    from pathlib import Path
    import pandas as pd
    from artifacts import checkpoint_read,checkpoint_write,paper_json
    from config import config_digest
    from reference import file_digest
    from features import AlphaStallFeatureExtractor
    from workflow import video_file_states
    from data.video import decode_bounded
    from data.prefetch import bounded_map,frame_reservation
    root=Path(root);output=Path(output);output.mkdir(parents=True,exist_ok=True)
    frame=pd.read_csv(manifest if manifest is not None else root/'data/manifests/active'/dataset/'evaluation.csv',keep_default_na=False)
    parameter_path=root/'precomputed/stall_params_vatex_dino_v3.npz'
    with np.load(parameter_path,allow_pickle=False) as z:parameters={k:z[k].copy() for k in z.files}
    states=video_file_states(root,frame)
    spec=dict(config=config,dataset=dataset,files=states,parameter_sha256=file_digest(parameter_path),
        code_sha256=file_digest(Path(__file__)),window_rule='official video_index seed42 second draw for 2 seconds',
        frame_batch=frame_batch_size,scoring='official NumPy parameter dtype',manifest_sha256=config_digest(frame.to_dict('records')))
    if frame_batch_size not in (8,32):raise ValueError('官方前向仅允许既有batch8控制或默认batch32')
    if frame_batch_size!=8 or not pad_tail:spec['pad_tail']=pad_tail
    if window_frames==8:spec.update(window_rule='official seed42 first draw for 1 second',window_frames=8)
    elif window_frames!=16:raise ValueError('官方窗口长度非法')
    identity=config_digest(spec);marker=output/f'identity_rank_{rank}.json'
    if marker.exists():
        if json.loads(marker.read_text())!=spec:raise ValueError('官方比较恢复身份改变')
    else:paper_json(marker,spec)
    items=[]
    for i,row in enumerate(frame.itertuples()):
        if i%world_size!=rank:continue
        indices=official_window(json.loads(row.downsample_idxs),window_frames)
        # 已有字段必须与官方生成规则相符，拒绝默默换窗。
        saved=frame.iloc[i].get('1_sec_idxs' if window_frames==8 else '2_sec_idxs','')
        if saved and json.loads(saved)!=indices:raise ValueError(f'{row.video_id}原单窗与官方seed规则不同')
        path=output/'raw'/f'{i:06d}.json'
        if path.exists():checkpoint_read(path,identity,row.video_id)
        else:items.append((i,row,indices))
    if file_digest(root/config['encoder']['weights'])!=config['encoder']['weights_sha256']:raise ValueError('权重身份错误')
    model=AlphaStallFeatureExtractor(config['runtime']['device'],dino_repo=str(root/config['encoder']['repo']),
        dino_weights=str(root/config['encoder']['weights']),pad_tail_batch=pad_tail)
    rt=config['runtime'];start=time.perf_counter()
    stream=bounded_map(items,lambda x:model.prepare_frames(decode_bounded(root/x[1].video_path,x[2])),
        lambda x:frame_reservation(root/x[1].video_path,window_frames),workers=rt['decode_workers'],depth=rt['prefetch_depth'],budget=rt['prefetch_memory_mb']*2**20)
    try:
        for n,((i,row,indices),frames) in enumerate(stream,1):
            g=model.frames_to_global_embeddings([frames],batch_size=frame_batch_size)[0]
            scores=official_numpy_score(g[None],parameters)
            checkpoint_write(output/'raw'/f'{i:06d}.json',identity,dict(video_id=row.video_id,frame_indices=indices,
                **{k:float(v[0]) for k,v in scores.items()}))
            if n%16==0 or n==len(items):
                paper_json(output/f'progress_rank_{rank}.json',dict(completed=n,pending_at_start=len(items),elapsed_seconds=time.perf_counter()-start))
                print(f'[official:{rank}] {n}/{len(items)}',flush=True)
    finally:stream.close()
    if states!=video_file_states(root,frame):raise ValueError('官方评分中视频改变')
    paper_json(output/f'completed_rank_{rank}.json',dict(identity=identity,world_size=world_size,rank=rank,elapsed_seconds=time.perf_counter()-start))
