import numpy as np
import pytest
import torch
from looped_video.stream import global_microbatches


@pytest.mark.parametrize('size',[1,7,8,16,17,23,32,33,65])
def test_ddp_partition_and_tail_weight(size):
    draw=list(range(size));seen=[]
    for p in global_microbatches(draw):
        for ids,count in zip(p['indices'],p['counts']):seen.extend(ids[:count])
        assert sum(p['counts'])<=16
        assert 1<=p['denom']<=32
    assert seen==draw


def test_resume_keeps_same_global_draw_order():
    full=list(global_microbatches(list(range(97))))
    resumed=list(global_microbatches(list(range(97)),start_update=2))
    assert resumed==[p for p in full if p['update']>=2]


def test_ddp_weighted_gradients_equal_single_model():
    # DDP取各rank梯度平均，故局部loss必须乘world/global实际样本数。
    torch.manual_seed(4);x=torch.randn(17,3);y=torch.randn(17,1)
    a=torch.nn.Linear(3,1);b=torch.nn.Linear(3,1);b.load_state_dict(a.state_dict())
    ((a(x)-y)**2).mean().backward()
    rank_grads=[]
    for rank in [0,1]:
        b.zero_grad()
        for p in global_microbatches(list(range(17))):
            ids=p['indices'][rank];loss=((b(x[ids])-y[ids])**2).sum()
            (loss*(2/p['denom'])*(p['counts'][rank]>0)).backward()
        rank_grads.append([p.grad.clone() for p in b.parameters()])
    for i,p in enumerate(a.parameters()):torch.testing.assert_close(p.grad,(rank_grads[0][i]+rank_grads[1][i])/2)


def test_ram_reader_preserves_values_and_deduplicates(tmp_path):
    import pandas as pd
    from looped_video.stream import SharedReader
    from looped_video.data import PatchDataset,collate_videos
    records={};rows=[]
    for i,t in enumerate([8,16]):
        p=tmp_path/f'{i}.npy';np.save(p,np.random.default_rng(i).normal(size=(t,4,12)).astype('float32'));s=p.stat()
        records[str(i)]=dict(bytes=s.st_size,mtime_ns=s.st_mtime_ns,window_positions=[list(range(t))])
        rows.append(dict(key=str(i),subset='real'))
    meta=pd.DataFrame(rows);old=PatchDataset(meta,tmp_path,records);reader=SharedReader(tmp_path,records,1,2)
    reader.configure(meta,np.array([.5,.5]))
    try:
        expected=collate_videos([old[0],old[1]])
        new=reader.batches(meta,[[0,1],[1,0]])[0]
        for k in expected:torch.testing.assert_close(new[k],expected[k],rtol=0,atol=0)
        assert reader.misses==2
        reader.batches(meta,[[0,1],[1,0]])
        assert reader.hits==2 and reader.misses==2
    finally:reader.close()


@pytest.mark.skipif(torch.cuda.device_count()<2,reason='需要两张CUDA卡')
def test_real_ddp_training_tail_and_checkpoint(tmp_path):
    import json
    import pandas as pd
    from pathlib import Path
    from looped_video.ddp_train import run_task
    from looped_video.stream import SharedReader
    root=Path(__file__).resolve().parents[1];out=tmp_path/'run';cache=tmp_path/'cache'
    out.mkdir();cache.mkdir();(out/'prepared.json').write_text('{}');(cache/'manifest.json').write_text('{}')
    rows=[];roles=[];records={}
    for i in range(21):
        p=cache/f'{i}.npy';np.save(p,np.random.default_rng(i).normal(size=(8,4,1024)).astype('float32'));s=p.stat()
        records[str(i)]=dict(bytes=s.st_size,mtime_ns=s.st_mtime_ns,window_positions=[list(range(8))])
        rows.append(dict(video_id=str(i),key=str(i),dataset='comgenvid',subset='real' if i%2==0 else 'annotated',
            source_model='real' if i%2==0 else 'fake',source_group=str(i),split_group=str(i),length=8))
        roles.append(dict(fold='pooled',video_id=str(i),role='train' if i<17 else 'validation'))
    c=dict(cache_directory=str(cache),width=24,heads=4,loops=4,tile=2,mlp_ratio=2,learning_rate=.0003,
        weight_decay=.01,epochs=2,warmup_epochs=1,amp=True)
    rt=dict(world_size=2,local_batch=8,global_batch=32,ram_gib=1,reader_workers=2,checkpoint_updates=1)
    task=dict(key='pooled__looped__s17',fold='pooled',variant='looped',seed=17)
    reader=SharedReader(cache,records,1,2)
    try:run_task(root,out,c,task,pd.DataFrame(rows),pd.DataFrame(roles),reader,rt)
    finally:reader.close()
    dest=out/'training'/task['key'];r=json.loads((dest/'manifest.json').read_text())
    assert r['status']=='trained' and not r['test_used']
    state=torch.load(dest/'last.pt',map_location='cpu',weights_only=True)
    assert state['phase']=='epoch_complete' and state['epoch']==2
    assert sum(x[1] for x in state['rank_stats'])==17
