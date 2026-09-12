import pytest
import torch
from looped_video.model import LoopedDetector, tiles, untiles


def small(**kwargs):
    return LoopedDetector(input_dim=12, width=24, heads=4, loops=4, **kwargs)


@pytest.mark.parametrize('temporal', [True, False])
def test_tile_inverse(temporal):
    x = torch.arange(2*3*16*5).reshape(2,3,16,5)
    assert torch.equal(x, untiles(tiles(x,2,temporal),x.shape,2,temporal))


def test_padding_is_not_evidence_and_gradients_reach_memory():
    torch.manual_seed(9)
    m = small().eval()
    x = torch.randn(2,4,16,12,requires_grad=True)
    valid = torch.tensor([[1,1,0,0],[1,1,1,1]],dtype=torch.bool)
    y = m(x,valid)
    altered=x.detach().clone();altered[~valid]=1000
    torch.testing.assert_close(y,m(altered,valid),rtol=0,atol=0)
    torch.testing.assert_close(y[:1],m(x[:1,:2],valid[:1,:2]),rtol=1e-5,atol=1e-6)
    y.sum().backward()
    assert x.grad[~valid].abs().sum()==0
    assert m.blocks[0].read.kv.weight.grad.abs().sum()>0
    assert torch.isfinite(x.grad).all()


def test_shared_and_unrolled_equal_when_weights_equal():
    torch.manual_seed(12)
    a=small(); b=small(variant='untied')
    b.projection.load_state_dict(a.projection.state_dict())
    b.readout.load_state_dict(a.readout.state_dict())
    for block in b.blocks:block.load_state_dict(a.blocks[0].state_dict())
    x=torch.randn(2,4,16,12);valid=torch.ones(2,4,dtype=torch.bool)
    torch.testing.assert_close(a(x,valid),b(x,valid))
    a(x,valid).sum().backward();b(x,valid).sum().backward()
    for name,p in a.blocks[0].named_parameters():
        expected=sum(dict(block.named_parameters())[name].grad for block in b.blocks)
        torch.testing.assert_close(p.grad,expected,rtol=2e-4,atol=2e-6)


@pytest.mark.parametrize('variant',['looped','untied','no_refresh','spatial_only','state_only'])
def test_variants_and_video_aggregation(variant):
    m=small(variant=variant)
    x=torch.randn(3,4,16,12);valid=torch.ones(3,4,dtype=torch.bool)
    window=m(x,valid)
    video=m(x,valid,torch.tensor([0,0,1]),2)
    torch.testing.assert_close(video,torch.stack([window[:2].mean(),window[2]]))
    video.sum().backward()
    assert all(p.grad is None or torch.isfinite(p.grad).all() for p in m.parameters())


def test_collator_keeps_short_frames_and_window_owners():
    from looped_video.data import collate_videos
    a=torch.ones(2,8,4,12);b=torch.full((1,16,4,12),2.)
    batch=collate_videos([(a,0,9),(b,1,2)])
    assert batch['owners'].tolist()==[0,0,1]
    assert batch['indices'].tolist()==[9,2]
    assert batch['valid'].sum(1).tolist()==[8,8,16]
    assert batch['patches'][:2,8:].abs().sum()==0


@pytest.mark.skipif(not torch.cuda.is_available(),reason='训练端到端检查需要CUDA')
def test_training_checkpoint_and_cached_restart(tmp_path):
    import json
    from pathlib import Path
    import numpy as np
    import pandas as pd
    from looped_video.training import train_job
    root=Path(__file__).resolve().parents[1]
    out=tmp_path/'run';cache=tmp_path/'cache';out.mkdir();cache.mkdir()
    (out/'prepared.json').write_text('{}')
    (cache/'manifest.json').write_text('{}')
    rows=[];records={};roles=[]
    for i in range(8):
        p=cache/f'{i}.npy';np.save(p,np.random.default_rng(i).normal(size=(8,4,1024)).astype('float32'))
        s=p.stat();records[str(i)]=dict(bytes=s.st_size,mtime_ns=s.st_mtime_ns,window_positions=[list(range(8))])
        rows.append(dict(video_id=str(i),dataset='comgenvid',subset='real' if i%2==0 else 'annotated',
            source_model='real' if i%2==0 else 'fake',source_group=str(i),split_group=str(i),length=8,key=str(i)))
        roles.append(dict(fold='pooled',video_id=str(i),role='train' if i<4 else 'validation'))
    c=dict(width=24,heads=4,loops=4,tile=2,mlp_ratio=2,learning_rate=.0003,weight_decay=.01,
        loader_workers=0,batch_size=2,accumulation=2,epochs=2,warmup_epochs=1,amp=True)
    task=dict(key='pooled__looped__s17',fold='pooled',variant='looped',seed=17)
    train_job(root,out,cache,records,c,pd.DataFrame(rows),pd.DataFrame(roles),task,'cuda:0')
    dest=out/'training'/task['key'];receipt=json.loads((dest/'manifest.json').read_text())
    assert receipt['status']=='trained' and not receipt['test_used']
    last=torch.load(dest/'last.pt',map_location='cpu',weights_only=True)
    assert last['epoch']==2 and len(last['history'])==2
    before=(dest/'model.pt').read_bytes()
    train_job(root,out,cache,records,c,pd.DataFrame(rows),pd.DataFrame(roles),task,'cuda:0')
    assert (dest/'model.pt').read_bytes()==before


def test_readiness_rejects_changed_identity_indices_and_files(tmp_path):
    import numpy as np
    from reference import file_digest
    from looped_video.readiness import validate_entry
    p=tmp_path/'sample.npy';np.save(p,np.zeros((2,196,1024),dtype=np.float32))
    st=p.stat();job=dict(key='sample',scope='development',windows=[[3,7]])
    r=dict(key='sample',scope='development',identity='producer',sha256=file_digest(p),
        bytes=st.st_size,mtime_ns=st.st_mtime_ns,frame_indices=[3,7],window_positions=[[0,1]],
        shape=[2,196,1024],global_exact=True)
    validate_entry(tmp_path,job,r,{'producer'})
    for bad in [dict(r,identity='other'),dict(r,window_positions=[[1,0]]),dict(r,global_exact=False),dict(r,sha256='')]:
        with pytest.raises(ValueError):validate_entry(tmp_path,job,bad,{'producer'})
    import os
    os.utime(p,ns=(st.st_atime_ns,st.st_mtime_ns+1000000))
    with pytest.raises(ValueError,match='发生变化'):validate_entry(tmp_path,job,r,{'producer'})
