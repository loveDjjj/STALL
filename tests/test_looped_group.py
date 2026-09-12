"""独立与共用批次的数值等价；真实双GPU覆盖五变体及已有模型接续。"""
import json
from pathlib import Path
import numpy as np
import pandas as pd
import pytest
import torch
from looped_video.model import LoopedDetector,VARIANTS


def test_models_have_disjoint_parameters():
    models=[LoopedDetector(input_dim=12,width=24,heads=4,variant=v) for v in VARIANTS]
    seen=set()
    for model in models:
        pointers={p.data_ptr() for p in model.parameters()}
        assert not seen&pointers;seen|=pointers


def compare_nested(a,b):
    if isinstance(a,torch.Tensor):torch.testing.assert_close(a,b,rtol=2e-5,atol=2e-6)
    elif isinstance(a,dict):
        assert a.keys()==b.keys()
        for k in a:compare_nested(a[k],b[k])
    elif isinstance(a,(list,tuple)):
        assert len(a)==len(b)
        for x,y in zip(a,b):compare_nested(x,y)
    else:assert a==b


@pytest.mark.skipif(torch.cuda.device_count()<2,reason='需要两GPU')
def test_group_equivalence_and_catch_up(tmp_path):
    import shutil
    from looped_video.ddp_train import run_task
    from looped_video.group_train import run_group
    from looped_video.stream import SharedReader
    root=Path(__file__).resolve().parents[1];cache=tmp_path/'cache';cache.mkdir()
    (cache/'manifest.json').write_text('{}');records={};rows=[];roles=[]
    for i in range(25):
        p=cache/f'{i}.npy';np.save(p,np.random.default_rng(i).normal(size=(8,4,1024)).astype('float32'));s=p.stat()
        records[str(i)]=dict(bytes=s.st_size,mtime_ns=s.st_mtime_ns,window_positions=[list(range(8))])
        rows.append(dict(video_id=str(i),key=str(i),dataset='comgenvid',subset='real' if i%2==0 else 'annotated',
            source_model='real' if i%2==0 else 'fake',source_group=str(i),split_group=str(i),length=8))
        if i<21:roles.append(dict(fold='pooled',video_id=str(i),role='train' if i<17 else 'validation'))
    meta=pd.DataFrame(rows[:21]);fm=pd.DataFrame(rows[21:]).reset_index(drop=True);roles=pd.DataFrame(roles)
    pairs=fm[['video_id','dataset','subset']].assign(generator='fake')
    c=dict(cache_directory=str(cache),width=24,heads=4,loops=4,tile=2,mlp_ratio=2,learning_rate=.0003,
        weight_decay=.01,epochs=2,warmup_epochs=1,amp=True)
    rt=dict(world_size=2,local_batch=8,global_batch=32,ram_gib=1,reader_workers=2,checkpoint_updates=1)
    independent=tmp_path/'independent';grouped=tmp_path/'grouped'
    for out in [independent,grouped]:out.mkdir();(out/'prepared.json').write_text('{}')
    names=list(VARIANTS)+['loop1','loop2','loop8','wide']
    c['variants']=names;c['bootstrap_replicates']=10
    c['experiment_overrides']={'loop1':dict(variant='looped',loops=1),'loop2':dict(variant='looped',loops=2),
        'loop8':dict(variant='looped',loops=8),'wide':dict(variant='looped',width=48,heads=4)}
    tasks=[dict(key=f'pooled__{v}__s17',fold='pooled',variant=v,seed=17) for v in names]
    reader=SharedReader(cache,records,1,2)
    try:
        for task in tasks[:2]:run_task(root,independent,c,task,meta,roles,reader,rt)
        # 模拟已有checkpoint；不重复已经完成的looped更新，其他四个从同一seed初始化补齐。
        dest=grouped/'training'/tasks[0]['key'];dest.parent.mkdir(parents=True)
        shutil.copytree(independent/'training'/tasks[0]['key'],dest)
        (dest/'manifest.json').unlink()
        run_group(root,grouped,c,tasks,meta,roles,reader,rt,fm,pairs)
    finally:reader.close()
    for task in tasks[:2]:
        a=torch.load(independent/'training'/task['key']/'last.pt',map_location='cpu',weights_only=True)
        b=torch.load(grouped/'training'/task['key']/'last.pt',map_location='cpu',weights_only=True)
        compare_nested(a['model'],b['model']);compare_nested(a['optimizer'],b['optimizer'])
        assert [h['validation_objective'] for h in a['history']]==[h['validation_objective'] for h in b['history']]
    for task in tasks:
        r=json.loads((grouped/'training'/task['key']/'manifest.json').read_text())
        assert not r['test_used'] and (grouped/'evaluation'/task['key']/'scores.csv').exists()
    from looped_video.postprocess import process_group
    process_group(root,grouped,c,tasks,fm,pairs)
    assert json.loads((grouped/'groups/pooled__s17/postprocess.json').read_text())['status']=='passed'
