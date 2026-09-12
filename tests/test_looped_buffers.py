import numpy as np
import pandas as pd
import pytest
import torch
from looped_video.buffers import BatchBuffers
from looped_video.data import collate_videos


def test_persistent_slots_equal_reference_and_mask_stale_data():
    records={'a':dict(window_positions=[list(range(8)),list(range(4,12))]),
             'b':dict(window_positions=[list(range(16))])}
    raw={'a':np.random.default_rng(1).normal(size=(12,4,12)).astype('float32'),
         'b':np.random.default_rng(2).normal(size=(16,4,12)).astype('float32')}
    meta=pd.DataFrame([dict(key='a',subset='real'),dict(key='b',subset='annotated')])
    ring=BatchBuffers(2,4,12)
    ptr=ring.slots[0]['patches'].data_ptr()
    for order in ([0,1],[1],[1,0],[0]):
        ring.slots[0]['patches'].fill_(float('nan'))
        info=ring.fill(0,meta,order,raw,records);new=ring.view(info)
        expected=collate_videos([(torch.from_numpy(raw[meta.iloc[i].key][records[meta.iloc[i].key]['window_positions']]),
                                 int(meta.iloc[i].subset=='annotated'),i) for i in order])
        for key in ['valid','owners','labels']:torch.testing.assert_close(new[key],expected[key],rtol=0,atol=0)
        clean=new['patches'].masked_fill(~new['valid'][...,None,None],0)
        torch.testing.assert_close(clean,expected['patches'],rtol=0,atol=0)
        assert info['indices']==list(order) and ring.slots[0]['patches'].data_ptr()==ptr


def test_invalid_indices_cannot_be_silently_clipped():
    ring=BatchBuffers(1,4,12);meta=pd.DataFrame([dict(key='a',subset='real')])
    with pytest.raises(ValueError,match='帧位置非法'):
        ring.fill(0,meta,[0],{'a':np.zeros((8,4,12),np.float32)}, {'a':dict(window_positions=[[0,99]])})
