"""新参考来源、分块恢复和窗口匹配的关键边界。"""
import json
import numpy as np
import pytest
from config import config_digest
from evaluation.confirmation_data import media_key,source_group,order
from evaluation.confirmation_engine import Chunks


def test_source_id_catches_multiple_msvd_clips_and_vatex_overlap():
    a='datasets/comgenvid/real/MSVD/abcdefghijk_1_4.mp4';b='datasets/comgenvid/real/MSVD/abcdefghijk_8_12.mp4'
    assert source_group('comgenvid',a)==source_group('comgenvid',b)
    assert media_key('comgenvid',a)==media_key('vatex','datasets/vatex_bank/abcdefghijk.mp4')
    assert order(17,'comgenvid',a)!=order(29,'comgenvid',a)


def test_chunks_roundtrip_infinity_and_reject_corruption(tmp_path):
    x=Chunks(tmp_path,'id');x.add(dict(key='a',score=float('inf')));x.add(dict(key='b',score=1.5));x.flush()
    y=Chunks(tmp_path,'id');assert float(y.records['a']['score'])==float('inf')
    assert y.records['b']['score']==1.5
    with pytest.raises(ValueError):y.add(dict(key='a',score=4))
    with pytest.raises(ValueError):Chunks(tmp_path,'other')
    p=tmp_path/'000000.json';v=json.loads(p.read_text());v['records'][1]['score']=2.;p.write_text(json.dumps(v))
    with pytest.raises(ValueError):Chunks(tmp_path,'id')


def test_shared_slot_window_layout_has_no_tail_leak():
    rng=np.random.default_rng(9);unique=rng.normal(size=(24,4,7)).astype('float32');slot=np.full((3,16,4,7),np.nan,dtype='float32')
    picks=[list(range(8)),list(range(8,16))]
    for i,ix in enumerate(picks):np.take(unique,ix,axis=0,out=slot[i,:8],mode='clip')
    actual=np.ascontiguousarray(slot[:2,:8]);expected=np.stack([unique[ix] for ix in picks])
    np.testing.assert_array_equal(actual,expected)
    np.testing.assert_array_equal(actual[:,2:]-2*actual[:,1:-1]+actual[:,:-2],expected[:,2:]-2*expected[:,1:-1]+expected[:,:-2])
