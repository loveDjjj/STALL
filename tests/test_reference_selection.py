"""新主线数值合同；旧算法结果冻结为fixture，不依赖历史代码。"""
from pathlib import Path
import sys
import numpy as np
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from selection import feature_change_windows, uniform_windows
from reference import ReferenceBundle, SCHEMA, fit_gaussian, local_video_cdfs, percentile, save_bundle, load_bundle
from math_utils import StableGaussianParams


def test_selection_matches_previous():
    expected={16:([0],[4.9744415283203125]),17:([0,1],[4.282526016235352]*2),
        20:([0,4],[2.7412118911743164]*2),31:([8,12,15],[4.686454772949219,4.686454772949219,4.227546215057373]),
        48:([8,12,0],[4.155154228210449,4.155154228210449,4.090712547302246]),
        93:([48,52,64],[3.992065906524658,3.992065906524658,3.9449479579925537])}
    rng=np.random.default_rng(17)
    for n in (16,17,20,31,48,93):
        indices=list(range(0,3*n,3));features=rng.normal(size=(len(indices[::8]),7)).astype(np.float32)
        after=feature_change_windows(indices,features)
        assert [w.start_position for w in after]==expected[n][0]
        assert [w.change for w in after]==expected[n][1]
        starts=list(dict.fromkeys(np.rint(np.linspace(0,n-16,3)).astype(int)))
        assert [list(w.frame_indices) for w in uniform_windows(indices)]==[indices[s:s+16] for s in starts]


def test_reference_cdf_and_roundtrip(tmp_path):
    cdfs=local_video_cdfs([[0.,2.,4.],[2.,4.,6.]])
    np.testing.assert_array_equal(cdfs[1],[0.,2.])
    np.testing.assert_array_equal(cdfs[2],[2.,4.])
    gs=StableGaussianParams(np.zeros(2),np.eye(2),np.array([0.,2.]))
    gt=StableGaussianParams(np.zeros(2),np.eye(2),np.array([0.,np.inf]))
    bundle=ReferenceBundle(gs,gt,gs,cdfs,dict(schema=SCHEMA))
    path=tmp_path/'reference.npz';save_bundle(path,bundle)
    result=load_bundle(path).score_video([1.],[np.inf],[2.])
    assert result['global_score']==.75 and result['local_score']==1. and result['final_score']==.875
    with pytest.raises(ValueError):percentile([np.nan],[0.,1.])


def test_balanced_fit_preserves_covariance():
    rng=np.random.default_rng(29);videos=[rng.normal(size=(8,4)) for _ in range(3)]
    params=fit_gaussian(videos)
    expected=np.cov(np.concatenate(videos).T)+1e-5*np.eye(4)
    np.testing.assert_allclose(params.whitening@params.whitening.T,np.linalg.inv(expected),rtol=1e-12,atol=1e-12)


def test_observation_contract_rejects_wrong_reference():
    from workflow import VideoScorer
    class Reference:
        metadata={}
    scorer=VideoScorer.__new__(VideoScorer)
    scorer.reference=Reference()
    with pytest.raises(ValueError,match='参考包'):
        scorer.score('unused.mp4',list(range(32)),selector='uniform')


def test_window_mean_preserves_legacy_group_reduction():
    import pandas as pd
    from reference import window_mean
    rng=np.random.default_rng(17)
    different=False
    for _ in range(100):
        values=rng.integers(0,2001,3)/2000.
        expected=pd.DataFrame({'video':['v']*3,'score':values}).groupby('video').score.mean().iloc[0]
        assert window_mean(values)==expected
        different |= float(np.mean(values))!=expected
    assert different
