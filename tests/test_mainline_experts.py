"""主线专家的权重与抽样位置合同。"""
import numpy as np
from mainline_experts.models import sample_indices,weighted_moments
from reference_fit import sample_d2_positions
from mainline_experts.models import fit_experts


def test_sample_window_indices_recover_original_positions():
    a=np.arange(3*14*196)[:,None];identity='original/path.mp4'
    idx=sample_indices(identity,3)
    np.testing.assert_array_equal(sample_d2_positions(a,identity).ravel(),idx)
    assert ((idx//(14*196))<3).all()


def test_conditional_weights_do_not_reassign_full_video_mass():
    x=np.array([[0.,1.],[1.,2.],[5.,3.],[6.,5.]]);w=np.array([.5,.5,.25,.25])
    mu,c=weighted_moments(x,w)
    np.testing.assert_allclose(mu,(x*w[:,None]).sum(0)/w.sum())
    xc=x-mu;expected=(xc*w[:,None]).T@xc/(w.sum()-(w*w).sum()/w.sum())
    np.testing.assert_allclose(c,expected)


def test_M1_and_full_pool_shrinkage_are_exact_fallbacks():
    rng=np.random.default_rng(12);assets=[]
    for i in range(8):
        c=np.array([[1.,.01*i],[-1.,.01*i]]);c/=np.linalg.norm(c,axis=1,keepdims=True)
        assets.append(dict(context=c,gt=rng.normal(size=(10,4)),lt=rng.normal(size=(12,4)),
            gt_window=np.repeat([0,1],5),lt_window=np.repeat([0,1],6),source_group=f's{i}',video_id=f'v{i}'))
    for k,rho in [(1,.5),(2,1.)]:
        _,models,_=fit_experts(assets,k=k,pool_weight=rho)
        for branch,items in models.items():
            for expert in items[1:]:
                assert expert is items[0]


def test_four_cells_change_only_the_requested_branches():
    from mainline_experts.evaluation import video_values
    meta={'expected':[dict(gs=1.,gt=1.,lt=1.),dict(gs=2.,gt=2.,lt=2.)]}
    actual={'windows':[dict(route=0,gt_experts=[3.,0.],lt_experts=[3.,0.]),dict(route=1,gt_experts=[0.,3.],lt_experts=[0.,3.])]}
    refs=dict(gs=np.arange(5.),gt0=np.arange(5.),gt1=np.arange(5.),local0={2:np.arange(5.)},local1={2:np.arange(5.)})
    v=video_values(meta,actual,refs)
    np.testing.assert_allclose(v['R3']-v['R0'],v['R1']-v['R0']+v['R2']-v['R0'],rtol=0,atol=1e-15)
    assert v['R1']>v['R0'] and v['R2']>v['R0']
