"""专家统计的数学等价、路由与随机控制，不镜像模型实现结果。"""
import numpy as np
import torch
from statistical_experts.gaussian import fitted,energy,score_pair,context,transitions,spherical_kmeans,neighbor_indices,random_indices


def test_gaussian_energy_equals_direct_quadratic_and_m1():
    torch.manual_seed(8);x=torch.randn(40,8,dtype=torch.float64);q=torch.randn(3,5,8,dtype=torch.float64)
    model=fitted(x);shrunk=fitted(x,model['covariance'],.5)
    assert torch.equal(model['mean'],shrunk['mean']) and torch.equal(model['chol'],shrunk['chol'])
    covariance=model['covariance']+1e-5*torch.eye(8,dtype=torch.float64)
    r=q-model['mean'];expected=-.5*(8*np.log(2*np.pi)+torch.einsum('btd,de,bte->bt',r,torch.linalg.inv(covariance),r))
    torch.testing.assert_close(energy(q,model['mean'],model['chol']),expected,rtol=1e-10,atol=1e-10)


def test_t1_matches_official_numpy_and_static_boundary():
    torch.manual_seed(9);g=torch.randn(2,8,8);g[0,3]=g[0,2]
    diff=np.diff(g.numpy(),axis=1);norm=np.linalg.norm(diff,axis=-1,keepdims=True)
    expected=diff/np.where(norm==0,1.,norm)
    t,zero=transitions(g);np.testing.assert_array_equal(t.numpy(),expected.astype(np.float64))
    assert zero[0,2]
    model=fitted(torch.randn(40,8,dtype=torch.float64))
    s=score_pair(torch.ones(2,8,8),model,model)
    assert torch.isfinite(s[:,0]).all() and torch.isposinf(s[:,1]).all()


def test_context_ignores_order_but_not_temporal_observations():
    torch.manual_seed(10);g=torch.randn(8,16)
    torch.testing.assert_close(context(g),context(g.flip(0)))
    assert not torch.equal(transitions(g)[0],transitions(g.flip(0))[0])


def test_knn_ties_and_random_neighbors_are_deterministic():
    fit=torch.tensor([[1.,0.],[1.,0.],[0.,1.],[-1.,0.]],dtype=torch.float64)
    q=torch.tensor([[1.,0.]],dtype=torch.float64)
    assert neighbor_indices(q,fit,2).tolist()==[[0,1]]
    a=random_indices(['a'*64,'b'*64],100,20,17)
    assert np.array_equal(a,random_indices(['a'*64,'b'*64],100,20,17))
    assert all(len(set(row))==20 for row in a)
    assert not np.array_equal(a[0],a[1])


def test_spherical_groups_depend_only_on_fit_content():
    x=torch.tensor([[1.,.01],[1.,-.01],[-1.,.01],[-1.,-.01]],dtype=torch.float64);x=x/x.norm(dim=1,keepdim=True)
    centers,labels,_=spherical_kmeans(x,2,17)
    assert labels[0]==labels[1] and labels[2]==labels[3] and labels[0]!=labels[2]
    torch.testing.assert_close(centers.norm(dim=1),torch.ones(2,dtype=torch.float64))
