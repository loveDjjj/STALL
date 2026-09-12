"""条件模型、打乱控制和边界的数学检查。"""
import numpy as np
import torch
from statistical_experts.predictive import fit_model,position_scores,derangement,query_scores
from statistical_experts.gaussian import fitted,energy


def test_derangement_preserves_marginals_and_excludes_same_source():
    p=derangement(31);assert np.array_equal(np.sort(p),np.arange(31));assert (p!=np.arange(31)).all()
    assert np.array_equal(p,derangement(31))


def test_zero_predictor_is_marginal_gaussian():
    torch.manual_seed(4);t=torch.randn(30,6,8,dtype=torch.float64);q=torch.randn(3,6,8,dtype=torch.float64)
    m=fit_model(t,'marginal');base=fitted(t[:,1:].flatten(0,1))
    torch.testing.assert_close(position_scores(q,m),energy(q[:,1:],base['mean'],base['chol']),rtol=1e-12,atol=1e-12)


def test_temporal_relation_is_learned_and_shuffling_destroys_it():
    torch.manual_seed(5);t=torch.empty(400,7,4,dtype=torch.float64);t[:,0]=torch.randn(400,4,dtype=torch.float64)
    for k in range(1,7):t[:,k]=.8*t[:,k-1]+.15*torch.randn(400,4,dtype=torch.float64)
    model=fit_model(t,'predictive');shuffled=fit_model(t,'shuffled')
    assert (model['B']-.8*torch.eye(4)).norm()<.1
    assert shuffled['B'].norm()<.2


def test_all_three_score_identical_valid_positions():
    torch.manual_seed(6);models={k:fit_model(torch.randn(30,7,4,dtype=torch.float64),k) for k in ('marginal','predictive','shuffled')}
    g=torch.randn(2,8,4);g[0]=1.;g[1,3]=g[1,2]
    scores,count=query_scores(g,models,'cpu');assert count.tolist()==[0,4]
    assert all(np.isposinf(s[0]) and np.isfinite(s[1]) for s in scores.values())
