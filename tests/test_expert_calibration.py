"""专家CDF的数学不变量，防止将换量纲或跨专家参考误作算法增益。"""
import numpy as np
import pytest
from statistical_experts.controls import expert_percentiles


def test_same_expert_reference_is_invariant_to_expert_offsets():
    refs=np.array([[0.,10.],[1.,11.],[2.,12.]])
    query=np.array([1.,11.,2.,10.]); routes=np.array([0,1,0,1])
    expected=np.array([2/3,2/3,1.,1/3])
    np.testing.assert_array_equal(expert_percentiles(query,routes,refs),expected)
    offset=np.array([500.,-900.])
    np.testing.assert_array_equal(expert_percentiles(query+offset[routes],routes,refs+offset),expected)


def test_identical_experts_collapse_to_pooled_cdf_with_ties_and_infinity():
    refs=np.array([1.,1.,2.,np.inf]); query=np.array([0.,1.,2.,np.inf])
    routes=np.array([0,1,1,0])
    expected=np.array([0.,.5,.75,1.])
    np.testing.assert_array_equal(expert_percentiles(query,routes,np.column_stack([refs,refs])),expected)


def test_invalid_router_and_scores_are_rejected():
    with pytest.raises(ValueError):expert_percentiles([1.],np.array([2]),np.ones((3,2)))
    with pytest.raises(ValueError):expert_percentiles([np.nan],np.array([0]),np.ones((3,2)))
    with pytest.raises(ValueError):expert_percentiles([1.],np.array([0.]),np.ones((3,2)))


def test_raw_metric_order_preserves_infinity_and_ties():
    from statistical_experts.moment_controls import raw_order_scores
    raw=np.array([2.,np.inf,-4.,2.,np.inf])
    rank=raw_order_scores(raw)
    assert np.isfinite(rank).all()
    np.testing.assert_array_equal(raw[:,None]<=raw[None,:],rank[:,None]<=rank[None,:])
