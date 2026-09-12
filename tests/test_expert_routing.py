"""验证连续精度的数学含义、聚合顺序和条件参考支持。"""
import numpy as np
import pytest
from statistical_experts.routing_study import temperature,soft_weights,combine_positions,cluster_percentile


def test_temperature_and_hard_uniform_limits():
    x=np.array([[3.,1.,0.],[2.,1.,0.],[4.,1.,0.]])
    tau,d=temperature(x);assert tau==2. and d['sources']==3
    np.testing.assert_array_equal(soft_weights(x,np.inf),np.full((3,3),1/3))
    np.testing.assert_array_equal(soft_weights(x,1e-8),np.array([[1.,0.,0.]]*3))
    tau,d=temperature(np.zeros((3,4)));assert tau==1e-6 and d['floor_used']


def test_combination_equals_precision_quadratic_not_average_window_min():
    rng=np.random.default_rng(12);u=rng.normal(size=(2,5,3));a=rng.normal(size=(4,3,3));a=a@a.transpose(0,2,1)
    w=soft_weights(rng.normal(size=(2,4)),.3)
    e=-.5*np.einsum('btd,mde,bte->btm',u,a,u)
    expected=-.5*np.einsum('btd,bde,bte->bt',u,np.einsum('bm,mde->bde',w,a),u)
    np.testing.assert_allclose(combine_positions(e,np.zeros((2,5),bool),w),expected.min(1),rtol=1e-14,atol=1e-14)
    cross=np.array([[[-1.,-5.],[-5.,-1.]]])
    assert combine_positions(cross,np.zeros((1,2),bool),np.array([[.5,.5]]))[0]==-3.
    assert cross.min(axis=1).mean()==-5.


def test_zero_transitions_and_conditional_population():
    e=np.array([[[-1.,-2.],[-3.,-4.]]]);w=np.array([[1.,0.]])
    assert np.isposinf(combine_positions(e,np.ones((1,2),bool),w)[0])
    assert combine_positions(e,np.array([[False,True]]),w)[0]==-1.
    q=np.array([2.,2.,np.inf]);r=np.array([0,1,1]);ref=np.array([1.,2.,5.,np.inf]);rr=np.array([0,0,1,1])
    np.testing.assert_array_equal(cluster_percentile(q,r,ref,rr),[1.,0.,1.])
    with pytest.raises(ValueError):cluster_percentile(q,r,ref,rr,minimum=3)
