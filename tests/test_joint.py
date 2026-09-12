"""联合逐对聚合与条件Gaussian分解，不把窗口min直接相加。"""
import torch
from statistical_experts.joint_study import aggregate_joint


def test_joint_aggregation_order_and_mask():
    x=torch.tensor([[-1.,-5.],[-1.,-2.]]);y=torch.tensor([[-5.,-1.],[-2.,-1.]])
    valid=torch.tensor([[True,True],[False,False]])
    q=aggregate_joint(x,y,valid)
    assert q[0]==-6. and x[0].min()+y[0].min()==-10.
    assert torch.isposinf(q[1])


def test_gaussian_chain_rule_for_joint_block_covariance():
    torch.manual_seed(7);d=4;b=torch.randn(d,d,dtype=torch.float64)*.2
    cx=torch.eye(d,dtype=torch.float64)*2;rr=torch.eye(d,dtype=torch.float64)*.7
    cov=torch.cat([torch.cat([cx,cx@b],1),torch.cat([b.T@cx,rr+b.T@cx@b],1)],0)
    xy=torch.randn(5,2*d,dtype=torch.float64);x,y=xy[:,:d],xy[:,d:]
    joint=torch.distributions.MultivariateNormal(torch.zeros(2*d,dtype=torch.float64),covariance_matrix=cov).log_prob(xy)
    marginal=torch.distributions.MultivariateNormal(torch.zeros(d,dtype=torch.float64),covariance_matrix=cx).log_prob(x)
    cond=torch.distributions.MultivariateNormal(x@b,covariance_matrix=rr).log_prob(y)
    torch.testing.assert_close(joint,marginal+cond)
