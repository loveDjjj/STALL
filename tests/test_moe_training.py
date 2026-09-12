"""预热必须与均匀专家计算等价，且按时解冻产生路由梯度。"""
import torch
from discriminative_moe.models import Classifier
from moe_training.training import set_epoch


def test_warmup_matches_uniform_and_unfreezes():
    torch.manual_seed(17);u=Classifier(8,'uniform',2,16,0.)
    torch.manual_seed(17);w=Classifier(8,'moe',2,16,0.)
    set_epoch(w,'warm',1,10)
    x=torch.randn(4,3,8);m=torch.ones(4,3,dtype=torch.bool);y=torch.tensor([0.,1.,1.,0.])
    a=u(x,m);b=w(x,m);torch.testing.assert_close(a['p_fake'],b['p_fake'],rtol=0,atol=0)
    u.loss(a,y).backward();w.loss(b,y).backward()
    for a,b in zip(u.experts.parameters(),w.experts.parameters()):torch.testing.assert_close(a.grad,b.grad,rtol=0,atol=0)
    assert all(p.grad is None for p in w.router.parameters())
    w.zero_grad();set_epoch(w,'warm',11,10);w.loss(w(x,m),y).backward()
    assert w.router.weight.grad.abs().sum()>0


def test_joint_router_enabled_from_start():
    m=Classifier(8,'moe',2,16,0.)
    set_epoch(m,'joint',1,10)
    assert all(p.requires_grad for p in m.router.parameters())
