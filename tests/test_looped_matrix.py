"""循环/容量/时间顺序控制与冻结矩阵覆盖。"""
import pytest
import torch
from looped_video.model import LoopedDetector
from looped_video.run import configuration
from looped_video.training import model_arguments


def test_ten_model_matrix_and_capacity():
    c=configuration();assert len(c['variants'])==10
    expected={'looped':(256,4,1189121),'loop1':(256,1,1189121),'loop2':(256,2,1189121),
        'loop8':(256,8,1189121),'wide512':(512,4,4211201),'no_time_position':(256,4,1189121),
        'untied':(256,4,3565313)}
    for name,(width,loops,n) in expected.items():
        args=model_arguments(c,name);m=LoopedDetector(**args)
        assert m.width==width and m.loops==loops and sum(p.numel() for p in m.parameters())==n
    assert len(c['folds'])*len(c['seeds'])*len(c['variants'])==120


def test_no_time_position_is_invariant_to_joint_frame_permutation():
    torch.manual_seed(4)
    a=LoopedDetector(input_dim=12,width=24,heads=4,variant='no_time_position')
    x=torch.randn(2,4,16,12);valid=torch.tensor([[1,1,0,0],[1,1,1,1]],dtype=torch.bool)
    ix=torch.tensor([2,0,3,1])
    torch.testing.assert_close(a(x,valid),a(x[:,ix],valid[:,ix]),rtol=1e-5,atol=1e-6)
    ordered=LoopedDetector(input_dim=12,width=24,heads=4,variant='looped');ordered.load_state_dict(a.state_dict())
    assert (ordered(x,valid)-ordered(x[:,ix],valid[:,ix])).abs().max()>1e-5


@pytest.mark.parametrize('loops',[1,2,8])
def test_requested_depth_reuses_parameters_and_receives_gradients(loops):
    m=LoopedDetector(input_dim=12,width=24,heads=4,loops=loops)
    calls=[];hook=m.blocks[0].register_forward_hook(lambda *args:calls.append(1))
    m(torch.randn(2,4,16,12),torch.ones(2,4,dtype=torch.bool)).sum().backward();hook.remove()
    assert len(calls)==loops and len(m.blocks)==1
    assert m.blocks[0].read.q.weight.grad.abs().sum()>0
