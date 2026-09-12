"""短片不能通过重复帧、空粗差分或不一致预算进入正式实验。"""
import pytest

from evaluation.short_video import ShortWindowMixin


def test_short_window_same_observation_for_all_budgets():
    scorer=ShortWindowMixin()
    ids=list(range(0,32,4))
    plans=[]
    for selector,k in [('uniform',1),('uniform',3),('feature_change',1),('feature_change',3)]:
        prepared=scorer.prepare_coarse('unused',ids,selector=selector,k=k)
        plan=scorer.plan_from_prepared(ids,prepared,selector=selector,k=k)
        assert plan['coarse_count']==0
        assert len(plan['windows'])==1
        plans.append(plan['windows'][0].frame_indices)
    assert plans==[tuple(ids)]*4


@pytest.mark.parametrize('ids',[list(range(7)),list(range(16)),[0,1,2,3,4,5,6,6]])
def test_short_window_rejects_implicit_length_and_duplicates(ids):
    with pytest.raises(ValueError):
        ShortWindowMixin().prepare_coarse('unused',ids)


def test_official_short_is_first_random_draw():
    import numpy as np
    from evaluation.official_baseline import official_window
    indices=list(range(80))
    rng=np.random.RandomState(42)
    first=rng.randint(0,73);second=rng.randint(0,65)
    assert official_window(indices,8)==indices[first:first+8]
    assert official_window(indices)==indices[second:second+16]
