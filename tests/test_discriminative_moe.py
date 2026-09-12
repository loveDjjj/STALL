"""监督协议的数学摘要与源级隔离；不是实现镜像或性能测试。"""
import numpy as np
import pandas as pd
import torch
from discriminative_moe.features import moments, projection, global_descriptors, local_descriptors
from discriminative_moe.data import grouped_roles
from discriminative_moe.models import Classifier, training_standardizer
from discriminative_moe.training import sampling_weights, validation_pairs
from discriminative_moe.evaluation import choose_capacity
from discriminative_moe.analysis import RankedMetrics
from discriminative_moe.audit import independent_probability
from sklearn.metrics import roc_auc_score, average_precision_score


def test_moments_preserve_cross_channel_information():
    a=torch.tensor([[1.,1.],[-1.,-1.]])
    b=torch.tensor([[1.,-1.],[-1.,1.]])
    np.testing.assert_array_equal(moments(a)[:4],moments(b)[:4])
    assert moments(a,torch.eye(2))[-1]==1
    assert moments(b,torch.eye(2))[-1]==-1
    np.testing.assert_allclose(projection(8,4).T@projection(8,4),np.eye(4),atol=1e-6)


def test_static_features_and_d2_are_finite_without_fake_observations():
    g=np.ones((1,8,4),np.float32)
    a,t=global_descriptors(g,projection(4,2))
    assert a.shape==(1,8) and t.shape==(1,9) and np.count_nonzero(t)==0
    p=torch.ones((1,8,3,4));s=local_descriptors(p,projection(4,2))
    assert s.shape==(1,9) and torch.count_nonzero(s)==0


def test_grouped_split_keeps_crops_and_cross_domain_sources_together():
    rows=[]
    for d in ['a','b','c']:
        for label in ['real','annotated']:
            for i in range(10):
                for crop in ['short','long']:
                    rows.append(dict(video_id=f'{d}/{label}/{i}/{crop}',dataset=d,subset=label,source_model=label,
                        split_group=f'{d}/{label}/{i}'))
    f=pd.DataFrame(rows);f.loc[f.video_id.str.startswith('b/real/0/'),'split_group']='a/real/0'
    s=grouped_roles(f)
    for fold,q in s.groupby('fold'):
        allowed=q[q.role!='excluded_shared_source']
        assert allowed.groupby('split_group').role.nunique().max()==1
    assert s[(s.fold=='a') & s.video_id.str.startswith('b/real/0/')].role.eq('excluded_shared_source').all()
    pd.testing.assert_frame_equal(s,grouped_roles(f))


def test_probability_mixture_and_padding_do_not_change_video_score():
    torch.manual_seed(17)
    m=Classifier(4,'moe',experts=2,hidden_budget=8,dropout=0).eval()
    x=torch.randn(2,3,4);mask=torch.tensor([[True,False,False],[True,True,True]])
    y=m(x,mask)
    expected=(y['expert_probability'].mean(-1)*mask).sum(1)/mask.sum(1)
    torch.testing.assert_close(y['p_fake'],expected)
    altered=x.clone();altered[0,1:]=10000
    torch.testing.assert_close(m(altered,mask)['p_fake'],y['p_fake'])
    torch.testing.assert_close(y['log_fake'].exp()+y['log_real'].exp(),torch.ones(2))
    loss=m.loss(y,torch.tensor([0.,1.]));loss.backward()
    assert m.router.weight.grad is not None and m.router.weight.grad.abs().sum()>0


def test_standardization_weights_video_instead_of_window_count():
    x=torch.tensor([[[1.],[900.],[900.]],[[3.],[3.],[3.]]])
    mask=torch.tensor([[True,False,False],[True,True,True]])
    mean,scale=training_standardizer(x,mask)
    torch.testing.assert_close(mean,torch.tensor([2.]))
    torch.testing.assert_close(scale,torch.tensor([1.]))


def test_training_weights_do_not_reward_repeated_source_crops():
    f=pd.DataFrame([
        dict(dataset='a',subset='real',source_model='camera',split_group='r',video_id='r1'),
        dict(dataset='a',subset='real',source_model='camera',split_group='r',video_id='r2'),
        dict(dataset='a',subset='annotated',source_model='g1',split_group='f1',video_id='f1'),
        dict(dataset='a',subset='annotated',source_model='g2',split_group='f2',video_id='f2')])
    f['length']=16
    np.testing.assert_allclose(sampling_weights(f),[.25,.25,.25,.25])


def test_training_and_validation_match_observation_length():
    rows=[]
    for length in (8,16):
        for label in ('real','annotated'):
            for i in range(3 if label=='annotated' else 5):
                rows.append(dict(video_id=f'{length}/{label}/{i}',dataset='a',subset=label,
                    source_model='camera' if label=='real' else f'generator{length}',split_group=f'{label}/{i}',length=length))
    f=pd.DataFrame(rows);f['weight']=sampling_weights(f)
    totals=f.groupby(['length','subset']).weight.sum()
    np.testing.assert_allclose(totals,[.25,.25,.25,.25])
    pairs=validation_pairs(f);lengths=f.set_index('video_id').loc[pairs.video_id,'length']
    np.testing.assert_array_equal(lengths,pairs.length)
    counts=pairs.groupby(['generator','length','subset']).size().unstack()
    np.testing.assert_array_equal(counts['real'],counts['annotated'])
    extra=f[(f.subset=='annotated')&(f.length==16)].copy()
    extra['video_id']=extra.video_id+'extra';extra['source_model']='third_generator';extra['split_group']=extra.split_group+'extra'
    more=pd.concat([f,extra],ignore_index=True);more['weight']=sampling_weights(more)
    np.testing.assert_allclose(more[more.subset=='annotated'].groupby('source_model').weight.sum(),[1/6]*3)
    totals=more.groupby(['length','subset']).weight.sum().unstack()
    np.testing.assert_allclose(totals['real'],totals['annotated'])


def test_capacity_rule_prefers_small_within_seed_variation():
    n,_=choose_capacity({2:[.8,.81,.79],4:[.801,.811,.791]})
    assert n==2
    n,_=choose_capacity({2:[.8,.81,.79],4:[.83,.84,.82]})
    assert n==4


def test_ranked_bootstrap_matches_sklearn_with_ties_and_zero_weights():
    rng=np.random.default_rng(17)
    y=np.array([False,True]*20);scores=rng.integers(0,5,len(y))/4
    ranked=RankedMetrics(y,scores)
    for _ in range(20):
        w=rng.poisson(1,len(y))
        expected=[roc_auc_score(y,scores,sample_weight=w),average_precision_score(y,scores,sample_weight=w)]
        np.testing.assert_allclose(ranked(w),expected,rtol=0,atol=1e-12)


def test_independent_numpy_forward_matches_all_classifier_types():
    rng=np.random.default_rng(17);x=rng.normal(size=(3,3,8)).astype(np.float32)
    mask=np.array([[1,0,0],[1,1,0],[1,1,1]],dtype=bool)
    for kind,n in [('linear',1),('mlp',1),('uniform',2),('moe',4)]:
        torch.manual_seed(17);m=Classifier(8,kind,n,hidden_budget=16,dropout=.1).eval()
        if m.router is not None:
            with torch.no_grad():m.router.weight.normal_(0,.1)
        state=dict(model=m.state_dict(),kind=kind,experts=n,mean=torch.zeros(8),scale=torch.ones(8))
        actual=m(torch.from_numpy(x),torch.from_numpy(mask))['p_fake'].detach().numpy()
        np.testing.assert_allclose(independent_probability(state,x,mask),actual,rtol=0,atol=1e-6)
