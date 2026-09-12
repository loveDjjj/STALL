"""按源选片段及分组留出，保证统计预算不受重复片段或输入行序影响。"""
import numpy as np
import pandas as pd
from statistical_experts.source_data import select_sources


def test_one_clip_per_source_and_stable_folds():
    f=pd.DataFrame([dict(source_group=f's{i}',legacy_video_id=f'v{i}_{j}',subset='real') for i in range(13) for j in range(1+i%3)])
    a=select_sources(f);b=select_sources(f.sample(frac=1,random_state=4))
    pd.testing.assert_frame_equal(a,b)
    assert a.source_group.nunique()==len(a)==13
    assert a.legacy_video_id.str.endswith('_0').all()
    assert set(a.fold)==set(range(5)) and a.fold.value_counts().max()-a.fold.value_counts().min()<=1
    for k in range(5):assert not set(a.loc[a.fold==k,'source_group'])&set(a.loc[a.fold!=k,'source_group'])


def test_nll_contains_logdet_and_video_average():
    import torch
    from statistical_experts.gaussian import fitted
    from statistical_experts.source_study import nll_per_video
    torch.manual_seed(5);x=torch.randn(40,4,dtype=torch.float64);q=torch.randn(3,7,4,dtype=torch.float64)
    model=fitted(x)
    distribution=torch.distributions.MultivariateNormal(model['mean'],scale_tril=model['chol'])
    expected=-distribution.log_prob(q).mean(1)
    torch.testing.assert_close(nll_per_video(q,model),expected,rtol=1e-13,atol=1e-13)
