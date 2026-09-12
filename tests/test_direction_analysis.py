"""预排序加权指标需吻合官方评价库，包含并列、零权重和差值交互。"""
import numpy as np
from sklearn.metrics import roc_auc_score,average_precision_score
from evaluation.direction_analysis import RankedMetrics


def test_weighted_metrics_match_sklearn_with_ties():
    rng=np.random.default_rng(45)
    for n in (8,50,250):
        y=np.arange(n)%2==0;s=rng.integers(0,7,n).astype(float);m=RankedMetrics(y,s)
        for _ in range(30):
            w=rng.poisson(1,n)
            if not w[y].sum() or not w[~y].sum():continue
            expected=[roc_auc_score(y,s,sample_weight=w),average_precision_score(y,s,sample_weight=w)]
            np.testing.assert_allclose(m(w),expected,atol=2e-15,rtol=0)


def test_shared_source_contrasts_match_existing_bootstrap():
    import pandas as pd
    from evaluation.direction_analysis import shared_contrasts
    from evaluation.bootstrap import paired_source_contrast
    rows=[];pairs=[];rng=np.random.default_rng(12)
    for d in ('comgenvid','genvideo','videofeedback'):
        for i in range(8):
            vid=f'{d}:{i}';real=i<4;gen='a' if i<6 else 'b'
            rows.append(dict(video_id=vid,dataset=d,subset='real' if real else 'annotated',source_model='real' if real else gen,
                             final_score=float(rng.integers(0,5))))
        for gen,ids in [('a',[0,1,4,5]),('b',[0,1,6,7])]:
            for i in ids:pairs.append(dict(video_id=f'{d}:{i}',dataset=d,subset='real' if i<4 else 'annotated',generator=gen))
    a=pd.DataFrame(rows);b=a.copy();b.final_score=rng.integers(0,5,len(b));pairs=pd.DataFrame(pairs)
    groups=pd.Series(a.video_id.to_numpy(),index=a.video_id)
    point,ci=paired_source_contrast(a,b,pairs,groups,seed=17,iterations=30)
    expected=point.merge(ci,on=['dataset','metric']).replace({'dataset':{'Macro-3':'Average'}}).set_index(['dataset','metric'])
    actual=shared_contrasts({'a':a,'b':b,'a_copy':a.copy(),'b_copy':b.copy()},pairs,groups,
        {'simple':{'a':1,'b':-1},'interaction':{'a':1,'b':-1,'a_copy':-1,'b_copy':1}},iterations=30)
    zero=actual[actual.contrast.eq('interaction')]
    np.testing.assert_allclose(zero[['delta','ci95_low','ci95_high']],0,atol=1e-15)
    fields=['delta','ci95_low','ci95_high']
    found=actual[actual.contrast.eq('simple')].set_index(['dataset','metric']).loc[expected.index]
    np.testing.assert_allclose(found[fields],expected[fields],atol=1e-15)
