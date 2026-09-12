import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
import numpy as np
from evaluation.operating_points import real_threshold,wilson
from evaluation.metrics import binary_metrics
import pandas as pd


def test_ties_do_not_exceed_real_fpr_budget():
    for scores in (np.arange(2000),np.repeat(np.arange(20),100),np.zeros(2000)):
        for fpr in (.001,.01):
            tau=real_threshold(scores,fpr)
            assert np.mean(scores<tau)<=fpr


def test_zero_errors_still_have_uncertainty():
    low,high=wilson(0,85)
    assert abs(low)<1e-12 and high>0.04


def test_collinear_roc_thresholds_are_not_discarded():
    scores=np.arange(2000,dtype=float)/2000
    frame=pd.DataFrame(dict(subset=['real']*2000+['annotated']*2000,final_score=np.r_[scores,scores]))
    metrics=binary_metrics(frame)
    assert metrics['fake_tpr_at_0_1pct_real_fpr']==.001
    assert metrics['fake_tpr_at_1pct_real_fpr']==.01
    assert metrics['real_fpr_at_95pct_fake_tpr']==.95
