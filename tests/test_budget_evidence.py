import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
import numpy as np
from selection import Window
from math_utils import StableGaussianParams,GaussianMeanCandidateScorerFloat64
from evaluation.budget import BudgetEvidenceScorer
from evaluation.evidence import SharedEvidenceScorer
from evaluation.representations import RepresentationScorers


def test_budget_path_matches_shared_target_evidence():
    rng=np.random.default_rng(17);g=rng.normal(size=(16,5)).astype('float32');p=rng.normal(size=(16,3,5)).astype('float32')
    target=StableGaussianParams(rng.normal(size=5),np.eye(5),np.empty(0))
    source=StableGaussianParams(rng.normal(size=5),np.eye(5),np.empty(0))
    class Extractor:
        def frames_to_global_patch_embeddings(self,*args,**kwargs):return [dict(global_=g,patch=p,**{'global':g})]
    plan=dict(windows=[Window(0,tuple(range(16)))],union=list(range(16)),coarse_count=2,selector='feature_change')
    budget=BudgetEvidenceScorer.__new__(BudgetEvidenceScorer)
    budget.gs=target;budget.gt=target;budget.device='cpu';budget.extractor=Extractor()
    budget.local_scorer=GaussianMeanCandidateScorerFloat64([target],source.mean,'cpu')
    shared=SharedEvidenceScorer.__new__(SharedEvidenceScorer);shared.device='cpu';shared.extractor=Extractor()
    shared.global_models={'target':(target,target)}
    shared.representations=RepresentationScorers({'local_d2':{'source':source,'target':target}},'cpu')
    a=budget.score_dense(plan,None)['windows'][0];b=shared.score_dense(plan,None)['windows'][0]
    assert a['global_models']==b['global_models'] and a['frame_indices']==b['frame_indices']
    np.testing.assert_allclose(a['representations']['local_d2']['target'],b['representations']['local_d2']['target'],rtol=1e-12,atol=1e-12)
