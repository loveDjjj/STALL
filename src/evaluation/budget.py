"""观察预算只评分固定主线，避免重复计算无关表示消融。"""
import time
from workflow import RawVideoScorer
from math_utils import GaussianMeanCandidateScorerFloat64


class BudgetEvidenceScorer(RawVideoScorer):
    def __init__(self,gs,gt,lt,center,device,**kwargs):
        super().__init__(gs,gt,lt,device,**kwargs)
        self.local_scorer=GaussianMeanCandidateScorerFloat64([lt],center,device)

    def global_from_prepared(self,window,frames):
        raw=super().global_from_prepared(window,frames)['windows'][0]
        return dict(frame_indices=raw['frame_indices'],global_models={'target':dict(gs=raw['global_spatial_raw'],gt=raw['global_temporal_raw'])})

    def score_dense(self,plan,frames):
        start=time.perf_counter();raw=super().score_dense(plan,frames)
        return dict(selector=raw['selector'],coarse_frames=raw['coarse_frames'],dense_unique_frames=raw['dense_unique_frames'],
            dense_seconds=time.perf_counter()-start,
            windows=[dict(rank=w['rank'],start_position=w['start_position'],frame_indices=w['frame_indices'],change=w['change'],
                global_models={'target':dict(gs=w['global_spatial_raw'],gt=w['global_temporal_raw'])},
                representations={'local_d2':{'target':w['local_raw']}}) for w in raw['windows']])
