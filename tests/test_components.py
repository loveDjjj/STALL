"""组件移除只改变既定融合，配对和源分支尺度保持不变。"""
import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from evaluation.tables import component_ablation_tables


def test_fixed_component_equations_and_pairs():
    frame=pd.DataFrame([dict(video_id='r',dataset='test',subset='real',source_model='real',
        global_score=.5,global_spatial_score=.1,global_temporal_score=.9,local_score=.5,final_score=.5),
        dict(video_id='f',dataset='test',subset='annotated',source_model='g',
        global_score=.4,global_spatial_score=.8,global_temporal_score=0.,local_score=.4,final_score=.4)])
    pairs=frame[['video_id','dataset','subset']].assign(generator='g')
    tables=component_ablation_tables(frame,pairs)
    videos=tables['video_scores'].set_index(['variant','video_id'])
    assert videos.loc[('without_gs','r'),'final_score']==.7
    assert videos.loc[('without_gt','f'),'final_score']==pytest.approx(.6)
    assert len(tables['generator_metrics'])==5
    metrics=tables['generator_metrics'].set_index('variant')
    assert metrics.loc['full','auc']==1 and metrics.loc['without_gt','auc']==0
    broken=frame.copy();broken.loc[0,'final_score']=.51
    with pytest.raises(ValueError,match='等权'):component_ablation_tables(broken,pairs)
    with pytest.raises(ValueError,match='缺少评分'):component_ablation_tables(frame.iloc[:1],pairs)
